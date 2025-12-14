import torch
import torch.nn as nn
import torch.optim as optim
import time
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

from utility import accuracy_from_logits, apply_masks_to_model
import wandb

from evaluation import evaluate

# ------------------------------
# MODEL CREATION
# ------------------------------
def create_model(num_classes=10, pretrained=True):
    weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = mobilenet_v2(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


# ------------------------------
# TRAIN ONE EPOCH
# ------------------------------
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0
    correct = total = 0

    for imgs, labels in dataloader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        c, t = accuracy_from_logits(outputs, labels)
        correct += c; total += t

    return running_loss/total, 100*correct/total


# ------------------------------
# BATCHNORM CALIBRATION
# ------------------------------
def recalibrate_batchnorm(model, loader, device, num_batches=20):
    model.train()
    with torch.no_grad():
        for i, (imgs, _) in enumerate(loader):
            imgs = imgs.to(device)
            model(imgs)
            if i >= num_batches:
                break
    model.eval()


# ------------------------------
# FINETUNE WITH MASKS
# ------------------------------
def finetune_with_masks(
    model, masks, train_loader, val_loader,
    device, epochs=2, lr=1e-3, weight_decay=5e-4
):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

    best_val = -1
    best_state = None
    named_params = dict(model.named_parameters())

    for ep in range(1, epochs+1):
        model.train()
        running_loss = 0
        correct = total = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()

            for pname, p in named_params.items():
                if pname in masks and p.grad is not None:
                    p.grad.data.mul_(masks[pname].to(p.grad.device))

            optimizer.step()

            with torch.no_grad():
                for pname, p in named_params.items():
                    if pname in masks:
                        p.data.mul_(masks[pname].to(p.data.device))

            running_loss += loss.item() * imgs.size(0)
            c, t = accuracy_from_logits(out, labels)
            correct += c; total += t

        train_loss = running_loss/total
        train_acc = 100*correct/total

        from evaluation import evaluate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    return best_val, best_state

def train_model(model, train_loader, val_loader, config, device,with_wandb):
    """
    Two-phase training routine:
    Phase 1: Train classifier head (freeze feature extractor)
    Phase 2: Fine-tune entire network
    
    Args:
        model: MobileNet model
        train_loader: dataloader
        val_loader: dataloader
        config: config object with hyperparameters
        device: torch device
    
    Returns:
        best_state_dict: weights with best validation accuracy
        history: dict of training logs
        best_val_acc: best validation accuracy
    """

    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0
    best_state_dict = None

    HEAD_EPOCHS = config["head_epochs"]
    TOTAL_EPOCHS = config["epochs"]

    history = {
        "epoch": [],
        "phase": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    # ------------------------------------------------------------
    # PHASE 1 — Train classifier head only
    # ------------------------------------------------------------
    print("=== Phase 1: Train classifier head only ===")

    # Freeze features
    for param in model.features.parameters():
        param.requires_grad = False

    # Optimizer for head training
    if config["optimizer"] == "SGD":
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config["learning_rate_HT"],
            momentum=config["momentum"],
            weight_decay=config["weight_decay"]
        )
    else:
        raise ValueError("Unsupported optimizer")

    head_epochs = min(HEAD_EPOCHS, TOTAL_EPOCHS)

    for epoch in range(1, head_epochs + 1):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        elapsed = time.time() - start_time

        # Logging
        history["epoch"].append(epoch)
        history["phase"].append("head")
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        if(with_wandb):
            wandb.log({
                "epoch": epoch,
                "phase": "head",
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
        })

        print(
            f"[Head Epoch {epoch}/{TOTAL_EPOCHS}] "
            f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%  "
            f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.2f}%  "
            f"Time: {elapsed:.1f}s"
        )

        # Track best validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_dict = model.state_dict()

    # ------------------------------------------------------------
    # PHASE 2 — Fine-tune all layers
    # ------------------------------------------------------------
    remaining_epochs = TOTAL_EPOCHS - head_epochs

    if remaining_epochs > 0:
        print("\n=== Phase 2: Fine-tune all layers ===")

        # Unfreeze all layers
        for param in model.features.parameters():
            param.requires_grad = True

        # Optimizer for FT
        if config["optimizer"] == "SGD":
            optimizer = optim.SGD(
                model.parameters(),
                lr=config["learning_rate_FT"],
                momentum=config["momentum"],
                weight_decay=config["weight_decay"]
            )

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=remaining_epochs
        )

        for e in range(1, remaining_epochs + 1):
            epoch = head_epochs + e
            start_time = time.time()

            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            elapsed = time.time() - start_time

            scheduler.step()

            # Logging
            history["epoch"].append(epoch)
            history["phase"].append("finetune")
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            
            if(with_wandb):
                wandb.log({
                    "epoch": epoch,
                    "phase": "finetune",
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                })

            print(
                f"[FT Epoch {epoch}/{TOTAL_EPOCHS}] "
                f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%  "
                f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.2f}%  "
                f"Time: {elapsed:.1f}s"
            )

            # Track best validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state_dict = model.state_dict()

    print(f"\nBest Validation Accuracy = {best_val_acc:.2f}%")
    #wandb.run.summary["final_val_accuracy"] = best_val_acc

    return best_state_dict, history, best_val_acc




