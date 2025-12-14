import torch
import torch.nn as nn
import torch.optim as optim
import time
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

from utility import accuracy_from_logits, apply_masks_to_model

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

