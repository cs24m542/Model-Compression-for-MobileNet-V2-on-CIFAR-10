import torch
import torch.nn as nn
from utility import accuracy_from_logits

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0
    correct = total = 0

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            loss = criterion(out, labels)

            running_loss += loss.item() * imgs.size(0)
            c, t = accuracy_from_logits(out, labels)
            correct += c; total += t

    return running_loss/total, 100*correct/total

