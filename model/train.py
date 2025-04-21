# model/train.py
import os
import zipfile
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import models
from torch.utils.data import random_split, DataLoader

import kagglehub
from dataset import BrainTumorDataset, transform  # your existing class

# 1) Download & extract via Kagglehub
def prepare_dataset() -> str:
    # This will download (or use cached) brain‑tumor‑mri‑scans.zip
    zip_path = kagglehub.dataset_download("rm1000/brain-tumor-mri-scans")
    # Define the extraction root
    cache_root = os.path.expanduser("~/.cache/kagglehub/datasets/rm1000/brain-tumor-mri-scans/versions/1")
    if not os.path.isdir(cache_root):
        os.makedirs(cache_root, exist_ok=True)
        print(f"Extracting {zip_path} → {cache_root} …")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(cache_root)
    else:
        print(f"Using cached dataset at {cache_root}")
    return cache_root

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # download / cache / extract
    data_root = prepare_dataset()

    # 2) Load dataset
    full_ds = BrainTumorDataset(root_dir=data_root, transform=transform)
    total = len(full_ds)
    train_sz = int(0.7 * total)
    val_sz   = int(0.15 * total)
    test_sz  = total - train_sz - val_sz

    train_ds, val_ds, test_ds = random_split(full_ds, [train_sz, val_sz, test_sz])
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=2)

    num_classes = len(full_ds.class_names)

    # 3) Build & freeze model
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for p in model.parameters():
        p.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    opt_head  = optim.Adam(model.fc.parameters(), lr=1e-3)

    # 4) Phase‑1: train head only
    for epoch in range(5):
        model.train()
        running = 0.0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            opt_head.zero_grad()
            loss = criterion(model(imgs), lbls)
            loss.backward()
            opt_head.step()
            running += loss.item()
        print(f"[Head Epoch {epoch+1}/5] loss={running/len(train_loader):.4f}")

    # 5) Phase‑2: fine‑tune layer4 + fc
    for p in model.layer4.parameters():
        p.requires_grad = True
    opt_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)

    for epoch in range(15):
        model.train()
        running = 0.0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            opt_ft.zero_grad()
            loss = criterion(model(imgs), lbls)
            loss.backward()
            opt_ft.step()
            running += loss.item()
        print(f"[FT Epoch {epoch+1}/15] loss={running/len(train_loader):.4f}")

    # 6) Save checkpoint
    ckpt = {
        "model_state_dict": model.state_dict(),
        "num_classes":      num_classes,
        "class_names":      full_ds.class_names
    }
    os.makedirs("model/checkpoints", exist_ok=True)
    torch.save(ckpt, "model/checkpoints/brain_tumor_model.pth")
    print("✅ Trained and saved → model/checkpoints/brain_tumor_model.pth")

if __name__ == "__main__":
    train()
