import argparse
import os
import pandas as pd
from PIL import Image
import numpy as np
import kagglehub

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import (
    Dataset,
    DataLoader,
    random_split,
    ConcatDataset
)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from datasets import load_dataset

# ==============================================================================
# 1. GTSRB TEST CSV DATASET
# ==============================================================================
class GTSRBTestDataset(Dataset):
    def __init__(self, csv_file, root_dir, class_to_idx, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[idx, 7])
        image = Image.open(img_path).convert("RGB")
        
        label_str = str(self.annotations.iloc[idx, 6])
        label = int(self.class_to_idx[label_str]) # Tensor 대신 int로 반환

        if self.transform:
            image = self.transform(image)
        return image, label

# ==============================================================================
# 2. SYNSET DATASET
# ==============================================================================
class SynsetDataset(Dataset):
    def __init__(self, split="train", transform=None):
        target_split = "validation" if split == "test" else "train"
        self.ds = load_dataset(
            "FraunhoferIOSB/Synset-Signset-Germany",
            "Cycles",
            split=target_split,
            trust_remote_code=True
        )
        labels = self.ds["label"]
        self.indices = [i for i, label in enumerate(labels) if int(label) < 43]
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        item = self.ds[real_idx]
        image = item["image"].convert("RGB")
        label = int(item["label"]) # int로 반환

        if self.transform:
            image = self.transform(image)
        return image, label

# ==============================================================================
# 3. BASELINE CNN
# ==============================================================================
class PureBaselineCNN(nn.Module):
    def __init__(self, num_classes=43):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ==============================================================================
# 4. MAIN
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=10)
    args, _ = parser.parse_known_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- 실행 시작 (Device: {device}) ---")

    path = kagglehub.dataset_download("meowmeowmeowmeowmeow/gtsrb-german-traffic-sign")
    
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
    ])

    print("데이터 로딩 중...")
    gtsrb_train = datasets.ImageFolder(os.path.join(path, "Train"), transform=transform)
    gtsrb_test = GTSRBTestDataset(os.path.join(path, "Test.csv"), path, gtsrb_train.class_to_idx, transform)
    synset_train = SynsetDataset(split="train", transform=transform)
    synset_test = SynsetDataset(split="test", transform=transform)

    full_dataset = ConcatDataset([gtsrb_train, gtsrb_test, synset_train, synset_test])
    
    train_size = int(len(full_dataset) * 0.8)
    val_size = int(len(full_dataset) * 0.1)
    test_size = len(full_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    # 에러 방지를 위해 num_workers를 2로 설정 (문제가 계속되면 0으로 수정하세요)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = PureBaselineCNN(num_classes=43).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler()

    print("\n--- 학습 시작 ---")
    for epoch in range(1, args.epochs + 1):
        model.train()
        t_correct, t_total = 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            t_correct += (outputs.argmax(1) == labels).sum().item()
            t_total += labels.size(0)

        print(f"Epoch [{epoch}/{args.epochs}] Train Acc: {t_correct/t_total:.4f}")

    print("\n--- 완료 ---")

if __name__ == "__main__":
    main()
