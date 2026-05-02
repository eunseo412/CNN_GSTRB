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
from torch.utils.data import DataLoader, Dataset, random_split
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.metrics import confusion_matrix, classification_report 

# ==============================================================================
# 1. Test.csv를 읽어서 정답을 맵핑해주는 커스텀 데이터셋 클래스
# ==============================================================================
class GTSRBTestDataset(Dataset):
    def __init__(self, csv_file, root_dir, class_to_idx, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 7])
        image = Image.open(img_path).convert("RGB")
        original_class_id_str = str(self.annotations.iloc[index, 6])
        mapped_label = self.class_to_idx[original_class_id_str]
        y_label = torch.tensor(mapped_label, dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, y_label

# ==============================================================================
# 2. 순수한 베이스라인 CNN 모델
# ==============================================================================
class PureBaselineCNN(nn.Module):
    def __init__(self, num_classes: int = 43):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ==============================================================================
# 3. 메인 실행 함수
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    path = kagglehub.dataset_download("meowmeowmeowmeowmeow/gtsrb-german-traffic-sign")
    train_dir = os.path.join(path, "Train")
    test_csv_path = os.path.join(path, "Test.csv")
    
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    # ==============================================================================
    # 💡 [질문자님 아이디어 적용 구간] 데이터 분리 방식 전면 수정!
    # ==============================================================================
    # 1. Train 데이터: 100% 온전히 다 쓴다! (성능 업그레이드)
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)

    # 2. Test 데이터 전체 불러오기
    full_test_dataset = GTSRBTestDataset( 
        csv_file=test_csv_path, 
        root_dir=path, 
        class_to_idx=train_dataset.class_to_idx, 
        transform=transform
    )

    # 3. Test 데이터를 50:50으로 쪼개서 Validation과 진짜 Test로 나눈다! (데이터 누수 완벽 차단)
    val_size = len(full_test_dataset) // 2
    test_size = len(full_test_dataset) - val_size
    val_dataset, test_dataset = random_split(full_test_dataset, [val_size, test_size])

    # DataLoader 세팅
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    # ==============================================================================

    model = PureBaselineCNN(num_classes=43).to(device)

    criterion = nn.CrossEntropyLoss()
    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    history_train_loss, history_val_loss = [], []
    history_train_acc, history_val_acc = [], []
    
    print(f"--- Starting Training (Validation is now strictly from Test Set!) ---")

    for epoch in range(1, args.epochs + 1):
        # --- Training ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
        train_loss /= len(train_loader.dataset)
        train_acc = train_correct / train_total

        # --- Validation ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / val_total

        history_train_loss.append(train_loss)
        history_val_loss.append(val_loss)
        history_train_acc.append(train_acc)
        history_val_acc.append(val_acc)
        
        print(f"Epoch [{epoch}/{args.epochs}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # ==============================================================================
    # 4. Evaluation & 시각화 결과물 저장
    # ==============================================================================
    print("\nGenerating evaluation plots and files...")
    
    # 1) Learning Curves (Loss & Accuracy)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history_train_loss, label='Train Loss')
    plt.plot(history_val_loss, label='Val Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history_train_acc, label='Train Acc')
    plt.plot(history_val_acc, label='Val Acc')
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.legend()
    plt.tight_layout()
    plt.savefig('model1-random_learning_curves.png')
    plt.close()

    # ==============================================================================
    # 🚨 진짜 Test 데이터 (나머지 50%)로 최종 평가 진행!
    # ==============================================================================
    print("--- Evaluating on Final Test Dataset ---")
    model.eval()
    test_correct, test_total = 0, 0
    all_preds, all_labels = [], []
    correct_examples, failure_examples = [], []
    
    with torch.no_grad():
        for images, labels in test_loader: 
            images_gpu, labels_gpu = images.to(device), labels.to(device)
            outputs = model(images_gpu)
            _, predicted = torch.max(outputs, 1)
            
            test_total += labels_gpu.size(0)
            test_correct += (predicted == labels_gpu).sum().item()
            
            preds_cpu = predicted.cpu().numpy()
            labels_cpu = labels_gpu.cpu().numpy()
            all_preds.extend(preds_cpu)
            all_labels.extend(labels_cpu)
            
            for i in range(len(labels_cpu)):
                img_np = images[i].permute(1, 2, 0).numpy()
                if preds_cpu[i] == labels_cpu[i] and len(correct_examples) < 5:
                    correct_examples.append((img_np, preds_cpu[i], labels_cpu[i]))
                elif preds_cpu[i] != labels_cpu[i] and len(failure_examples) < 5:
                    failure_examples.append((img_np, preds_cpu[i], labels_cpu[i]))

    final_test_acc = test_correct / test_total
    print(f"\n=======================================================")
    print(f"⭐ REAL Final Test Accuracy: {final_test_acc:.4f} ⭐")
    print(f"=======================================================\n")

    # 2) Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds, normalize='true')
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=False, cmap='Blues', vmin=0, vmax=1)
    plt.title('Normalized Confusion Matrix (Proportion)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('model1-random_confusion_matrix.png')
    plt.close()

    # 3) Class-wise performance (.txt)
    report = classification_report(all_labels, all_preds, digits=4)
    with open('model1-random_class_wise_performance.txt', 'w') as f:
        f.write("=== Class-wise Performance (Test Dataset) ===\n")
        f.write(report)

    # 4) Examples of Correct & Failure Cases
    def save_image_grid(examples, filename, title_prefix):
        if not examples: return
        fig, axes = plt.subplots(1, len(examples), figsize=(15, 3))
        for idx, (img, pred, true) in enumerate(examples):
            axes[idx].imshow(np.clip(img, 0, 1))
            axes[idx].set_title(f"P:{pred} / T:{true}")
            axes[idx].axis('off')
        plt.suptitle(title_prefix)
        plt.savefig(filename)
        plt.close()

    save_image_grid(correct_examples, 'model1-random_correct_cases.png', 'Examples of Correct Predictions')
    save_image_grid(failure_examples, 'model1-random_failure_cases.png', 'Examples of Failure Cases')

    print("All evaluation files generated successfully!")

if __name__ == "__main__":
    main()
