import os
import timm
import torch
import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import time
import wandb
from torchvision import models
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from einops import rearrange

bs = 32  # batch size
imsize = (224, 224)  # 必须是224x224
patch = 16  # patch size必须是16
dim = 768  # 预训练模型的维度
n_epochs = 71
lr = 1e-4
num_classes = 2

# **1️⃣ 训练参数**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# **2️⃣ 数据预处理**
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# **3️⃣ 加载数据**
root_dir = "/kaggle/input/eye111/eye"  # 替换为您的数据集路径
dataset = ImageFolder(root=root_dir, transform=transform)
train_size = int(0.6 * len(dataset))
val_size = int(0.3 * len(dataset))
test_size = len(dataset) - train_size - val_size
trainset, valset, testset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
val_loader = DataLoader(valset, batch_size=64, shuffle=False, num_workers=2)
test_loader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

# **7️⃣ 迁移学习，仅训练新增模块**
# model = SimpleViT(num_classes=2,image_size=224, patch_size=16, dim=768, depth=12, heads=12,
#             mlp_dim=3072, dim_head=64).to(device)
model = models.mobilenet_v3_large(num_classes=2).to(device)
# model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=2).to(device)
# model = mobilevit_xxs().to(device)
if torch.cuda.device_count() > 1:
    print(f"使用 {torch.cuda.device_count()} 个GPU训练")
    model = nn.DataParallel(model)

# **8️⃣ 训练配置**
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)


def calculate_metrics(outputs, labels):
    preds = outputs.argmax(1).cpu().numpy()
    labels = labels.cpu().numpy()
    precision = precision_score(labels, preds, average='macro')
    recall = recall_score(labels, preds, average='macro')
    f1 = f1_score(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return precision, recall, f1, acc


# **9️⃣ 训练函数**
def train_model(epoch):
    model.train()
    total_loss, correct = 0, 0
    all_preds = []
    all_labels = []

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()

        all_preds.append(outputs.argmax(1).cpu())
        all_labels.append(labels.cpu())

    # 计算指标
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    acc = 100. * correct / len(train_loader.dataset)
    avg_loss = total_loss / len(train_loader)

    print(f"Epoch [{epoch + 1}], Train Loss: {avg_loss:.4f}, Train Acc: {acc:.4f}, "
          f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    return avg_loss, acc, precision, recall, f1


# **验证函数**
def validate(epoch):
    model.eval()
    total_loss, correct = 0, 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

            all_preds.append(outputs.argmax(1).cpu())
            all_labels.append(labels.cpu())

    # 计算指标
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    acc = 100. * correct / len(val_loader.dataset)
    avg_loss = total_loss / len(val_loader)

    print(f"Epoch [{epoch + 1}], Val Loss: {avg_loss:.4f}, Val Acc: {acc:.4f}, "
          f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    return avg_loss, acc, precision, recall, f1


# **测试函数**
def test_model(epoch):
    model.eval()
    total_loss, correct = 0, 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

            all_preds.append(outputs.argmax(1).cpu())
            all_labels.append(labels.cpu())

    # 计算指标
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    acc = 100. * correct / len(test_loader.dataset)
    avg_loss = total_loss / len(test_loader)

    print(f"Epoch [{epoch + 1}], Test Loss: {avg_loss:.4f}, Test Acc: {acc:.4f}, "
          f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    return avg_loss, acc, precision, recall, f1


# **主函数**
def main():
    for epoch in range(n_epochs):
        start_time = time.time()
        train_loss, train_acc, train_precision, train_recall, train_f1 = train_model(epoch)
        val_loss, val_acc, val_precision, val_recall, val_f1 = validate(epoch)
        test_loss, test_acc, test_precision, test_recall, test_f1 = test_model(epoch)

        scheduler.step()

        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'train_precision': train_precision,
            'train_recall': train_recall,
            'train_f1': train_f1,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_f1': val_f1,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'lr': optimizer.param_groups[0]["lr"],
            'epoch_time': time.time() - start_time
        })


if __name__ == "__main__":
    wandb.init(project="eye_training")
    main()
    wandb.finish()