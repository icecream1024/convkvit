import os
import torch
import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import time
import wandb
from gai import CustomModel

n_epochs = 100

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
# 加载自定义数据集
root_dir = "/content/isic-2019-skin-disease/final"  # 替换为您自己的数据集根目录

dataset = ImageFolder(root=root_dir, transform=transform)

# 加载训练集、验证集和测试集
train_dir = os.path.join(root_dir, "train")  # 训练集路径
val_dir = os.path.join(root_dir, "val")      # 验证集路径
test_dir = os.path.join(root_dir, "test")    # 测试集路径

# 检查数据集路径是否存在
if not all(os.path.exists(d) for d in [train_dir, val_dir, test_dir]):
    raise FileNotFoundError("One or more dataset directories do not exist.")
# **3️⃣ 数据预处理**


train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

# **4️⃣ 加载预训练 ViT**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=8)
model = CustomModel(pretrained=True, num_classes=8)
model = model.to(device)

# **5️⃣ 训练和验证**
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
# **训练函数**
def train_model(epoch):
    model.train()
    total_loss, correct = 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()

    acc = 100. * correct / len(train_loader.dataset)
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}], Train Loss: {avg_loss:.4f}, Train Acc: {acc:.4f}")
    return acc, avg_loss

# **验证函数**
def validate(epoch):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

    acc = 100. * correct / len(val_loader.dataset)
    avg_loss = total_loss / len(val_loader)
    print(f"Epoch [{epoch+1}], Val Loss: {avg_loss:.4f}, Val Acc: {acc:.4f}")
    return avg_loss, acc

# **测试函数**
def test_model(epoch):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

    acc = 100. * correct / len(test_loader.dataset)
    avg_loss = total_loss / len(test_loader)
    print(f"Epoch [{epoch+1}], Test Loss: {avg_loss:.4f}, Test Acc: {acc:.4f}")
    return avg_loss, acc
# **主函数**
def main():
    for epoch in range(n_epochs):
        start_time = time.time()
        train_acc, train_loss = train_model(epoch)
        val_loss, val_acc = validate(epoch)
        test_loss, test_acc = test_model(epoch)

        scheduler.step()

        wandb.log({
            'epoch': epoch + 1,
            'train_acc': train_acc,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'lr': optimizer.param_groups[0]["lr"],
            'epoch_time': time.time() - start_time
        })

# **运行主函数**
if __name__ == "__main__":
    wandb.init(project="stomach-vit")
    main()
    wandb.finish()
