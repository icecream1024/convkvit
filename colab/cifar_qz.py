import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
import time
import wandb
import timm
from gai import CustomModel
# 配置参数
bs = 32 # batch size
imsize = (224,224)  # image size
patch = 16  # patch size for ViT
dimhead = 1024  # dimension of heads
n_epochs = 100  # number of epochs
lr = 1e-4  # learning rate

# 检查设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 数据预处理
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.Resize(imsize),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize(imsize),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# 准备数据集
full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

# 从训练集划分出验证集（例如，20%的数据作为验证集）
val_size = int(0.2 * len(full_trainset))
train_size = len(full_trainset) - val_size

trainset, valset = random_split(full_trainset, [train_size, val_size])

# 数据加载器
train_loader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=8)
val_loader = torch.utils.data.DataLoader(valset, batch_size=100, shuffle=False, num_workers=8)
test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)

model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=10)
# model = CustomModel(pretrained=True, num_classes=10)
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
    wandb.init(project="vit")
    main()
    wandb.finish()

