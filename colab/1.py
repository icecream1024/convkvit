import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader,random_split
import time
import wandb
# from cnnKvit import ViT
# from w import ViT
# from xin import ViT
# from new import ViT
from vit import ViT

# 配置参数
bs = 32  # batch size
imsize = (224, 224)  # image size
patch = 16  # patch size for ViT
dim = 2048  # dimension of heads
n_epochs = 100  # number of epochs
lr = 1e-4  # learning rate

# 检查设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 数据预处理
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.Resize(imsize),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# 加载自定义数据集
root_dir = "/content/data"  # 替换为您自己的数据集根目录
# if not os.path.exists(root_dir):
#     raise FileNotFoundError(f"Dataset directory {root_dir} does not exist.")

dataset = ImageFolder(root=root_dir, transform=transform)

# 划分训练、验证和测试集
train_size = int(0.6 * len(dataset))
val_size = int(0.3 * len(dataset))
test_size = len(dataset) - train_size - val_size
trainset, valset, testset = random_split(dataset, [train_size, val_size, test_size])

# 数据加载器
# trainloader = DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=8)
# valloader = DataLoader(valset, batch_size=bs, shuffle=False, num_workers=8)
# testloader = DataLoader(testset, batch_size=bs, shuffle=False, num_workers=8)
# 加载训练集、验证集和测试集
train_dir = os.path.join(root_dir, "train")  # 训练集路径
val_dir = os.path.join(root_dir, "val")      # 验证集路径
test_dir = os.path.join(root_dir, "test")    # 测试集路径

# 检查数据集路径是否存在
if not all(os.path.exists(d) for d in [train_dir, val_dir, test_dir]):
    raise FileNotFoundError("One or more dataset directories do not exist.")

# 创建 ImageFolder 数据集
trainset = ImageFolder(root=train_dir, transform=transform)
valset = ImageFolder(root=val_dir, transform=transform)
testset = ImageFolder(root=test_dir, transform=transform)

# 创建 DataLoader
trainloader = DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=8)
valloader = DataLoader(valset, batch_size=bs, shuffle=False, num_workers=8)
testloader = DataLoader(testset, batch_size=bs, shuffle=False, num_workers=8)

net = ViT(
    image_size=imsize,
    patch_size=patch,
    num_classes=len(dataset.classes),
    dim=dim,
    depth=4,
    heads=4,
    mlp_dim=1024,
    dropout=0.2,
    emb_dropout=0.1
)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(net.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

# 使用 AMP (自动混合精度)
use_amp = True
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

# 训练函数
def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = net(inputs)
            loss = criterion(outputs, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        print(f'Loss: {train_loss / (batch_idx + 1):.3f} | Acc: {100. * correct / total:.3f}% ({correct}/{total})')

    return train_loss / (batch_idx + 1)

# 验证函数
def validate(epoch):
    net.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_acc = 100. * correct / total
    print(f'Validation Acc: {val_acc:.3f}% ({correct}/{total})')
    return val_loss / (batch_idx + 1), val_acc

# 测试函数
def ts(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_acc = 100. * correct / total
    print(f'Test Acc: {test_acc:.3f}% ({correct}/{total})')
    return test_loss / (batch_idx + 1), test_acc

# 主函数
def main():
    net.to(device)

    for epoch in range(n_epochs):
        start_time = time.time()
        train_loss = train(epoch)
        val_loss, val_acc = validate(epoch)
        test_loss, test_acc = ts(epoch)

        scheduler.step()

        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'lr': optimizer.param_groups[0]["lr"],
            'epoch_time': time.time() - start_time
        })

# 运行主函数
if __name__ == "__main__":
    wandb.init(project="stomach-vit")
    main()
    wandb.finish()