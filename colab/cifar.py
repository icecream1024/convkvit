import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
import time
import wandb
from vit import ViT

# 配置参数
bs = 32 # batch size
imsize = (32,32)  # image size
patch = 4  # patch size for ViT
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
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=8)
valloader = torch.utils.data.DataLoader(valset, batch_size=100, shuffle=False, num_workers=8)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)

# 定义 ViT 模型
net = ViT(
    image_size=imsize,
    patch_size=patch,
    num_classes=10,
    dim=dimhead,
    depth=6,
    heads=8,
    mlp_dim=2048,
    dropout=0.2,
    emb_dropout=0.2
)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

# 使用 AMP (自动混合精度)
# 自动混合精度 是一种优化技术，用于在训练深度学习模型时，自动地选择使用较低精度的数据类型（如 float16）来加速计算，
# 同时保持模型的准确性。这样可以减少计算资源的消耗和内存使用。
use_amp = True
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)


# 训练函数
def train(epoch):
    # print(f'\nEpoch: {epoch}')
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
        for batch_idx, (inputs, targets) in enumerate(valloader):  # 使用训练集的 DataLoader
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
        for batch_idx, (inputs, targets) in enumerate(testloader):  # 使用测试集的 DataLoader
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
        val_loss, val_acc = validate(epoch)  # 使用训练集进行验证
        test_loss, test_acc = ts(epoch)  # 使用测试集进行评估

        scheduler.step()  # 调整学习率

        # 记录 WandB 日志
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'test_loss': test_loss,  # 记录测试集损失
            'test_acc': test_acc,    # 记录测试集准确率
            'lr': optimizer.param_groups[0]["lr"],
            'epoch_time': time.time() - start_time
        })

# 运行主函数
if __name__ == "__main__":
    wandb.init(project="vit")
    main()
    wandb.finish()

