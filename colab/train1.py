import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
import time
import wandb
import os
from PIL import Image
from torchvision import models
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# 配置参数
bs = 32  # batch size
imsize = (224, 224)  # 必须是224x224
# patch = 16  # patch size必须是16
# dim = 768  # 预训练模型的维度
n_epochs = 100
lr = 1e-4

# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        self.images = []
        self.labels = []
        self.disease_classes = set()
        self.class_counts = {}

        print("\n数据集统计信息:")
        print("-" * 50)

        # 直接获取四个大类文件夹
        class_folders = os.listdir(data_dir)
        for class_folder in class_folders:
            class_path = os.path.join(data_dir, class_folder)
            if not os.path.isdir(class_path):
                continue

            # 遍历具体疾病文件夹
            disease_folders = os.listdir(class_path)
            for disease_folder in disease_folders:
                disease_path = os.path.join(class_path, disease_folder)
                if not os.path.isdir(disease_path):
                    continue

                # 将大类和具体疾病名称组合
                disease_full_name = f"{class_folder}/{disease_folder}"
                self.disease_classes.add(disease_full_name)

        # 将疾病类型转换为排序后的列表并创建映射
        self.disease_classes = sorted(list(self.disease_classes))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.disease_classes)}

        # 收集图片和标签
        total_images = 0
        for class_folder in class_folders:
            class_path = os.path.join(data_dir, class_folder)
            if not os.path.isdir(class_path):
                continue

            for disease_folder in os.listdir(class_path):
                disease_path = os.path.join(class_path, disease_folder)
                if not os.path.isdir(disease_path):
                    continue

                disease_full_name = f"{class_folder}/{disease_folder}"
                disease_idx = self.class_to_idx[disease_full_name]

                # 统计该疾病类型的图片
                image_count = 0
                for img_name in os.listdir(disease_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(disease_path, img_name)
                        self.images.append(img_path)
                        self.labels.append(disease_idx)
                        image_count += 1

                self.class_counts[disease_full_name] = image_count
                total_images += image_count

        # 打印统计信息
        print("疾病类型统计:")
        major_class_counts = {}
        for full_name in self.disease_classes:
            major_class = full_name.split('/')[0]
            if major_class not in major_class_counts:
                major_class_counts[major_class] = 0
            major_class_counts[major_class] += self.class_counts[full_name]

        for major_class, count in major_class_counts.items():
            print(f"\n主要类别 {major_class}: 总计 {count} 张图片")
            # 打印该大类下的具体疾病
            for full_name in self.disease_classes:
                if full_name.startswith(major_class):
                    sub_class = full_name.split('/')[1]
                    print(f"  - {sub_class}: {self.class_counts[full_name]} 张图片")

        print("\n" + "-" * 50)
        print(f"总计图片数量: {total_images}")
        print(f"疾病类型总数: {len(self.disease_classes)}")
        print("-" * 50)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, label


# 数据预处理
transform_train = transforms.Compose([
    transforms.Resize((384, 384)),  # 先调整到较大尺寸
    transforms.RandomCrop(224),  # 随机裁剪到224x224
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),  # 添加随机旋转
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 添加颜色增强
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 检查设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# 加载数据集
def load_data(data_dir, batch_size):
    full_dataset = CustomDataset(data_dir, transform=transform_train)

    # 检查数据集的基本信息
    print("\n数据集检查:")
    print("-" * 50)
    print(f"数据集总大小: {len(full_dataset)}")

    # 检查第一个样本的形状
    first_image, first_label = full_dataset[0]
    print(f"图像形状: {first_image.shape}")
    print(f"标签值: {first_label}")

    # 检查标签范围
    all_labels = [label for _, label in full_dataset]
    unique_labels = sorted(set(all_labels))
    print(f"标签取值范围: {min(all_labels)} 到 {max(all_labels)}")
    print(f"唯一标签数量: {len(unique_labels)}")
    print(f"所有唯一标签: {unique_labels}")

    # 检查每个类别的样本数量
    label_counts = {}
    for label in all_labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    print("\n每个类别的样本数量:")
    for label in sorted(label_counts.keys()):
        print(f"类别 {label}: {label_counts[label]} 个样本")

    # 数据集划分
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size]
    )

    # 修改DataLoader的参数
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,  # 增加工作进程数
        pin_memory=True  # 使用固定内存
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 检查第一个batch的形状
    print("\n数据加载器检查:")
    print("-" * 50)
    sample_batch, sample_labels = next(iter(train_loader))
    print(f"Batch 图像形状: {sample_batch.shape}")
    print(f"Batch 标签形状: {sample_labels.shape}")
    print(f"Batch 中的唯一标签: {torch.unique(sample_labels).tolist()}")
    print("-" * 50)

    return train_loader, val_loader, test_loader


# 训练函数
def train(epoch, net, trainloader, criterion, optimizer, scaler, use_amp):
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    # 用于计算指标
    all_preds = []
    all_targets = []

    print(f"\nEpoch: {epoch + 1}/{n_epochs}")
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        with torch.amp.autocast('cuda', enabled=use_amp):
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

        # 收集预测结果和真实标签
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

        if batch_idx % 10 == 0:  # 每10个batch打印一次
            print(f'Batch: {batch_idx}/{len(trainloader)} | '
                  f'Loss: {train_loss / (batch_idx + 1):.3f} | '
                  f'Acc: {100. * correct / total:.2f}% ({correct}/{total})')

    # 计算指标
    train_acc = accuracy_score(all_targets, all_preds)
    train_precision = precision_score(all_targets, all_preds, average='macro')
    train_recall = recall_score(all_targets, all_preds, average='macro')
    train_f1 = f1_score(all_targets, all_preds, average='macro')

    print(f'Train Metrics:')
    print(f'Accuracy: {train_acc:.4f}')
    print(f'Precision: {train_precision:.4f}')
    print(f'Recall: {train_recall:.4f}')
    print(f'F1 Score: {train_f1:.4f}')

    return train_loss / (batch_idx + 1), train_acc, train_precision, train_recall, train_f1


# 验证函数
def validate(net, valloader, criterion):
    net.eval()
    val_loss = 0
    correct = 0
    total = 0

    # 用于计算指标
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # 收集预测结果和真实标签
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # 计算指标
    val_acc = accuracy_score(all_targets, all_preds)
    val_precision = precision_score(all_targets, all_preds, average='macro')
    val_recall = recall_score(all_targets, all_preds, average='macro')
    val_f1 = f1_score(all_targets, all_preds, average='macro')

    print(f'Validation Metrics:')
    print(f'Accuracy: {val_acc:.4f}')
    print(f'Precision: {val_precision:.4f}')
    print(f'Recall: {val_recall:.4f}')
    print(f'F1 Score: {val_f1:.4f}')

    return val_loss / (batch_idx + 1), val_acc, val_precision, val_recall, val_f1


# 测试函数
def test(net, testloader, criterion):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    # 用于计算指标
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # 收集预测结果和真实标签
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # 计算指标
    test_acc = accuracy_score(all_targets, all_preds)
    test_precision = precision_score(all_targets, all_preds, average='macro')
    test_recall = recall_score(all_targets, all_preds, average='macro')
    test_f1 = f1_score(all_targets, all_preds, average='macro')

    print(f'Test Metrics:')
    print(f'Accuracy: {test_acc:.4f}')
    print(f'Precision: {test_precision:.4f}')
    print(f'Recall: {test_recall:.4f}')
    print(f'F1 Score: {test_f1:.4f}')

    return test_loss / (batch_idx + 1), test_acc, test_precision, test_recall, test_f1


# 主函数
def main():
    data_dir = '/kaggle/input/eye111/eye'

    # 首先检查数据集的类别信息
    dataset = CustomDataset(data_dir, transform=None)
    num_classes = len(dataset.disease_classes)
    print("\n类别信息检查:")
    print("-" * 50)
    print(f"总类别数量: {num_classes}")
    print("\n类别对应关系:")
    for idx, class_name in enumerate(dataset.disease_classes):
        print(f"标签 {idx}: {class_name}")
    print("-" * 50)

    # 加载数据并进行检查
    trainloader, valloader, testloader = load_data(data_dir, bs)

    # 初始化模型时确保类别数量正确
    # net = ViT(
    #     image_size=imsize,
    #     patch_size=patch,
    #     num_classes=num_classes,  # 使用实际的类别数
    #     dim=768,
    #     depth=12,
    #     heads=12,
    #     mlp_dim=3072,
    #     dropout=0.1,
    #     emb_dropout=0.1,
    #     dim_head=64
    # ).to(device)
    # net = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=num_classes)
    # net = MobileNetV2(num_classes=23).to(device) #FLOPs: 13.675G Total Parameters: 2.253M
    # net = ResNet18().to(device)
    net = models.resnet50(num_classes=2).to(device)
    # net = SimpleViT(num_classes=23,image_size=224, patch_size=16, dim=768, depth=12, heads=12,
    #             mlp_dim=3072, dim_head=64).to(device)
    # net = SqueezeNet(num_classes=23).to(device)

    # 在训练前打印 FLOPs 和参数量
    # input_tensor = torch.randn(1, 3, 224, 224).to(device)  # 假设输入是224x224的RGB图像
    # flops, params = profile(net, inputs=(input_tensor,))
    # flops, params = clever_format([flops, params], "%.3f")
    #
    # print("\n模型复杂度信息:")
    # print("-" * 50)
    # print(f"FLOPs: {flops}")
    # print(f"Total Parameters: {params}")
    # print("-" * 50)

    # 如果使用多 GPU 训练，将模型包装为 DataParallel
    if torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 个GPU训练")
        net = nn.DataParallel(net)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    # 混合精度训练
    use_amp = True
    scaler = torch.cuda.amp.GradScaler()

    # 训练循环
    for epoch in range(n_epochs):
        start_time = time.time()

        # 训练、验证和测试
        train_loss, train_acc, train_precision, train_recall, train_f1 = train(epoch, net, trainloader, criterion,
                                                                               optimizer, scaler, use_amp)
        val_loss, val_acc, val_precision, val_recall, val_f1 = validate(net, valloader, criterion)
        test_loss, test_acc, test_precision, test_recall, test_f1 = test(net, testloader, criterion)

        scheduler.step()

        # 记录日志到wandb
        wandb.log({
            'epoch': epoch,
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

        # 打印当前epoch的所有指标
        print(f'\nEpoch Summary:')
        print(
            f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}% | Train Precision: {train_precision:.4f} | Train Recall: {train_recall:.4f} | Train F1: {train_f1:.4f}')
        print(
            f'Val Loss: {val_loss:.3f} | Val Acc: {val_acc:.2f}% | Val Precision: {val_precision:.4f} | Val Recall: {val_recall:.4f} | Val F1: {val_f1:.4f}')
        print(
            f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.2f}% | Test Precision: {test_precision:.4f} | Test Recall: {test_recall:.4f} | Test F1: {test_f1:.4f}')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        print(f'Time taken: {time.time() - start_time:.2f}s')


if __name__ == "__main__":
    wandb.init(project="eye_training")
    # wandb.init(project="custom_dataset_training")
    main()
    wandb.finish()