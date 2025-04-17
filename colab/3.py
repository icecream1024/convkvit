import torch
from torchvision import models
from thop import profile, clever_format

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建模型
net = models.vit_b_16(num_classes=3).to(device)


# 创建随机输入张量
input_tensor = torch.randn(1, 3, 224, 224).to(device)

# 计算FLOPs和参数量
flops, params = profile(net, inputs=(input_tensor,))
flops, params = clever_format([flops, params], "%.3f")

# 打印结果
print("\n模型复杂度信息:")
print("-" * 50)
print(f"FLOPs: {flops}")  # 浮点运算次数
print(f"Total Parameters: {params}")  # 总参数量
print("-" * 50)