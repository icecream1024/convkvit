import torch
import torch.nn as nn
from torchvision.models._utils import _make_divisible

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)
        )
        # 继承来自nn.Sequential，不需要写forward函数
        # groups = 1则为普通卷积，groups设置成输入特征矩阵的深度则为dw卷积；padding根据kernel_size来设置。
class InvertedResidual(nn.Module):
    def __init__(self, in_channel, out_channel, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        hidden_channel = in_channel * expand_ratio
        self.use_shortcut = stride == 1 and in_channel == out_channel

        layers = []  # 定义层列表
        if expand_ratio != 1:
            # 1x1 pointwise conv
            layers.append(ConvBNReLU(in_channel, hidden_channel, kernel_size=1))
        layers.extend([
            # 3x3 depthwise conv
            ConvBNReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
            # 1x1 pointwise conv(linear)
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)
# use_shortcut ：需要满足两个条件，stride == 1并且 in_channel == out_channel
# layer的第一层：判断expand_ratio 是否为1，如果为1则不需要这一层，若不为1则输入为in_channel，输出为hidden_channel（就是这一层的卷积核个数），kernel_size=1
# layer的第二层：dw卷积，因此设置groups=hidden_channel，即group为输入通道数
# layer的第三层：没有直接使用前面定义的ConvBNReLU类，这是一因为最后一层没有使用relu激活函数。因为线性层相当与y=x，因此不需要额外添加一个线性层。
# 将layer通过位置参数传入Sequential（），打包组合在一起取名叫self.conv

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=23, alpha=1.0, round_nearest=8):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = _make_divisible(32 * alpha, round_nearest)          # 将卷积核个数调整到最接近8的整数倍数
        last_channel = _make_divisible(1280 * alpha, round_nearest)

        # inverted_residual_setting定义了每个倒残差模块的参数
        # （t：expand_ratio，c：output_channel，n：重复次数，s：stride）
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 1],
            [6, 32, 3, 2],
            [6, 64, 4, 1],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        features = []
        # conv1 layer
        features.append(ConvBNReLU(3, input_channel, stride=1))
        # building inverted residual residual blockes
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * alpha, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, last_channel, 1))
        # combine feature layers
        self.features = nn.Sequential(*features)

        # building classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes)
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)         # 初始化均值为0
                nn.init.zeros_(m.bias)          # 初始化方差为1
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# def main():
#     blk = ConvBNReLU(64, 128)
#     tmp = torch.randn(2, 64, 224, 224)
#     out = blk(tmp)
#     print('block:', out.shape)
#
#
#     model = MobileNetV2(8)
#     tmp = torch.randn(2, 3, 224, 224)
#     out = model(tmp)
#     print('resnet:', out.shape)
#
#     p = sum(map(lambda p:p.numel(), model.parameters()))
#     print('parameters size:', p)


if __name__ == '__main__':
    _ = MobileNetV2()
    # main()