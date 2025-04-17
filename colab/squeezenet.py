import torch
from torch import nn

class Fire(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels, version=1):
        super(Fire, self).__init__()
        self.in_channels = in_channels
        self.version = version
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.bn_squeeze = nn.BatchNorm2d(squeeze_channels)
        self.expand1x1 = nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1)
        self.activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1)
        self.bn_out = nn.BatchNorm2d(expand1x1_channels + expand3x3_channels)

    def forward(self, x):
        # x1 = self.squeeze_activation(self.squeeze(x))
        # e1 = self.expand1x1(x1)
        # e2 = self.expand3x3(x1)
        x1 = self.activation(self.bn_squeeze(self.squeeze(x)))
        e1 = self.expand1x1(x1)
        e2 = self.expand3x3(x1)
        y = torch.cat([e1, e2], 1)
        if self.version == 1:
            y = self.activation(self.bn_out(y))
        elif self.version == 2:
            # y = self.expand_activation(y + x)
            y = self.activation(self.bn_out(y + x))
        return y

class SqueezeNet(nn.Module):
    def __init__(self, version=1, num_classes=10):
        super(SqueezeNet, self).__init__()
        self.num_classes = num_classes
        self.version = version
        self.net = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            Fire(96, 16, 64, 64),
            Fire(128, 16, 64, 64),
            Fire(128, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2),
            Fire(256, 32, 128, 128),
            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192),
            Fire(384, 64, 256, 256),
            nn.MaxPool2d(kernel_size=3, stride=2),
            Fire(512, 64, 256, 256),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(512, self.num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # 自适应平均池化，指定输出（H，W）
        )

    def forward(self, x):
        x = self.net(x)
        x = self.classifier(x)
        y = torch.flatten(x, 1)
        return y

def main():
    model = SqueezeNet(10)
    tmp = torch.randn(2, 3, 224, 224)
    out = model(tmp)
    print('SqueezeNet:', out.shape)

    p = sum(map(lambda p:p.numel(), model.parameters()))
    print('parameters size:', p)

if __name__ == '__main__':
    main()
