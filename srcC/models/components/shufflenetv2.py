import torch
import torch.nn as nn

# --------------------------------- #
# （1）通道重排
# --------------------------------- #

def channel_shuffle(x, groups):
    # 获取输入特征图的shape=[b,c,h,w]
    batch_size, num_channels, height, width = x.size()
    # 均分通道，获得每个组对应的通道数
    channels_per_group = num_channels // groups
    # 特征图shape调整 [b,c,h,w]==>[b,g,c_g,h,w]
    x = x.view(batch_size, groups, channels_per_group, height, width)
    # 维度调整 [b,g,c_g,h,w]==>[b,c_g,g,h,w]；将调整后的tensor以连续值的形式保存在内存中
    x = torch.transpose(x, 1, 2).contiguous()
    # 将调整后的通道拼接回去 [b,c_g,g,h,w]==>[b,c,h,w]
    x = x.view(batch_size, -1, height, width)
    # 完成通道重排
    return x


# ------------------------------------ #
# （2）倒残差结构
# ------------------------------------ #

class InvertedResidual(nn.Module):
    # 初始化，输入特征图通道数，输出特征图通道数，DW卷积的步长=1或2
    def __init__(self, input_c, output_c, stride):
        super(InvertedResidual, self).__init__()
        # 属性分配
        self.stride = stride
        # 特征图的通道数必须是2的整数倍，保证平分和拼接后的通道数不变
        assert output_c % 2 == 0
        # 每个分支对应的通道数
        branch_features = output_c // 2
        # 如果stride==1，输入特征图的通道数是输出特征图的2倍
        assert (self.stride != 1) or (input_c == branch_features * 2)

        # ------------------------------------------- #
        # 步长为2, 下采样模块, 左分支第二个1*1卷积调整通道数，右分支第一个1*1卷积调整通道
        # ------------------------------------------- #

        if self.stride == 2:
            # 左分支DW卷积+逐点卷积
            self.branch1 = nn.Sequential(
                # DW卷积，输入和输出特征图的通道数相同
                self.depthwise_conv(input_c, input_c, kernel_s=3, stride=self.stride, padding=1),
                # 在特征图周围填充一圈0，卷积后的size不变
                nn.BatchNorm2d(input_c),  # 对输出特征图的每个通道做BN
                # 1*1卷积调整通道数，下降为一半。有BN就不要偏置
                nn.Conv2d(input_c, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True)  # 覆盖输入数据，节省内存
            )
            # 右分支1*1卷积+DW卷积+1*1卷积
            self.branch2 = nn.Sequential(
                # 1*1卷积下降通道数，下降一半
                nn.Conv2d(in_channels=input_c, out_channels=branch_features,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),  # 对输出的每个通道做BN
                nn.ReLU(inplace=True),
                # 3*3 DW卷积，输入和输出通道数相同
                self.depthwise_conv(branch_features, branch_features,
                                    kernel_s=3, stride=self.stride, padding=1, bias=False),
                nn.BatchNorm2d(branch_features),
                # 1*1普通卷积
                nn.Conv2d(in_channels=branch_features, out_channels=branch_features,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True)
            )

        # --------------------------------------------- #
        # 步长为1, 基本模块，跟在下采样模块后面，左分支不做任何处理
        # --------------------------------------------- #

        else:
            # 左分支
            self.branch1 = nn.Sequential()
            # 右分支1*1卷积+DW卷积+1*1卷积
            self.branch2 = nn.Sequential(
                # 1*1卷积通道数不变
                nn.Conv2d(in_channels=branch_features, out_channels=branch_features, kernel_size=1,
                          stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),  # 对输出的每个通道做BN
                nn.ReLU(inplace=True),
                # 3*3 DW卷积，输入和输出通道数相同
                self.depthwise_conv(branch_features, branch_features,
                                    kernel_s=3, stride=self.stride, padding=1, bias=False),
                nn.BatchNorm2d(branch_features),
                # 1*1普通卷积
                nn.Conv2d(in_channels=branch_features, out_channels=branch_features,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True)
            )

    # ------------------------------------ #
    # DW卷积
    # ------------------------------------ #

    def depthwise_conv(self, input_c, output_c, kernel_s,
                       stride=1, padding=0, bias=False):
        # 深度可分离卷积，卷积核对每张通道做卷积运算
        return nn.Conv2d(in_channels=input_c, out_channels=output_c, kernel_size=kernel_s,
                         stride=stride, padding=padding, bias=bias,
                         groups=input_c)

    # ------------------------------------ #
    # 前向传播
    # ------------------------------------ #

    def forward(self, x):  # x代表输入特征图
        # 基本单元
        if self.stride == 1:
            # 将输入特征图在通道维度上均分2份
            x1, x2 = x.chunk(2, dim=1)
            # 分别对左右分支做前向传播，通道数不变
            x1 = self.branch1(x1)
            x2 = self.branch2(x2)
            # 将输出特征图在通道维度上堆叠，通道数还原
            out = torch.cat((x1, x2), dim=1)

        # 下采样模块
        if self.stride == 2:
            # 对输入特征图分别做左右分支的前传
            x1 = self.branch1(x)
            x2 = self.branch2(x)
            # 将输出特征图堆叠
            out = torch.cat((x1, x2), dim=1)

        # 通道重排
        out = channel_shuffle(out, 2)

        return out


# ------------------------------------ #
# （3）主干网络
# ------------------------------------ #

class ShuffleNetV2(nn.Module):
    # 初始化
    def __init__(self,
                 num_classes=10,  # 分类数
                 ):
        super(ShuffleNetV2, self).__init__()

        # 输入特征图通道数RGB
        input_channels = 3
        # 第一个卷积块的输出特征图通道数24
        output_channels = 24

        # 1*1普通卷积调整通道数
        self.conv1 = nn.Sequential(
            # [b,3,224,224]==>[b,24,112,112]
            nn.Conv2d(in_channels=input_channels, out_channels=output_channels,
                      kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

        # 最大池化层 [b,24,112,112]==>[b,24,56,56]
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 主干的三个卷积块
        inverted_block = [
            # input_c, output_c, stride
            # 下采样 [b,24,56,56] ==> [b,116,28,28]
            InvertedResidual(24, 116, 2),
            # [b,116,28,28]==>[b,116,28,28]
            InvertedResidual(116, 116, 1),
            InvertedResidual(116, 116, 1),
            InvertedResidual(116, 116, 1),
            # 下采样 [b,116,28,28]==>[b,232,14,14]
            InvertedResidual(116, 232, 2),
            # [b,232,14,14]==>[b,232,14,14]
            InvertedResidual(232, 232, 1),
            InvertedResidual(232, 232, 1),
            InvertedResidual(232, 232, 1),
            InvertedResidual(232, 232, 1),
            InvertedResidual(232, 232, 1),
            InvertedResidual(232, 232, 1),
            InvertedResidual(232, 232, 1),
            # 下采样 [b,232,14,14]==>[b,464,7,7]
            InvertedResidual(232, 464, 2),
            # [b,464,7,7]==>[b,464,7,7]
            InvertedResidual(464, 464, 1),
            InvertedResidual(464, 464, 1),
            InvertedResidual(464, 464, 1),
        ]

        # 将堆叠的倒残差结构以非关键字参数返回
        self.inverted_block = nn.Sequential(*inverted_block)

        # 1*1卷积调整通道 [b,464,7,7]==>[b,1024,7,7]
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=464, out_channels=1024,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

        # [b,1024,1,1]==>[b,1000]
        self.fc = nn.Linear(1024, num_classes)

    # 前相传播
    def forward(self, x):  # x输入特征图
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.inverted_block(x)
        x = self.conv5(x)
        # 全局池化[b,1024,7,7]==>[b,1024,1,1]
        x = x.mean([2, 3])
        # [b,1024,1,1]==>[b,1000]
        x = self.fc(x)
        return x

# def main():
#     blk = InvertedResidual(64, 128, 2)
#     tmp = torch.randn(2, 64, 224, 224)
#     out = blk(tmp)
#     print('block:', out.shape)
#
#
#     model = ShuffleNetV2(8)
#     tmp = torch.randn(2, 3, 224, 224)
#     out = model(tmp)
#     print('mobile:', out.shape)
#
#     p = sum(map(lambda p:p.numel(), model.parameters()))
#     print('parameters size:', p)
#
#
if __name__ == '__main__':
    # main()
    _ = ShuffleNetV2()

