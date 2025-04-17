import timm
import torch
from torch import nn
from sklearn.cluster import KMeans
from thop import profile, clever_format

# 保留原有的模块定义
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class AsymmetricConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.horizontal_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size), padding=(0, padding))
        self.vertical_conv = nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_size, 1), padding=(padding, 0))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.horizontal_conv(x)
        x = self.vertical_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class DilatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=2, padding=2):
        super().__init__()
        self.dilated_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dilated_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ComplexConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        self.downsample = downsample
        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = AsymmetricConv(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv3 = DilatedConv(out_channels, out_channels, kernel_size=3, dilation=2, padding=2)
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = self.residual(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x + residual
        x = self.bn(x)
        x = self.relu(x)
        return x

class ImageProcessingNet(nn.Module):
    def __init__(self, input_dim=224):
        super().__init__()
        # self.conv1 = ComplexConvModule(3, 32)
        # self.conv2 = ComplexConvModule(32, 64)
        # self.pool = nn.AdaptiveAvgPool2d((input_dim, input_dim))
        # self.fc = nn.Linear(64 * input_dim * input_dim, 768)  # 输出维度调整为 768
        self.conv1 = ComplexConvModule(3, 32, downsample=True)  # [32,112,112]
        self.conv2 = ComplexConvModule(32, 64, downsample=True)  # [64,56,56]
        self.pool = nn.AdaptiveAvgPool2d((7, 7))  # [64,7,7]
        self.fc = nn.Linear(64 * 7 * 7, 768)  # 输出[768]

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5  # 缩放因子

        inner_dim = dim_head * heads
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key=None, value=None):
        # 如果 key 和 value 为空，默认使用 query 作为 key 和 value
        if key is None:
            key = query
        if value is None:
            value = query

        # 检查 query 形状，确保其为 (batch, num_patches, dim)
        if query.dim() == 2:
            query = query.unsqueeze(1)  # 添加 num_patches 维度

        b, n, _ = query.shape  # (batch_size, num_patches, dim)

        # 计算 Q, K, V
        q = self.to_q(query).view(b, n, self.heads, -1)
        k = self.to_k(key).view(b, n, self.heads, -1)
        v = self.to_v(value).view(b, n, self.heads, -1)

        # 交换维度以适配 PyTorch 的多头注意力计算
        q, k, v = map(lambda x: x.permute(0, 2, 1, 3), (q, k, v))

        # 计算注意力分数
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # 计算最终的加权和
        out = (attn @ v).permute(0, 2, 1, 3).reshape(b, n, -1)

        # 输出层
        out = self.to_out(out)
        return out

class KMeansModule(nn.Module):
    def __init__(self, num_clusters, feature_dim):
        super().__init__()
        self.num_clusters = num_clusters
        self.feature_dim = feature_dim
        self.cluster_transform = nn.Linear(feature_dim, feature_dim)

    def forward(self, x):
        batch_size, num_patches, feature_dim = x.shape
        x_flat = x.view(-1, feature_dim)

        kmeans = KMeans(n_clusters=self.num_clusters, random_state=0)
        cluster_labels = kmeans.fit_predict(x_flat.detach().cpu().numpy())
        cluster_centers = torch.tensor(kmeans.cluster_centers_, device=x.device, dtype=x.dtype)

        clustered_features = cluster_centers[cluster_labels]
        clustered_features = clustered_features.view(batch_size, num_patches, feature_dim)

        transformed_features = self.cluster_transform(clustered_features)
        return transformed_features

class TransformerWithKMeans(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, num_clusters, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
        self.kmeans = KMeansModule(num_clusters=num_clusters, feature_dim=dim)

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        x = self.kmeans(x)
        return x

class CrossAttentionModule(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)

    def forward(self, x, y):
        attn_x = self.attn(x, y, y)
        attn_y = self.attn(y, x, x)
        return attn_x + attn_y

class CustomModel(nn.Module):
    def __init__(self, pretrained, num_classes):
        super().__init__()
        self.pretrained = pretrained
        self.num_classes = num_classes

        # 图像处理网络
        self.image_processing = ImageProcessingNet(input_dim=64)

        # 加载预训练 ViT 模型
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=pretrained, num_classes=num_classes)
        if pretrained:
            # 忽略不匹配的权重
            state_dict = self.vit.state_dict()
            pretrained_state_dict = torch.hub.load_state_dict_from_url(
                "https://huggingface.co/timm/vit_base_patch16_224.augreg2_in21k_ft_in1k/resolve/main/pytorch_model.bin"
            )
            # 过滤掉分类头权重
            pretrained_state_dict = {k: v for k, v in pretrained_state_dict.items() if not k.startswith('head.')}
            state_dict.update(pretrained_state_dict)
            self.vit.load_state_dict(state_dict, strict=False)

            # 冻结 ViT 主干
            for param in self.vit.parameters():
                param.requires_grad = False

        # 其他模块
        self.transformer_kmeans = TransformerWithKMeans(dim=768, dim_head=64, depth=12, heads=12, mlp_dim=3072, num_clusters=10)
        self.cross_attention = CrossAttentionModule(dim=768)
        # **新增一个分类层**
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.image_processing(x)
        x = self.transformer_kmeans(x)
        x = self.cross_attention(x, x)
        # **取第一个 token（或者均值池化）**
        x = x[:, 0, :]  # shape [batch_size, 768]

        # **最终分类**
        x = self.classifier(x)  # shape [batch_size, num_classes]
        return x

    def print_trainable_parameters(self):
        trainable_params = 0
        all_params = 0
        for name, param in self.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                print(f"可训练层: {name}")

        print(f'\n可训练参数数量: {trainable_params:,}')
        print(f'总参数数量: {all_params:,}')
        print(f'可训练参数占比: {100 * trainable_params / all_params:.2f}%')

# 测试模型
# if __name__ == "__main__":
#     model = CustomModel(pretrained=True, num_classes=5)
#     model.print_trainable_parameters()
#
#     X = torch.randn(12, 3, 224, 224)
#     y = model(X)
#     print(f"\n输出张量形状: {y.shape}")

# if __name__ == "__main__":
#     # 测试模型
#     model = CustomModel(pretrained=True, num_classes=5)
#
#     # 打印模型结构和可训练参数信息
#     print("\n模型结构:")
#     print(model)
#
#     print("\n可训练参数信息:")
#     model.print_trainable_parameters()
#
#     # 测试前向传播
#     X = torch.randn(12, 3, 224, 224)
#     y = model(X)
#     print(f"\n输出张量形状: {y.shape}")

    ## 可视化计算图
    # from torchviz import make_dot
    # dot = make_dot(y.mean(), params=dict(model.named_parameters()))
    # dot.render("vit_gai_graph", format="png")
    # print("\n计算图已保存为 'vit_gai_graph.png'")

# 示例用法
if __name__ == "__main__":
    model = CustomModel(num_classes=23,pretrained=False)
    input_tensor = torch.randn(4, 3, 224, 224)
    # 计算FLOPs和参数量
    flops, params = profile(model, inputs=(input_tensor,))
    flops, params = clever_format([flops, params], "%.3f")

    # 打印结果
    print("\n模型复杂度信息:")
    print("-" * 50)
    print(f"FLOPs: {flops}")  # 浮点运算次数
    print(f"Total Parameters: {params}")  # 总参数量
    print("-" * 50)

