import torch
from torch import nn
from einops import rearrange, repeat
from sklearn.cluster import KMeans
from einops.layers.torch import Rearrange

# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# ----------------- Image Processing Modules -----------------
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
    def __init__(self, in_channels, out_channels):
        super().__init__()
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

# class ImageProcessingNet(nn.Module):
#     def __init__(self, input_dim):
#         super().__init__()
#         self.conv1 = ComplexConvModule(3, 32)
#         self.conv2 = ComplexConvModule(32, 64)
#         self.fc = nn.Linear(64 * 128 * 128, 1024)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = x.flatten(1)
#         x = self.fc(x)
#         return x
class ImageProcessingNet(nn.Module):
    def __init__(self, input_dim=224):
        super().__init__()
        self.conv1 = ComplexConvModule(3, 32)
        self.conv2 = ComplexConvModule(32, 64)
        self.pool = nn.AdaptiveAvgPool2d((input_dim, input_dim))  # 适配 ViT 输入
        self.fc = nn.Linear(64 * input_dim * input_dim, 1024)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)  # 调整大小
        x = x.flatten(1)  # 展平
        x = self.fc(x)
        return x

# ----------------- Vision Transformer Modules -----------------
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

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)  # k 和 v 一起处理
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key=None, value=None):
        # 如果 key 和 value 没有提供，默认为 query
        if key is None:
            key = query
        if value is None:
            value = query

        b, n = query.shape[:2]
        h = self.heads

        # 计算 query, key, value
        q = self.to_q(query).view(b, n, h, -1)
        k, v = self.to_kv(key).chunk(2, dim=-1)  # 分割 k 和 v
        k = k.view(b, n, h, -1)  # 确保 h 维度正确
        v = v.view(b, n, h, -1)  # 同样处理 v

        # Scaled dot-product attention
        dots = torch.einsum('bqhd,bkhd->bhqk', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.einsum('bhqk,bkhd->bqhd', attn, v).reshape(b, n, -1)
        return self.to_out(out)

class KMeansModule(nn.Module):
    """
            初始化 K-Means 模块。
            :param num_clusters: 聚类簇的数量 K
            :param feature_dim: 输入特征的维度 D
    """
    def __init__(self, num_clusters, feature_dim):
        super().__init__()
        self.num_clusters = num_clusters
        self.feature_dim = feature_dim
        self.cluster_transform = nn.Linear(feature_dim, feature_dim)

    def forward(self, x):
        """
                执行前向传播。
                :param x: 输入特征，形状为 (B, N, D)
                :return: 经过 K-Means 聚类和线性变换后的特征，形状为 (B, N, D)
        """
        batch_size, num_patches, feature_dim = x.shape
        x_flat = x.view(-1, feature_dim)

        kmeans = KMeans(n_clusters=self.num_clusters, random_state=0)
        cluster_labels = kmeans.fit_predict(x_flat.detach().cpu().numpy())
# kmeans.fit_predict()：这个方法首先执行 K-Means 聚类，并返回每个样本点所属的簇标签  C
        cluster_centers = torch.tensor(kmeans.cluster_centers_, device=x.device, dtype=x.dtype)
# kmeans.cluster_centers_ 是 K-Means 聚类算法在训练过程之后计算出来的簇中心（即质心），它是每个簇的平均值。

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
        # x 和 y 分别是来自两个路径的特征
        # Cross Attention: x 关注 y，反之亦然
        # 使用 query 和 key-value 来进行注意力计算
        attn_x = self.attn(x, y, y)  # x 作为查询，y 作为键值
        attn_y = self.attn(y, x, x)  # y 作为查询，x 作为键值
        return attn_x + attn_y  # 两个路径的特征相加融合


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3, dim_head=64, num_clusters=10, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = TransformerWithKMeans(dim, depth, heads, dim_head, mlp_dim, num_clusters, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        self.image_processing_net = ImageProcessingNet(input_dim=dim)

        # 修正: 初始化交叉注意力模块
        self.cross_attention = CrossAttentionModule(dim)

    def forward(self, img):
        # Patch embedding and adding positional encoding
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        # Vision Transformer processing
        x = self.transformer(x)

        # 聚合ViT的输出
        if self.pool == 'mean':
            x = x.mean(dim=1)
        else:
            x = x[:, 0]

        x = self.to_latent(x)

        # 卷积路径
        conv_features = self.image_processing_net(img)

        # 修正: 通过交叉注意力机制融合两个路径的特征
        combined_features = self.cross_attention(x.unsqueeze(1), conv_features.unsqueeze(1))

        # 通过MLP头输出
        return self.mlp_head(combined_features.squeeze(1))

