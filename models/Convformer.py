import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torch import Tensor


class Config(object):
    def __init__(self, dataset):
        self.model_name = 'Convformer'
        self.data_path = dataset + '/raw'
        self.num_classes = 3
        self.f1_save_path = dataset + '/saved_dict/f1_' + self.model_name + '.ckpt'  # 模型训练结果
        self.auc_save_path = dataset + '/saved_dict/auc_' + self.model_name + '.ckpt'  # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
        self.class_list = [x.strip() for x in open(
            dataset + '/class.txt', encoding='utf-8').readlines()]  # 类别名单

        # for model
        self.learning_rate = 1e-4
        self.num_epoch = 40
        self.require_improvement = 1000
        self.batch_size = 32
        self.dropout = 0.5


class PatchEmbedding(nn.Sequential):
    def __init__(self, embed_size):
        super(PatchEmbedding, self).__init__()
        # [B, 1, C, S]
        self.ConvNet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (22, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),
            nn.Dropout(0.5),
        )
        self.projection = nn.Sequential(
            nn.Conv2d(40, embed_size, (1, 1), stride=(1, 1)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x):
        x = self.ConvNet(x)
        x = self.projection(x)
        return x


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res  # residual add
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.embed_size = emb_size
        self.num_heads = num_heads
        self.K = nn.Linear(emb_size, emb_size)
        self.Q = nn.Linear(emb_size, emb_size)
        self.V = nn.Linear(emb_size, emb_size)
        self.drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.Q(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.K(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.V(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
        scale = self.embed_size ** (1/2)
        att = F.softmax(energy / scale, dim=-1)
        att = self.drop(att)
        out = torch.einsum('bhal, bhlv -> bhav', att, values)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.projection(out)
        return out


class FeedForwardBlock(nn.Module):
    def __init__(self, embed_size, expansion, drop_p):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embed_size, expansion * embed_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * embed_size, embed_size)
        )

    def forward(self, x):
        out = self.fc(x)
        return out


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_size, num_heads=5, drop_p=0.5, forward_expansion=4, forward_drop_p=0.5):
        super(TransformerEncoderBlock, self).__init__()
        self.attention = ResidualAdd(nn.Sequential(
            nn.LayerNorm(embed_size),
            MultiHeadAttention(embed_size, num_heads, drop_p),
            nn.Dropout(drop_p),
        ))
        self.feedforward = ResidualAdd(nn.Sequential(
            nn.LayerNorm(embed_size),
            FeedForwardBlock(
                embed_size, expansion=forward_expansion, drop_p=forward_drop_p
            ),
            nn.Dropout(drop_p),
        ))

    def forward(self, x):
        x = self.attention(x)
        x = self.feedforward(x)
        return x


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, embed_size):
        super().__init__(*[TransformerEncoderBlock(embed_size) for _ in range(depth)])


class ClassificationHead(nn.Module):
    def __init__(self, embed_size, n_classes):
        super(ClassificationHead, self).__init__()
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(embed_size),
            nn.Linear(embed_size, n_classes),
        )
        self.fc = nn.Sequential(
            nn.Linear(21960, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes)
        )

    def forward(self, x):
        x = x.contiguous().view(x.shape[0], -1)
        out = self.fc(x)
        return out


class Model(nn.Module):
    def __init__(self, config, embed_size=40, depth=3, n_classes=3, **kwargs):
        super(Model, self).__init__()
        self.conv = PatchEmbedding(embed_size)
        self.transformer = TransformerEncoder(depth, embed_size)
        self.class_fc = ClassificationHead(embed_size, n_classes)
        self.config = config

    def forward(self, x):
        # [batch_size, 1, 30, 1001]
        x = x.permute(0, 2, 1)  # [batch_size channels sample]
        x = x.view(x.shape[0], 1, 30, -1)  # [batch_size, 1, channels, sample]
        x = self.conv(x)
        x = self.transformer(x)
        x = self.class_fc(x)
        return x
