# 用于进行消融实验
# model1：仅convEmbedding+cf

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torch import Tensor
import numpy as np


def square(x):
    return x * x


def safe_log(x, eps=1e-6):
    """ Prevents :math:`log(0)` by using :math:`log(max(x, eps))`."""
    return torch.log(torch.clamp(x, min=eps))


class Config(object):
    def __init__(self, dataset):
        self.model_name = 'DSANet'
        self.data_path = dataset + '/raw'
        self.num_classes = 3
        self.f1_save_path = dataset + '/saved_dict/'  # 模型训练结果
        self.auc_save_path = dataset + '/saved_dict/'  # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
        self.class_list = [x.strip() for x in open(
            dataset + '/class.txt', encoding='utf-8').readlines()]  # 类别名单

        # for model
        self.learning_rate = 1e-3
        self.num_epoch = 100
        self.require_improvement = 1000
        self.batch_size = 32
        self.dropout = 0.5


class ConvEmbedding(nn.Sequential):
    def __init__(self, embed_size):
        super(ConvEmbedding, self).__init__()
        self.n_classes = 3
        self.in_chans = 30
        self.samples = 1001
        self.dropout = 0.5
        self.n_filters_time = 25
        self.n_filters_spat = 25
        self.filter_time_length = 10
        self.pool_time_length = 3
        self.pool_time_stride = 3
        self.n_filters_2 = 50
        self.filter_length_2 = 10
        self.n_filters_3 = 100
        self.filter_length_3 = 10
        self.n_filters_4 = 200
        self.filter_length_4 = 10
        self.drop_prob = 0.5
        self.batch_norm_alhpa = 0.1
        self.n_filters_conv = self.n_filters_spat
        self.conv_stride = 1
        self.conv_nonlinear = square
        # block 1
        self.conv_time = nn.Conv2d(1, self.n_filters_time, (1, self.filter_time_length))
        self.conv_spat = nn.Conv2d(self.n_filters_time, self.n_filters_spat, (self.in_chans, 1))
        self.bn1 = nn.BatchNorm2d(self.n_filters_conv, momentum=self.batch_norm_alhpa, affine=True, eps=1e-5)
        # self.activ1 = nn.ELU()
        self.activ1 = Expression(self.conv_nonlinear)
        self.pool1 = nn.MaxPool2d((1, self.pool_time_length), stride=(1, self.pool_time_stride))
        # block 2
        self.block2 = self._make_conv_pool_layer(self.n_filters_conv, self.n_filters_2, self.filter_length_2)

        # block 3
        # self.block3 = self._make_conv_pool_layer(self.n_filters_2, self.n_filters_3, self.filter_length_3)

        # block 4
        # self.block4 = self._make_conv_pool_layer(self.n_filters_3, self.n_filters_4, self.filter_length_4)

        # projection
        self.projection = nn.Sequential(
            nn.Conv2d(self.n_filters_2, embed_size, (1, 1), stride=(1, 1)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x):
        # [batch_size, 1, 30, 1001]
        # block 1
        x = self.conv_time(x)
        # print(x.shape)
        x = self.conv_spat(x)
        x = self.bn1(x)
        x = self.activ1(x)
        x = self.pool1(x)
        # print(x.shape)

        # block 2, 3, 4
        x = self.block2(x)
        # print(x.shape)
        # x = self.block3(x)
        # print(x.shape)
        # x = self.block4(x)
        # print(x.shape)
        out = self.projection(x)
        # print(out.shape)
        return out

    def _make_conv_pool_layer(self, in_chans, out_chans, filter_length):
        return nn.Sequential(*[
            nn.Dropout(self.drop_prob),
            nn.Conv2d(in_chans, out_chans, (1, filter_length), stride=(self.conv_stride, 1)),
            nn.BatchNorm2d(out_chans, momentum=self.batch_norm_alhpa, affine=True, eps=1e-5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, self.pool_time_length), stride=(1, self.pool_time_stride))
        ])


class PatchEmbedding(nn.Sequential):
    def __init__(self, embed_size):
        super(PatchEmbedding, self).__init__()
        # [B, 1, C, S]
        self.ConvNet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (30, 1), (1, 1)),
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


class ResidualAdd(nn.Sequential):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res  # residual add
        return x


class MultiHeadAttention(nn.Sequential):
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
        scale = self.embed_size ** (1 / 2)
        att = F.softmax(energy / scale, dim=-1)
        att = self.drop(att)
        out = torch.einsum('bhal, bhlv -> bhav', att, values)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.projection(out)
        return out


class FeedForwardBlock(nn.Sequential):
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


class TransformerEncoderBlock(nn.Sequential):
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


class ClassificationHead(nn.Sequential):
    def __init__(self, embed_size, n_classes):
        super(ClassificationHead, self).__init__()
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(embed_size),
            nn.Linear(embed_size, n_classes),
        )
        self.fc = nn.Sequential(
            nn.Linear(4280, 256),
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


class Model(nn.Sequential):
    def __init__(self, config, embed_size=40, depth=3, n_classes=3, **kwargs):
        super(Model, self).__init__()
        self.conv = ConvEmbedding(embed_size)
        # self.pos_embeding = Positional_Encoding(embed_size, 7, 0.5, config.device)
        self.transformer = TransformerEncoder(depth, embed_size)
        self.class_fc = ClassificationHead(embed_size, n_classes)
        self.config = config

    def forward(self, x):
        # [batch_size, 1, 30, 1001]
        x = x.permute(0, 2, 1)  # [batch_size channels sample]
        x = x.view(x.shape[0], 1, 30, -1)  # [batch_size, 1, channels, sample]
        x = self.conv(x)
        # print(x.shape)
        # x = self.pos_embeding(x)
        x = self.transformer(x)
        x = self.class_fc(x)
        return x


# 位置编码
class Positional_Encoding(nn.Module):
    def __init__(self, embed, pad_size, dropout, device):
        super(Positional_Encoding, self).__init__()
        self.device = device
        self.pe = torch.tensor(
            [[pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)] for pos in range(pad_size)])
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = x + nn.Parameter(self.pe, requires_grad=False).to(self.device)
        # out = x + nn.Parameter(self.pe, requires_grad=False)
        out = self.dropout(out)
        return out


class Expression(nn.Module):
    """Compute given expression on forward pass.

    Parameters
    ----------
    expression_fn : callable
        Should accept variable number of objects of type
        `torch.autograd.Variable` to compute its output.
    """

    def __init__(self, expression_fn):
        super(Expression, self).__init__()
        self.expression_fn = expression_fn

    def forward(self, *x):
        return self.expression_fn(*x)

    def __repr__(self):
        if hasattr(self.expression_fn, "func") and hasattr(
                self.expression_fn, "kwargs"
        ):
            expression_str = "{:s} {:s}".format(
                self.expression_fn.func.__name__, str(self.expression_fn.kwargs)
            )
        elif hasattr(self.expression_fn, "__name__"):
            expression_str = self.expression_fn.__name__
        else:
            expression_str = repr(self.expression_fn)
        return (
                self.__class__.__name__ +
                "(expression=%s) " % expression_str
        )
