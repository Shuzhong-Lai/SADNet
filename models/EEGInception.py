import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import prod


class Config(object):
    def __init__(self, dataset):
        self.model_name = 'EEGInception'
        self.data_path = dataset+'/raw'
        self.num_classes = 3
        self.f1_save_path = dataset + '/saved_dict/f1_' + self.model_name + '.ckpt'  # 模型训练结果
        self.auc_save_path = dataset + '/saved_dict/auc_' + self.model_name + '.ckpt'  # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
        self.class_list = [x.strip() for x in open(
            dataset + '/class.txt', encoding='utf-8').readlines()]  # 类别名单

        # for model
        self.learning_rate = 1e-3
        self.num_epoch = 40
        self.require_improvement = 1000
        self.batch_size = 64


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.in_channels = 30
        self.n_classes = 3
        self.sfreq = 500
        self.drop_prob = 0.5
        self.scales_samples_s = (0.5, 0.25, 0.125)
        self.n_filters = 8
        self.activation = nn.ELU()
        self.batch_norm_alpha = 0.01
        self.depth_multiplier = 2
        self.pooling_sizes = (4, 2, 2, 2)
        self.scales_samples = tuple(
            int(size_s * self.sfreq) for size_s in self.scales_samples_s
        )
        self.input_window_samples = 1001

        # block1
        self.block11 = self._get_inception_branch_1(
            self.in_channels,
            self.n_filters,
            kernel_length=self.scales_samples[0],
            alpha_momentum=self.batch_norm_alpha,
            activation=self.activation,
            drop_out=self.drop_prob,
            depth_mutliplier=self.depth_multiplier
        )
        self.block12 = self._get_inception_branch_1(
            self.in_channels,
            self.n_filters,
            kernel_length=self.scales_samples[1],
            alpha_momentum=self.batch_norm_alpha,
            activation=self.activation,
            drop_out=self.drop_prob,
            depth_mutliplier=self.depth_multiplier
        )
        self.block13 = self._get_inception_branch_1(
            self.in_channels,
            self.n_filters,
            kernel_length=self.scales_samples[2],
            alpha_momentum=self.batch_norm_alpha,
            activation=self.activation,
            drop_out=self.drop_prob,
            depth_mutliplier=self.depth_multiplier
        )
        self.inception_block1 = _InceptionBlock((self.block11, self.block12, self.block13))
        self.avg_pool1 = nn.AvgPool2d((1, self.pooling_sizes[0]))

        self.n_concat_filters = len(self.scales_samples) * self.n_filters
        self.n_concat_dw_filters = self.n_concat_filters * self.depth_multiplier

        # block2
        self.block21 = self._get_inception_branch_2(
            in_channels=self.n_concat_dw_filters,
            out_channels=self.n_filters,
            kernel_length=self.scales_samples[0] // 4,
            alpha_momentum=self.batch_norm_alpha,
            activation=self.activation,
            drop_prob=self.drop_prob
        )
        self.block22 = self._get_inception_branch_2(
            in_channels=self.n_concat_dw_filters,
            out_channels=self.n_filters,
            kernel_length=self.scales_samples[1] // 4,
            alpha_momentum=self.batch_norm_alpha,
            activation=self.activation,
            drop_prob=self.drop_prob
        )
        self.block23 = self._get_inception_branch_2(
            in_channels=self.n_concat_dw_filters,
            out_channels=self.n_filters,
            kernel_length=self.scales_samples[2] // 4,
            alpha_momentum=self.batch_norm_alpha,
            activation=self.activation,
            drop_prob=self.drop_prob
        )

        self.inception_block2 = _InceptionBlock((self.block21, self.block22, self.block23))
        self.avg_pool2 = nn.AvgPool2d((1, self.pooling_sizes[1]))

        # block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(self.n_concat_filters, self.n_concat_filters // 2, (1, 8), padding='same', bias=False),
            nn.BatchNorm2d(self.n_concat_filters // 2, momentum=self.batch_norm_alpha),
            self.activation,
            nn.Dropout(self.drop_prob),
            nn.AvgPool2d((1, self.pooling_sizes[2])),
            nn.Conv2d(self.n_concat_filters // 2, self.n_concat_filters // 4, (1, 4), padding='same', bias=False),
            nn.BatchNorm2d(self.n_concat_filters // 4, momentum=self.batch_norm_alpha),
            self.activation,
            nn.Dropout(self.drop_prob),
            nn.AvgPool2d((1, self.pooling_sizes[3]))
        )

        # classifier
        spatial_dim_last_layer = (
                self.input_window_samples // prod(self.pooling_sizes))
        n_channels_last_layer = self.n_filters * len(self.scales_samples) // 4

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(spatial_dim_last_layer * n_channels_last_layer, self.n_classes),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        # [batch_size, 1, 30, 1001]
        x = x.permute(0, 2, 1)  # [batch_size channels sample]
        x = x.view(x.shape[0], 1, 30, -1)  # [batch_size, 1, channels, sample]

        # inception Block
        x = self.inception_block1(x)  # [batch_size, 48, 1, 1001]
        x = self.avg_pool1(x)
        x = self.inception_block2(x)  # [batch_size, 24, 1, 250]
        x = self.avg_pool2(x)

        # final block and fc softmax
        x = self.block3(x)  # [batch_size, 6, 1, 31]
        out = self.fc(x)
        return out

    @staticmethod
    def _get_inception_branch_1(in_channels, out_channels, kernel_length, alpha_momentum, drop_out, activation,
                                depth_mutliplier):
        return nn.Sequential(
            nn.Conv2d(
                1, out_channels, kernel_size=(1, kernel_length), padding='same', bias=True),
            nn.BatchNorm2d(out_channels, momentum=alpha_momentum),
            activation,
            nn.Dropout(drop_out),
            _DepthwiseConv2d(out_channels, kernel_size=(in_channels, 1), depth_multiplier=depth_mutliplier, bias=False,
                             padding='valid'),
            nn.BatchNorm2d(depth_mutliplier * out_channels, momentum=alpha_momentum),
            activation,
            nn.Dropout(drop_out)
        )

    @staticmethod
    def _get_inception_branch_2(in_channels, out_channels, kernel_length, alpha_momentum, drop_prob, activation):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_length), padding='same', bias=False),
            nn.BatchNorm2d(out_channels, momentum=alpha_momentum),
            activation,
            nn.Dropout(drop_prob)
        )


class _DepthwiseConv2d(torch.nn.Conv2d):
    def __init__(
            self,
            in_channels,
            depth_multiplier=2,
            kernel_size=3,
            stride=1,
            padding=0,
            dilation=1,
            bias=True,
            padding_mode="zeros",
    ):
        out_channels = in_channels * depth_multiplier
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
            padding_mode=padding_mode,
        )


class _InceptionBlock(nn.Module):
    def __init__(self, branches):
        super().__init__()
        self.branches = nn.ModuleList(branches)

    def forward(self, x):
        return torch.cat([branch(x) for branch in self.branches], 1)
