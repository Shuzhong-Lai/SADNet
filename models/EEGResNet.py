# from DeepConvNet resnet vision
import torch.nn as nn
import numpy as np
import torch
from torch._C import _infer_size


class Config(object):
    def __init__(self, dataset):
        self.model_name = 'EEGResNet'
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
        self.num_epoch = 200
        self.require_improvement = 1000
        self.batch_size = 32


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.in_chans = 30
        self.input_window_samples = 1001
        self.n_classes = 3
        self.final_pool_length = 2
        self.n_first_filters = 8
        self.first_filter_length = 2
        self.n_layers_per_block = 2
        self.nonlinearity = nn.ELU()
        self.batch_norm_alpha = 0.1
        self.batch_norm_epsilon = 1e-4
        self.n_filter_conv = self.n_first_filters

        # structure
        self.conv_time = nn.Conv2d(1, self.n_first_filters, (1, self.first_filter_length))
        self.conv_spat = nn.Conv2d(self.n_first_filters, self.n_first_filters, (self.in_chans, 1))
        self.bn1 = nn.BatchNorm2d(self.n_filter_conv, momentum=self.batch_norm_alpha, affine=True, eps=1e-5)
        self.conv_nonlin = nn.ELU()
        self.avgpool1 = nn.AdaptiveAvgPool2d((1, 250))
        cur_dilation = np.array([1, 1])
        n_cur_filters = self.n_filter_conv
        self.residual_block = nn.Sequential(
            _ResidualBlock(n_cur_filters, n_cur_filters, dilation=cur_dilation),
            _ResidualBlock(n_cur_filters, n_cur_filters, dilation=cur_dilation),
        )
        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 32))
        self.conv_classifier = nn.Conv2d(n_cur_filters, self.n_classes, (1, 32), bias=True)
        self.flatten = nn.Flatten()
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # [batch_size, 1, 30, 1001]
        x = x.permute(0, 2, 1)  # [batch_size channels sample]
        x = x.view(x.shape[0], 1, 30, -1)  # [batch_size, 1, channels, sample]

        # time and spat conv
        x = self.conv_time(x)
        x = self.conv_spat(x)
        x = self.bn1(x)
        x = self.conv_nonlin(x)
        x = self.avgpool1(x)

        # residual block
        x = self.residual_block(x)

        # conv_classifier
        x = self.avgpool2(x)
        x = self.conv_classifier(x)
        out = self.flatten(x)
        # out = self.softmax(x)
        return out


class _ResidualBlock(nn.Module):
    """
    create a residual learning building block with two stacked 3x3 convlayers as in paper
    """

    def __init__(self, in_filters,
                 out_num_filters,
                 dilation,
                 filter_time_length=3,
                 nonlinearity=nn.ELU(),
                 batch_norm_alpha=0.1, batch_norm_epsilon=1e-4):
        super(_ResidualBlock, self).__init__()
        self.n_pad_chans = out_num_filters - in_filters

        self.conv_1 = nn.Conv2d(
            in_filters, out_num_filters, (1, filter_time_length), stride=(1, 1),
            dilation=dilation,
            padding=(0, 1))
        self.bn1 = nn.BatchNorm2d(
            out_num_filters, momentum=batch_norm_alpha, affine=True,
            eps=batch_norm_epsilon)
        self.conv_2 = nn.Conv2d(
            out_num_filters, out_num_filters, (1, filter_time_length), stride=(1, 1),
            dilation=dilation,
            padding=(0, 1))
        self.bn2 = nn.BatchNorm2d(
            out_num_filters, momentum=batch_norm_alpha,
            affine=True, eps=batch_norm_epsilon)
        # also see https://mail.google.com/mail/u/0/#search/ilya+joos/1576137dd34c3127
        # for resnet options as ilya used them
        self.nonlinearity = nonlinearity

    def forward(self, x):
        stack_1 = self.nonlinearity(self.bn1(self.conv_1(x)))
        stack_2 = self.bn2(self.conv_2(stack_1))  # next nonlin after sum
        if self.n_pad_chans != 0:
            zeros_for_padding = torch.autograd.Variable(
                torch.zeros(x.size()[0], self.n_pad_chans // 2, x.size()[2], x.size()[3]))
            if x.is_cuda:
                zeros_for_padding = zeros_for_padding.cuda()
            x = torch.cat((zeros_for_padding, x, zeros_for_padding), dim=1)
        out = self.nonlinearity(x + stack_2)
        return out


