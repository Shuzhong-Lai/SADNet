import torch
import torch.nn as nn
import torch.nn.functional as F


class Config(object):
    def __init__(self, dataset):
        self.model_name = 'DeepConvNet'
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
        self.num_epoch = 100
        self.require_improvement = 1000
        self.batch_size = 32


class Model(torch.nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
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

        # block 1
        self.conv_time = nn.Conv2d(1, self.n_filters_time, (1, self.filter_time_length))
        self.conv_spat = nn.Conv2d(self.n_filters_time, self.n_filters_spat, (self.in_chans, 1))
        self.bn1 = nn.BatchNorm2d(self.n_filters_conv, momentum=self.batch_norm_alhpa, affine=True, eps=1e-5)
        self.activ1 = nn.ELU()
        self.pool1 = nn.MaxPool2d((1, self.pool_time_length), stride=(1, self.pool_time_stride))

        # block 2
        self.block2 = self._make_conv_pool_layer(self.n_filters_conv, self.n_filters_2, self.filter_length_2)

        # block 3
        self.block3 = self._make_conv_pool_layer(self.n_filters_2, self.n_filters_3, self.filter_length_3)

        # block 4
        self.block4 = self._make_conv_pool_layer(self.n_filters_3, self.n_filters_4, self.filter_length_4)

        # fc
        # params(kernel_size) of conv_fc is not clear now
        self.conv_fc = nn.Conv2d(self.n_filters_4, self.n_classes, (1, 7))
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # [batch_size, 1, 30, 1001]
        x = x.permute(0, 2, 1)  # [batch_size channels sample]
        x = x.view(x.shape[0], 1, 30, -1)  # [batch_size, 1, channels, sample]

        # block 1
        x = self.conv_time(x)
        # print(x.shape)
        x = self.conv_spat(x)
        x = self.bn1(x)
        x = self.activ1(x)
        x = self.pool1(x)

        # block 2, 3, 4
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        # fc
        x = self.conv_fc(x)     # [batch_size, 3, 1, 1]
        out = x.reshape(x.shape[0], -1)
        # out = self.softmax(x)
        return out

    def _make_conv_pool_layer(self, in_chans, out_chans, filter_length):
        return nn.Sequential(*[
            nn.Dropout(self.drop_prob),
            nn.Conv2d(in_chans, out_chans, (1, filter_length), stride=(self.conv_stride, 1)),
            nn.BatchNorm2d(out_chans, momentum=self.batch_norm_alhpa, affine=True, eps=1e-5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, self.pool_time_length), stride=(1, self.pool_time_stride))
        ])
