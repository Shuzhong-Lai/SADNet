import torch


class Config(object):
    def __init__(self, dataset):
        self.model_name = 'EEGNet'
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
        self.dropout = 0.5

        # for EEGNet
        self.dropout = 0.5
        self.kernLength = 250  # half of the samples
        self.F1 = 8
        self.D = 2
        self.F2 = self.F1 * self.D
        self.norm_rate = 0.25
        self.samples = 500
        self.nb_classes = 3
        self.dropout = 0.5
        self.C = 30
        self.T = 1001


class Model(torch.nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        # block 1
        self.conv1 = torch.nn.Conv2d(1, self.config.F1, (1, 1))
        self.bn1 = torch.nn.BatchNorm2d(self.config.F1, track_running_stats=False)
        self.depthWiseConv2d = torch.nn.Conv2d(self.config.F1, self.config.D * self.config.F1, kernel_size=(1, 1),
                                               padding=0, groups=self.config.F1)
        self.bn2 = torch.nn.BatchNorm2d(self.config.D * self.config.F1, track_running_stats=False)
        self.activation1 = torch.nn.ELU()
        self.GAP1 = torch.nn.AvgPool2d((1, 4))
        self.dropout1 = torch.nn.Dropout(self.config.dropout)

        # block 2
        self.pointWiseConv2d = torch.nn.Conv2d(self.config.D * self.config.F1, self.config.F2, (self.config.C, 1))
        self.bn3 = torch.nn.BatchNorm2d(self.config.F2)
        self.activation2 = torch.nn.ELU()
        self.GAP2 = torch.nn.AvgPool2d((1, 8))
        self.dropout2 = torch.nn.Dropout(self.config.dropout)

        # fc
        self.fc = torch.nn.Linear(self.config.F2 * (self.config.T // 32), self.config.nb_classes)
        # self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        # [batch_size, 1, 30, 1001]
        x = x.permute(0, 2, 1)  # [batch_size channels sample]
        x = x.view(x.shape[0], 1, 30, -1)  # [batch_size, 1, channels, sample]
        # block 1
        x = self.conv1(x)
        x = self.bn1(x)
        # print(x.shape)
        x = self.depthWiseConv2d(x)
        # print(x.shape)
        x = self.bn2(x)
        x = self.activation1(x)
        x = self.GAP1(x)
        # print(x.shape)
        x = self.dropout1(x)

        # block 2
        x = self.pointWiseConv2d(x)
        # print(x.shape)
        x = self.bn3(x)
        x = self.activation2(x)
        x = self.GAP2(x)
        # print(x.shape)
        x = self.dropout2(x)

        # fc
        # print(x.shape)
        x = x.view(x.shape[0], -1)
        # print(x.shape)
        out = self.fc(x)
        # print(x.shape)
        # out = self.softmax(x)
        return out
