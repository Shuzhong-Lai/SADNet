import torch


class Config(object):
    def __init__(self, dataset):
        self.model_name = 'InterpretableCNN'
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


class Model(torch.nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.classes = 3
        self.sampleChannel = 30
        self.sampleLength = 1001
        self.N1 = 16
        self.d = 2
        self.kernelLength = 250  # half of the sample rate
        self.conv1 = torch.nn.Conv2d(1, self.N1, (self.sampleChannel, 1))
        self.conv2 = torch.nn.Conv2d(self.N1, self.N1 * self.d, (1, self.kernelLength))
        self.activate = torch.nn.ReLU()
        self.batchNorm = torch.nn.BatchNorm2d(self.d * self.N1, track_running_stats=False)
        self.GAP = torch.nn.AvgPool2d(1, self.sampleLength - self.kernelLength + 1)
        self.fc = torch.nn.Linear(self.d * self.N1, self.classes)
        # self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, x):
        # [batch_size, 1, 30, 1001]
        x = x.permute(0, 2, 1)  # [batch_size channels sample]
        x = x.view(x.shape[0], 1, 30, -1)  # [batch_size, 1, channels, sample]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.activate(x)
        x = self.batchNorm(x)
        x = self.GAP(x)
        x = x.view(x.shape[0], -1)
        out = self.fc(x)
        # out = self.softmax(x)
        return out
