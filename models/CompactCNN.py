import torch


class Config(object):
    def __init__(self, dataset):
        self.model_name = 'CompactCNN'
        self.data_path = dataset+'/raw'
        self.num_classes = 3
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'  # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
        self.class_list = [x.strip() for x in open(
            dataset + '/class.txt', encoding='utf-8').readlines()]  # 类别名单

        # for model
        self.learning_rate = 1e-2
        self.num_epoch = 40
        self.require_improvement = 1000
        self.batch_size = 32
        self.dropout = 0.5


class Model(torch.nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.kernelLength = 250  # half of the sample rate 500HZ
        self.channels = 32
        self.sampleLength = 1001
        self.num_classes = 3
        self.conv = torch.nn.Conv2d(1, self.channels, (1, self.kernelLength))
        self.batch = Batchlayer(self.channels)
        self.GAP = torch.nn.AvgPool2d((30, self.sampleLength - self.kernelLength + 1))
        self.fc = torch.nn.Linear(self.channels, self.num_classes)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [batch_size channels sample]
        x = x.view(x.shape[0], 1, 30, -1)  # [batch_size, 1, channels, sample]
        x = self.conv(x)  # [batch_size, 32, 30, 752]
        x = self.batch(x)
        x = torch.nn.ELU()(x)
        x = self.GAP(x)  # [batch_size, 32, 1, 1]
        x = x.view(x.shape[0], -1)
        out = self.fc(x)
        out = self.softmax(out)
        # print(out.shape)
        return out


class Batchlayer(torch.nn.Module):
    def __init__(self, dim):
        super(Batchlayer, self).__init__()
        self.gamma = torch.nn.Parameter(torch.Tensor(1, dim, 1, 1))
        self.beta = torch.nn.Parameter(torch.Tensor(1, dim, 1, 1))
        self.gamma.data.uniform_(-0.1, 0.1)
        self.beta.data.uniform_(-0.1, 0.1)

    def forward(self, input):
        data = normalizelayer(input)
        gammamatrix = self.gamma.expand(int(data.size(0)), int(data.size(1)), int(data.size(2)), int(data.size(3)))
        betamatrix = self.beta.expand(int(data.size(0)), int(data.size(1)), int(data.size(2)), int(data.size(3)))

        return data * gammamatrix + betamatrix


def normalizelayer(data):
    eps = 1e-05
    a_mean = data - torch.mean(data, [0, 2, 3], True).expand(int(data.size(0)), int(data.size(1)), int(data.size(2)),
                                                             int(data.size(3)))
    b = torch.div(a_mean, torch.sqrt(torch.mean((a_mean) ** 2, [0, 2, 3], True) + eps).expand(int(data.size(0)),
                                                                                              int(data.size(1)),
                                                                                              int(data.size(2)),
                                                                                              int(data.size(3))))

    return b
