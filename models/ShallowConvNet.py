import torch
import torch.nn as nn
import torch.nn.functional as F


def square(x):
    return x * x


def safe_log(x, eps=1e-6):
    """ Prevents :math:`log(0)` by using :math:`log(max(x, eps))`."""
    return torch.log(torch.clamp(x, min=eps))


class Config(object):
    def __init__(self, dataset):
        self.model_name = 'ShallowConvNet'
        self.data_path = dataset + '/raw'
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
        self.batch_size = 32


class Model(torch.nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.in_chans = 30
        self.n_classes = 3
        self.n_filters_time = 40
        self.fitler_time_length = 25
        self.n_filters_spat = 40
        self.pool_time_length = 75
        self.pool_time_stride = 15
        self.final_conv_length = 30
        self.batch_norm = True
        self.batch_norm_alpha = 0.1
        self.drop_prob = 0.5
        self.conv_nonlin = square   # better
        self.pool_nonlin = safe_log
        # self.conv_nonlin = nn.ReLU()

        # block
        self.conv_time = nn.Conv2d(1, self.n_filters_time, (1, self.fitler_time_length))
        self.conv_spat = nn.Conv2d(self.n_filters_time, self.n_filters_spat, (self.in_chans, 1))
        self.bn1 = nn.BatchNorm2d(self.n_filters_spat, momentum=self.batch_norm_alpha, affine=True)
        self.conv_nonlin_exp = Expression(self.conv_nonlin)
        self.pool1 = nn.AvgPool2d(kernel_size=(1, self.pool_time_length), stride=(1, self.pool_time_stride)) # a little bit better
        self.pool_nonlin_log = Expression(self.pool_nonlin)
        self.drop = nn.Dropout(self.drop_prob)
        # self.pool1 = nn.MaxPool2d(kernel_size=(1, self.pool_time_length), stride=(1, self.pool_time_stride))
        self.conv_classifier = nn.Conv2d(self.n_filters_spat, self.n_classes, (1, 61), bias=True)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # [batch_size, 1, 30, 1001]
        x = x.permute(0, 2, 1)  # [batch_size channels sample]
        x = x.view(x.shape[0], 1, 30, -1)  # [batch_size, 1, channels, sample]
        x = self.conv_time(x)
        # print(x.shape)
        x = self.conv_spat(x)
        # print(x.shape)
        x = self.bn1(x)
        x = self.conv_nonlin_exp(x)
        x = self.pool1(x)  # [batch_size, 40, 1, 60]
        # print(x.shape)
        x = self.pool_nonlin_log(x)
        x = self.drop(x)
        x = self.conv_classifier(x)
        # print(x.shape)
        out = x.reshape(x.shape[0], -1)
        # out = self.softmax(x)
        # print(out.shape)
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
