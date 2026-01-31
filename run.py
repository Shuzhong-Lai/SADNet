# coding: UTF-8
import time
import argparse
from importlib import import_module

from train_test import train_MAG
from utils import get_time_dif, build_datasets, set_seed, build_cross_datasets
from train_test import init_network
import os
import warnings
import wandb

parser = argparse.ArgumentParser(description='bci project')
parser.add_argument('--model', type=str, default='CompactCNN')
# two mode：inner, outer
# for one subject to split the train and test dataset
# leave-one-subject-out
parser.add_argument('--mode', type=bool, default=False)
args = parser.parse_args()


# get all subjects path
def get_all_subjects(dataset):
    data_list = os.listdir(dataset + '/raw')
    all_subjects = []
    for subject_name in data_list:
        data_path = dataset + '/raw/' + subject_name  # data/raw/sxx
        all_subjects.append(data_path)
    return all_subjects

# inner subject
def inner_subject_train():
    dataset = 'data'
    # model_name = ['CompactCNN', 'DeepConvNet', 'EEGInception', 'EEGNet', 'EEGResNet', 'InterpretableCNN',
    #               'ShallowConvNet']
    model_name = ['EEGNet']
    all_subjects = get_all_subjects(dataset)
    for index in range(len(model_name)):
        x = import_module('models.' + model_name[index])
        config = x.Config(dataset)
        start_time = time.time()

        # loading data
        # for 27 subjects
        # data_list = os.listdir(dataset + '/raw')
        data_list = all_subjects
        # print(data_list)
        for subject_name in data_list:
            drop_last = True
            print('loading data...')
            # config.data_path = dataset + '/raw/' + subject_name  # data/raw/sxx
            config.data_path = subject_name
            print(config.data_path)
            train_iter, test_iter, dev_iter = build_datasets(config, subject_name[-3:], mode=False, oversample=True,
                                                             drop_last=drop_last)
            time_dif = get_time_dif(start_time)
            print("loading data usage:", time_dif)

            # train and test
            model = x.Model(config).to(config.device)
            print(model.parameters)
            # train(config, model, train_iter, dev_iter, test_iter)
            train_MAG(config, model, train_iter, dev_iter, test_iter, subject_name[-3:], "inter")


# cross subject
def leave_one_subject_out():
    dataset = 'cross'
    model_name = ['EEGNet']
    # 首先得到所有被试的文件名
    all_subjects = get_all_subjects(dataset)
    # 选model
    for index in range(len(model_name)):
        data_list = all_subjects

        x = import_module('models.' + model_name[index])
        config = x.Config(dataset)
        start_time = time.time()

        # 选一个数据集作为测试集，其余的混合后作为训练集和验证集
        for subject_name in data_list:
            print('loading data...')
            # config.data_path = dataset + '/raw/' + subject_name  # data/raw/sxx
            config.data_path = subject_name
            print(config.data_path)
            train_iter, test_iter, dev_iter = build_cross_datasets(config, subject_name[-3:])
            time_dif = get_time_dif(start_time)
            print("loading data usage:", time_dif)

            # train and test
            model = x.Model(config).to(config.device)
            print(model.parameters)
            # train(config, model, train_iter, dev_iter, test_iter)
            train_MAG(config, model, train_iter, dev_iter, test_iter, subject_name[-3:], "cross")


if __name__ == '__main__':
    set_seed()
    # ignore the warning
    warnings.filterwarnings('ignore')
    # offline
    os.environ["WANDB_API_KEY"] = '5f0253465a590ba45b8cd6f7115b70704b12deb2'
    os.environ['WANDB_MODE'] = 'offline'
    inner_subject_train()
    # leave_one_subject_out()
