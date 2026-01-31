# coding: UTF-8
import collections
import os
import random

import torch
import numpy as np
import pandas as pd
import time
from datetime import timedelta
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import collections
from sklearn.model_selection import train_test_split


# 设置随机数种子，方便复现
def set_seed():
    np.random.seed(27)
    torch.manual_seed(27)  # 设置CPU生成随机数的种子，方便下次复现实验结果
    torch.cuda.manual_seed_all(27)  # 在GPU中设置生成随机数的种子。当设置的种子固定下来的时候，之后依次pytorch生成的随机数序列也被固定下来
    torch.backends.cudnn.deterministic = True  # 保证每次运行网络的时候相同输入的输出是固定的
    random.seed(27)
    os.environ["PYTHONHASHSEED"] = str(27)


# 获取时间差
def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def split_data(X, y):
    # transfer to 2-d array
    y_index = X.reshape(X.shape[0], -1).shape[1]
    data = np.concatenate((X.reshape(X.shape[0], -1), y.reshape(-1, 1)), axis=1)
    data = pd.DataFrame(data)
    # sort by y_label
    data = data.sort_values(y_index)
    # reset new index
    data.reset_index(drop=True, inplace=True)
    # collect the number of labels
    c = collections.Counter(data[y_index])
    num = c[0]
    data_0 = data.iloc[:num, :]
    data_1 = data.iloc[num:2 * num, :]
    data_2 = data.iloc[-num:, :]
    # split the data
    x0_train, x0_test, y0_train, y0_test = train_test_split(data_0, data_0[y_index], test_size=0.2, random_state=27)
    x1_train, x1_test, y1_train, y1_test = train_test_split(data_1, data_1[y_index], test_size=0.2, random_state=27)
    x2_train, x2_test, y2_train, y2_test = train_test_split(data_2, data_2[y_index], test_size=0.2, random_state=27)

    x0_dev, x0_test, y0_dev, y0_test = train_test_split(x0_test, y0_test, test_size=0.5, random_state=27)
    x1_dev, x1_test, y1_dev, y1_test = train_test_split(x1_test, y1_test, test_size=0.5, random_state=27)
    x2_dev, x2_test, y2_dev, y2_test = train_test_split(x2_test, y2_test, test_size=0.5, random_state=27)

    # concatenate
    train_data = x0_train.append(x1_train)
    dev_data = x0_dev.append(x1_dev)
    test_data = x0_test.append(x1_test)

    train_data = train_data.append(x2_train)
    dev_data = dev_data.append(x2_dev)
    test_data = test_data.append(x2_test)

    train_data = train_data.astype(np.float32)
    dev_data = dev_data.astype(np.float32)
    test_data = test_data.astype(np.float32)

    # print(train_data.shape)
    # print(dev_data.shape)
    # print(test_data.shape)
    # print(len(collections.Counter(test_data[30030])))
    return train_data, dev_data, test_data


def trans2DataLoader(data, batch_size=32, shuffle=True, drop_last=False):
    # drop_last：当batch_size较大时，可能会drop掉某类别的全部标签导致分类出错
    X = data.iloc[:, :-1].to_numpy()
    y = data.iloc[:, -1:].to_numpy()
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(X.reshape(X.shape[0], -1, 30)), torch.from_numpy(y))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return loader


# use for leave-one-subject-experiment
def build_cross_datasets(config, subject_name):
    # subject_name : sxx
    print("build datasets...")
    batch_size = config.batch_size
    shuffle = True
    drop_last = True

    train_x = np.load(config.data_path+'/train_'+subject_name+'_x.npy')
    train_y = np.load(config.data_path+'/train_'+subject_name+'_y.npy')

    dev_x = np.load(config.data_path+'/dev_'+subject_name+'_x.npy')
    dev_y = np.load(config.data_path+'/dev_'+subject_name+'_y.npy')

    test_x = np.load(config.data_path+'/test_'+subject_name+'_x.npy')
    test_y = np.load(config.data_path+'/test_'+subject_name+'_y.npy')

    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    dev_dataset = torch.utils.data.TensorDataset(torch.from_numpy(dev_x), torch.from_numpy(dev_y))
    test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

    return train_loader, dev_loader, test_loader


def build_datasets(config, subject_name, mode=False, oversample=True, drop_last=False):
    global X, y
    print("build datasets...")
    if not mode:
        X = np.load(config.data_path + '/' + subject_name + '_x.npy')
        y = np.load(config.data_path + '/' + subject_name + '_y.npy')
        print(X.shape)
        print(y.shape)
    # if oversample:
    #     ros = RandomOverSampler(random_state=27)
    # else:
    #     ros = RandomUnderSampler(random_state=27)
    # X_res, y_res = ros.fit_resample(X.reshape(X.shape[0], -1), y.reshape(-1, 1))
    # print(X_res.shape)
    # print(y_res.shape)
    # X_res = X_res.reshape(X_res.shape[0], -1, 30)  # reshape
    # train_data, test_data, dev_data = split_data(X_res, y_res)
    train_data, test_data, dev_data = split_data(X, y)
    train_loader = trans2DataLoader(train_data, config.batch_size, drop_last=drop_last)
    dev_loader = trans2DataLoader(dev_data, config.batch_size, drop_last=drop_last)
    test_loader = trans2DataLoader(test_data, config.batch_size, drop_last=drop_last)
    return train_loader, dev_loader, test_loader


def sync_offline2WB(path):
    data_list = os.listdir(path)
    for name in data_list:
        command = 'wandb sync wandb/c/' + name
        os.system(command)
    print("finish offline upload!")


if __name__ == '__main__':
    sync_offline2WB('wandb/c')
