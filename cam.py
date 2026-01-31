from importlib import import_module

import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
from einops import rearrange
from torchcam.utils import overlay_mask
# from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from grad_cam import GradCAM, show_cam_on_image
from pytorch_grad_cam import GuidedBackpropReLUModel

from run import get_all_subjects

import mne


def reshape_transform(tensor):
    new_tensor = rearrange(tensor, 'b (h w) c -> b (h) c (w)', h=1)
    return new_tensor


def get_predict(model, x_data, y_data):
    subject_predict_normal = []
    subject_predict_poor = []
    subject_predict_optimal = []
    error = []
    correct = []
    for i in range(x_data.shape[0]):
        input_tensor = torch.Tensor(x_data[i]).unsqueeze(0)
        label = y_data[i]
        out = model(input_tensor)
        if int(label) == 0:
            subject_predict_normal.append(i)
        elif int(label) == 1:
            subject_predict_poor.append(i)
        elif int(label) == 2:
            subject_predict_optimal.append(i)

        if int(label) != int(out.detach().numpy().argmax(axis=-1)):
            error.append(i)
        elif int(label) == int(out.detach().numpy().argmax(axis=-1)):
            if int(label) == 1 or int(label) == 2:
                correct.append(i)

    return subject_predict_normal, subject_predict_poor, subject_predict_optimal, error, correct


if __name__ == '__main__':
    state = ['normal', 'poor', 'optimal']
    dataset = 'data'
    model_name = 'DSANet'
    all_subjects = get_all_subjects(dataset)
    x = import_module('models.' + model_name)
    config = x.Config(dataset)
    config.batch_size = 1
    model = x.Model(config)
    checkpoint = torch.load(config.f1_save_path)
    model.load_state_dict(checkpoint)
    model.eval()
    subject = 's53'
    data_path = 'data/raw/'
    x_data = np.load(data_path + subject + '/' + subject + '_x.npy')
    y_data = np.load(data_path + subject + '/' + subject + '_y.npy')
    id = 0
    # print(model.parameters)
    input_tensor = torch.Tensor(x_data[id]).unsqueeze(0)
    label = y_data[id]
    out = model(input_tensor)

    normal, poor, optimal, error, correct = get_predict(model, x_data, y_data)
    # print(out)
    # print(label)
    # #     # print('Label:'+state[(int(label))], 'Likelihood'+str(out.detach().numpy()))
    print(error)
    print(len(error))
    print(optimal)
    print(len(optimal))
    print(poor)
    print(len(poor))
    print(correct)
    print(len(correct))
    target_layers = [model.transformer[-1]]
    cam = GradCAM(model, target_layers=target_layers, use_cuda=False)
    grayscale_cam = cam(input_tensor=input_tensor)
    grayscale_cam = grayscale_cam[0, :]

    channelnames = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FT7', 'FC3', 'FCz', 'FC4', 'FT8', 'T3', 'C3', 'Cz',
                    'C4', 'T4', 'TP7', 'CP3', 'CPz', 'CP4', 'TP8', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'Oz', 'O2']
    montage = 'standard_1020'
    info = mne.create_info(ch_names=channelnames, sfreq=500., ch_types='eeg')
    info.set_montage(montage)
    reverse_x_data = x_data[id].reshape(30, -1) ** 2  # 电压的平方表示当前区域的能量
    reverse_cam_data = grayscale_cam.reshape(30, -1)
    topox = np.mean(reverse_x_data, axis=1)

    # print(reverse_cam_data)
    # print(reverse_x_data)
    # reverse_cam_data = np.multiply(reverse_x_data, reverse_cam_data)
    # print(reverse_cam_data)
    # print(reverse_cam_data)
    topoHeatmap = np.mean(reverse_cam_data, axis=1)
    # hypHeadtmap = topox * topoHeatmap
    # print(topox)
    # print(topoHeatmap)
    # print(hypHeadtmap)

    fig, [ax1, ax2] = plt.subplots(nrows=2)
    plt.subplot(211)
    im, cn = mne.viz.plot_topomap(topox, info, show=False, axes=ax1, res=1200, names=channelnames, outlines='head',
                                  cmap='viridis',
                                  )
    plt.subplot(212)
    im, cn = mne.viz.plot_topomap(topoHeatmap, info, show=False, axes=ax2, res=1200, names=channelnames,
                                  outlines='head',
                                  cmap='viridis',
                                  )
    fig.colorbar(im, ax=[ax1, ax2])
    fig.suptitle(
        'Subject:' + subject + ' ' + 'Label:' + state[(int(label))] + ' ' + 'Likelihood:' + str(out.detach().numpy()))
    plt.show()
