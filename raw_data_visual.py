
from importlib import import_module

import mne
import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.collections import LineCollection
import matplotlib.gridspec as gridspec

from grad_cam import GradCAM
from run import get_all_subjects
from scipy import signal

if __name__ == '__main__':
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
    # print(x_data.shape)
    sample_data = x_data.reshape(-1, 30, 1001)
    # f_fft = signal.resample(sample_data[0][0], 384)
    # x = np.linspace(0, 10001, 1001, endpoint=False)
    # xnew = np.linspace(0, 10001, 384, endpoint=False)
    # print(f_fft.shape)
    # plt.plot(xnew, f_fft, 'b.-')
    # plt.plot(x, sample_data[0][0])
    # plt.legend(['resample', 'data'], loc='best')
    # plt.show()

    id = 152

    input_tensor = torch.Tensor(x_data[id]).unsqueeze(0)
    label = y_data[id]
    out = model(input_tensor)
    # print(out)
    # print(label)
    target_layers = [model.transformer[-1]]
    cam = GradCAM(model, target_layers=target_layers, use_cuda=False)
    grayscale_cam = cam(input_tensor=input_tensor)
    heatmap = grayscale_cam[0, :].reshape(30, 1001)
    # print(grayscale_cam.shape)

    # sampleInput = x_data[id].reshape(30, 1001)
    # print(sampleInput.shape)
    sampleInput = x_data[id]
    sampleChannel = 30
    sampleLength = 384
    thespan = np.percentile(sampleInput, 98)
    xx = np.arange(1, sampleLength + 1)
    fig = plt.figure(figsize=(24, 6))
    gridlayout = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)
    ax0 = fig.add_subplot(gridlayout[0, 0])
    sampleInput = signal.resample(sampleInput, sampleLength)
    sampleInput = sampleInput.reshape(30, -1)

    print(heatmap.shape)
    resample_heatmap = signal.resample(heatmap.reshape(-1, 30), sampleLength).reshape(30, -1)
    for i in range(0, sampleChannel):
        y = sampleInput[i, :] + thespan * (sampleChannel - 1 - i)
        # print(y.shape)
        dydx = heatmap[i, :]
        points = np.array([xx, y]).T.reshape(-1, 1, 2)
        # print(points.shape)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        # print(segments.shape)
        norm = plt.Normalize(-1, 1)
        lc = LineCollection(segments, norm=norm)
        lc.set_linewidth(2)
        lc.set_array(dydx)
        ax0.add_collection(lc)
    yttics = np.zeros(sampleChannel)
    for gi in range(sampleChannel):
        yttics[gi] = gi * thespan
    ax0.set_ylim([-thespan, thespan*sampleChannel])
    ax0.set_xlim([0, sampleLength+1])
    # ax0.set_xticks([1, 200, 400, 600, 800, 1001])
    ax0.set_xticks([1, 100, 200, 300, 384])
    channelnames = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FT7', 'FC3', 'FCz', 'FC4', 'FT8', 'T3', 'C3', 'Cz',
                    'C4', 'T4', 'TP7', 'CP3', 'CPz', 'CP4', 'TP8', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'Oz', 'O2']
    inversechannelnames = []
    for i in range(sampleChannel):
        inversechannelnames.append(channelnames[sampleChannel-1-i])
    plt.sca(ax0)
    plt.yticks(yttics, inversechannelnames)
    plt.show()
    # plt.savefig('showData.jpg')