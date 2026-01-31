"merge cam.py psd.py raw_data_visual.py with one plot"
from importlib import import_module

import torch
import numpy as np
import matplotlib.pyplot as plt

import mne
from matplotlib import gridspec
from mne.time_frequency import psd_array_multitaper
from scipy.integrate import simps
from scipy.signal import resample
from matplotlib.collections import LineCollection

from grad_cam import GradCAM
from run import get_all_subjects
from cam import get_predict

state = ['normal', 'poor', 'optimal']
dataset = 'data'


def save_visualization(id, model, subject, x_data, y_data, mode, ll):
    # get label and predict
    input_tensor = torch.Tensor(x_data[id]).unsqueeze(0)
    label = y_data[id]
    out = model(input_tensor)

    # make info
    channelnames = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FT7', 'FC3', 'FCz', 'FC4', 'FT8', 'T3', 'C3', 'Cz',
                    'C4', 'T4', 'TP7', 'CP3', 'CPz', 'CP4', 'TP8', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'Oz', 'O2']
    montage = 'standard_1020'
    info = mne.create_info(ch_names=channelnames, sfreq=500., ch_types='eeg')
    info.set_montage(montage)

    # get camHeatMap
    target_layers = [model.transformer[-1]]
    cam = GradCAM(model, target_layers=target_layers, use_cuda=False)
    grayscale_cam = cam(input_tensor=input_tensor)
    grayscale_cam = grayscale_cam[0, :]

    # psd
    rawSignal = x_data[id]
    sampleLength = 1001
    sampleChannel = 30
    maxvalue = np.max(np.abs(rawSignal))
    rawsignal = rawSignal.reshape(30, 1001)
    deltapower = np.zeros(sampleChannel)
    thetapower = np.zeros(sampleChannel)
    alphapower = np.zeros(sampleChannel)
    betapower = np.zeros(sampleChannel)
    for kk in range(sampleChannel):
        psd, freqs = psd_array_multitaper(rawsignal[kk, :], 500, adaptive=True, normalization='full', verbose=0)
        freq_res = freqs[1] - freqs[0]
        totalpower = simps(psd, dx=freq_res)
        if totalpower < 0.00000001:
            deltapower[kk] = 0
            thetapower[kk] = 0
            alphapower[kk] = 0
            betapower[kk] = 0
        else:
            idx_band = np.logical_and(freqs >= 1, freqs <= 4)
            deltapower[kk] = simps(psd[idx_band], dx=freq_res) / totalpower
            idx_band = np.logical_and(freqs >= 4, freqs <= 8)
            thetapower[kk] = simps(psd[idx_band], dx=freq_res) / totalpower
            idx_band = np.logical_and(freqs >= 8, freqs <= 12)
            alphapower[kk] = simps(psd[idx_band], dx=freq_res) / totalpower
            idx_band = np.logical_and(freqs >= 12, freqs <= 30)
            betapower[kk] = simps(psd[idx_band], dx=freq_res) / totalpower
    mixpower = np.zeros((4, sampleChannel))
    mixpower[0, :] = deltapower
    mixpower[1, :] = thetapower
    mixpower[2, :] = alphapower
    mixpower[3, :] = betapower
    vmax = np.percentile(mixpower, 95)

    # plot
    fig = plt.figure(figsize=(18, 6))
    gridlayout = gridspec.GridSpec(ncols=6, nrows=2, figure=fig, wspace=0.05, hspace=0.005)
    ax0 = fig.add_subplot(gridlayout[0:2, 0:2])
    ax11 = fig.add_subplot(gridlayout[0:1, 2:4])
    ax12 = fig.add_subplot(gridlayout[1:2, 2:4])
    ax21 = fig.add_subplot(gridlayout[0, 4])
    ax22 = fig.add_subplot(gridlayout[0, 5])
    ax23 = fig.add_subplot(gridlayout[1, 4])
    ax24 = fig.add_subplot(gridlayout[1, 5])

    # raw_data_visual including resample
    resampleLength = 202  # 128Hz
    heatmap = grayscale_cam
    sampleInput = x_data[id]
    thespan = np.percentile(sampleInput, 98)
    xx = np.arange(1, resampleLength + 1)
    sampleInput = resample(sampleInput, resampleLength).reshape(30, -1)
    heatmap = resample(heatmap, resampleLength).reshape(30, -1)
    for i in range(0, sampleChannel):
        y = sampleInput[i, :] + thespan * (sampleChannel - 1 - i)
        # print(y.shape)
        dydx = heatmap[i, :]
        points = np.array([xx, y]).T.reshape(-1, 1, 2)
        # print(points.shape)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        # print(segments.shape)
        norm = plt.Normalize(-1, 1)
        lc = LineCollection(segments, norm=norm, cmap='RdBu_r')
        lc.set_linewidth(2)
        lc.set_array(dydx)
        ax0.add_collection(lc)

    yttics = np.zeros(sampleChannel)
    for gi in range(sampleChannel):
        yttics[gi] = gi * thespan

    ax0.set_ylim([-thespan, thespan * sampleChannel])
    ax0.set_xlim([0, resampleLength + 1])
    # ax0.set_xticks([1, 200, 400, 600, 800, 1001])
    # ax0.set_xticks([1, 100, 200, 300, 384])
    ax0.set_xticks([1, 60, 120, 180, 202])
    # ax0.set_xticks([1, 30, 60, 90, 101])
    ax0.set_title('EEG signal map', y=-0.1)

    inversechannelnames = []
    for i in range(sampleChannel):
        inversechannelnames.append(channelnames[sampleChannel - 1 - i])
    plt.sca(ax0)
    plt.yticks(yttics, inversechannelnames)

    reverse_x_data = x_data[id].reshape(30, -1) ** 2  # 电压的平方表示当前区域的能量
    reverse_cam_data = grayscale_cam.reshape(30, -1)
    topox = np.mean(reverse_x_data, axis=1)
    topoHeatmap = np.mean(reverse_cam_data, axis=1)

    im, cn = mne.viz.plot_topomap(topox, pos=info, show=False, axes=ax11, res=1200, names=channelnames, outlines='head',
                                  cmap='RdBu_r',
                                  )
    im, cn = mne.viz.plot_topomap(topoHeatmap, pos=info, show=False, axes=ax12, res=1200, names=channelnames,
                                  outlines='head',
                                  cmap='RdBu_r',
                                  )
    ax11.set_title('PSD', y=-0.09)
    ax12.set_title('Grad-cam', y=-0.1)
    # fig.colorbar(im, ax=[ax11, ax12])
    fig.suptitle(
        'Subject:' + subject + '    ' + 'Label:' + state[(int(label))] + '    ' + 'Likelihood:' + str(
            out.detach().numpy().reshape(3)))

    ax21.set_title('Delta', y=-0.1)
    ax22.set_title('Theta', y=-0.1)
    ax23.set_title('Alpha', y=-0.1)
    ax24.set_title('Beta', y=-0.1)
    im, cn = mne.viz.plot_topomap(data=deltapower, pos=info, axes=ax21, names=channelnames,
                                  outlines='head', cmap='RdBu_r', show=False)
    im, cn = mne.viz.plot_topomap(data=thetapower, pos=info, axes=ax22, names=channelnames,
                                  outlines='head', cmap='RdBu_r', show=False)
    im, cn = mne.viz.plot_topomap(data=alphapower, pos=info, axes=ax23, names=channelnames,
                                  outlines='head', cmap='RdBu_r', show=False)
    im, cn = mne.viz.plot_topomap(data=betapower, pos=info, axes=ax24, names=channelnames,
                                  outlines='head', cmap='RdBu_r', show=False)
    # fig.colorbar(im, ax=[ax21, ax22, ax23, ax24])
    plt.savefig('save_fig/' + subject + '/' + str(ll) + '/' + str(mode) + '_' + str(id) + '.jpg')
    print(str(id) + ' finished.')


if __name__ == '__main__':
    # load data and model
    model_name = 'DSANet'
    subject = 's53'
    data_path = 'data/raw/'
    mode = "cross"
    # mode = "inter"
    x = import_module('models.' + model_name)
    config = x.Config(dataset)
    config.batch_size = 1
    model = x.Model(config)
    config.f1_save_path += config.model_name + '/f1_' + subject + '_' + mode + '.ckpt'
    config.auc_save_path += config.model_name + '/auc_' + subject + '_' + mode + '.ckpt'
    # checkpoint = torch.load(config.auc_save_path)
    checkpoint = torch.load(config.f1_save_path)
    model.load_state_dict(checkpoint)
    model.eval()
    x_data = np.load(data_path + subject + '/' + subject + '_x.npy')
    y_data = np.load(data_path + subject + '/' + subject + '_y.npy')
    ids = [0, 1, 2, 3, 6, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 121, 122, 123, 124, 125, 128, 129, 152, 170, 171, 173, 174, 175, 176, 177, 215, 216, 217, 227, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, ]
    # statistics
    # normal, poor, optimal, error, correct = get_predict(model, x_data, y_data)
    # print("error:")
    # print(error)
    # print(len(error))
    # print("optimal:")
    # print(optimal)
    # print(len(optimal))
    # print("poor:")
    # print(poor)
    # print(len(poor))
    # print("normal:")
    # print(normal)
    # print(len(normal))
    # print("correct:")
    # print(correct)
    # print(len(correct))

    for id in ids:
        save_visualization(id, model, subject, x_data, y_data, mode, "optimal")
