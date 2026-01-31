import mne
import numpy as np
from matplotlib import gridspec
from mne.time_frequency import psd_array_multitaper
from scipy.integrate import simps
import matplotlib.pyplot as plt

if __name__=='__main__':
    dataset = 'data'
    subject = 's40'
    data_path = 'data/raw/'
    x_data = np.load(data_path + subject + '/' + subject + '_x.npy')
    y_data = np.load(data_path + subject + '/' + subject + '_y.npy')
    id = 202
    rawsignal = x_data[id]
    channelnum = 30
    samplelength = 1001
    maxvalue = np.max(np.abs(rawsignal))
    rawsignal = rawsignal.reshape(30, 1001)
    sampleChannel = channelnum
    deltapower = np.zeros(sampleChannel)
    thetapower = np.zeros(sampleChannel)
    alphapower = np.zeros(sampleChannel)
    betapower = np.zeros(sampleChannel)
    for kk in range(channelnum):
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
    channelnames = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FT7', 'FC3', 'FCz', 'FC4', 'FT8', 'T3', 'C3', 'Cz',
                    'C4', 'T4', 'TP7', 'CP3', 'CPz', 'CP4', 'TP8', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'Oz', 'O2']
    montage = 'standard_1020'
    info = mne.create_info(ch_names=channelnames, sfreq=500., ch_types='eeg')
    info.set_montage(montage)
    ch_names = channelnames

    fig = plt.figure(figsize=(14, 6))

    gridlayout = gridspec.GridSpec(ncols=2, nrows=2, figure=fig, wspace=0.05, hspace=0.1)

    ax1 = fig.add_subplot(gridlayout[0, 0])
    ax2 = fig.add_subplot(gridlayout[0, 1])
    ax3 = fig.add_subplot(gridlayout[1, 0])
    ax4 = fig.add_subplot(gridlayout[1, 1])
    ax1.set_title('Delta', y=-0.1)
    ax2.set_title('Theta', y=-0.1)
    ax3.set_title('Alpha', y=-0.1)
    ax4.set_title('Beta', y=-0.1)
    print(deltapower.shape)
    plt.subplot(221)
    im, cn = mne.viz.plot_topomap(data=deltapower, pos=info, axes=ax1, names=ch_names,
                                  outlines='head', cmap='viridis', show=False, res=1200)
    plt.subplot(222)
    im, cn = mne.viz.plot_topomap(data=thetapower, pos=info, axes=ax2, names=ch_names,
                                  outlines='head', cmap='viridis', show=False, res=1200)
    plt.subplot(223)
    im, cn = mne.viz.plot_topomap(data=alphapower, pos=info, axes=ax3, names=ch_names,
                                  outlines='head', cmap='viridis', show=False, res=1200)
    plt.subplot(224)
    im, cn = mne.viz.plot_topomap(data=betapower, pos=info, axes=ax4, names=ch_names,
                                  outlines='head', cmap='viridis', show=False, res=1200)
    # fig.colorbar(im, ax=[ax1, ax2, ax3, ax4])
    plt.show()