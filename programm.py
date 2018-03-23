from my_ceemdan import ceemdan
from my_emd import emd
from scipy.signal import welch
import matplotlib.pyplot as plt
import numpy as np


def noise_check(imfs):
    '''
    m = np.eye(len(imfs), len(imfs))
    for i in range(len(imfs)):
        for j in range(len(imfs)):
            m[i,j] = round(sum(imfs[i]*imfs[j]), 3)

    ort = []
    for i in np.arange(1, len(imfs) -1):
        for j in np.arange(0, i):
            ort.append(np.abs(m[i,j]))

    print(max(ort), min(ort))
    '''
    print("_____________________")
    print("|  imf  ||   period  |")
    periods = []
    w = []
    n = 0
    for imf in imfs:
        w.append(np.sum(imf**2))
        cross = 0
        for i in range(len(imf) - 1):
            if imf[i]*imf[i+1] < 0 or imf[i] == 0:
                cross += 1
        print("|   {}   ||   {}   |".format(n, cross / len(imf)))
        if (cross != 0):
            periods.append(len(imf) / cross)
        else:
            periods.append(0)
        n+=1
    print("_____________________")
    plt.figure("lnW_lnP")
    plt.title("Зависимость энергии моды от ее периода, двойной логорифмический масштаб")
    plt.xscale("log")
    plt.xlabel("lnP")
    plt.yscale("log")
    plt.ylabel("lnW")
    plt.plot(periods, w, color='red', marker='o')
    plt.show()


def main():
    # region INPUT
    # periodic ; delta ; sin ; sample_signal ; gauss_A
    fname = "gauss_20000_A"
    I = 199
    bc_type = 'natural'
    max_extremas = 2
    sd = 0.1
    show_images = True
    save_images = False
    # endregion

    signal = np.load(fname + ".npy")
    imfs = ceemdan(signal, bc=bc_type, sd=sd, max_extr=max_extremas, I=I)
    # noise_check(imfs)

    # region Plotting
    if save_images or show_images:
        plt.figure("INPUT")
        plt.plot(signal)
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        if save_images:
            plt.savefig("img/INPUT_" + fname)

        plt.figure("IMFs")
        l = len(imfs)
        max_amp = max([max(np.abs(imf)) for imf in imfs])
        for i in range(l):
            plt.subplot(l, 1, i + 1)
            title = str(i) + " imf"
            plt.ylabel(title)
            plt.ylim([-max_amp - 0.05 * max_amp, max_amp + 0.05 * max_amp])
            plt.plot(imfs[i])
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        if save_images:
            plt.savefig("img/IMFs_" + fname + "_" + bc_type + "_" + str(I))

        plt.figure("POWER SPECTRAL DESTINY")
        max_psd = max([max(welch(imf)[1]) for imf in imfs])
        for i in range(l):
            plt.subplot(l, 1, i + 1)
            title = str(i) + " imf"
            plt.ylabel(title)
            freq, psd = welch(imfs[i])
            plt.ylim(0, max_psd)
            plt.plot(freq, psd)
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        if save_images:
            plt.savefig("img/PSD_" + fname + "_" + bc_type + "-" + str(I))

        plt.figure("POWER SPECTRAL DESTINY, LOG")
        for i in range(l):
            plt.subplot(l, 1, i + 1)
            title = str(i) + " imf"
            plt.ylabel(title)
            freq, psd = welch(imfs[i])
            plt.xscale('log')
            plt.ylim(0, max_psd)
            plt.plot(freq, psd)
            mng = plt.get_current_fig_manager()
            mng.window.showMaximized()
        if save_images:
            plt.savefig("img/PSD_LOG_" + fname + "_" + bc_type + "-" + str(I))

        if show_images:
            plt.show()
    # endregion


if __name__ == "__main__":
    main()
