import numpy as np
import matplotlib.pyplot as plt
from my_emd import emd
from scipy.signal import welch
from scipy import signal


def imf_n(data, num, bc="natural"):
    imfs = emd(data, bc=bc)
    if len(imfs) >= num:
        return np.array(imfs[num - 1])
    else:
        return None


def ceemdan(data, I=500, sd=0.1, max_extr=2, bc="natural"):
    epsilon = 0.08
    size = data.shape[0]
    imfs = []
    reside = data.copy()
    imfs1 = []
    Imfs_max = 50

    for i in range(int(I)):
        imf1 = imf_n(reside + epsilon * np.random.normal(0, 1, size), 1, bc=bc)
        imfs1.append(imf1)
    imf1 = sum(imfs1) / I
    imfs.append(imf1)
    print("imf 0 is ready")

    num = 1
    finish = False
    for iteration in range(Imfs_max):
        iteration += 1
        imf_prev = imfs[num - 1]
        reside -= imf_prev
        imfs_num = []
        for i in range(int(I)):
            imf_num = imf_n(reside + epsilon * imf_n(np.random.normal(0, 1, size), num, bc=bc), 1, bc=bc)
            if imf_num is None:
                finish = True
                break
            imfs_num.append(imf_num)
        if not finish:
            imf_num = sum(imfs_num) / I

        # finish criterion by number of extremas
        if (len(signal.argrelmax(imf_num, order=1)[0]) + len(signal.argrelmin(imf_num, order=1)[0])) <= max_extr:
            finish = True

        # finish criterion by difference in amps
        sd_ = np.sum((imf_num - imf_prev) ** 2) / np.sum(imf_prev ** 2)
        if sd_ < sd:
            break

        if finish:
            imfs.append(np.array(reside))
            print("last imf is ready")
            break
        imfs.append(imf_num)
        print("imf", num, "is ready")
        num += 1
    return imfs


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

    #region INPUT
    # periodic ; delta ; sin ; sample_signal ; gauss_A
    fname = "gauss_20000_A"
    I = 199
    bc_type = 'natural'
    max_extremas = 2
    sd = 0.1
    show_images = True
    save_images = False
    #endregion

    signal = np.load(fname + ".npy")
    imfs = ceemdan(signal, bc=bc_type, sd=sd, max_extr=max_extremas, I=I)
    #noise_check(imfs)

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
            plt.ylim([-max_amp - 0.05*max_amp, max_amp + 0.05*max_amp])
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
            plt.title(title)
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
            plt.title(title)
            freq, psd = welch(imfs[i])
            plt.xscale('log')
            plt.ylim(0, max_psd)
            plt.plot(freq, psd)
            mng = plt.get_current_fig_manager()
            mng.window.showMaximized()
        if save_images:
            plt.savefig("img/PSD_LOG_"+fname+"_"+bc_type+"-"+str(I))

        if show_images:
            plt.show()
    # endregion

if __name__ == "__main__":
    main()
# vim: set tw=100:


