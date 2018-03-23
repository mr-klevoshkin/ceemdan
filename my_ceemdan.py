import numpy as np
from my_emd import emd
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
    last_imf = False
    for iteration in range(Imfs_max):
        iteration += 1
        imf_prev = imfs[num - 1]
        reside -= imf_prev
        imfs_num = []
        for i in range(int(I)):
            imf_num = imf_n(reside + epsilon * imf_n(np.random.normal(0, 1, size), num, bc=bc), 1, bc=bc)
            if imf_num is None:
                last_imf = True
                break
            imfs_num.append(imf_num)
        if not last_imf:
            imf_num = sum(imfs_num) / I

        # finish criterion by number of extremas
        if (len(signal.argrelmax(imf_num, order=1)[0]) + len(signal.argrelmin(imf_num, order=1)[0])) <= max_extr:
            last_imf = True

        # finish criterion by difference in amps
        sd_ = np.sum((imf_num - imf_prev) ** 2) / np.sum(imf_prev ** 2)
        if sd_ < sd:
            last_imf = True

        if last_imf:
            imfs.append(np.array(reside))
            print("last imf is ready")
            break
        imfs.append(imf_num)
        print("imf", num, "is ready")
        num += 1
    return imfs
