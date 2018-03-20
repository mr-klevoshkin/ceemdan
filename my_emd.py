import numpy as np
from scipy import signal, interpolate
import matplotlib.pyplot as plt


def emd(data, sd=0.1, bc="natural"):
    t = np.arange(data.shape[0])

    imfs = []
    last_imf = False
    residue = data.copy()
    for i in range(20):
        h_prev = residue

        _i = 0
        for _ in range(50):
            maxima_idx = signal.argrelmax(h_prev, order=1)[0]
            minima_idx = signal.argrelmin(h_prev, order=1)[0]
            #maxima_idx = np.insert(maxima_idx, 0, 0)
            #maxima_idx = np.append(maxima_idx, len(h_prev) - 1)
            #minima_idx = np.insert(minima_idx, 0, 0)
            #minima_idx = np.append(minima_idx, len(h_prev) - 1)

            if (len(maxima_idx) + len(minima_idx)) <= 3:
                last_imf = True
                break

            maxima_vals = interpolate.CubicSpline(maxima_idx, h_prev[maxima_idx], bc_type=bc)(t)
            minima_vals = interpolate.CubicSpline(minima_idx, h_prev[minima_idx], bc_type=bc)(t)

            mean = 0.5*(maxima_vals + minima_vals)
            #mode_ampl = 0.5*(maxima_vals - minima_vals)
            #eval_func = np.abs(mean / mode_ampl)
            h = h_prev - mean
            _i += 1

            # sifting criterion
            sd_ = np.sum((h - h_prev)**2) / np.sum(h_prev**2)
            if sd_ < sd:
                break

            h_prev = h.copy()

        if last_imf:
            break

        imfs.append(np.nan_to_num(h))
        residue -= h

    # Add residue
    imfs.append(data - sum(imfs))
    return imfs


def main():
    sign = np.load("sin.npy")
    imfs = emd(sign)

    plt.figure()
    plt.plot(sign)

    plt.figure()
    l = len(imfs)
    for i in range(l):
        plt.subplot(l, 1, i+1)
        plt.plot(imfs[i])

    plt.figure()
    l = len(imfs)
    m = len(imfs[0])
    out = np.zeros((m), np.float16)
    for i in range(m):
        for j in range(l):
            out[i] += imfs[j][i]

    plt.plot(out)

    plt.show()


if __name__ == "__main__":
    main()
# vim: set tw=100: