import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy

config = {
    "font.family": "serif",
    "font.size": 14,
    "mathtext.fontset": "stix",
    # "font.serif": ["SimSun"],
    "font.serif": ["Times New Roman"],
    "axes.unicode_minus": False,
}
plt.rcParams.update(config)

if __name__ == "__main__":
    fig = plt.figure(figsize=(12, 6))
    fig.suptitle("FFT results 3D")

    start = 0
    end = 801  # 200 kHz
    # end = 1201  # 300 kHz
    # (a) 3 sccm
    ax = fig.add_subplot(121, projection="3d")
    labels = ["2.5A", "3A", "3.5A", "4A", "4.5A", "5A", "5.5A"]
    # 反向遍历防止覆盖
    for i in range(7, 0, -1):
        csv_data = pd.read_csv("res/current_FFT/tek000" + str(i) + "ALL.csv", header=1)
        Fre = csv_data.iloc[start:end, 0] / 1e3
        FFT_absn = csv_data.iloc[start:end, 1]
        ax.plot(
            Fre,
            FFT_absn,
            zs=2 + 0.5 * i,
            zdir="y",
            label=labels[i - 1],
        )
        smooth_dimention = 1
        window_size = int(5e1)
        FFT_fitted = scipy.signal.savgol_filter(FFT_absn, window_size, smooth_dimention)
        ax.plot(
            Fre,
            FFT_fitted,
            zs=2 + 0.5 * i,
            zdir="y",
            label=labels[i - 1],
            color="grey",
        )
        # ax.set_xlim(0, 3e5)
        ax.set_ylim(2.5, 5.5)
        ax.set_zlim(0, 2)
        ax.set_xlabel("Frequency (kHz)")
        ax.set_ylabel("Current (A)")
        ax.set_zlabel("Amplitude (A)")
        ax.set_title("(a) 3 sccm")
        index1 = Fre < 3e1
        max_A1 = max(FFT_absn[index1])
        max_A1_pos = Fre[FFT_absn == max_A1].values[0]
        l = np.linspace(0, max_A1, 100)
        ax.plot(
            [max_A1_pos] * len(l),
            [2 + 0.5 * i] * len(l),
            l,
            color="black",
            linestyle="--",
        )
        index2 = Fre > 3e1
        max_A2 = max(FFT_absn[index2])
        max_A2_pos = Fre[FFT_absn == max_A2].values[0]
        l = np.linspace(0, max_A2, 100)
        ax.plot(
            [max_A2_pos] * len(l),
            [2 + 0.5 * i] * len(l),
            l,
            color="black",
            linestyle="--",
        )

    # # (b) 6 sccm
    ax = fig.add_subplot(122, projection="3d")
    z_current = [5, 6, 7, 8, 9, 10, 11]
    for i in range(29, 16, -2):
        csv_data = pd.read_csv("res/current_FFT/tek00" + str(i) + "ALL.csv", header=1)
        Fre = csv_data.iloc[start:end, 0] / 1e3
        FFT_absn = csv_data.iloc[start:end, 1]
        ax.plot(
            Fre,
            FFT_absn,
            zs=z_current[int((i - 17) / 2)],
            zdir="y",
        )
        smooth_dimention = 1
        window_size = int(5e1)
        FFT_fitted = scipy.signal.savgol_filter(FFT_absn, window_size, smooth_dimention)
        ax.plot(
            Fre,
            FFT_fitted,
            zs=z_current[int((i - 17) / 2)],
            zdir="y",
            color="grey",
        )
        # ax.set_xlim(0, 3e5)
        ax.set_ylim(5, 11)
        ax.set_zlim(0, 2)
        ax.set_xlabel("Frequency (kHz)")
        ax.set_ylabel("Current (A)")
        ax.set_zlabel("Amplitude (A)")
        ax.set_title("(b) 6 sccm")
        max_A1 = max(FFT_absn)
        max_A1_pos = Fre[FFT_absn == max_A1].values[0]
        l = np.linspace(0, max_A1, 100)
        ax.plot(
            [max_A1_pos] * len(l),
            [z_current[int((i - 17) / 2)]] * len(l),
            l,
            color="black",
            linestyle="--",
        )
    # plt.savefig("res/FFT_3D.png")
    plt.show()
