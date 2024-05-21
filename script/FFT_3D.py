import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import cycler
import scipy

config = {
    "font.family": "serif",
    "font.size": 14,
    "mathtext.fontset": "stix",
    "font.serif": ["SimSun"],
    # "font.serif": ["Times New Roman"],
    "axes.unicode_minus": False,
}
plt.rcParams.update(config)


def FFT_3D_1():
    fig = plt.figure(figsize=(12, 6))
    fig.suptitle("FFT results 3D")

    start = 0
    end = 801  # 800 kHz
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


def FFT_3D_2():
    fig = plt.figure(figsize=(6, 6))
    plt.subplots_adjust(
        left=0.00,
    )

    max_Fre = 300  # kHz
    ax = fig.add_subplot(111, projection="3d")
    data_list = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    # data_list = [i + 5 for i in data_list]
    labels = [
        "12 sccm",
        "14 sccm",
        "16 sccm",
        "18 sccm",
        "20 sccm",
        "22 sccm",
        "24 sccm",
        "26 sccm",
        "28 sccm",
        "30 sccm",
    ]
    zs_list = [12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
    # 从后往前绘制，防止覆盖
    data_list.reverse()
    labels.reverse()
    zs_list.reverse()

    colormap = cm.get_cmap(name="coolwarm")
    colorcycles = 10
    cnt = -1
    for i in range(0, len(data_list)):
        cnt += 1
        path = "res/FFT/20240512/tek" + str(data_list[i]).zfill(4) + "ALL.csv"
        csv_data = pd.read_csv(path, header=1)

        Fre = csv_data.iloc[:, 0] / 1e3
        fre_index = Fre < max_Fre
        Fre = Fre[fre_index]
        FFT_absn = csv_data.iloc[:, 1]
        FFT_absn = FFT_absn[fre_index]
        FFT_absn = FFT_absn * 1000
        ax.set_box_aspect((1, 1, 0.3))  # 3D图形长宽高比例
        ax.plot(
            Fre,
            FFT_absn,
            zs=zs_list[i],
            zdir="y",
            label=labels[i],
            color=colormap(cnt / colorcycles),
        )
        smooth_dimention = 1
        window_size = int(5e1)
        FFT_fitted = scipy.signal.savgol_filter(FFT_absn, window_size, smooth_dimention)
        ax.plot(
            Fre,
            FFT_fitted,
            zs=zs_list[i],
            zdir="y",
            # label=labels[i],
            color="grey",
        )
        ax.set_xlabel(r"频率 $\mathrm{(kHz)}$")
        ax.set_ylabel(r"流量 $\mathrm{(sccm)}$")
        ax.set_zlabel(r"振幅 $\mathrm{(mA\cdot s)}$")
        try:
            index1 = Fre < 150
            max_A1 = max(FFT_absn[index1])
            max_A1_pos = Fre[FFT_absn == max_A1].values[0]
            l = np.linspace(0, max_A1, 100)
            ax.plot(
                [max_A1_pos] * len(l),
                [zs_list[i]] * len(l),
                l,
                color="black",
                linestyle="--",
            )
        except:
            pass
        try:
            index2 = Fre > 150
            max_A2 = max(FFT_absn[index2])
            max_A2_pos = Fre[FFT_absn == max_A2].values[0]
            l = np.linspace(0, max_A2, 100)
            ax.plot(
                [max_A2_pos] * len(l),
                [zs_list[i]] * len(l),
                l,
                color="black",
                linestyle="--",
            )
        except:
            pass
    # ax.legend(loc=(0.6, 0.5))
    plt.savefig("res/FFT_3D_for_20240513_Ie0.png")
    plt.show()


def FFT_3D_3():
    fig = plt.figure(figsize=(6, 6))
    plt.subplots_adjust(
        left=0.00,
    )

    max_Fre = 100  # kHz
    ax = fig.add_subplot(111, projection="3d")
    data_list = [496, 200, 486, 350]
    labels = ["10 A", "12.5 A", "15 A", "20 A"]
    zs_list = [10, 12.5, 15, 20]
    data_list.reverse()
    labels.reverse()
    zs_list.reverse()

    for i in range(0, len(data_list)):
        path = "res/FFT/20240414/tek" + str(data_list[i]).zfill(4) + "ALL.csv"
        csv_data = pd.read_csv(path, header=1)

        Fre = csv_data.iloc[:, 0] / 1e3
        fre_index = Fre < max_Fre
        Fre = Fre[fre_index]
        FFT_absn = csv_data.iloc[:, 1]
        FFT_absn = FFT_absn[fre_index]
        FFT_absn = FFT_absn
        ax.set_box_aspect((1, 1, 0.8))  # 3D图形长宽高比例
        ax.plot(Fre, FFT_absn, zs=zs_list[i], zdir="y", label=labels[i])
        smooth_dimention = 1
        window_size = int(5e1)
        FFT_fitted = scipy.signal.savgol_filter(FFT_absn, window_size, smooth_dimention)
        ax.plot(
            Fre,
            FFT_fitted,
            zs=zs_list[i],
            zdir="y",
            # label=labels[i],
            color="grey",
        )
        ax.set_xlabel(r"频率 $\mathrm{(kHz)}$")
        ax.set_ylabel(r"电流 $\mathrm{(A)}$")
        ax.set_zlabel(r"振幅 $\mathrm{(A\cdot s)}$")
        try:
            index1 = Fre < 18
            max_A1 = max(FFT_absn[index1])
            max_A1_pos = Fre[FFT_absn == max_A1].values[0]
            l = np.linspace(0, max_A1, 100)
            ax.plot(
                [max_A1_pos] * len(l),
                [zs_list[i]] * len(l),
                l,
                color="black",
                linestyle="--",
            )
        except:
            pass
        try:
            index2 = Fre > 18
            max_A2 = max(FFT_absn[index2])
            max_A2_pos = Fre[FFT_absn == max_A2].values[0]
            l = np.linspace(0, max_A2, 100)
            ax.plot(
                [max_A2_pos] * len(l),
                [zs_list[i]] * len(l),
                l,
                color="black",
                linestyle="--",
            )
        except:
            pass
    ax.legend(loc="upper right")
    plt.savefig("res/FFT_3D_for_20240414.png")
    plt.show()


if __name__ == "__main__":
    FFT_3D_3()
