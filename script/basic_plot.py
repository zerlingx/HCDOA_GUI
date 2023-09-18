import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import rfft, rfftfreq
from sklearn import preprocessing

config = {
    "font.family": "serif",
    "font.size": 20,
    "mathtext.fontset": "stix",
    # "font.serif": ["SimSun"],
    "font.serif": ["Times New Roman"],
    "axes.unicode_minus": False,
}
plt.rcParams.update(config)

import sys

sys.path.append("./script")
import data


def FFT(dT, FFT_data):
    """
    dT采样间隔, FFT_data为时域数据
    """
    FFT_y = rfft(np.int16(FFT_data / max(FFT_data) * 32767))
    Fre = rfftfreq(len(FFT_data), dT)
    return Fre, FFT_y


def plot_curve_and_FFT(data_points, plot_range=0.01, title="", fre_range=[1e1, 1e6]):
    res = data_points

    fig, ax = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(16, 7),
    )
    fig.suptitle(title)

    # 设置绘制时间范围
    # plot_time = [-3e-5, 3e-5]
    start_time = res[0][0]
    end_time = res[0][len(res[0]) - 1]
    step_time = res[0][1] - res[0][0]
    mid_time = (start_time + end_time + step_time) / 2
    plot_time = [
        mid_time - (end_time - start_time) * plot_range / 2,
        mid_time + (end_time - start_time) * plot_range / 2,
    ]
    plot_index = (res[0] >= plot_time[0]) & (res[0] <= plot_time[1])

    for i in range(1, len(res)):
        try:
            ax[0].plot(res[0][plot_index], res[i][plot_index], label="CH" + str(i))
        except:
            pass
    ax[0].set_title("(a) Scope data (Anode Voltage and Current).")
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Voltage (V) / Current (A)")
    ax[0].legend(loc="upper right")
    ax[0].grid()

    FFT_data = res[4]
    Fre, FFT_y = FFT(dT=4e-10, FFT_data=FFT_data)
    # 设置要绘制频率范围
    fre_min = fre_range[0]
    fre_max = fre_range[1]
    index = (Fre > fre_min) & (Fre < fre_max)
    Fre = Fre[index]
    FFT_abs = np.abs(FFT_y)[index]
    # print(len(Fre), len(FFT_abs))
    ax[1].plot(Fre, FFT_abs / len(FFT_y))
    # 显示频谱功率最大值
    max_A = max(FFT_abs)
    max_A_pos = Fre[FFT_abs == max_A]
    ax[1].vlines(
        x=[max_A_pos],
        ymin=0,
        ymax=max_A / len(FFT_y),
        colors="r",
        linestyles="dashed",
    )
    ax[1].annotate(
        "max_A_pos=" + str(max_A_pos) + "\nmax_A=" + str(max_A / len(FFT_y)),
        xy=(max_A_pos * 1.1, max_A / len(FFT_y) * 0.7),
    )
    ax[1].set_title("(b) FFT_current")
    ax[1].set_xlabel("Frequency (Hz)")
    ax[1].set_ylabel("Amplitude")
    ax[1].grid()

    return fig, ax


if __name__ == "__main__":
    dir = "D:/001_zerlingx/notes/literature/HC/007_experiments/2023-07 一号阴极测试/2023-08-30 点火与单探针测试/data/RAW/"
    path = "tek0011ALL.csv"
    default_path = dir + path
    data_obj = data.data(default_path)
    data_obj.read_range = [0, 1e7]  # 计算使用的采样点范围，一般来说越多计算越精确
    data_points = data_obj.read()
    # data_points = data_obj.normalize()
    fig, ax = plot_curve_and_FFT(data_points)
    plt.savefig("res/fig.png")
    plt.show()
