import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.fft import rfft, rfftfreq
from scipy.ndimage import maximum_filter1d
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
    # FFT_y = rfft(np.int16(FFT_data / max(FFT_data) * 32767))
    FFT_y = rfft(np.array(FFT_data))
    # 振幅归一化
    # FFT_y = FFT_y / len(FFT_y) * 2
    Fre = rfftfreq(len(FFT_data), dT)
    return Fre, FFT_y


def plot_curve_and_FFT(
    data_points,
    plot_range=0.01,
    title="",
    fre_range=[1e1, 1e6],
    plot_channels=[1, 2, 3, 4],
    FFT_channel=4,
    save_FFT_csv=False,
    save_FFT_path="res/FFT/",
):
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

    # for i in range(1, len(res)):
    for i in plot_channels:
        try:
            ax[0].plot(res[0][plot_index], res[i][plot_index], label="CH" + str(i))
        except:
            pass
    ax[0].set_title("(a) Scope data.")
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Voltage (V) / Current (A) / Normalized (a.u.)")
    ax[0].legend(loc="upper right")
    ax[0].grid()

    # 默认用CH4电流做FFT算频谱
    FFT_data = res[FFT_channel]
    dT = res[0][1] - res[0][0]
    Fre, FFT_y = FFT(dT=dT, FFT_data=FFT_data)
    # 设置要绘制频率范围
    fre_min = fre_range[0]
    fre_max = fre_range[1]
    index = (Fre > fre_min) & (Fre < fre_max)
    Fre = Fre[index]
    FFT_abs = np.abs(FFT_y)[index]
    FFT_absn = FFT_abs / len(FFT_y) * 2
    # 保存FFT结果
    if save_FFT_csv:
        # if True:
        np.savetxt(
            save_FFT_path + title,
            np.array([Fre, FFT_absn]).T,
            delimiter=",",
            fmt="%f",
            header="Fre,FFT_absn",
        )
    # FFT曲线滤波
    smooth_dimention = 1
    window_size = int(1e2)
    # 可以考虑最值滤波，只是我感觉均值滤波找到的峰值更准确
    # FFT_fitted = maximum_filter1d(FFT_abs, window_size)
    FFT_fitted = scipy.signal.savgol_filter(FFT_absn, window_size, smooth_dimention)

    ax[1].plot(Fre, FFT_absn, label="FFT")
    # ax[1].plot(Fre, FFT_fitted, label="FFT_fitted")
    # 显示频谱幅值最大值
    max_A = max(FFT_absn)
    max_A_pos = Fre[FFT_absn == max_A][0]
    ax[1].vlines(
        x=[max_A_pos],
        ymin=0,
        ymax=max_A,
        colors="r",
        linestyles="dashed",
    )
    ax[1].annotate(
        "peak_freq="
        + str(int(max_A_pos))
        + " Hz\npeak_A="
        + "{:.2e}".format(max_A)
        + " A",
        xy=(max_A_pos * 1.1, max_A * 0.7),
    )
    ax[1].set_title("(b) FFT")
    ax[1].set_xlabel("Frequency (Hz)")
    ax[1].set_ylabel("Amplitude")
    ax[1].legend(loc="upper right")
    ax[1].grid()

    return fig, ax


if __name__ == "__main__":
    dir = "D:/001_zerlingx/archive/for_notes/HC/07_experiments/2024-03 一号阴极测试/2024-05-12 羽流诊断与色散关系测试/data/RAW/"
    path = "tek0000ALL.csv"
    default_path = dir + path
    data_obj = data.data(default_path)
    # data_obj.read_range = [0, 1e7]  # 计算使用的采样点范围，一般来说越多计算越精确
    data_points = data_obj.read()
    # data_points = data_obj.normalize()
    fig, ax = plot_curve_and_FFT(
        data_points,
        fre_range=[1e1, 5e6],
        FFT_channel=3,
        save_FFT_csv=True,
        title=path,
    )
    # plt.savefig("res/20231203_NO10_fre_5e6.png")
    plt.show()
