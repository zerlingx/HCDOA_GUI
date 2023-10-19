import matplotlib.pyplot as plt
import numpy as np
import scipy
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.fft import rfft, rfftfreq
from scipy.ndimage import maximum_filter1d

import sys

sys.path.append("./script")
import data
import basic_plot


def load_data_points(dir_path, read_range=[], normalize=False):
    data_obj = data.data(dir_path)
    data_obj.read_range = read_range
    data_points = data_obj.read()
    if normalize:
        data_points = data_obj.normalize()
    return data_points


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


def FFT_and_save(
    data_points,
    plot_range=0.01,
    title="",
    fre_range=[1e1, 1e6],
    plot_channels=[1, 2, 3, 4],
    save_FFT_csv=False,
    save_FFT_path="res/current_FFT/",
):
    res = data_points
    FFT_data = res[4]
    Fre, FFT_y = FFT(dT=4e-10, FFT_data=FFT_data)
    fre_min = fre_range[0]
    fre_max = fre_range[1]
    index = (Fre > fre_min) & (Fre < fre_max)
    Fre = Fre[index]
    FFT_abs = np.abs(FFT_y)[index]
    FFT_absn = FFT_abs / len(FFT_y) * 2
    # 保存FFT结果
    save_FFT_csv = True
    if save_FFT_csv:
        np.savetxt(
            save_FFT_path + title,
            np.array([Fre, FFT_absn]).T,
            delimiter=",",
            fmt="%f",
            header="Fre,FFT_absn",
        )
    return Fre, FFT_absn


if __name__ == "__main__":
    dir = "D:/001_zerlingx/notes/literature/HC/007_experiments/2023-07 一号阴极测试/2023-10-17 点火与单探针测试/data/RAW/"
    file = "tek0025ALL.csv"
    data_points = load_data_points(dir + file, read_range=[0, 1e7], normalize=False)
    Fre, FFT_absn = FFT_and_save(data_points, title=file)

    # 绘图
    fig, ax = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(16, 7),
    )
    smooth_dimention = 1
    window_size = int(1e2)
    FFT_fitted = scipy.signal.savgol_filter(FFT_absn, window_size, smooth_dimention)
    ax[1].plot(Fre, FFT_absn, label="FFT_current")
    ax[1].plot(Fre, FFT_fitted, label="FFT_fitted")
    # 显示频谱幅值最大值
    max_A = max(FFT_absn)
    max_A_pos = Fre[FFT_absn == max_A]
    ax[1].vlines(
        x=[max_A_pos],
        ymin=0,
        ymax=max_A,
        colors="r",
        linestyles="dashed",
    )
    ax[1].annotate(
        "max_A_pos=" + str(max_A_pos) + "\nmax_A=" + str(max_A),
        xy=(max_A_pos * 1.1, max_A * 0.7),
    )
    ax[1].set_title("(b) FFT_current")
    ax[1].set_xlabel("Frequency (Hz)")
    ax[1].set_ylabel("Amplitude")
    ax[1].legend(loc="upper right")
    ax[1].grid()
    plt.show()
