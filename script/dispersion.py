import matplotlib.pyplot as plt
import numpy as np
import scipy
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.fft import rfft, rfftfreq
from scipy.stats import gaussian_kde

import sys

sys.path.append("./script")
import data
import basic_plot

# 字体设置
config = {
    "font.family": "serif",
    "font.size": 18,
    "mathtext.fontset": "stix",
    "font.serif": ["SimSun"],
    "axes.unicode_minus": False,
}
plt.rcParams.update(config)


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


def get_FFT(dT, FFT_data, fre_range):
    Fre, FFT_y = FFT(dT=dT, FFT_data=FFT_data)
    fre_min = fre_range[0]
    fre_max = fre_range[1]
    index = (Fre > fre_min) & (Fre < fre_max)
    Fre = Fre[index]
    FFT_abs = np.abs(FFT_y[index])
    FFT_absn = FFT_abs / len(FFT_y) * 2
    FFT_y = FFT_y[index]
    return Fre, FFT_y, FFT_absn


# Deltax_x为探针间距
def init_k_func_omega(Fre, FFT_sig1, FFT_sig2, Deltax_x=3.5e-3):
    global table_k
    k_func_omega = (
        1
        / Deltax_x
        * np.arctan(
            # 论文里这一项用了个有歧义的符号，即\\mathcal{F}^*，似乎是共轭的意思
            (FFT_sig2 * FFT_sig1.conj()).imag
            / (FFT_sig2 * FFT_sig1.conj()).real
            # (FFT_sig2 * FFT_sig1).imag
            # / (FFT_sig2 * FFT_sig1).real
        )
    )
    zipped = zip(Fre, k_func_omega)
    table_k = dict(zipped)


def k(omega):
    return table_k.get(omega)


# 核估计法，将散点数据变为密度图
def use_KDE_plot(x, y, values):
    xy = np.vstack([x, y])
    kde = gaussian_kde(xy, weights=values)
    # 在规则网格上评估 KDE
    grid_x, grid_y = np.mgrid[
        np.min(x) : np.max(x) : 100j, np.min(y) : np.max(y) : 100j
    ]  # 100j 表示100x100的网格
    # grid_x, grid_y = np.mgrid[
    #     0 : np.max(x) : 100j, 0 : np.max(y) : 100j
    # ]  # 100j 表示100x100的网格
    grid = np.vstack([grid_x.ravel(), grid_y.ravel()])
    kde_values = kde(grid).reshape(grid_x.shape)
    kde_values = kde_values / np.max(kde_values)
    # kde_values = np.log(kde_values)
    #
    # 绘制密度/热力图
    plt.figure(figsize=(6, 5))
    plt.subplots_adjust(
        top=0.97,
        bottom=0.15,
        left=0.15,
        right=1.0,
        hspace=0.2,
        wspace=0.2,
    )
    plt.pcolormesh(
        grid_x,
        grid_y / 1000,
        kde_values,
        shading="auto",
        cmap="coolwarm",  # coolwarm/viridis效果比较好
    )
    plt.colorbar()
    # plt.colorbar(label="归一化功率谱")
    plt.xlabel(r"波数 $\mathrm{(m^{-1})}$")
    plt.ylabel(r"频率 $\mathrm{(kHz)}$")
    plt.grid()
    plt.savefig("res/dispersion.jpg")
    plt.show()


def dispersion(
    data_points,
    plot_range=0.01,
    title="",
    fre_range=[1e1, 1e6],
    plot_channels=[1, 2, 3, 4],
    FFT_channels=[1, 2],
    save_FFT_csv=False,
    save_FFT_path="res/FFT/",
    Deltax_x=3.5e-3,
):
    # 获取FFT数据
    res = data_points
    dT = res[0][1] - res[0][0]
    FFT_y = []
    FFT_psd_norm = []
    for i in FFT_channels:
        # mean_res = np.mean(res[i])
        # FFT_data = (res[i] - mean_res) / mean_res
        FFT_data = res[i]
        Fre, FFT_y_tmp, FFT_absn_tmp = get_FFT(dT, FFT_data, fre_range)
        FFT_y.append(FFT_y_tmp)
        # 功率谱的选取方式
        FFT_psd_norm.append((FFT_absn_tmp / max(FFT_absn_tmp)) ** 2)
    # 计算色散关系
    init_k_func_omega(
        # 探针正反顺序搞错会改变色散分布的x轴方向
        Fre,
        FFT_y[1],
        FFT_y[0],
        Deltax_x,
    )
    ks = []
    for omega in Fre:
        ks.append(k(omega))
    # 每个点的权重
    values = []
    for i in range(len(Fre)):
        # values.append((FFT_psd_norm[0][i] + FFT_psd_norm[1][i]) / 2)
        values.append(1)
    # KDE
    use_KDE_plot(ks, Fre, values)
    # 散点图
    # plt.scatter(ks, Fre, c=(FFT_psd_norm[0] + FFT_psd_norm[1]) / 2, s=0.1)
    # plt.colorbar(label="Value")
    # plt.grid()
    # plt.show()

    # plt.plot(Fre, FFT_psd_norm[0])
    # plt.plot(Fre, FFT_psd_norm[1])
    # plt.grid()
    # plt.show()


if __name__ == "__main__":
    dir = "D:/001_zerlingx/archive/for_notes/HC/07_experiments/2024-03 一号阴极测试/2024-05-05 羽流诊断与色散关系测试/data/RAW/"
    path = "tek0012ALL.csv"
    # dir = "D:/001_zerlingx/archive/for_notes/HC/07_experiments/2024-09 一号阴极测试/2024-11-13 双探针离子声波测量/data/RAW/"
    # path = "tek0009ALL.csv"
    default_path = dir + path
    data_obj = data.data(default_path)
    data_obj.read_range = [0, 1e7]
    data_points = data_obj.read()
    dispersion(data_points, fre_range=[1e1, 3e6], Deltax_x=0.98e-3)
