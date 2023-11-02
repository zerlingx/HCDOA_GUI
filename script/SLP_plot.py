import pytest
import numpy as np
import sys
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import time


# 卡尔曼滤波器
class KalmanFilter1D:
    def __init__(
        self, process_variance, measurement_variance, est_error, initial_value=0
    ):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimate = initial_value
        self.estimate_error = est_error

    def update(self, measurement):
        # 预测
        prediction = self.estimate
        prediction_error = self.estimate_error + self.process_variance

        # 更新
        kalman_gain = prediction_error / (prediction_error + self.measurement_variance)
        self.estimate = prediction + kalman_gain * (measurement - prediction)
        self.estimate_error = (1 - kalman_gain) * prediction_error

        return self.estimate


def use_kalman_filter(RAW_data):
    process_variance = 0.0001
    measurement_variance = 1000  # 测量误差
    est_error = 2
    initial_value = RAW_data[0]
    kf = KalmanFilter1D(
        process_variance, measurement_variance, est_error, initial_value
    )
    filtered_values = []
    for measurement in RAW_data:
        filtered_value = kf.update(measurement)
        filtered_values.append(filtered_value)
    return np.array(filtered_values)


def test_SLP_plot():
    # 数据读取
    dir = "D:/001_zerlingx/notes/literature/HC/007_experiments/2023-07 一号阴极测试/2023-10-22 点火与单探针测试/data/RAW/"
    title = "tek0000ALL.csv"
    path = dir + title
    with open(path, "r") as file:
        csv_data = pd.read_csv(
            file,
            header=19,
        )
    time = csv_data.loc[:, "TIME"]
    voltage = csv_data.loc[:, "CH2"]
    current = csv_data.loc[:, "CH1"]
    voltage = np.array(voltage)
    current = np.array(current)
    # 降采样
    restep = int(1e3)
    resize = int(len(voltage) / restep)
    time = time[::restep]
    voltage = voltage[::restep]
    current = current[::restep]
    # 滤波
    smooth_dimention = 1
    window_size = int(resize / 100)
    voltage = scipy.signal.savgol_filter(voltage, window_size, smooth_dimention)
    current = scipy.signal.savgol_filter(current, window_size, smooth_dimention)
    # 找每个周期的起始点
    eps = 1
    start_points = abs(voltage) <= eps
    start_points = np.where(start_points == True)[0]
    start_list = []
    for i in start_points:
        # 跳过重复的点，若扫描频率为5kHz，那么一个周期对应(1/5k) / 4e-10 = 5e5个点
        if start_list == [] and voltage[i + int(resize * 0.2)] < 0:
            start_list.append(i)
        elif (
            i > start_list[-1] + int(resize * 0.2)
            and voltage[i - int(resize * 0.1)] > 0
        ):
            start_list.append(i)

    start_times = np.array(time).take(start_list)

    print(start_list)
    print(start_times)

    # 零点获取绘图
    # voltage = voltage / max(voltage)
    # plt.plot(time, voltage, label="voltage")
    # plt.vlines(
    #     x=start_times,
    #     ymin=min(voltage),
    #     ymax=max(voltage),
    #     colors="r",
    #     linestyles="dashed",
    # )
    # plt.legend()
    # plt.grid()
    # plt.show()
    # return

    # 截取一个扫描周期计算
    voltage = voltage[start_list[0] : start_list[1]]
    current = current[start_list[0] : start_list[1]]
    time = time[start_list[0] : start_list[1]]
    # 字体和绘图设置
    config = {
        "font.family": "serif",
        "font.size": 20,
        "mathtext.fontset": "stix",
        # "font.serif": ["SimSun"],
        "font.serif": ["Times New Roman"],
        "axes.unicode_minus": False,
    }
    plt.rcParams.update(config)
    fig, ax = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(20, 6),
    )
    plt.subplots_adjust(wspace=0.6)
    # (a)
    axplt1 = ax[0].plot(time, voltage)
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Voltage (V)")
    ax[0].grid()
    axtwin = ax[0].twinx()
    axplt2 = axtwin.plot(time, current, color="red")
    axtwin.set_ylabel("Current (A)")
    # axtwin.grid()
    # axtwin.set_ylim(-0.005, 0.06)
    ax[0].set_title("(a) V-t and I-t")
    axplts = axplt1 + axplt2
    labels = ["Voltage", "Current"]
    ax[0].legend(axplts, labels, loc="upper left")
    # (b)
    # 为方便绘制I-V曲线及后续处理，按电压排序并对电流滤波
    tmp = [list(t) for t in zip(voltage, current)]
    tmp.sort()
    tmp = np.array(tmp)
    voltage = tmp[:, 0]
    current = tmp[:, 1]
    # 找V_f
    current = scipy.signal.savgol_filter(
        current, int(0.05 * len(current)), smooth_dimention
    )
    zeropoint = abs(current) == min(abs(current))
    zeropoint = np.where(zeropoint == True)[0]
    if len(zeropoint) > 1:
        zeropoint = zeropoint[0]
    zeropoint = float(voltage[zeropoint])
    ax[1].vlines(
        x=zeropoint,
        ymin=min(current) * 0.2,
        ymax=max(current) * 0.2,
        colors="r",
        linestyles="dashed",
    )
    ax[1].text(
        x=zeropoint + max(voltage) * 0.1,
        y=max(current) * 0.2,
        s="V_f=" + str(round(zeropoint, 2)),
    )
    # 滤波，画I-V曲线和计算dI/dV
    axplt1 = ax[1].plot(voltage, current)
    # dstep = 2
    # dI = np.diff(current[:: int(resize / dstep)]) / np.diff(
    #     voltage[:: int(resize / dstep)]
    # )
    # dI = scipy.signal.savgol_filter(dI, int(0.05 * len(dI)), smooth_dimention)
    # dIV = voltage[:: int(resize / 10)]
    # axtwin = ax[1].twinx()
    # axplt2 = ax[1].plot(dIV[1:], dI)
    # axtwin.set_ylabel("dI/dV (mA/V)")
    # axplts = axplt1 + axplt2
    # labels = ["Current", "dI/dV"]
    # ax[1].legend(axplts, labels, loc="upper left")
    ax[1].set_xlabel("Voltage (V)")
    ax[1].set_ylabel("Current (A)")
    ax[1].grid()
    ax[1].set_title("(b) I-V")
    # (c) ln_I to k
    # 选取过渡段
    ln_I_start = np.where(voltage > 20)[0][0]
    ln_I_end = np.where(voltage > 0.5 * max(voltage))[0][0]
    voltage = voltage[ln_I_start:ln_I_end]
    current = current[ln_I_start:ln_I_end]
    ln_I = np.log(current)
    [k, b] = np.polyfit(voltage, ln_I, 1)
    k_BT_e = 1 / k  # 电子温度，单位eV
    ax[2].scatter(voltage, ln_I)
    ax[2].plot(voltage, k * voltage + b, "r")
    ax[2].set_xlabel("voltage")
    ax[2].set_ylabel("ln(I)")
    ax[2].legend(["ln(I) - V", "k * V + b"], loc="upper left")
    ax[2].grid()
    # 电子数密度
    # 暂时还没有测到饱和电子电流，把测到的最大电流作为饱和电子电流吧 (A)
    I_e0 = max(current)
    e = 1.6e-19
    # 探针直径、长度 (m)
    d_p = 0.8e-3
    l_p = 5.5e-3
    # 面积，计算侧面加一个端面
    A_p = np.pi * d_p * l_p + np.pi / 4.0 * d_p**2
    m_e = 9.1e-31
    T_e = k_BT_e
    n_e = I_e0 / (e * A_p) * np.sqrt(2 * np.pi * m_e / (e * T_e))
    ax[2].text(
        x=voltage[0] + 0.4 * (voltage[-1] - voltage[0]),
        y=ln_I[0] + 0.1 * (ln_I[-1] - ln_I[0]),
        s="k="
        + str(round(k, 4))
        + "\nk_BT_e="
        + str(round(k_BT_e, 1))
        + " eV\nn_e="
        + "{:.2e}".format(n_e),
    )
    ax[2].set_title("(c) ln(I) - V")
    plt.savefig("res/SLP_plot/" + title.split(".")[0] + ".jpg")
    plt.show()


if __name__ == "__main__":
    strattime = time.time()
    test_SLP_plot()
    endtime = time.time()
    print("run time =", endtime - strattime)
