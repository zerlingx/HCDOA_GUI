import pytest
import numpy as np
import sys
import matplotlib.pyplot as plt
import pandas as pd
import scipy


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


def test_SLP_plot():
    # 数据读取
    path = "D:/001_zerlingx/notes/literature/HC/007_experiments/2023-07 一号阴极测试/2023-09-26 点火与单探针测试/data/RAW/tek0006ALL.csv"
    with open(path, "r") as file:
        csv_data = pd.read_csv(
            file,
            header=19,
        )
    gap = 3e5
    start = 5e6 - gap
    end = 5e6 + gap
    time = csv_data.loc[start:end, "TIME"]
    voltage = csv_data.loc[start:end, "CH2"]
    current = csv_data.loc[start:end, "CH1"]
    voltage = np.array(voltage)
    current = np.array(current)
    # 滤波，不滤波的话边缘检测容易出问题
    # smooth_dimention = 1
    # window_size = int(1e3)
    # voltage = scipy.signal.savgol_filter(voltage, window_size, smooth_dimention)
    # 卡尔曼滤波，感觉快一些
    process_variance = 0.0001
    measurement_variance = 1000  # 测量误差
    est_error = 2
    initial_value = voltage[0]
    kf = KalmanFilter1D(
        process_variance, measurement_variance, est_error, initial_value
    )
    filtered_values = []
    for measurement in voltage:
        filtered_value = kf.update(measurement)
        filtered_values.append(filtered_value)
    # 找每个周期的起始点
    eps = 1e-1
    dv = np.diff(filtered_values)
    # dv = np.diff(voltage)
    # dv = scipy.signal.savgol_filter(dv, window_size, smooth_dimention)
    start_points = np.append((dv < 0), False) & (abs(voltage) <= eps)
    start_points = np.where(start_points == True)[0]
    start_list = []
    for i in start_points:
        # 周期间隔大于1e5，跳过重复的点
        if start_list == [] or i > start_list[-1] + 1e5:
            start_list.append(i)
    start_times = np.array(time).take(start_list)

    print(start_list)
    print(start_times)

    plt.plot(time, voltage, label="voltage")
    plt.plot(time, filtered_values, label="filtered_values")
    plt.vlines(
        x=start_times,
        ymin=min(voltage),
        ymax=max(voltage),
        colors="r",
        linestyles="dashed",
    )
    plt.grid()
    plt.show()

    voltage, unique_index = np.unique(voltage, return_index=True)
    return
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
        figsize=(21, 6),
    )
    plt.subplots_adjust(wspace=0.5)
    # (a)
    ax[0].plot(time, voltage)
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Voltage (V)")
    ax[0].grid()
    axtwin = ax[0].twinx()
    axtwin.plot(time, current, color="red")
    axtwin.set_ylabel("Current (A)")
    ax[0].set_title("(a) V-t and I-t")
    # (b)
    # 电压电压排序
    # voltage, current = zip(*sorted(zip(voltage, current)))
    # voltage = np.array(voltage)
    # current = np.array(current)
    ax[1].plot(voltage, current)
    ax[1].set_xlabel("Voltage (V)")
    ax[1].set_ylabel("Current (A)")
    ax[1].grid()
    ax[1].set_title("(b) I-V")
    # (c) ln_I to k
    ln_I_range = voltage > 30
    ln_I = np.log(current[ln_I_range])
    [k, b] = np.polyfit(voltage[ln_I_range], ln_I, 1)
    k_BT_e = 1 / k  # 电子温度，单位eV
    ax[2].scatter(voltage[ln_I_range], ln_I)
    ax[2].plot(voltage[ln_I_range], k * voltage[ln_I_range] + b, "r")
    ax[2].set_xlabel("voltage")
    ax[2].set_ylabel("ln(I)")
    ax[2].legend(["ln(I) - V", "k * V + b"])
    ax[2].grid()
    ax[2].text(
        x=37,
        y=-4,
        s="k=" + str(round(k, 4)) + "\n" + "k_BT_e=" + str(round(k_BT_e, 1)) + " eV",
    )
    ax[2].set_title("(c) ln(I) - V")
    plt.show()


if __name__ == "__main__":
    test_SLP_plot()
