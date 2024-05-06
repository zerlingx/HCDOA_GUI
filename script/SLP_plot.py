import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import time
import sys

sys.path.append("./")
import constant.plasma_parameters as pp

sys.path.append("./script")
import data


def SLP_read_and_plot(
    data_points,
    title="",
):
    # 数据读取
    # dir = "D:/001_zerlingx/notes/literature/HC/007_experiments/2024-03 一号阴极测试/2024-03-07 羽流诊断与色散关系测试/data/RAW/"
    # title = "tek0000ALL.csv"
    # path = dir + title
    # with open(path, "r") as file:
    #     csv_data = pd.read_csv(
    #         file,
    #         header=19,
    #     )
    # time = csv_data.loc[:, "TIME"]
    # voltage = csv_data.loc[:, "CH1"]
    # current = csv_data.loc[:, "CH2"]
    time = data_points[0]
    voltage = data_points[1]
    current = data_points[2]

    voltage = np.array(voltage)
    current = np.array(current)
    rescale = 100
    restep = int(len(current) / rescale)
    dI = current[::restep]
    dI_t = time[::restep]
    dI = np.diff(dI)
    max_dI = max(dI)
    # dI = scipy.signal.savgol_filter(dI, 3, 1)
    # plt.plot(dI)
    # plt.show()
    # return
    # 将大于0.5*max_dI作为锯齿波周期起始的判据
    stages = np.where(dI > 0.6 * max_dI)
    stage_1 = stages[0][0] * restep + 1000
    stage_2 = stages[0][1] * restep - 1000
    time = time[stage_1:stage_2]
    voltage = voltage[stage_1:stage_2]
    current = current[stage_1:stage_2]
    smooth_dimention = 1
    window_size = int(len(voltage) / 100)
    voltage = scipy.signal.savgol_filter(voltage, window_size, smooth_dimention)
    current = scipy.signal.savgol_filter(current, window_size, smooth_dimention)

    # plt.plot(time, voltage / max(voltage), time, current / max(current))
    # plt.grid()
    # plt.show()
    # return

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
    plt.subplots_adjust(
        wspace=0.5,
        left=0.05,
        right=0.98,
    )
    # (a)
    axplt1 = ax[0].plot(time, voltage)
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Voltage (V)")
    axtwin = ax[0].twinx()
    axplt2 = axtwin.plot(time, current, color="red")
    axtwin.set_ylabel("Current (A)")
    axtwin.grid()
    ax[0].set_title("(a) V-t and I-t")
    axplts = axplt1 + axplt2
    labels = ["Voltage", "Current"]
    ax[0].legend(axplts, labels, loc="upper left")
    # (b)
    # (b-1) I-V曲线，找电流零点时电压为悬浮电压V_f
    # 为方便绘制I-V曲线及后续处理，按电压排序并对电流滤波
    tmp = [list(t) for t in zip(voltage, current)]
    tmp.sort()
    tmp = np.array(tmp)
    voltage = tmp[:, 0]
    current = tmp[:, 1]
    # 减掉等效串联耦合电阻的电流
    # R_empty = 8747
    # current = current - voltage / R_empty
    # 找V_f
    current = scipy.signal.savgol_filter(
        current, int(len(current) / 100), smooth_dimention
    )
    zeropoint = abs(current) == min(abs(current))
    zeropoint = np.where(zeropoint == True)[0]
    if len(zeropoint) > 1:
        zeropoint = zeropoint[0]
    V_f = float(voltage[zeropoint][0])
    range_I = max(current) - min(current)
    ax[1].vlines(
        x=V_f,
        ymin=0 - 0.1 * range_I,
        ymax=0 + 0.1 * range_I,
        colors="r",
        linestyles="dashed",
    )
    ax[1].text(
        x=V_f + max(voltage) * 0.1,
        y=max(current) * 0.1,
        s="V_f=" + str(round(V_f, 2)) + " V",
    )
    axplt1 = ax[1].plot(voltage, current)
    # (b-2) dI/dV-V曲线，找其拐点为等离子体电势V_p
    # 降采样
    dstep = 100
    start = int(0.1 * len(current))
    end = int(0.9 * len(current))
    dI = np.diff(current[start:end:dstep]) / np.diff(voltage[start:end:dstep])
    dI = scipy.signal.savgol_filter(dI, int(len(dI) / 20), smooth_dimention)
    dIV = voltage[start:end:dstep]
    axtwin = ax[1].twinx()
    axplt2 = axtwin.plot(dIV[1:], dI, color="orange")
    axtwin.set_ylabel("dI/dV (mA/V)")
    axplts = axplt1 + axplt2
    labels = ["Current", "dI/dV"]
    turn_point = np.where(dI == max(dI))[0]
    V_p = float(dIV[turn_point][0])
    range_dI = max(dI) - min(dI)
    axtwin.vlines(
        x=V_p,
        ymin=max(dI) - 0.1 * range_dI,
        ymax=max(dI) + 0.1 * range_dI,
        colors="r",
        linestyles="dashed",
    )
    axtwin.text(
        x=V_p + max(dIV) * 0.1,
        y=max(dI),
        s="V_p=" + str(round(V_p, 2)) + " V",
    )
    ax[1].legend(axplts, labels, loc="upper left")
    ax[1].set_xlabel("Voltage (V)")
    ax[1].set_ylabel("Current (A)")
    ax[1].grid()
    ax[1].set_title("(b) I-V and dI/dV-V")
    # (b-3) 找离子、电子饱和电流
    start = np.where(voltage > 0.9 * min(voltage))[0][0]
    end = np.where(voltage > 0.1 * min(voltage))[0][0]
    I_i0 = np.mean(current[start:end])
    ax[1].hlines(
        xmin=0.9 * min(voltage),
        xmax=0.1 * min(voltage),
        y=I_i0,
        colors="r",
        linestyles="dashed",
    )
    ax[1].text(
        x=0.9 * min(voltage),
        y=I_i0 + (max(current) - min(current)) * 0.1,
        s="I_i0=" + str(round(I_i0, 5)) + " A",
    )
    # (c) ln_I to k
    # 选取过渡段
    ln_I_start = np.where(voltage > V_f + 1)[0][0]
    ln_I_end = np.where(voltage > V_p)[0][0]
    voltage = voltage[ln_I_start:ln_I_end]
    current = current[ln_I_start:ln_I_end]
    # 过渡段特性记得把离子电流加上
    ln_I = np.log(current + abs(I_i0))
    [k, b] = np.polyfit(voltage, ln_I, 1)
    # 这种方法计算出来的电子温度T_e实际上是k_B*T_e/e，单位eV
    T_e = 1 / k
    ax[2].scatter(voltage, ln_I)
    ax[2].plot(voltage, k * voltage + b, "r")
    ax[2].set_xlabel("voltage")
    ax[2].set_ylabel("ln(I)")
    ax[2].legend(["ln(I)", "k*V+b"], loc="upper left")
    ax[2].grid()
    # 电子数密度
    # 暂时还没有测到饱和电子电流，把测到的最大电流作为饱和电子电流吧 (A)
    I_e0 = max(current)
    e = pp.Plasma().constants["E_ELEC"]
    m_e = pp.Plasma().constants["M_ELECTRON"]
    k_B = pp.Plasma().constants["K_BOLTZMANN"]
    # 探针直径、长度 (m)，\phi 0.12 mm细钨丝，暴露长度3 mm
    d_p = 0.12e-3
    l_p = 3e-3
    # 面积，计算侧面加一个端面
    A_p = np.pi * d_p * l_p + np.pi / 4.0 * d_p**2

    n_e = I_e0 / (e * A_p) * np.sqrt(2 * np.pi * m_e / (e * T_e))
    # n_e = 3.7e8 * I_e0 * 1e3 / (A_p * 1e4 * np.sqrt(T_e)) * 1e6
    ax[2].text(
        x=voltage[0] + 0.4 * (voltage[-1] - voltage[0]),
        y=ln_I[0] + 0.1 * (ln_I[-1] - ln_I[0]),
        s="k="
        + str(round(k, 4))
        + "\nk_BT_e="
        + str(round(T_e, 1))
        + " eV\nn_e="
        + "{:.2e}".format(n_e)
        + " m^-3",
    )
    ax[2].set_title("(c) ln(I)-V")
    # plt.savefig("res/SLP_plot/" + title.split(".")[0] + ".jpg")
    # plt.show()

    return fig, V_f, T_e, n_e


if __name__ == "__main__":
    strattime = time.time()
    dir = "D:/001_zerlingx/archive/for_notes/HC/07_experiments/2024-03 一号阴极测试/2024-03-07 羽流诊断与色散关系测试/data/RAW/"
    path = "tek0015ALL.csv"
    default_path = dir + path
    data_obj = data.data(default_path)
    data_points = data_obj.read()
    fig, V_f, T_e, n_e = SLP_read_and_plot(data_points)
    plt.savefig("res/SLP_tmp_plot.jpg")
    plt.plot()
    plt.show()
    endtime = time.time()
    print("run time =", endtime - strattime)
