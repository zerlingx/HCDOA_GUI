# Single-Langmuir Probe (SLP)
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
import sys

sys.path.append("./")
import constant.plasma_parameters as pp
import constant.reference_quantity_hollow_cathode as rqhc

sys.path.append("./script")
import data

"""
Brief: Single-Langmuir Probe, basic plasma diagnostic device.
Refe.: 
Set  : 
Calc.: 
"""


class SLP:
    def __init__(self) -> None:
        self.ref_parameters = {
            "D": 0.12,  # 探针直径        mm
            "L": 3,  # 探针长度        mm
        }

    def find_periods(self, data_points, if_print=False):
        """
        Brief: 找到锯齿波周期
        Args:
            data_points: list, [time, voltage, current]
        Returns:
            starts: int, 锯齿波周期起始
            ends: int, 锯齿波周期结束
        """
        # 读取数据
        time = data_points[0]
        voltage = data_points[1]
        current = data_points[2]
        voltage = np.array(voltage)
        current = np.array(current)
        # 寻找峰值和谷值作为周期始末
        peaks, _ = scipy.signal.find_peaks(
            voltage,
            distance=len(voltage) / 11,  # 最小周期记录波形的1/10
            height=max(voltage) * 0.7,  # 最小峰值为波形最大值的70%
        )
        lows, _ = scipy.signal.find_peaks(
            -voltage,
            distance=len(voltage) / 11,
            height=max(voltage) * 0.7,
        )
        if np.mean(peaks) < np.mean(lows):
            starts = peaks
            ends = lows
        else:
            ends = peaks
            starts = lows
        starts_num = len(starts)
        ends_num = len(ends)
        # 测试数据残缺问题
        # starts = np.delete(starts, [0, 1, 4, 5, 9])
        # ends = np.delete(ends, [0, 1, 4, 6, 9])
        # print("starts num=", starts_num)
        # print("ends num=", ends_num)
        # 若峰谷数不一致，尝试删除空缺间隔
        # 若峰值较少，删除逻辑为删枚举峰值，删除时间超前的谷值点
        if starts_num < ends_num:
            for i in range(len(starts)):
                while starts[i] > ends[i]:
                    ends = np.delete(ends, i)
            ends = ends[:starts_num]
        # 若谷值较少，删除逻辑为枚举谷值，删除时间差大于1.5最小周期的峰值点
        elif starts_num > ends_num:
            min_period = min(min(np.diff(starts)), min(np.diff(ends)))
            for i in range(len(ends)):
                while ends[i] - starts[i] > 1.5 * min_period:
                    starts = np.delete(starts, i)
            starts = starts[:ends_num]

        # 绘图展示周期查找结果
        if if_print:
            print("starts num=", len(starts))
            print("ends num=", len(ends))
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            axplt1 = ax.plot(time, voltage, color="orange")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Voltage (V)")
            axtwin = ax.twinx()
            axplt2 = axtwin.plot(time, current, color="blue")
            axtwin.set_ylabel("Current (A)")
            axtwin.grid()
            ax.set_title("(a) V-t and I-t")
            axplts = axplt1 + axplt2
            labels = ["Voltage", "Current"]
            ax.legend(axplts, labels, loc="upper right")
            ax.plot(time[starts], voltage[starts], "x", color="red")
            ax.plot(time[ends], voltage[ends], "o", color="red")
            plt.show()

        return starts, ends

    def cal(self, data_points, title="", if_print=False):
        """
        Brief: 朗缪尔单探针计算,输出绘图对象,V_f, T_e, n_e
        Args:
            data_points: list, [time, voltage, current]
            title: str, 图片标题
        Returns:
            fig: matplotlib.figure.Figure, 绘图对象
            V_f: float, 悬浮电压
            T_e: float, 电子温度
            n_e: float, 电子密度
        """
        time = data_points[0]
        VOLTAGE = data_points[1]
        CURRENT = data_points[2]

        # 结果：V_p, T_e, n_e
        V_ps = []
        T_es = []
        n_es = []
        starts, ends = self.find_periods(data_points, if_print=if_print)
        # 分别计算每个周期
        for i in range(len(starts)):
            stage_1 = starts[i]
            stage_2 = ends[i]
            time = time[stage_1:stage_2]
            voltage = VOLTAGE[stage_1:stage_2]
            current = CURRENT[stage_1:stage_2]
            # 平滑滤波
            smooth_dimention = 1
            window_size = int(len(voltage) / 100)
            voltage = scipy.signal.savgol_filter(voltage, window_size, smooth_dimention)
            current = scipy.signal.savgol_filter(current, window_size, smooth_dimention)
            # 如果电压从高到低，反转，默认电压递增为正序
            if voltage[0] > voltage[-1]:
                time = np.flip(time)
                voltage = np.flip(voltage)
                current = np.flip(current)
            # 计算用于找悬浮电势和空间电势的电流梯度dI和ddI
            # 降采样
            dstep = int(len(voltage) / 100)
            start = int(0.1 * len(current))
            end = int(0.9 * len(current))
            dI = np.diff(current[start:end:dstep]) / np.diff(voltage[start:end:dstep])
            dI = scipy.signal.savgol_filter(
                dI,
                int(len(dI) / 20 + 2),
                smooth_dimention,
            )
            dIV = voltage[start:end:dstep]
            ddI = np.diff(dI)
            ddI = scipy.signal.savgol_filter(
                ddI,
                int(len(ddI) / 20 + 2),
                smooth_dimention,
            )
            ddI = abs(ddI)
            # 测试绘图
            # fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            # axplt1 = ax.plot(voltage, color="orange")
            # ax.set_xlabel("Time (s)")
            # ax.set_ylabel("Voltage (V)")
            # axtwin = ax.twinx()
            # axplt2 = axtwin.plot(current, color="blue")
            # axtwin.set_ylabel("Current (A)")
            # axtwin.grid()
            # axplts = axplt1 + axplt2
            # labels = ["Voltage", "Current"]
            # ax.legend(axplts, labels, loc="upper right")

            # plt.plot(dI)
            # plt.plot(ddI)
            # plt.legend(["dI", "ddI"])
            # plt.grid()
            # plt.show()
            # return

            # 找V_f
            ddI_peaks, _ = scipy.signal.find_peaks(
                ddI,
                distance=len(ddI) / 30,  # 最小周期记录波形的1/10
                height=max(ddI) * 0.3,  # 最小峰值为波形最大值的70%
            )
            # 1、电流零点作为V_f
            zero_point = abs(current) == min(abs(current))
            zero_point = np.where(zero_point == True)[0]
            V_f1 = float(voltage[zero_point][0])
            # 2、电流梯度突然增大的点作为V_f
            V_f2 = float(dIV[ddI_peaks][0])
            V_f = max(V_f1, V_f2)

            # 找V_p
            # 梯度突然减小的点为V_p
            V_p = float(dIV[ddI_peaks][1])
            # turn_point = np.where(dI == max(dI))[0]
            # V_p = float(dIV[turn_point][0])
            # 找离子、电子饱和电流
            # 将0.9*min(voltage)和0.1*min(voltage)段的均值作为离子饱和电流
            start_for_I_i0 = np.where(voltage > 0.9 * min(voltage))[0][0]
            end_for_I_i0 = np.where(voltage > 0.1 * min(voltage))[0][0]
            I_i0 = np.mean(current[start_for_I_i0:end_for_I_i0])
            # 将最大电流作为电子饱和电流
            I_e0 = max(current)
            # print("V_f=", V_f)
            # print("V_p=", V_p)
            # print("I_i0=", I_i0)
            # print("I_e0=", I_e0)
            # return
            # 选取过渡段
            ln_I_start = np.where(voltage > V_f + 1)[0][0]
            ln_I_end = np.where(voltage > V_p)[0][0]
            trans_stage_vol = voltage[ln_I_start:ln_I_end]
            trans_stage_cur = current[ln_I_start:ln_I_end]
            # 过渡段特性记得把离子电流加上
            ln_I = np.log(trans_stage_cur + abs(I_i0))
            [k, b] = np.polyfit(trans_stage_vol, ln_I, 1)
            # 这种方法计算出来的电子温度T_e实际上是k_B*T_e/e，单位eV
            T_e = 1 / k
            e = pp.Plasma().constants["E_ELEC"]
            m_e = pp.Plasma().constants["M_ELECTRON"]
            k_B = pp.Plasma().constants["K_BOLTZMANN"]
            # 探针直径、长度，注意self.ref_parameters中单位为mm
            d_p = self.ref_parameters["D"] * 1e-3
            l_p = self.ref_parameters["L"] * 1e-3
            # 面积，计算侧面加一个端面
            A_p = np.pi * d_p * l_p + np.pi / 4.0 * d_p**2
            # 计算电子数密度
            n_e = I_e0 / (e * A_p) * np.sqrt(2 * np.pi * m_e / (e * T_e))
            V_ps.append(V_p)
            T_es.append(T_e)
            n_es.append(n_e)
            if if_print:
                print("\n-----num=", i, "-----")
                print("V_f=", V_f)
                print("V_p=", V_p)
                print("I_i0=", I_i0)
                print("I_e0=", I_e0)
                print("T_e=", T_e)
                print("n_e=", n_e)
        if if_print:
            print("\n-----Summary-----")
            print("V_p=", np.mean(V_ps))
            print("T_e=", np.mean(T_es))
            print("n_e=", np.mean(n_es))
        return V_f, T_e, n_e
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
        plt.savefig("res/SLP_plot/" + title.split(".")[0] + ".jpg")
        # plt.show()

        return fig, V_f, T_e, n_e


if __name__ == "__main__":
    dir = "D:/001_zerlingx/archive/for_notes/HC/07_experiments/2024-03 一号阴极测试/2024-04-14 羽流诊断与色散关系测试/data/RAW/"
    path = "tek0276ALL.csv"
    default_path = dir + path
    data_obj = data.data(default_path)
    data_points = data_obj.read()
    SLP_example = SLP()
    SLP_example.cal(data_points, if_print=True)
    # fig, V_f, T_e, n_e = SLP_example.cal(data_points)
    # # plt.savefig("res/SLP_tmp_plot.jpg")
    # plt.plot()
    # plt.show()
