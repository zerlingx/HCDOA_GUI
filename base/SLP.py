# Single-Langmuir Probe (SLP)
import numpy as np
import scipy
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
            "D": 0.8,  # 探针直径        mm
            "L": 2,  # 探针长度        mm
        }

    def find_periods(
        self,
        data_points,
        periods_num=10,
        peak_height_rate=0.7,
        if_print=False,
    ):
        """
        Brief: 找到锯齿波周期
        Args:
            data_points: list, [time, voltage, current]
            periods_num: int, 数据包含的周期数,决定了峰值查找函数中的distance
            peak_height_rate: float, 峰值高度比例,决定了峰值查找函数中的height
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
            # 电压取为正值便于查找峰值
            voltage + abs(min(voltage)),
            # 最小周期记录波形的1/(periods_num+1)
            distance=len(voltage) / (periods_num + 1),
            # 最小峰值为波形最大值乘peak_height_rate
            height=max(voltage) * peak_height_rate,
        )
        lows, _ = scipy.signal.find_peaks(
            -voltage + abs(min(-voltage)),
            distance=len(voltage) / (periods_num + 1),
            height=max(voltage) * peak_height_rate,
        )
        # 排除因电平设置可能造成的谷值前置
        # 若峰谷位置小于半个扫描周期（50 ms），即视为谷值前置
        scan_freq = 10  # Hz
        scan_period = int(1000 / scan_freq) * 1e-3  # ms
        if abs(time[peaks[0]] - time[lows[0]]) < 0.5 * scan_period:
            lows = np.delete(lows, 0)
        # 检查峰谷时刻电压，删除电压离群值
        mean_peaks_voltage = np.mean(voltage[peaks])
        mean_lows_voltage = np.mean(voltage[lows])
        error_limit = 0.1
        error_peaks_index = []
        for i in range(len(peaks)):
            if (
                abs(voltage[peaks[i]] - mean_peaks_voltage) / abs(mean_peaks_voltage)
                > error_limit
            ):
                error_peaks_index.append(i)
        # 找到峰值离群点，删除
        # 注意要把对应的谷值也删除
        peaks = np.delete(peaks, error_peaks_index)
        lows = np.delete(lows, error_peaks_index)
        error_lows_index = []
        for i in range(len(lows)):
            if (
                abs(voltage[lows[i]] - mean_lows_voltage) / abs(mean_lows_voltage)
                > error_limit
            ):
                error_lows_index.append(i)
        lows = np.delete(lows, error_lows_index)
        peaks = np.delete(peaks, error_lows_index)
        # 由峰谷分布前后确定数据读取起始
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

    def cal(self, data_points, title="", if_print=False, if_print_first=False):
        """
        Brief: 朗缪尔单探针计算,输出绘图对象,V_p, T_e, n_e
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
            try:
                stage_1 = starts[i]
                stage_2 = ends[i]
                time = time[stage_1:stage_2]
                voltage = VOLTAGE[stage_1:stage_2]
                current = CURRENT[stage_1:stage_2]
                # 将示波器量程截断产生的无效值转换为0
                voltage = np.nan_to_num(voltage, nan=0, posinf=0, neginf=0)
                current = np.nan_to_num(current, nan=0, posinf=0, neginf=0)
                # 平滑滤波
                smooth_dimention = 1
                window_size = int(len(voltage) / 100)
                voltage = scipy.signal.savgol_filter(
                    voltage, window_size, smooth_dimention
                )
                current = scipy.signal.savgol_filter(
                    current, window_size, smooth_dimention
                )
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
                # 一阶导
                dI = np.diff(current[start:end:dstep]) / np.diff(
                    voltage[start:end:dstep]
                )
                dI = scipy.signal.savgol_filter(
                    dI, int(len(dI) / 20 + 2), smooth_dimention
                )
                dIV = voltage[start:end:dstep]
                dIV = dIV[1:]
                ddI = np.diff(dI)
                ddI = scipy.signal.savgol_filter(
                    ddI, int(len(ddI) / 20 + 2), smooth_dimention
                )
                ddI = abs(ddI)
                # 测试绘图 1
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
                # return
                # 测试绘图 2
                # plt.plot(
                #     dIV,
                #     current[start:end:dstep][1:] / max(current[start:end:dstep][1:]),
                #     label="current",
                # )
                # plt.plot(dIV, dI / max(dI), label="dI")
                # plt.plot(dIV[1:], ddI / max(ddI), label="ddI")
                # plt.xlim([0, 30])
                # plt.legend()
                # plt.grid()
                # plt.show()
                # return
                # 测试绘图 3
                # EEDF
                # plt.legend(["dI", "ddI"])
                # index = dIV > 0
                # V_EEDF = dIV[index][:-1]
                # ddI_EEDF = ddI[index[:-1]]
                # f_EEDF = []
                # for i in range(len(ddI_EEDF)):
                #     f_EEDF.append(1 * np.sqrt(V_EEDF[i]) * ddI_EEDF[i])
                # plt.plot(V_EEDF, f_EEDF)
                # plt.grid()
                # plt.show()
                # return
                # 找V_f
                ddI_peaks, _ = scipy.signal.find_peaks(
                    ddI,
                    distance=len(ddI) / 30,  # 最小周期记录波形的1/10
                    height=max(ddI) * 0.3,  # 最小峰值为波形最大值的30%
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
                # 若出现异常值，跳过
                if (
                    V_p < V_f
                    or np.isfinite(V_p) == False
                    or np.isfinite(T_e) == False
                    or np.isfinite(n_e) == False
                ):
                    continue
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
                    # 若if_print_first为Ture则绘制第一个周期的SLP分析过程
                    if if_print_first == True:
                        try:
                            # 绘图展示SLP分析过程
                            fig, ax = plt.subplots(
                                nrows=1,
                                ncols=3,
                                figsize=(12, 4),
                            )
                            plt.subplots_adjust(
                                wspace=0.5,
                                left=0.055,
                                right=0.98,
                                bottom=0.12,
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
                            # (b-1) I-V曲线和两个悬浮电势V_f
                            range_I = max(current) - min(current)
                            ax[1].vlines(
                                x=V_f1,
                                ymin=0 - 0.1 * range_I,
                                ymax=0 + 0.1 * range_I,
                                colors="r",
                                linestyles="dashed",
                            )
                            ax[1].text(
                                x=V_f1 + max(voltage) * 0.1,
                                y=0.1 * range_I,
                                s="V_f1=" + str(round(V_f1, 2)) + " V",
                            )
                            range_I = max(current) - min(current)
                            ax[1].vlines(
                                x=V_f2,
                                ymin=0 - 0.1 * range_I,
                                ymax=0 + 0.1 * range_I,
                                colors="r",
                                linestyles="dashed",
                            )
                            ax[1].text(
                                x=V_f2 + max(voltage) * 0.1,
                                y=0.02 * range_I,
                                s="V_f2=" + str(round(V_f2, 2)) + " V",
                            )
                            axplt1 = ax[1].plot(voltage, current)
                            # (b-2) dI/dV-V曲线，找其拐点为等离子体电势V_p
                            # 降采样
                            axtwin = ax[1].twinx()
                            axplt2 = axtwin.plot(dIV, dI, color="orange")
                            axtwin.set_ylabel("dI/dV (mA/V)")
                            axplts = axplt1 + axplt2
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
                            labels = ["I-V", "dI/dV-V"]
                            ax[1].legend(axplts, labels, loc="upper left")
                            ax[1].set_xlim((-30, 50))  # 绘制电压范围
                            ax[1].set_xlabel("Voltage (V)")
                            ax[1].set_ylabel("Current (A)")
                            ax[1].grid()
                            ax[1].set_title("(b) I-V and dI/dV-V")
                            # (b-3) 找饱和离子电流
                            ax[1].hlines(
                                xmin=0.9 * min(voltage),
                                xmax=0.1 * min(voltage),
                                y=I_i0,
                                colors="r",
                                linestyles="dashed",
                            )
                            ax[1].text(
                                # x=0.9 * min(voltage),
                                x=-30,
                                y=I_i0 + (max(current) - min(current)) * 0.1,
                                s="I_i0=" + str(round(I_i0, 5)) + " A",
                            )
                            # (c) ln_I to k
                            ax[2].scatter(trans_stage_vol, ln_I)
                            ax[2].plot(trans_stage_vol, k * trans_stage_vol + b, "r")
                            ax[2].set_xlabel("trans_stage_vol")
                            ax[2].set_ylabel("ln(I)")
                            ax[2].legend(["ln(I)", "k*V+b"], loc="upper left")
                            ax[2].grid()
                            ax[2].text(
                                x=trans_stage_vol[0]
                                + 0.4 * (trans_stage_vol[-1] - trans_stage_vol[0]),
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
                            plt.plot()
                            plt.show()
                            if_print_first = False
                        except:
                            if_print_first = True
            except:
                pass  # 若计算失败，跳过
        V_p = np.mean(V_ps)
        T_e = np.mean(T_es)
        n_e = np.mean(n_es)
        V_p_std = np.std(V_ps)
        T_e_std = np.std(T_es)
        n_e_std = np.std(n_es)
        if if_print:
            print("\n-----Summary-----")
            print("V_p=", V_p)
            print("T_e=", T_e)
            print("n_e=", n_e)
            print("V_p_std (%)=", "{:.2f}".format(V_p_std / abs(V_p) * 100), "%")
            print("T_e_std (%)=", "{:.2f}".format(T_e_std / abs(T_e) * 100), "%")
            print("n_e_std (%)=", "{:.2f}".format(n_e_std / abs(n_e) * 100), "%")

        return V_p, T_e, n_e, V_p_std, T_e_std, n_e_std


if __name__ == "__main__":
    dir = "D:/001_zerlingx/archive/for_notes/HC/07_experiments/2024-03 一号阴极测试/2024-04-14 羽流诊断与色散关系测试/data/RAW/"
    path = "tek0246ALL.csv"
    # dir = "D:/001_zerlingx/archive/for_notes/HC/07_experiments/2024-03 一号阴极测试/2024-05-12 羽流诊断与色散关系测试/data/RAW/"
    # path = "tek0102ALL.csv"

    default_path = dir + path
    data_obj = data.data(default_path)
    data_points = data_obj.read()
    SLP_example = SLP()
    SLP_example.ref_parameters = {
        "D": 0.25,
        "L": 8,
    }
    V_p, T_e, n_e, V_p_std, T_e_std, n_e_std = SLP_example.cal(
        data_points, if_print=True, if_print_first=True
    )
