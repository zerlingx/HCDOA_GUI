# Single-Langmuir Probe (SLP)
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
import sys

sys.path.append("./")
import constant.plasma_parameters as pp
import constant.reference_quantity_hollow_cathode as rqhc

"""
Brief: Single-Langmuir Probe, basic plasma diagnostic device.
Refe.: 
Set  : 
Calc.: 
"""


class SLP:
    def __init__(self) -> None:
        HC = rqhc.HollowCathode()
        self.ref_parameters = {
            "D": 0.3,  # 探针直径        mm
            "L": 10,  # 探针长度        mm
        }

    def cal(self, voltage, current, print_result=False):
        """
        Brief: 朗缪尔单探针计算，输出电子温度、电子密度、等离子体电势（电子饱和电流位置）
        Args:
            voltage (_np.array or list_): diagnostic voltage data   V
            current (_np.array or list_): diagnostic current data   A
            print_result (bool, optional): print result or not. Defaults to False.
        Returns:
            _float64_: k_BT_e, n_e, I_e0_position
        """
        # 输入数据滤波，否则dIdV噪声过大
        # 但是对于从论文读取的数据，滤波会使得过渡区曲线过于平滑，以至于电子温度计算不准确
        window_size = 5
        smooth_dimention = 1
        current = scipy.signal.savgol_filter(current, window_size, smooth_dimention)
        # 差分近似导数并滤波
        dIdV = np.diff(current) / np.diff(voltage)
        dIdV = np.insert(dIdV, 0, dIdV[0])
        dIdV = scipy.signal.savgol_filter(dIdV, window_size, smooth_dimention)
        # dIdV最大值对应的电流值等于电子饱和电流
        dIdV_max = max(dIdV)
        I_e0 = current[dIdV == dIdV_max][0]
        I_e0_position = np.array(voltage)[dIdV == dIdV_max][0]  # 等离子体电势，即电子饱和电流对应的电压
        # 过渡区ln(I) - V曲线斜率的倒数等于电子温度k_B*T_e
        ln_I_range = np.logical_and(voltage < I_e0_position, current > 0)
        ln_I = np.log(current[ln_I_range])
        [k, b] = np.polyfit(voltage[ln_I_range], ln_I, 1)
        k_BT_e = 1 / k  # 电子温度，单位eV
        e = pp.Plasma().constants["E_ELEC"]
        m_e = pp.Plasma().constants["M_ELECTRON"]
        D = self.ref_parameters["D"] * 1e-3
        L = self.ref_parameters["L"] * 1e-3
        A = np.pi * D * L + D**2 / 4 * np.pi
        # 两个公式均可，注意电子温度单位
        # n_e = I_e0 / (e * A) * np.sqrt((2 * np.pi * m_e) / (k_BT_e * e))
        n_e = 3.73 * pow(10, 13) * I_e0 / (A * np.sqrt(k_BT_e))
        # 悬浮电位
        min_I = min(abs(current))
        V_f = np.array(voltage)[abs(current) == min_I][0]

        if print_result:
            print("\n----------in SLP.py, print result----------")
            print("电子饱和电流为：%e" % I_e0)
            print("电子温度为：", k_BT_e)
            print("电子密度为：%e" % n_e)
            print("等离子体电势为：%.2f" % I_e0_position)
            print("悬浮电位为：%.2f" % V_f)

            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            plt.subplots_adjust(wspace=0.5)

            # (a) SLP I-V curve and dIdV-V curve
            axplt1 = ax[0].plot(voltage, current, label="I-V curve")
            ax[0].set_xlabel("voltage")
            ax[0].set_ylabel("current")
            ax[0].set_title("(a) SLP I-V curve and dIdV-V curve")
            ax[0].grid()
            ax_twin = ax[0].twinx()
            axplt_twin = ax_twin.plot(voltage, dIdV, "r", label="dIdV-V curve")
            ax_twin.set_ylabel("dIdV")
            axplt2 = ax[0].vlines(
                x=I_e0_position,
                ymin=min(current),
                ymax=max(current),
                color="g",
                linestyles="dashed",
                label="I_e0_position",
            )

            # 一起显示图例
            axplt_total = axplt1
            axplt_total.extend(axplt_twin)  # 不能用append，否则会出现错误
            axplt_total.append(axplt2)  # 不能用extend，否则会出现错误
            ax[0].legend(axplt_total, [l.get_label() for l in axplt_total])

            # (b) ln(I) - V curve and k * V + b curve
            ax[1].scatter(voltage[ln_I_range], ln_I)
            ax[1].plot(voltage[ln_I_range], k * voltage[ln_I_range] + b, "r")
            ax[1].set_xlabel("voltage")
            ax[1].set_ylabel("ln(I)")
            ax[1].set_title("(b) ln(I) - V scatter and k * V + b curve")
            ax[1].legend(["ln(I) - V", "k * V + b"])
            ax[1].grid()

            plt.show()
            print("----------in SLP.py, print result end----------\n")

        return k_BT_e, n_e, I_e0_position
