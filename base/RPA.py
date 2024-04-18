import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import scipy
from scipy.interpolate import griddata

sys.path.append("./script")
import data

sys.path.append("./base")
import SLP


class RPA:
    def __init__(self) -> None:
        self.ref_parameters = {
            "d": 0.9,  # 栅网孔径       mm
            "D": 28,  # 收集极直径      mm
        }

    def cal(self, data_points, if_print=False):
        """
        Brief: 读取栅极扫描电压和收集极电流，计算离子能量分布
        Args:
            data_points: list, [time, voltage, current]
        Returns:
            fig: matplotlib.figure.Figure, 绘图对象
            T_i: float, 离子温度
            f_ni: float, 离子密度分布函数
        """
        time = data_points[0]
        VOLTAGE = data_points[1]
        CURRENT = data_points[2]
        # 调用SLP类的find_periods方法，查找电压扫描周期
        tmp_SLP = SLP.SLP()
        starts, ends = tmp_SLP.find_periods(data_points, 2, 0.7, if_print=False)

        for i in range(len(starts)):
            # 选取一个周期的数据,删除始末段扫描电压突变时的数据
            stage_1 = starts[i]
            stage_2 = ends[i]
            period = stage_2 - stage_1
            stage_1 = stage_1 + int(0.001 * period)
            stage_2 = stage_2 - int(0.001 * period)
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
            # 计算dI/dV
            # 降采样
            dstep = int(len(voltage) / 100)
            start = int(0.001 * len(current))
            end = int(0.999 * len(current))
            dI = np.diff(current[start:end:dstep]) / np.diff(voltage[start:end:dstep])
            dI = scipy.signal.savgol_filter(dI, int(len(dI) / 20 + 2), smooth_dimention)
            dIV = voltage[start:end:dstep]
            dIV = dIV[1:]
            # 测试绘图
            fig, ax = plt.subplots(1, 2, figsize=(16, 6))
            axplt1 = ax[0].plot(voltage, color="orange")
            ax[0].set_xlabel("Time (s)")
            ax[0].set_ylabel("Voltage (V)")
            axtwin = ax[0].twinx()
            axplt2 = axtwin.plot(current, color="blue")
            axtwin.set_ylabel("Current (A)")
            axtwin.grid()
            axplts = axplt1 + axplt2
            labels = ["Voltage", "Current"]
            ax[0].legend(axplts, labels, loc="upper right")
            f_ni = scipy.signal.savgol_filter(-dI, int(len(dI) / 10), smooth_dimention)
            ax[1].plot(dIV, f_ni)
            ax[1].legend(["dI/dV"])
            ax[1].grid()
            plt.show()
            return


if __name__ == "__main__":
    dir = "D:/001_zerlingx/archive/for_notes/HC/07_experiments/2024-03 一号阴极测试/2024-04-14 羽流诊断与色散关系测试/data/RAW/"
    path = "tek0017ALL.csv"
    default_path = dir + path
    data_obj = data.data(default_path)
    data_points = data_obj.read()
    RPA_example = RPA()
    RPA_example.cal(data_points, if_print=True)
