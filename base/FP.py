# Faraday Probe (FP)
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("./")
import constant.plasma_parameters as pp
import constant.reference_quantity_hollow_cathode as rqhc

"""
Brief: Faraday Probe, basic plasma diagnostic device.
Refe.: 
Set  : 
Calc.: 
"""


class FP:
    def __init__(self) -> None:
        HC = rqhc.HollowCathode()
        self.ref_parameters = {
            "D_COLLECTOR": 12,  # 收集极直径        mm
            "THICK_COLLECTOR": 0.5,  # 收集极厚度        mm
            "GAP_INSULATOR": 1,  # 绝缘套管间隙      mm
            "D_SHIELD": 14,  # 屏蔽极内径        mm
            "THICK_SHIELD": 0.5,  # 屏蔽极厚度        mm
        }

    def cal(self, angle, current, print_result=False):
        """
        Brief: 计算羽流发散角
        Args :
            angle   : 羽流发散角    °
            current : 电流         mA/A
        Return:
            plume_angle   : 羽流发散角
        """
        max_current = max(current)
        k_boundary = 0.9  # 电流边界系数，即羽流发散角的边界为电流最大值的90%
        angle1 = min(angle[current > k_boundary * max_current])
        angle2 = max(angle[current > k_boundary * max_current])
        plume_angle = angle2 - angle1

        if print_result:
            print("\n----------in FP.py, print result----------")
            print("angle1:", angle1)
            print("angle2:", angle2)
            print("羽流发散角为：", plume_angle)
            fig, ax = plt.subplots()
            ax.plot(angle, current)
            ax.set_xlabel("angle")
            ax.set_ylabel("current")
            ax.set_title("FP I-angle curve")
            ax.grid()
            max_position = angle[current == max(current)]
            ax.vlines(
                x=[
                    max_position - plume_angle / 2,
                    max_position + plume_angle / 2,
                ],
                ymin=0,
                ymax=max(current),
                colors="r",
                linestyles="dashed",
            )
            ax.legend(["I-angle curve", "90% current boundary"])
            plt.show()
            print("----------in FP.py, print result end----------\n")

        return plume_angle
