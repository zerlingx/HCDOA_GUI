import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


class condition:
    def __init__(
        self,
        date,  # 日期 (yyyymmdd_number)
        cathode,  # 阴极型号
        propellant,  # 工质类型 (Xe,Kr,Ar)
        mass_flow,  # 质量流量 (sccm)
        anode_current,  # 阳极电流 (A)
        anode_voltage,  # 阳极电压 (V)
        discharge_mode,  # 放电模式(plume,spot)
        cycle,  # 测试行程(forward,reverse,single)
    ) -> None:
        self.date = date
        self.cathode = cathode
        self.propellant = propellant
        self.mass_flow = mass_flow
        self.anode_current = anode_current
        self.anode_voltage = anode_voltage
        self.discharge_mode = discharge_mode
        self.cycle = cycle


def discharge_mode_2D():
    path = "src/conditions.csv"
    with open(path, "r") as file:
        csv_data = pd.read_csv(
            file,
            header=0,
        )
    columns = [
        "date",
        "cathode",
        "propellant",
        "mass_flow",
        "anode_current",
        "anode_voltage",
        "discharge_mode",
        "cycle",
    ]
    conditions = []
    len_csv = len(csv_data)
    # print(len_csv)
    # print(csv_data)
    for i in range(len_csv):
        conditions.append(
            condition(
                csv_data.loc[i, "date"],
                csv_data.loc[i, "cathode"],
                csv_data.loc[i, "propellant"],
                csv_data.loc[i, "mass_flow"],
                csv_data.loc[i, "anode_current"],
                csv_data.loc[i, "anode_voltage"],
                csv_data.loc[i, "discharge_mode"],
                csv_data.loc[i, "cycle"],
            )
        )

    cathode_index = "HIT01"
    conditions = [i for i in conditions if i.cathode == cathode_index]

    # 字体和绘图设置
    config = {
        "font.family": "serif",
        "font.size": 12,
        "mathtext.fontset": "stix",
        # "font.serif": ["SimSun"],
        "font.serif": ["Times New Roman"],
        "axes.unicode_minus": False,
    }
    plt.rcParams.update(config)
    # plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    for i in conditions:
        # plt.arrow config
        head_width = 0.2
        head_length = 1
        if i.cycle == "forward":
            dx = 2
            dy = 0.3
            x = i.anode_current
            y = i.mass_flow
        else:
            dx = -2
            dy = 0
            x = i.anode_current - dx + head_length
            y = i.mass_flow
        if i.discharge_mode == "plume":
            color = "red"
        else:
            color = "green"
        # plt.scatter(i.anode_current, i.mass_flow, color=color)
        ax[0].arrow(
            x,
            y,
            dx,
            dy,
            head_width=head_width,
            head_length=head_length,
            fc=color,
            ec=color,
        )
    line_x = [10, 51]
    line_y = [92, 74]
    ax[0].plot(line_x, line_y, color="blue", linestyle="--")
    ax[0].grid()
    ax[0].set_xlabel("anode current (A)")
    ax[0].set_ylabel("mass flow (sccm)")
    ax[0].set_title("discharge mode 20231105_HIT01")
    # 自定义图例
    legend_elements = [
        Line2D([0], [0], color="red", lw=4, label="plume mode"),
        Line2D([0], [0], color="green", lw=4, label="spot mode"),
    ]
    ax[0].legend(handles=legend_elements, loc="upper right")

    # ax[1] 画电压电流曲线图，虽然第一临界点附近特性和预期的不一样，也画出来看看
    # plt.savefig("res/discharge_mode.png")
    plt.show()


if __name__ == "__main__":
    discharge_mode_2D()
