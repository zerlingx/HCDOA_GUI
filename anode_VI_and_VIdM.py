import numpy as np
import matplotlib.pyplot as plt

config = {
    "font.family": "serif",
    "font.size": 20,
    "mathtext.fontset": "stix",
    # "font.serif": ["SimSun"],
    "font.serif": ["Times New Roman"],
    "axes.unicode_minus": False,
}
plt.rcParams.update(config)

anode_3sccm = [
    [2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5],
    [49, 48, 48, 49, 51, 56, 63, 67, 76, 91],
]

anode_6sccm = [
    [3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11],
    [34, 35, 35, 35, 35, 35, 36, 35, 35, 35, 34, 35, 38, 40, 42, 45, 49],
]

if __name__ == "__main__":
    fig, ax = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(12, 6),
    )
    fig.suptitle("Anode Voltage-Current/IdM")
    ax[0].plot(anode_3sccm[0], anode_3sccm[1], "-o", label="3 sccm")
    ax[0].plot(anode_6sccm[0], anode_6sccm[1], "-o", label="6 sccm")
    ax[0].vlines(
        x=[4],
        ymin=40,
        ymax=60,
        colors="r",
        linestyles="dashed",
    )
    ax[0].vlines(
        x=[8],
        ymin=30,
        ymax=40,
        colors="r",
        linestyles="dashed",
    )
    ax[0].set_xlabel("Current (A)")
    ax[0].set_ylabel("Voltage (V)")
    ax[0].grid()
    ax[0].legend(loc="upper left")
    ax[0].set_title("(a) U-I")
    IM_3sccm = np.divide(anode_3sccm[0], 3)
    IM_6sccm = np.divide(anode_6sccm[0], 6)
    ax[1].plot(IM_3sccm, anode_3sccm[1], "-o", label="3 sccm")
    ax[1].plot(IM_6sccm, anode_6sccm[1], "-o", label="6 sccm")
    ax[1].vlines(
        x=[1.33],
        ymin=30,
        ymax=60,
        colors="r",
        linestyles="dashed",
    )
    ax[1].set_xlabel("Current / Mass Flow (A/sccm)")
    ax[1].set_ylabel("Voltage (V)")
    ax[1].grid()
    ax[1].legend(loc="upper left")
    ax[1].set_title("(b) V-IdM")
    plt.savefig("res/anode_VI_and_VIdM.jpg")
    plt.show()
