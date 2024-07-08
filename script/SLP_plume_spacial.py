import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import sys
from scipy.interpolate import griddata


sys.path.append("./script")
import data

sys.path.append("./base")
import SLP

# 字体设置
config = {
    "font.family": "serif",
    "font.size": 14,
    "mathtext.fontset": "stix",
    # "font.serif": ["SimSun"],
    "font.serif": ["Times New Roman"],
    "axes.unicode_minus": False,
}
plt.rcParams.update(config)


# 读取px_index.csv中的测量点坐标数据
def read_coordinates(path, header=0):
    with open(path, "r") as file:
        csv_data = pd.read_csv(
            file,
            header=header,
        )
        num = np.array(csv_data.loc[:, "num"])
        r = np.array(csv_data.loc[:, "r"])
        z = np.array(csv_data.loc[:, "z"])
        return num, r, z


def save_index_and_SLP_data(
    save_path, num, r, z, V_ps, T_es, n_es, V_p_stds, T_e_stds, n_e_stds
):
    np.savetxt(
        save_path,
        np.array([num, r, z, V_ps, T_es, n_es, V_p_stds, T_e_stds, n_e_stds]).T,
        delimiter=",",
        fmt="%f",
    )
    names = ["num", "r", "z", "V_p", "T_e", "n_e", "V_p_std", "T_e_std", "n_e_std"]
    with open(save_path, "r+") as file:
        content = file.read()
        file.seek(0, 0)
        file.write(",".join(names) + "\n" + content)


def analyze_and_save():
    # 读取px_index.csv位置坐标
    index_dir = "D:/001_zerlingx/archive/for_notes/HC/07_experiments/2024-03 一号阴极测试/2024-04-14 羽流诊断与色散关系测试/analysis/"
    index_path = "p1_index.csv"
    Num, R, Z = read_coordinates(index_dir + index_path)
    # 读取px的SLP诊断数据并计算
    dir = "D:/001_zerlingx/archive/for_notes/HC/07_experiments/2024-03 一号阴极测试/2024-04-14 羽流诊断与色散关系测试/data/RAW/"
    paths = []
    for i in Num:
        paths.append("tek" + str(i).zfill(4) + "ALL.csv")
    V_ps = []
    T_es = []
    n_es = []
    V_p_stds = []
    T_e_stds = []
    n_e_stds = []
    num = []
    r = []
    z = []
    for i in range(len(paths)):
        single_path = paths[i]
        data_obj = data.data(dir + single_path)
        data_points = data_obj.read()
        single_SLP_point = SLP.SLP()
        single_SLP_point.ref_parameters = {
            "D": 0.8,
            "L": 2,
        }
        V_p, T_e, n_e, V_p_std, T_e_std, n_e_std = single_SLP_point.cal(data_points)
        # 跳过无效数据
        if (
            np.isfinite(V_p) == False
            or np.isfinite(T_e) == False
            or np.isfinite(n_e) == False
        ):
            print("Error in data: ", single_path)
            continue
        num.append(Num[i])
        r.append(R[i])
        z.append(Z[i])
        V_ps.append(V_p)
        T_es.append(T_e)
        n_es.append(n_e)
        V_p_stds.append(V_p_std)
        T_e_stds.append(T_e_std)
        n_e_stds.append(n_e_std)
        print("-----With data : ", single_path, "-----", end=" ")
        print("V_p=%.2f, T_e=%.3f, n_e=%.3e" % (V_p, T_e, n_e), end=", ")
        print("V_p_std (%)=", "{:.2f}".format(V_p_std / abs(V_p) * 100), "%", end=", ")
        print("T_e_std (%)=", "{:.2f}".format(T_e_std / abs(T_e) * 100), "%", end=", ")
        print("n_e_std (%)=", "{:.2f}".format(n_e_std / abs(n_e) * 100), "%")
    # 保存坐标和诊断结果
    save_index_and_SLP_data(
        save_path="res/SLP_spacial/p1_SLP_spacial_20240702.csv",
        num=num,
        r=r,
        z=z,
        V_ps=V_ps,
        T_es=T_es,
        n_es=n_es,
        V_p_stds=V_p_stds,
        T_e_stds=T_e_stds,
        n_e_stds=n_e_stds,
    )
    print("Data saved. nums=", len(num))


def get_plot_grid(r, z, values):
    grid_x, grid_y = np.mgrid[
        np.min(z) : np.max(z) : 100j, np.min(r) : np.max(r) : 100j
    ]
    grid_z = griddata((z, r), values, (grid_x, grid_y), method="linear")
    return grid_x, grid_y, grid_z


def plot_SLP_spacial():
    result_path = "res/SLP_spacial/p3_SLP_spacial.csv"
    with open(result_path, "r") as file:
        data = pd.read_csv(file)
        num = data.loc[:, "num"]
        r = data.loc[:, "r"]
        z = data.loc[:, "z"]
        V_ps = data.loc[:, "V_p"]
        T_es = data.loc[:, "T_e"]
        n_es = data.loc[:, "n_e"]
    # 绘图
    fig, ax = plt.subplots(1, 3, figsize=(16, 3.2))
    plt.subplots_adjust(
        top=0.9,
        bottom=0.18,
        left=0.05,
        right=0.98,
        hspace=0.2,
        wspace=0.2,
    )
    grid_x, grid_y, grid_z = get_plot_grid(r, z, V_ps)
    im = ax[0].contourf(20 - grid_x, grid_y + 259.5, grid_z, levels=50, cmap="jet")
    ax[0].set_title("Plasma potential, V")
    ax[0].set_xlabel("Z, mm")
    ax[0].set_ylabel("R, mm")
    # ax[0].set_title(r"空间电势 ($\mathrm{V}$)")
    # ax[0].set_xlabel(r"$\mathrm{Z\ (mm)}$")
    # ax[0].set_ylabel(r"$\mathrm{R\ (mm)}$")
    ax[0].set_xlim([0.5, 27.5])
    ax[0].set_ylim([0, 17])
    ax[0].set_aspect("equal")
    fig.colorbar(im, ax=ax[0])
    grid_x, grid_y, grid_z = get_plot_grid(r, z, T_es)
    im = ax[1].contourf(20 - grid_x, grid_y + 259.5, grid_z, levels=50, cmap="jet")
    ax[1].set_title("Electron temperature, eV")
    ax[1].set_xlabel("Z, mm")
    ax[1].set_ylabel("R, mm")
    # ax[1].set_title(r"电子温度 ($\mathrm{eV}$)")
    # ax[1].set_xlabel(r"$\mathrm{Z\ (mm)}$")
    # ax[1].set_ylabel(r"$\mathrm{R\ (mm)}$")
    ax[1].set_xlim([0.5, 27.5])
    ax[1].set_ylim([0, 17])
    ax[1].set_aspect("equal")
    fig.colorbar(im, ax=ax[1])
    grid_x, grid_y, grid_z = get_plot_grid(r, z, n_es)
    im = ax[2].contourf(20 - grid_x, grid_y + 259.5, grid_z, levels=50, cmap="jet")
    ax[2].set_title(r"Electron number density, $\mathrm{m^{-3}}$")
    ax[2].set_xlabel("Z, mm")
    ax[2].set_ylabel("R, mm")
    # ax[2].set_title(r"电子密度 ($\mathrm{m^{-3}}$)")
    # ax[2].set_xlabel(r"$\mathrm{Z\ (mm)}$")
    # ax[2].set_ylabel(r"$\mathrm{R\ (mm)}$")
    ax[2].set_xlim([0.5, 27.5])
    ax[2].set_ylim([0, 17])
    ax[2].set_aspect("equal")
    fig.colorbar(im, ax=ax[2])
    plt.savefig(result_path + ".jpg")
    plt.show()


def combine_SLP_spacial():
    config = {"font.size": 18}
    plt.rcParams.update(config)
    gs = gridspec.GridSpec(3, 1)
    fig = plt.figure(figsize=(20, 13.5))
    plt.subplots_adjust(
        top=0.984, bottom=0.016, left=0.015, right=0.985, hspace=0.0, wspace=0.04
    )
    ax = plt.subplot(gs[0, 0])
    img = Image.open("res/SLP_spacial/p1_SLP_spacial.csv.jpg")
    ax.imshow(img)
    ax.axis("off")
    ax.set_title("(a) 10 A, 15 V")
    ax = plt.subplot(gs[1, 0])
    img = Image.open("res/SLP_spacial/p2_SLP_spacial.csv.jpg")
    ax.imshow(img)
    ax.axis("off")
    ax.set_title("(b) 12.5 A, 26 V")
    ax = plt.subplot(gs[2, 0])
    img = Image.open("res/SLP_spacial/p3_SLP_spacial.csv.jpg")
    ax.imshow(img)
    ax.axis("off")
    ax.set_title("(c) 20 A, 17 V")
    plt.savefig("res/SLP_spacial/combined_SLP_spacial.jpg")
    plt.show()


# 获取沿Z或R方向数据（电子数密度）
def get_SLP_along_z_or_r(result_path):
    with open(result_path, "r") as file:
        data = pd.read_csv(file)
        num = data.loc[:, "num"]
        r = data.loc[:, "r"]
        z = data.loc[:, "z"]
        n_es = data.loc[:, "n_e"]
        n_e_stds = data.loc[:, "n_e_std"]
    index = r == min(r)
    res_z = z[index]
    z_n_es = n_es[index]
    z_n_es_std = n_e_stds[index]
    index = z > 19
    res_r = r[index]
    r_n_es = n_es[index]
    r_n_es_std = n_e_stds[index]
    return res_z, z_n_es, res_r, r_n_es, z_n_es_std, r_n_es_std


# 获取沿Z或R方向数据，改为读取电势
def get_SLP_along_z_or_r_V_p(result_path):
    with open(result_path, "r") as file:
        data = pd.read_csv(file)
        num = data.loc[:, "num"]
        r = data.loc[:, "r"]
        z = data.loc[:, "z"]
        V_ps = data.loc[:, "V_p"]
        V_p_stds = data.loc[:, "V_p_std"]
    index = r == min(r)
    res_z = z[index]
    z_n_es = V_ps[index]
    z_n_es_std = V_p_stds[index]
    index = z > 19
    res_r = r[index]
    r_n_es = V_ps[index]
    r_n_es_std = V_p_stds[index]
    return res_z, z_n_es, res_r, r_n_es, z_n_es_std, r_n_es_std


def plot_SLP_along_z_or_r():
    config = {"font.size": 18}
    plt.rcParams.update(config)
    result_path = "res/SLP_spacial/p1_SLP_spacial.csv"
    p1_z, p1_z_n_es, p1_r, p1_r_n_es, p1_z_n_es_std, p1_r_n_es_std = (
        get_SLP_along_z_or_r(result_path)
    )
    result_path = "res/SLP_spacial/p2_SLP_spacial.csv"
    p2_z, p2_z_n_es, p2_r, p2_r_n_es, p2_z_n_es_std, p2_r_n_es_std = (
        get_SLP_along_z_or_r(result_path)
    )
    result_path = "res/SLP_spacial/p3_SLP_spacial.csv"
    p3_z, p3_z_n_es, p3_r, p3_r_n_es, p3_z_n_es_std, p3_r_n_es_std = (
        get_SLP_along_z_or_r(result_path)
    )
    # 绘图
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    plt.subplots_adjust(
        top=0.92,
        bottom=0.14,
        left=0.06,
        right=0.99,
        hspace=0.2,
        wspace=0.2,
    )
    ax[0].plot(
        20 - p1_z,
        p1_z_n_es / max(p1_z_n_es) * 100,
        label=r"$\mathrm{10\ A}$",
        marker="s",
        color="#1f77b4",
    )
    ax[0].errorbar(
        20 - p1_z,
        p1_z_n_es / max(p1_z_n_es) * 100,
        yerr=p1_z_n_es_std / max(p1_z_n_es) * 100,
        ecolor="k",
        elinewidth=0.5,
        marker="s",
        mfc="#1f77b4",
        mec="k",
        mew=1,
        # ms=10,
        alpha=1,
        capsize=5,
        capthick=3,
        linestyle="none",
    )
    ax[0].plot(
        20 - p2_z,
        p2_z_n_es / max(p2_z_n_es) * 100,
        label=r"$\mathrm{12.5\ A}$",
        marker="s",
        color="#ff7f0e",
    )
    ax[0].errorbar(
        20 - p2_z,
        p2_z_n_es / max(p2_z_n_es) * 100,
        yerr=p2_z_n_es_std / max(p2_z_n_es) * 100,
        ecolor="k",
        elinewidth=0.5,
        marker="s",
        mfc="#ff7f0e",
        mec="k",
        mew=1,
        # ms=10,
        alpha=1,
        capsize=5,
        capthick=3,
        linestyle="none",
    )
    ax[0].plot(
        20 - p3_z,
        p3_z_n_es / max(p3_z_n_es) * 100,
        label=r"$\mathrm{20\ A}$",
        marker="s",
        color="#2ca02c",
    )
    ax[0].errorbar(
        20 - p3_z,
        p3_z_n_es / max(p3_z_n_es) * 100,
        yerr=p3_z_n_es_std / max(p3_z_n_es) * 100,
        ecolor="k",
        elinewidth=0.5,
        marker="s",
        mfc="#2ca02c",
        mec="k",
        mew=1,
        # ms=10,
        alpha=1,
        capsize=5,
        capthick=3,
        linestyle="none",
    )
    ax[0].set_title(r"(a) $\mathrm{n_e}$ Along Z axis")
    ax[0].set_xlabel("Z, mm")
    ax[0].set_ylabel("Normalized electron density, %")
    # ax[0].set_title(r"$\mathrm{(a)}$ 沿$\mathrm{Z}$轴")
    # ax[0].set_xlabel(r"$\mathrm{Z\ (mm)}$")
    # ax[0].set_ylabel(r"$\mathrm{n_e\ (\%)}$")
    ax[0].legend()
    ax[0].grid()
    # ax[0].set_yscale("log")
    # ax[0].grid(True, which="both", axis="both", ls="--")
    ax[1].plot(
        260 + p1_r,
        p1_r_n_es / max(p1_r_n_es) * 100,
        label=r"$\mathrm{10\ A}$",
        marker="s",
        color="#1f77b4",
    )
    ax[1].errorbar(
        260 + p1_r,
        p1_r_n_es / max(p1_r_n_es) * 100,
        yerr=p1_r_n_es_std / max(p1_r_n_es) * 100,
        ecolor="k",
        elinewidth=0.5,
        marker="s",
        mfc="#1f77b4",
        mec="k",
        mew=1,
        # ms=10,
        alpha=1,
        capsize=5,
        capthick=3,
        linestyle="none",
    )
    ax[1].plot(
        260 + p2_r,
        p2_r_n_es / max(p2_r_n_es) * 100,
        label=r"$\mathrm{12.5\ A}$",
        marker="s",
        color="#ff7f0e",
    )
    ax[1].errorbar(
        260 + p2_r,
        p2_r_n_es / max(p2_r_n_es) * 100,
        yerr=p2_r_n_es_std / max(p2_r_n_es) * 100,
        ecolor="k",
        elinewidth=0.5,
        marker="s",
        mfc="#ff7f0e",
        mec="k",
        mew=1,
        # ms=10,
        alpha=1,
        capsize=5,
        capthick=3,
        linestyle="none",
    )
    ax[1].plot(
        260 + p3_r,
        p3_r_n_es / max(p3_r_n_es) * 100,
        label=r"$\mathrm{20\ A}$",
        marker="s",
        color="#2ca02c",
    )
    ax[1].errorbar(
        260 + p3_r,
        p3_r_n_es / max(p3_r_n_es) * 100,
        yerr=p3_r_n_es_std / max(p3_r_n_es) * 100,
        ecolor="k",
        elinewidth=0.5,
        marker="s",
        mfc="#2ca02c",
        mec="k",
        mew=1,
        # ms=10,
        alpha=1,
        capsize=5,
        capthick=3,
        linestyle="none",
    )
    ax[1].set_title(r"(b) $\mathrm{n_e}$ Along R axis")
    ax[1].set_xlabel("R, mm")
    ax[1].set_ylabel("Normalized electron density, %")
    # ax[1].set_title(r"$\mathrm{(b)}$ 沿$\mathrm{R}$轴")
    # ax[1].set_xlabel(r"$\mathrm{R\ (mm)}$")
    # ax[1].set_ylabel(r"$\mathrm{n_e\ (\%)}$")
    ax[1].legend()
    ax[1].grid()
    # ax[1].set_yscale("log")
    # ax[1].grid(True, which="both", axis="both", ls="--")
    plt.savefig("res/SLP_spacial/SLP_spacial_along_z_or_r.jpg")
    plt.show()


def plot_SLP_along_r_logVp_and_logne():
    config = {"font.size": 18}
    plt.rcParams.update(config)

    # 绘图
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    plt.subplots_adjust(
        top=0.92,
        bottom=0.14,
        left=0.06,
        right=0.99,
        hspace=0.2,
        wspace=0.2,
    )

    # (a) 沿R电势
    result_path = "res/SLP_spacial/p1_SLP_spacial.csv"
    p1_z, p1_z_n_es, p1_r, p1_r_n_es, p1_z_n_es_std, p1_r_n_es_std = (
        get_SLP_along_z_or_r_V_p(result_path)
    )
    result_path = "res/SLP_spacial/p2_SLP_spacial.csv"
    p2_z, p2_z_n_es, p2_r, p2_r_n_es, p2_z_n_es_std, p2_r_n_es_std = (
        get_SLP_along_z_or_r_V_p(result_path)
    )
    result_path = "res/SLP_spacial/p3_SLP_spacial.csv"
    p3_z, p3_z_n_es, p3_r, p3_r_n_es, p3_z_n_es_std, p3_r_n_es_std = (
        get_SLP_along_z_or_r_V_p(result_path)
    )
    ax[0].plot(
        260 + p1_r,
        p1_r_n_es,
        label=r"$\mathrm{10\ A}$",
        marker="s",
        color="#1f77b4",
    )
    ax[0].errorbar(
        260 + p1_r,
        p1_r_n_es,
        yerr=p1_r_n_es_std,
        ecolor="k",
        elinewidth=0.5,
        marker="s",
        mfc="#1f77b4",
        mec="k",
        mew=1,
        # ms=10,
        alpha=1,
        capsize=5,
        capthick=3,
        linestyle="none",
    )
    ax[0].plot(
        260 + p2_r,
        p2_r_n_es,
        label=r"$\mathrm{12.5\ A}$",
        marker="s",
        color="#ff7f0e",
    )
    ax[0].errorbar(
        260 + p2_r,
        p2_r_n_es,
        yerr=p2_r_n_es_std,
        ecolor="k",
        elinewidth=0.5,
        marker="s",
        mfc="#ff7f0e",
        mec="k",
        mew=1,
        # ms=10,
        alpha=1,
        capsize=5,
        capthick=3,
        linestyle="none",
    )
    ax[0].plot(
        260 + p3_r,
        p3_r_n_es,
        label=r"$\mathrm{20\ A}$",
        marker="s",
        color="#2ca02c",
    )
    ax[0].errorbar(
        260 + p3_r,
        p3_r_n_es,
        yerr=p3_r_n_es_std,
        ecolor="k",
        elinewidth=0.5,
        marker="s",
        mfc="#2ca02c",
        mec="k",
        mew=1,
        # ms=10,
        alpha=1,
        capsize=5,
        capthick=3,
        linestyle="none",
    )
    ax[0].set_title(r"(a) $\mathrm{V_p}$ Along R axis")
    ax[0].set_xlabel("R, mm")
    ax[0].set_ylabel("Plasma potentian, V")
    # ax[1].set_title(r"$\mathrm{(b)}$ 沿$\mathrm{R}$轴")
    # ax[1].set_xlabel(r"$\mathrm{R\ (mm)}$")
    # ax[1].set_ylabel(r"$\mathrm{n_e\ (\%)}$")
    ax[0].legend()
    # ax[1].grid()
    # ax[0].set_yscale("log")
    ax[0].grid(True, which="both", axis="both", ls="--")
    # ax[0].set_yscale("log")
    # ax[0].grid(True, which="both", axis="both", ls="--")

    # (b) 沿R电子数密度
    result_path = "res/SLP_spacial/p1_SLP_spacial.csv"
    p1_z, p1_z_n_es, p1_r, p1_r_n_es, p1_z_n_es_std, p1_r_n_es_std = (
        get_SLP_along_z_or_r(result_path)
    )
    result_path = "res/SLP_spacial/p2_SLP_spacial.csv"
    p2_z, p2_z_n_es, p2_r, p2_r_n_es, p2_z_n_es_std, p2_r_n_es_std = (
        get_SLP_along_z_or_r(result_path)
    )
    result_path = "res/SLP_spacial/p3_SLP_spacial.csv"
    p3_z, p3_z_n_es, p3_r, p3_r_n_es, p3_z_n_es_std, p3_r_n_es_std = (
        get_SLP_along_z_or_r(result_path)
    )
    ax[1].plot(
        260 + p1_r,
        p1_r_n_es,
        label=r"$\mathrm{10\ A}$",
        marker="s",
        color="#1f77b4",
    )
    ax[1].errorbar(
        260 + p1_r,
        p1_r_n_es,
        yerr=p1_r_n_es_std,
        ecolor="k",
        elinewidth=0.5,
        marker="s",
        mfc="#1f77b4",
        mec="k",
        mew=1,
        # ms=10,
        alpha=1,
        capsize=5,
        capthick=3,
        linestyle="none",
    )
    ax[1].plot(
        260 + p2_r,
        p2_r_n_es,
        label=r"$\mathrm{12.5\ A}$",
        marker="s",
        color="#ff7f0e",
    )
    ax[1].errorbar(
        260 + p2_r,
        p2_r_n_es,
        yerr=p2_r_n_es_std,
        ecolor="k",
        elinewidth=0.5,
        marker="s",
        mfc="#ff7f0e",
        mec="k",
        mew=1,
        # ms=10,
        alpha=1,
        capsize=5,
        capthick=3,
        linestyle="none",
    )
    ax[1].plot(
        260 + p3_r,
        p3_r_n_es,
        label=r"$\mathrm{20\ A}$",
        marker="s",
        color="#2ca02c",
    )
    ax[1].errorbar(
        260 + p3_r,
        p3_r_n_es,
        yerr=p3_r_n_es_std,
        ecolor="k",
        elinewidth=0.5,
        marker="s",
        mfc="#2ca02c",
        mec="k",
        mew=1,
        # ms=10,
        alpha=1,
        capsize=5,
        capthick=3,
        linestyle="none",
    )
    ax[1].set_title(r"(b) $\mathrm{n_e}$ Along R axis (log yscale)")
    ax[1].set_xlabel("R, mm")
    ax[1].set_ylabel(r"Electron density, $\mathrm{m^{-3}}$")
    # ax[1].set_title(r"$\mathrm{(b)}$ 沿$\mathrm{R}$轴")
    # ax[1].set_xlabel(r"$\mathrm{R\ (mm)}$")
    # ax[1].set_ylabel(r"$\mathrm{n_e\ (\%)}$")
    ax[1].legend()
    # ax[1].grid()
    ax[1].set_yscale("log")
    ax[1].grid(True, which="both", axis="both", ls="--")
    plt.savefig("res/SLP_spacial/SLP_along_r_logVp_and_logne.jpg")
    plt.show()


if __name__ == "__main__":
    analyze_and_save()
    # plot_SLP_spacial()
    # combine_SLP_spacial()
    # plot_SLP_along_z_or_r()
    # plot_SLP_along_r_logVp_and_logne()
