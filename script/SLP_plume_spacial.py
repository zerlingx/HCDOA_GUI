import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from scipy.interpolate import griddata

sys.path.append("./script")
import data

sys.path.append("./base")
import SLP


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


def save_index_and_SLP_data(save_path, num, r, z, V_ps, T_es, n_es):
    np.savetxt(
        save_path,
        np.array([num, r, z, V_ps, T_es, n_es]).T,
        delimiter=",",
        fmt="%f",
    )
    names = ["num", "r", "z", "V_p", "T_e", "n_e"]
    with open(save_path, "r+") as file:
        content = file.read()
        file.seek(0, 0)
        file.write(",".join(names) + "\n" + content)


def analyze_and_save():
    # 读取px_index.csv位置坐标
    index_dir = "D:/001_zerlingx/archive/for_notes/HC/07_experiments/2024-03 一号阴极测试/2024-04-14 羽流诊断与色散关系测试/analysis/"
    index_path = "p3_index.csv"
    Num, R, Z = read_coordinates(index_dir + index_path)
    # 读取px的SLP诊断数据并计算
    dir = "D:/001_zerlingx/archive/for_notes/HC/07_experiments/2024-03 一号阴极测试/2024-04-14 羽流诊断与色散关系测试/data/RAW/"
    paths = []
    for i in Num:
        paths.append("tek" + str(i).zfill(4) + "ALL.csv")
    V_ps = []
    T_es = []
    n_es = []
    num = []
    r = []
    z = []
    for i in range(len(paths)):
        single_path = paths[i]
        data_obj = data.data(dir + single_path)
        data_points = data_obj.read()
        single_SLP_point = SLP.SLP()
        V_p, T_e, n_e = single_SLP_point.cal(data_points)
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
        print("-----With data : ", single_path, "-----", end=" ")
        print("V_p=%.2f, T_e=%.3f, n_e=%.3e" % (V_p, T_e, n_e))
    # 保存坐标和诊断结果
    save_index_and_SLP_data(
        save_path="res/SLP_spacial/p3_SLP_spacial.csv",
        num=num,
        r=r,
        z=z,
        V_ps=V_ps,
        T_es=T_es,
        n_es=n_es,
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
    fig, ax = plt.subplots(1, 3, figsize=(16, 3))
    plt.subplots_adjust(
        # top=0.899,
        # bottom=0.078,
        left=0.03,
        right=1,
        # hspace=0.2,
        wspace=0.05,
    )
    grid_x, grid_y, grid_z = get_plot_grid(r, z, V_ps)
    im = ax[0].contourf(grid_x, grid_y, grid_z, levels=50, cmap="jet")
    ax[0].set_title("V_p")
    ax[0].set_xlim([-8, 19.5])
    ax[0].set_ylim([-259, -242])
    ax[0].set_aspect("equal")
    fig.colorbar(im, ax=ax[0])
    grid_x, grid_y, grid_z = get_plot_grid(r, z, T_es)
    im = ax[1].contourf(grid_x, grid_y, grid_z, levels=50, cmap="jet")
    ax[1].set_title("T_e")
    ax[1].set_xlim([-8, 19.5])
    ax[1].set_ylim([-259, -242])
    ax[1].set_aspect("equal")
    fig.colorbar(im, ax=ax[1])
    grid_x, grid_y, grid_z = get_plot_grid(r, z, n_es)
    im = ax[2].contourf(grid_x, grid_y, grid_z, levels=50, cmap="jet")
    ax[2].set_title("n_e")
    ax[2].set_xlim([-8, 19.5])
    ax[2].set_ylim([-259, -242])
    ax[2].set_aspect("equal")
    fig.colorbar(im, ax=ax[2])
    plt.savefig(result_path + ".jpg")
    plt.show()


def get_SLP_along_z_or_r(result_path):
    with open(result_path, "r") as file:
        data = pd.read_csv(file)
        num = data.loc[:, "num"]
        r = data.loc[:, "r"]
        z = data.loc[:, "z"]
        V_ps = data.loc[:, "V_p"]
        T_es = data.loc[:, "T_e"]
        n_es = data.loc[:, "n_e"]
    index = r == min(r)
    res_z = z[index]
    z_n_es = n_es[index]
    index = z > 19
    res_r = r[index]
    r_n_es = n_es[index]
    return res_z, z_n_es, res_r, r_n_es


def plot_SLP_along_z_or_r():
    result_path = "res/SLP_spacial/p1_SLP_spacial.csv"
    p1_z, p1_z_n_es, p1_r, p1_r_n_es = get_SLP_along_z_or_r(result_path)
    result_path = "res/SLP_spacial/p2_SLP_spacial.csv"
    p2_z, p2_z_n_es, p2_r, p2_r_n_es = get_SLP_along_z_or_r(result_path)
    result_path = "res/SLP_spacial/p3_SLP_spacial.csv"
    p3_z, p3_z_n_es, p3_r, p3_r_n_es = get_SLP_along_z_or_r(result_path)
    # 绘图
    fig, ax = plt.subplots(1, 2, figsize=(16, 5))
    ax[0].plot(25 - p1_z, p1_z_n_es / max(p1_z_n_es) * 100, label="p1", marker="s")
    ax[0].plot(25 - p2_z, p2_z_n_es / max(p2_z_n_es) * 100, label="p2", marker="s")
    ax[0].plot(25 - p3_z, p3_z_n_es / max(p3_z_n_es) * 100, label="p3", marker="s")
    ax[0].set_title("n_e along z")
    ax[0].set_xlabel("z (mm)")
    ax[0].set_ylabel("n_e (%)")
    ax[0].legend()
    ax[0].grid()
    ax[1].plot(260 + p1_r, p1_r_n_es / max(p1_r_n_es) * 100, label="p1", marker="s")
    ax[1].plot(260 + p2_r, p2_r_n_es / max(p2_r_n_es) * 100, label="p2", marker="s")
    ax[1].plot(260 + p3_r, p3_r_n_es / max(p3_r_n_es) * 100, label="p3", marker="s")
    ax[1].set_title("n_e along r")
    ax[1].set_xlabel("r (mm)")
    ax[1].set_ylabel("n_e (%)")
    ax[1].legend()
    ax[1].grid()
    plt.savefig("res/SLP_spacial/SLP_spacial_along_z_or_r.jpg")
    plt.show()


if __name__ == "__main__":
    # analyze_and_save()
    # plot_SLP_spacial()
    plot_SLP_along_z_or_r()
