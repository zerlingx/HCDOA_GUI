import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from scipy.stats import gaussian_kde
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


def use_KDE(x, y, values):
    xy = np.vstack([x, y])
    kde = gaussian_kde(xy, weights=values)
    # 在规则网格上评估 KDE
    grid_x, grid_y = np.mgrid[
        np.min(x) : np.max(x) : 100j, np.min(y) : np.max(y) : 100j
    ]  # 100j 表示100x100的网格
    grid = np.vstack([grid_x.ravel(), grid_y.ravel()])
    kde_values = kde(grid).reshape(grid_x.shape)
    # kde_values = kde_values / np.max(kde_values)
    return grid_x, grid_y, kde_values


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
    result_path = "res/SLP_spacial/p2_SLP_spacial.csv"
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


if __name__ == "__main__":
    # analyze_and_save()
    plot_SLP_spacial()
