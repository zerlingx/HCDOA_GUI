import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import rfft, rfftfreq
from sklearn import preprocessing
import streamlit as st
import pandas as pd
import os

config = {
    "font.family": "serif",
    "font.size": 20,
    "mathtext.fontset": "stix",
    # "font.serif": ["SimSun"],
    "font.serif": ["Times New Roman"],
    "axes.unicode_minus": False,
}
plt.rcParams.update(config)

import sys

sys.path.append("./script")
import data
import SLP_plot

st.markdown(
    """
    # 02 诊断探针数据分析界面
    
    这一页进行诊断探针数据的分析。
    """
)
st.sidebar.markdown("# 02 诊断探针数据分析界面")

st.markdown(
    """
    ## 02-1 选择数据
    """
)
st.sidebar.markdown("## 02-1 选择数据")

# 选择数据文件夹
if "DIR" not in st.session_state:
    st.session_state["DIR"] = (
        "D:/001_zerlingx/archive/for_notes/HC/07_experiments/2024-03 一号阴极测试/2024-03-29 羽流诊断与色散关系测试/data/RAW/"
    )
dir = st.session_state["DIR"]
dir = st.text_input("输入数据文件夹路径", dir, help="路径建议使用正斜杠。")
# dir = eval(repr(dir).replace("\\", "/"))
dir = dir.replace("\\", "/")
if dir[len(dir) - 1] != "/":
    dir += "/"
st.session_state["DIR"] = dir

st.markdown("数据文件夹路径：" + dir)
try:
    data_pathes = os.listdir(dir)
except:
    st.markdown("无法读取该路径下文件！")

# 选择数据文件
if "PATH" not in st.session_state:
    st.session_state["PATH"] = "tek0000ALL.csv"
path = st.session_state["PATH"]
try:
    path = st.selectbox(
        label="选择分析数据",
        options=data_pathes,
    )
    st.markdown("当前选择数据文件：" + path)
except:
    st.markdown("尝试读取默认文件名：" + path)


# 数据缓存函数，如果不是第一次执行这个函数，就不会重新执行，而是直接读取缓存的返回值
@st.cache_data()
def load_data_points(dir_path, read_range=[], normalize=False):
    data_obj = data.data(dir_path)
    data_obj.read_range = read_range
    data_points = data_obj.read()
    if normalize:
        data_points = data_obj.normalize()
    return data_points


@st.cache_data()
def plot_data_points(plot_data):
    fig, V_f, T_e, n_e = SLP_plot.SLP_read_and_plot(
        data_points=plot_data,
        title=path,
    )
    return fig, V_f, T_e, n_e


# 加载输入并绘图
try:
    data_points = load_data_points(dir + path)
except:
    st.markdown("未加载数据。")

try:
    fig, V_f, T_e, n_e = plot_data_points(data_points)
    st.pyplot(fig)
except:
    st.markdown("绘图失败。")

if st.button("清除缓存"):
    # 点击该按钮时清除缓存，使得之后的程序中能重新加载
    st.cache_data.clear()
else:
    pass
