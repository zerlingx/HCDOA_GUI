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
import basic_plot

st.markdown(
    """
    # 01 阴极测试数据分析界面
    
    >2023-09-04, Version 0.1
    >
    >此界面用于阴极测试数据的分析，包括数据的读取、归一化、绘图、FFT等。
    """
)
st.sidebar.markdown("# 01 阴极放电数据分析界面")

st.markdown(
    """
    ## 01-1 选择数据
    """
)
st.sidebar.markdown("## 01-1 选择数据")

# 选择数据文件夹
if "DIR" not in st.session_state:
    st.session_state["DIR"] = (
        "D:/001_zerlingx/archive/for_notes/HC/07_experiments/2024-03 一号阴极测试/2024-05-12 羽流诊断与色散关系测试/data/RAW/"
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
def plot_data_points(
    plot_data,
    plot_range,
    fre_range,
    plot_channels=[4],
    FFT_channel=4,
    save_FFT_csv=False,
):
    fig, ax = basic_plot.plot_curve_and_FFT(
        data_points=plot_data,
        plot_range=plot_range,
        title=path,
        fre_range=fre_range,
        plot_channels=plot_channels,
        FFT_channel=FFT_channel,
        save_FFT_csv=save_FFT_csv,
    )
    return fig


# 选择读取数据范围
col1, col2, col3, col4 = st.columns(4)
with col1:
    read_start = st.number_input("读取数据起始点", value=0, format="%d")
with col2:
    read_end = st.number_input(
        "读取数据结束点", value=10000000, format="%d", help="根据示波器或计算需要选取。"
    )
read_range = [int(read_start), int(read_end)]
with col3:
    st.markdown("是否归一化", help="此选项将在读取数据时将其归一化，可能增加读取时间。")
    normalize = st.checkbox("数据归一化", value=False)
with col4:
    st.markdown("是否保存FFT", help="保存到'res/FFT/title + str(FFT_channel)'。")
    save_FFT_csv = st.checkbox("保存FFT数据", value=False)

col1, col2, col3 = st.columns(3)
with col1:
    plot_channels = st.multiselect("选择绘图通道", [1, 2, 3, 4], default=[2])
with col2:
    FFT_channel = st.selectbox("选择FFT通道", [1, 2, 3, 4])


# 选择绘图使用数据范围
col1, col2 = st.columns(2)
with col1:
    plot_range = st.slider(
        "绘图使用数据占比(%)",
        min_value=1,
        max_value=100,
        step=1,
        value=10,
        help="从读取数据时间中点向两侧延申。",
    )
with col2:
    plot_amp = st.slider(
        "波形放大倍率",
        min_value=1,
        max_value=100,
        value=1,
    )
plot_range = plot_range / 100 / plot_amp
# 选择绘图程序中FFT处理频率范围
log_axis = []
log_axis.append(0)
for i in np.arange(-1, 0, 0.1):
    log_axis.append(round(pow(10, i)))
for i in np.arange(0, 8.1, 0.1):
    log_axis.append(round(pow(10, i)))
fre_range = st.select_slider(
    "FFT处理频率范围",
    options=log_axis,
    value=[1e1, 1e6],
)

# 加载输入并绘图
try:
    data_points = load_data_points(dir + path, read_range, normalize)
    fig = plot_data_points(
        data_points,
        plot_range,
        fre_range,
        plot_channels,
        FFT_channel,
        save_FFT_csv,
    )
    st.pyplot(fig)
    # st.image("res/fig.png")
except:
    st.markdown("未加载数据。")

if st.button("清除缓存"):
    # 点击该按钮时清除缓存，使得之后的程序中能重新加载
    st.cache_data.clear()
else:
    pass
