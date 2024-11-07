# 空心阴极放电振荡分析简易图形化处理软件

>Hollow Cathode Discharge Oscillation Analysis GUI

## 01 简介

阴极振荡数据处理软件，主要功能有：

- 示波器csv放电数据读取；
- 数据曲线绘制；
- FFT处理与频谱绘制。

## 02 安装

`git clone`到本地，确保已安装python（原环境采用版本为3.8.10），并参考`requirements.txt`安装必要依赖包。

## 03 运行

使用GUI框架为`streamlit`，即一种类似web的框架，因此您的平台应具有必要的浏览器。

要运行程序，在终端根目录下使用命令`streamlit run app/st_app.py`，运行后终端将显示端口和程序内输出信息，GUI程序将自动在浏览器打开。

## 结果示例

`script/basic_plot.py`绘制图像。

![basic_plot](/res/fig.png)

图形化界面演示。

![res](/res/GUI_example.png)
