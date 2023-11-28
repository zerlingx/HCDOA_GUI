import numpy as np
import sys

sys.path.append("./base")
sys.path.append("./constant")
sys.path.append("./script")
import plasma_parameters as pp

k = pp.Plasma().constants["K_BOLTZMANN"]
e = pp.Plasma().constants["E_ELEC"]
sigma = pp.Plasma().constants["SIGMA"]
A_EFF = pp.Plasma().constants["A_EFF"]

# 钨丝
T = 2400  # K
rho = 5.3e-8
d = 0.12e-3  # m
l = 20e-3
S = np.pi * d * l  # m^2
omega = rho * l / (np.pi / 4 * d**2)  # ohm
print("室温电阻=", omega)
k_T = 0.0045  # 钨丝电阻温度系数
omega = omega * (1 + k_T * (T - 300))  # ohm
print(T, "K高温时电阻=", omega)

# 热发射
W = 4.54  # ev
J = A_EFF * T**2 * np.exp(-W * e / (k * T))
print("电流密度=", J)
print("电流=", J * S)

# 热平衡温度计算
S_expose = S * 0.5
P = sigma * S_expose * T**4
print("热辐射功率=", P)
P = 7.5  # W
# IV=P=sigma*S*e*T**4
T_cal = (P / (sigma * S_expose)) ** (1 / 4)
print("计算理论热平衡温度=", T_cal)
