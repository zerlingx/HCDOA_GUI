import numpy as np
import sys
import matplotlib.pyplot as plt

sys.path.append("./base")
sys.path.append("./constant")
sys.path.append("./script")
import plasma_parameters as pp

k = pp.Plasma().constants["K_BOLTZMANN"]
e = pp.Plasma().constants["E_ELEC"]
sigma = pp.Plasma().constants["SIGMA"]
A_EFF = pp.Plasma().constants["A_EFF"]

# 钨丝
T = 2900  # K
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
S_exposed = S * 0.25
P = sigma * S_exposed * T**4
print("热辐射功率=", P)
P = 7.5  # W
# IV=P=sigma*S*e*T**4
T_cal = (P / (sigma * S_exposed)) ** (1 / 4)
print("计算理论热平衡温度=", T_cal)

# LaB6发射体热发射
T = 1300
d1 = 5e-3
d2 = 3e-3
l = 15e-3
S = np.pi * (d1**2 - d2**2) / 4 * l
W = 2.7
J = A_EFF * T**2 * np.exp(-W * e / (k * T))
print("LaB6电流密度=", J)
print("LaB6电流=", J * S)

T = range(1000, 2000, 10)
J = A_EFF * np.array(T) ** 2 * np.exp(-W * e / (k * np.array(T)))
plt.plot(T, J * S)
plt.title("LaB6 thermionic emission - Temperature")
plt.xlabel("Temperature (K)")
plt.ylabel("Emission (A)")
plt.grid()
plt.show()
