import numpy as np
import sys

sys.path.append("./base")
sys.path.append("./constant")
sys.path.append("./script")
import plasma_parameters as pp

k = pp.Plasma().constants["K_BOLTZMANN"]
e = pp.Plasma().constants["E_ELEC"]
A = 1.2e6

# 钨丝
rho = 5.3e-8
d = 0.12e-3  # m
l = 3e-3
S = np.pi / 4 * d**2 + np.pi * d * l  # m^2
omega = rho * l / (np.pi / 4 * d**2)  # ohm
print("电阻", omega)

# 热发射
T = 3000  # K
W = 4.54  # ev
J = A * T**2 * np.exp(-W * e / (k * T))
print("电流密度=", J)
print("电流=", J * S)
