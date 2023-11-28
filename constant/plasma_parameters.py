# 等离子体常用常数
class Plasma:
    def __init__(self):
        self.constants = {
            # 定义等离子体常用常数
            "EPSILON_0": 8.859e-12,  # 真空介电常数  F/m
            "K_BOLTZMANN": 1.380649e-23,  # 玻尔兹曼常数  J/K
            "H_PLANCK": 6.62607015e-34,  # 普朗克常数    J·s
            "E_ELEC": 1.6021766208e-19,  # 元电荷        C
            "M_ELECTRON": 9.109e-31,  # 电子质量      kg
            "SIGMA": 5.670374419e-8,  # 斯特藩-玻尔兹曼常数 W/(m^2·K^4)
            "A_EFF": 1.2e6,  # 有效理查森常数 A/m^2·K^2
            # 空心阴极和霍尔推力器研究中的通用常数
            "M_I_XE": 2.18e-25,  # 氙离子质量    kg
        }

    def print_all(self):
        for key, value in self.constants.items():
            print(key, value)

    def get_constants(self):
        # 返回常量
        return self.constants
