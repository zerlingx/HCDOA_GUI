import numpy as np
import sys

sys.path.append("./")
import constant.plasma_parameters as pp


class HollowCathode:
    def __init__(self):
        # 阴极常用参考量
        self.ref_parameters = {
            # 出口稍远处（10 mm）参数，稍稀薄的等离子体才能使用探针测量
            "T_E_REF_HC": 2,  # 电子温度            eV
            "N_E_REF_HC": 6e14,  # 电子数密度       m^-3
            "N_I_REF_HC": 6e14,  # 离子数密度       m^-3
        }

        # 计算参考徳拜长度和等离子体频率（离子）
        plasma = pp.Plasma()
        pc = plasma.constants
        eps0 = pc["EPSILON_0"]
        kb = pc["K_BOLTZMANN"]
        h = pc["H_PLANCK"]
        e = pc["E_ELEC"]
        mi = pc["M_I_XE"]
        Te = self.ref_parameters["T_E_REF_HC"]
        Ne = self.ref_parameters["N_E_REF_HC"]
        Ni = self.ref_parameters["N_I_REF_HC"]
        # LAMBDA_D_REF_HC = 4.293e-4        # 徳拜长度          m
        self.ref_parameters["LAMBDA_D_REF_HC"] = np.sqrt(
            eps0 * kb * Te * 11605 / (Ne * e**2)
        )
        # OMEGA_P_REF_HC = 2.824e6          # 离子等离子体频率   Hz
        self.ref_parameters["OMEGA_P_REF_HC"] = np.sqrt(Ni * e**2 / (eps0 * mi))

    def print_all(self):
        for key, value in self.ref_parameters.items():
            print(key, value)

    def get_para(self):
        # 返回常量
        return self.ref_parameters
