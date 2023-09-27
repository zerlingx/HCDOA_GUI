# Triple-Langmuir Probe (TLP)
import numpy as np

import sys

sys.path.append("./")
import constant.plasma_parameters as pp
import constant.reference_quantity_hollow_cathode as rqhc

"""
Brief: Simplified TLP, where the floating probe P_2 is set to open circuit.
Refe.: Zhang Z, Zhang Z, Ling W Y L, et al. Time-resolved investigation of the asymmetric plasma plume in a pulsed plasma thruster[J]. Journal of Physics D: Applied Physics, 2020, 53(47): 475201.
Set  : P_2 as floating, P_1 as positive, P_3 as negative.
Calc.: Therefore I_2 = 0, I_1 = I_3 = I.
"""

# electron density
