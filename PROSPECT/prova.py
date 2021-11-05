import sys
import os
sys.path.append(os.getcwd()[:-5]+"/Common")

import numpy as np
import PROSPECT as PS
import Models
from scipy import integrate as integrate
import matplotlib.pyplot as plt
import time

fitter = PS.Prospect()
osc = Models.PlaneWaveSM()
noosc = Models.NoOscillations()

# print(fitter.calculate_naked_event_expectation_simple(osc,15,14))
# print(np.sum(fitter.get_expectation_unnorm_nobkg(osc).reshape(10,16),axis=1))
print(fitter.get_expectation(osc))
