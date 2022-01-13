import FitClass as FC
import numpy as np
import time

# This is an example program of how to obtain points of the chi2
# Here we use the variables and functions from FitClass.py

# Tune these arrays to the interval of parameters you wish to study
datangl1 = np.linspace(0,1,150)
datmass1 = np.linspace(0,10,150)

# The data is saved inside PlotData
dir = 'PlotData/'

begin = time.time()
fit = FC.SterileFit(wave_packet = False)
fit.write_data_table(datmass1,datangl1,dir+'PWSterileChi2_1.dat')
end = time.time()
print('Time = '+str(end-begin)[:6]+' s.')
