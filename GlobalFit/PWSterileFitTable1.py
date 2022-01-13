import FitClass as FC
import numpy as np
import time

# This is an example program of how to obtain points of the chi2
# Here we use the variables and functions from FitClass.py

# Tune these arrays to the interval of parameters you wish to study
datmass1 = np.logspace(-2,np.log10(0.149),160)
datangl1 = np.logspace(-3,0,160)

# The data is saved inside PlotData
dir = 'PlotData/'

begin = time.time()
fit = FC.SterileGlobalFit(wave_packet = False) # Here we choose PW formalism
fit.write_data_table(datmass1,datangl1,dir+'PWSterileChi2_1.dat')
end = time.time()
print('Time = '+str(end-begin)[:6]+' s.')
