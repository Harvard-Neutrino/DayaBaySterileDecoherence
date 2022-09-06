import FitClass as FC
import numpy as np
import time

# This is an example program of how to obtain points of the chi2
# Here we use the variables and functions from FitClass.py

# Tune these arrays to the interval of parameters you wish to study
# masses = np.logspace(0,1,3)

# i = 0
# datangl1 = np.logspace(np.log10(4e-3),0,30)
# datmass1 = np.logspace(np.log10(masses[i]),np.log10(masses[i+1]*9.9/10),25)

datmass1 = np.logspace(np.log10(1.6),1,100)
datangl1 = np.logspace(np.log10(0.003),np.log10(0.8),100)

# The data is saved inside PlotData
dir = 'PlotData/'

begin = time.time()
fit = FC.SterileFit(wave_packet = True)  # Here we choose WP formalism
fit.write_data_table(datmass1,datangl1,dir+'WPSterileChi2_2.1e-3_2.dat', sigma = 2.1e-3)
end = time.time()
print('Time = '+str(end-begin)[:6]+' s.')
