import FitClass as FC
import numpy as np
import time

# This is an example program of how to obtain points of the chi2
# Here we use the variables and functions from FitClass.py

# Tune these arrays to the interval of parameters you wish to study
masses = np.logspace(np.log10(0.08),np.log10(1), 3)

# datmass1 = np.logspace(np.log10(0.15793),np.log10(1.65),10)#80/90
i = 0
datmass1 = np.logspace(np.log10(masses[i]),np.log10(masses[i+1]*0.99), 30)
datangl1 = np.logspace(-2,-0.001,40)#90/80

# The data is saved inside PlotData
dir = 'PlotData/'

begin = time.time()
fit = FC.SterileGlobalFit(wave_packet = True) # Here we choose WP formalism
fit.write_data_table(datmass1,datangl1,dir+'WPSterileChi2_5.0e-4_'+str(i)+'.dat', sigma = 5e-4)
end = time.time()
print('Time = '+str(end-begin)[:6]+' s.')
