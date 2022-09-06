import FitClass as FC
import numpy as np
import time

# This is an example program of how to obtain points of the chi2
# Here we use the variables and functions from FitClass.py

# Tune these arrays to the interval of parameters you wish to study
datmass1 = np.logspace(np.log10(0.08),np.log10(1.65),90)#80/90
datangl1 = np.logspace(-2,-1,80)#90/80
# Trigarà 24h. presumiblement.

# The data is saved inside PlotData
dir = 'PlotData/'

begin = time.time()
fit = FC.SterileGlobalFit(wave_packet = True) # Here we choose WP formalism
fit.write_data_table(datmass1,datangl1,dir+'WPSterileChi2_2.1e-3_1.dat', sigma = 2.1e-3)
end = time.time()
print('Time = '+str(end-begin)[:6]+' s.')
