import FitClass as FC
import numpy as np
import time
import os

homedir = os.path.realpath(__file__)[:-len('PROSPECT/WPSterileFitTable1.py')]
# The data is saved inside PROSPECT/PlotData
datadir = homedir + 'PROSPECT/PlotData/'

# This is an example program of how to obtain points of the chi2
# Here we use the variables and functions from FitClass.py

# Tune these arrays to the interval of parameters you wish to study
datmass1 = np.logspace(np.log10(0.08),np.log10(1.99),100)
datangl1 = np.logspace(-2,-0.001,100)#90/80

begin = time.time()
fit = FC.SterileFit(wave_packet = True)  # Here we choose WP formalism
fit.write_data_table(datmass1,datangl1,datadir+'WPSterileChi2_x.dat', sigma = 5.0e-4)
end = time.time()
print('Time = '+str(end-begin)[:6]+' s.')
