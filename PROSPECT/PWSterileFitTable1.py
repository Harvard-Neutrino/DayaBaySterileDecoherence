import FitClass as FC
import numpy as np
import time
import os

homedir = os.path.realpath(__file__)[:-len('PROSPECT/PWSterileFitTable1.py')]
# The data is saved inside PROSPECT/PlotData
datadir = homedir + 'PROSPECT/PlotData/'

# This is an example program of how to obtain points of the chi2
# Here we use the variables and functions from FitClass.py

# Tune these arrays to the interval of parameters you wish to study
datangl1 = np.logspace(np.log10(4e-3),0,150)
datmass1 = np.logspace(np.log10(0.08),0,160)

begin = time.time()
fit = FC.SterileFit(wave_packet = False)  # Here we choose PW formalism
fit.write_data_table(datmass1,datangl1,datadir+'PWSterileChi2_x.dat')
end = time.time()
print('Time = '+str(end-begin)[:6]+' s.')
