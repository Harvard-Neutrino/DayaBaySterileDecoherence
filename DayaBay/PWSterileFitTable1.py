import FitClass as FC
import numpy as np
import time


datmass1 = np.logspace(-4,np.log10(0.1499),150)
datangl1 = np.logspace(-3,np.log10(0.8),150)


dir = 'PlotData/'

begin = time.time()
fit = FC.SterileFit(wave_packet = False)
fit.write_data_table(datmass1,datangl1,dir+'PWSterileChi2_1.dat')
end = time.time()
print('Time = '+str(end-begin)[:6]+' s.')
