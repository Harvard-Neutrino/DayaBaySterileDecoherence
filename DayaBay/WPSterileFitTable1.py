import FitClass as FC
import numpy as np
import time


datangl1 = np.logspace(np.log10(0.375),-0.02,7)
datmass1 = np.logspace(np.log10(0.26),np.log10(0.6),6)

dir = 'PlotData/'

begin = time.time()
fit = FC.SterileFit(wave_packet = True)
fit.write_data_table(datmass1,datangl1,dir+'WPSterileChi2_1.dat')
end = time.time()
print('Time = '+str(end-begin)[:6]+' s.')