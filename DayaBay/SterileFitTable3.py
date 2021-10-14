import FitClass as FC
import numpy as np
import time


datangl1 = np.logspace(-3,0,40)
datmass1 = np.logspace(np.log10(0.03314872),np.log10(0.12),30)

dir = 'PlotData/'

begin = time.time()
fit = FC.SterileFit(wave_packet = True)
fit.write_data_table(datmass1,datangl1,dir+'WPSterileChi2_3.dat')
end = time.time()
print('Time = '+str(end-begin)[:6]+' s.')
