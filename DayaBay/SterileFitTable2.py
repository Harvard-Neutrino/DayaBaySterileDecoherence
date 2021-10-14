import FitClass as FC
import numpy as np
import time


datangl1 = np.logspace(-3,np.log10(6e-2),20)
datmass1 = np.logspace(np.log10(0.00303558),-2,16)

dir = 'PlotData/'

begin = time.time()
fit = FC.SterileFit(wave_packet = True)
fit.write_data_table(datmass1,datangl1,dir+'WPSterileChi2_2.dat')
end = time.time()
print('Time = '+str(end-begin)[:6]+' s.')
