import FitClass as FC
import numpy as np
import time


datangl1 = np.array(np.logspace(np.log10(6e-3),np.log10(5e-1),8))
datmass1 = np.array(np.logspace(np.log10(0.17384575),np.log10(0.25),7))

dir = 'PlotData/'

begin = time.time()
fit = FC.SterileFit(wave_packet = True)
fit.write_data_table(datmass1,datangl1,dir+'WPSterileChi2_8.dat')
end = time.time()
print('Time = '+str(end-begin)[:6]+' s.')
