import FitClass as FC
import numpy as np
import time
dir = 'PlotData/'

datangl1 = np.linspace(0,1,150)
datmass1 = np.linspace(0,10,150)

# datmass1 = np.array([10.0])
# datangl1 = np.linspace(0.2,0.7,20)

# datmass1 = np.array([1.0])
# datangl1 = np.array([0.4])

# datmass1 = np.array([1])
# datangl1 = np.array([0.0])

begin = time.time()
fit = FC.SterileFit(wave_packet = False)
fit.write_data_table(datmass1,datangl1,dir+'PWSterileChi2_1.dat')
end = time.time()
print('Time = '+str(end-begin)[:6]+' s.')
