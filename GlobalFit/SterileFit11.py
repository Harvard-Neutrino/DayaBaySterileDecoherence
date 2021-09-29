import SterileFit as SF
import numpy as np
import time

datmass = np.logspace(np.log10(2.01),np.log10(10),5)
datangl = np.logspace(np.log10(0.008),np.log10(1),5)
print(datmass,datangl)

begin = time.time()
fit = SF.SterileGlobalFit()
fit.write_data_table(datmass,datangl,'PWSterileChi2_22.dat')
end = time.time()
print('Time = '+str(end-begin)[:6]+' s.')
