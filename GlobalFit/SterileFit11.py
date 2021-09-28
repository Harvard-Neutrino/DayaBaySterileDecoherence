import SterileFit as SF
import numpy as np
import time

datmass = np.logspace(-2,-0.5-0.078947,20)
datangl = np.logspace(-3,-1.5-0.078947,20)
print(datmass,datangl)

begin = time.time()
fit = SF.SterileGlobalFit()
fit.write_data_table(datmass,datangl,'PWSterileChi2_11.dat')
end = time.time()
print('Time = '+str(end-begin)[:6]+' s.')
