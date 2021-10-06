import SterileFit as SF
import numpy as np
import time

datmass1 = np.logspace(np.log10(0.01),np.log10(0.25),15)
datmass2 = np.logspace(np.log10(0.28),np.log10(0.5),9)
datmass3 = np.logspace(np.log10(0.53),np.log10(0.85),9)
datmass4 = np.logspace(np.log10(0.88),np.log10(1.20),9)
datangl1 = np.logspace(-3,0,30)

datmass5 = np.logspace(np.log10(1.23),np.log10(1.55),9)
datmass6 = np.logspace(np.log10(1.58),np.log10(1.90),9)
datmass7 = np.logspace(np.log10(1.93),np.log10(2.2),12)
datangl2 = np.logspace(np.log10(0.002),np.log10(0.2),30)

datmass8 = np.logspace(np.log10(2.3),np.log10(3.),20)
datmass9 = np.logspace(np.log10(4.6),np.log10(7.),25)
datangl3 = np.logspace(-2,0,90)


begin = time.time()
fit = SF.SterileGlobalFit(wave_packet = True)
fit.write_data_table(datmass1,datangl1,'WPSterileChi2_1.dat')
end = time.time()
print('Time = '+str(end-begin)[:6]+' s.')
