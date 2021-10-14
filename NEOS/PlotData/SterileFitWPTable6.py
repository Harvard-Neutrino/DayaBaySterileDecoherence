import FitClass as FC
import numpy as np
import time

# first gap: 2.8 < m < 4.6, 0.02 < t < 1.
# first gap masses: [2.81      , 2.93818831, 3.07222438, 3.212375  , 3.3589191 ,
       #             3.51214833, 3.67236767, 3.839896  , 4.01506673, 4.19822851,
       #             4.38974588, 4.59      ]
# second gap: 1.2 < m < 1.93, 0.01 < t < 0.7
# second gap masses: [1.21      , 1.26186846, 1.31596033, 1.37237094, 1.43119967,
       #              1.49255018, 1.55653058, 1.62325359, 1.69283678, 1.76540275,
       #              1.84107938, 1.92      ]

masses = np.logspace(0,1,9)

i = 4
datangl1 = np.logspace(np.log10(4e-3),0,50)
datmass1 = np.logspace(np.log10(masses[i]),np.log10(masses[i+1]*9/10),10)



begin = time.time()
fit = FC.SterileFit(wave_packet = True, use_HM = False)
fit.write_data_table(datmass1,datangl1,'WPSterileChi2_6.dat')
end = time.time()
print('Time = '+str(end-begin)[:6]+' s.')
