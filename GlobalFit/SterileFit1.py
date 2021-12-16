import SterileFit as SF
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

datmass1 = np.logspace(-2,np.log10(0.149),160)
datangl1 = np.logspace(-3,0,160)



begin = time.time()
fit = SF.SterileGlobalFit(wave_packet = False)
fit.write_data_table(datmass1,datangl1,'PWSterileChi2_1.dat')
end = time.time()
print('Time = '+str(end-begin)[:6]+' s.')
