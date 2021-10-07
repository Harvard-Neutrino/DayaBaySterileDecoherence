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

datmass1 = np.array([1.21      , 1.26186846, 1.31596033])
datmass2 = np.array([1.37237094, 1.43119967, 1.49255018])
datmass3 = np.array([1.55653058, 1.62325359, 1.69283678])
datmass4 = np.array([1.76540275, 1.84107938, 1.92      ])
datangl1 = np.logspace(np.log10(0.01),np.log10(0.7),30)

datmass5 = np.array([2.81      , 2.93818831, 3.07222438])
datmass6 = np.array([3.212375  , 3.3589191 , 3.51214833])
datmass7 = np.array([3.67236767, 3.839896  , 4.01506673])
datmass8 = np.array([4.19822851, 4.38974588, 4.59      ])
datangl2 = np.logspace(np.log10(0.02),0,30)


begin = time.time()
fit = SF.SterileGlobalFit(wave_packet = True)
fit.write_data_table(datmass1,datangl1,'WPSterileChi2_1.dat')
end = time.time()
print('Time = '+str(end-begin)[:6]+' s.')
