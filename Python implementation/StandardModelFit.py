import DayaBay as DB
import Models

import numpy as np

dayabay = DB.DayaBay()

datmass = np.arange(2.0,2.6,0.1)*1e-3
datangl = np.arange(0.05,0.11,0.01)

def getChi2(mass,angl):
    model = Models.PlaneWaveSM(Sin22Th13 = angl, DM2_31 = mass)
    chi2 = dayabay.get_poisson_chi2(model)
    return chi2

file = open('SMChi2.dat','w')
for m in datmass:
    for a in datangl:
        file.write('{0:1.5f},{1:1.5f},{2:7.4f}\n'.format(m,a,getChi2(m,a)))

file.close()
