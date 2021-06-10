import DayaBay as DB
import Models

import numpy as np

dayabay = DB.DayaBay()

datmass = np.linspace(2.0,3.3,40)*1e-3
datangl = np.linspace(0.01,0.15,40)

def getChi2(mass = 2.4e-3,angl = 0.0841):
    model = Models.PlaneWaveSM(Sin22Th13 = angl, DM2_31 = mass)
    chi2 = dayabay.get_poisson_chi2(model)
    print(mass,angl,chi2)
    return chi2

print("Referència best-fit DB:", 0.0841, 2.4e-3,getChi2())

file = open('SMChi2.dat','w')
for m in datmass:
    for a in datangl:
        file.write('{0:1.5f},{1:1.5f},{2:7.4f}\n'.format(m,a,getChi2(m,a)))

# HA QUEDAT ESCRIU A L'ARXIU!!!!!!!!

file.close()
