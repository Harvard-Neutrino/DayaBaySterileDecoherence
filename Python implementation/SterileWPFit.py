import DayaBay as DB
import Models

import numpy as np

dayabay = DB.DayaBay()

datmass = np.logspace(np.log10(1e-4),np.log10(0.25),40)
print(datmass)
datangl = np.logspace(np.log10(1e-3),np.log10(1.),40)

def getChi2(mass = 2.5e-3,angl = 0.0841):
    model = Models.WavePacketSterile(Sin22Th14 = angl, DM2_41 = mass)
    chi2 = dayabay.get_poisson_chi2(model)
    print(mass,angl,chi2)
    return chi2

# print("Referència best-fit DB:", 0.0841, 2.5e-3,getChi2())

file = open('SMWPSterileChi2.dat','w')
for m in datmass:
    for a in datangl:
        file.write('{0:1.5f},{1:1.5f},{2:7.4f}\n'.format(m,a,getChi2(m,a)))

# HA QUEDAT ESCRIU A L'ARXIU!!!!!!!!

file.close()
