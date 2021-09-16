import GlobalFit as GF
import Models
import time

import numpy as np

fitter = GF.GlobalFit()


datmass = np.logspace(-2,1,3)
datangl = np.logspace(-3,0,3)
print(np.log10(datmass),np.log10(datangl))

def getChi2(mass = 2.5e-3,angl = 0.0841):
    model = Models.PlaneWaveSterile(Sin22Th14 = angl, DM2_41 = mass)
    chi2 = fitter.get_poisson_chi2(model)
    print(mass,angl,chi2)
    return chi2

# print("Referència best-fit DB:", 0.0841, 2.5e-3,getChi2())
begin = time.time()
file = open('SMSterileChi2.dat','w')
for m in datmass:
    for a in datangl:
        file.write('{0:1.5f},{1:1.5f},{2:7.4f}\n'.format(m,a,getChi2(m,a)))
end = time.time()
print(end-begin)
# HA QUEDAT ESCRIT A L'ARXIU!!!!!!!!

file.close()
