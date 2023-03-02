import sys
import os
common_dir = '/Common_cython'
sys.path.append(os.getcwd()[:-8]+common_dir)

import DayaBay as DB
import Models

import numpy as np
dir = 'PlotData/'

dayabay = DB.DayaBay()

datmass = np.linspace(2.2,2.9,40)*1e-3
datangl = np.linspace(0.02,0.13,40)

def getChi2(mass = 2.5e-3,angl = 0.0841):
    model = Models.WavePacketSM(Sin22Th13 = angl, DM2_31 = mass)
    chi2 = dayabay.get_poisson_chi2(model)
    print(mass,angl,chi2)
    return chi2

print("Official best-fit from DB:", 0.0841, 2.5e-3,getChi2())

file = open(dir+'SMWPChi2.dat','w')
for m in datmass:
    for a in datangl:
        file.write('{0:1.5f},{1:1.5f},{2:7.4f}\n'.format(m,a,getChi2(m,a)))


file.close()
