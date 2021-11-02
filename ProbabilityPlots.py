import sys
import os
common_dir = '/Common_cython'
main_dir = os.getcwd()
sys.path.append(main_dir+common_dir)
sys.path.append(main_dir+"/DayaBay")
sys.path.append(main_dir+"/NEOS")

import matplotlib.pyplot as plt
import numpy as np
import matplotlib

import Models

path_to_style = main_dir + common_dir
dir = 'PlotData/'
plt.style.use(path_to_style+r"/paper.mplstyle")
matplotlib.rcParams.update({'text.usetex': True})
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

C = 80
dm_DB = 0.25
dm_NEOS = C/24
print(dm_NEOS)
angl = 0.4
SM_PW = Models.PlaneWaveSM()
SM_WP = Models.WavePacketSM()
DB_PW = Models.PlaneWaveSterile(DM2_41 = dm_DB, Sin22Th14 = angl)
DB_WP = Models.WavePacketSterile(DM2_41 = dm_DB, Sin22Th14 = angl)
NEOS_PW = Models.PlaneWaveSterile(DM2_41 = dm_NEOS, Sin22Th14 = angl)
NEOS_WP = Models.WavePacketSterile(DM2_41 = dm_NEOS, Sin22Th14 = angl)

L_DB = C/0.25
print(L_DB)
L_NEOS = 24
print(L_NEOS*dm_NEOS)

axis = [1.3,12.,0.85,1]
datx = np.arange(axis[0],axis[1],0.001)

prob_DB_PW =  [DB_PW.oscProbability(x,L_DB)/SM_PW.oscProbability(x,L_DB) for x in datx]
prob_DB_WP =  [DB_WP.oscProbability(x,L_DB)/SM_WP.oscProbability(x,L_DB) for x in datx]
prob_NEOS_PW =  [NEOS_PW.oscProbability(x,L_NEOS)/SM_PW.oscProbability(x,L_NEOS) for x in datx]
prob_NEOS_WP =  [NEOS_WP.oscProbability(x,L_NEOS)/SM_WP.oscProbability(x,L_NEOS) for x in datx]

margins = dict(left=0.14, right=0.96,bottom=0.15, top=0.9)
plot,axx =plt.subplots(figsize = (7,5),gridspec_kw=margins)

# axx.plot(datx,prob_DB_PW)
# axx.plot(datx,prob_DB_WP)
axx.grid(linestyle = '--')
axx.tick_params(axis='x')
axx.tick_params(axis='y')
axx.set_ylabel(r"$P_{3+1}/P_3$")
axx.set_xlabel(r"$E (\text{MeV})$")
axx.plot(datx,prob_NEOS_PW, label = 'Plane wave')
axx.plot(datx,prob_NEOS_WP, label = 'Wave packet')
axx.legend()
plot.suptitle(r'$L\cdot \Delta m^2_{41} = 80 \text{ m}\cdot\text{eV}^2,\ \sin^2 2\theta_{14} = 0.4$')
plot.savefig('Probability.pdf')
