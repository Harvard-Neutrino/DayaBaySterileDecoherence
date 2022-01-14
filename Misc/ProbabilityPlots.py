import sys
import os
common_dir = '/Common_cython'
main_dir = os.getcwd()[:-5]
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

color1 = '#FFB14E'
color2 = '#EA5F94'
color3 = '#0000FF'


C = 80
dm_DB = 0.25
dm_NEOS = C/24
# print(dm_NEOS)
angl = 0.4
SM_PW = Models.PlaneWaveSM()
SM_WP = Models.WavePacketSM()
DB_PW = Models.PlaneWaveSterile(DM2_41 = dm_DB, Sin22Th14 = angl)
DB_WP = Models.WavePacketSterile(DM2_41 = dm_DB, Sin22Th14 = angl)
NEOS_PW = Models.PlaneWaveSterile(DM2_41 = dm_NEOS, Sin22Th14 = angl)
NEOS_WP = Models.WavePacketSterile(DM2_41 = dm_NEOS, Sin22Th14 = angl)

L_DB = C/0.25
# print(L_DB)
L_NEOS = 24
# print(L_NEOS*dm_NEOS)

axis = [1.3,16.,0.85,1]
datx = np.arange(axis[0],axis[1],0.001)

prob_DB_PW =  [DB_PW.oscProbability(x,L_DB)/SM_PW.oscProbability(x,L_DB) for x in datx]
prob_DB_WP =  [DB_WP.oscProbability(x,L_DB)/SM_WP.oscProbability(x,L_DB) for x in datx]
prob_NEOS_PW =  [NEOS_PW.oscProbability(x,L_NEOS)/SM_PW.oscProbability(x,L_NEOS) for x in datx]
prob_NEOS_WP =  [NEOS_WP.oscProbability(x,L_NEOS)/SM_WP.oscProbability(x,L_NEOS) for x in datx]

margins = dict(left=0.14, right=0.96,bottom=0.19, top=0.9)
plot,axx =plt.subplots(figsize = (7,6),gridspec_kw=margins)

print(np.sqrt(L_NEOS*dm_NEOS/(4*np.sqrt(2)*2.1e-13))*1e-6)

# axx.plot(datx,prob_DB_PW)
# axx.plot(datx,prob_DB_WP)
axx.grid(linestyle = '--')
axx.tick_params(axis='x')
axx.tick_params(axis='y')
axx.set_xlim([axis[0],axis[1]])
axx.set_ylim([0.59,1.])
axx.set_ylabel(r"$P_{3+1}/P_3$", fontsize = 24)
axx.set_xlabel(r"$E (\text{MeV})$", fontsize = 24)
axx.plot(datx,prob_NEOS_PW, label = 'Plane wave', color = color2)
axx.plot(datx,prob_NEOS_WP, label = 'Wave package', color = color3)
axx.annotate(r'$E = E^{\text{coh}}$',(8.3,0.97), size = 20, color = 'k')
axx.vlines(np.sqrt(L_NEOS*dm_NEOS/(4*np.sqrt(2)*2.1e-13))*1e-6, ymin = 0.6, ymax = 1., linestyle = '--', color = 'gray')
axx.legend(bbox_to_anchor=(0,-0.31, 1, 0), loc="lower center", ncol = 2, fontsize = 20)
plot.suptitle(r'$L\cdot \Delta m^2_{41} = 80 \text{ m}\cdot\text{eV}^2,\ \sin^2 2\theta_{14} = 0.4$')
plot.savefig('Probability.pdf')

# -----------------------------------------------------------------

dm = 1
sin2 = 0.1
SM_PW = Models.PlaneWaveSM()
SM_WP = Models.WavePacketSM()
PW = Models.PlaneWaveSterile(DM2_41 = dm, Sin22Th14 = sin2)
WP = Models.WavePacketSterile(DM2_41 = dm, Sin22Th14 = sin2)

E = 2
rat_ax = np.arange(0,7,0.001)
# L = ratio*E
sigmax = 2.1e-13
Lcoh = 4*np.sqrt(2)*sigmax/(dm*1e-12)*E
print(Lcoh)

prob_PW =  [PW.oscProbability(E,x*E) for x in rat_ax]
prob_WP =  [WP.oscProbability(E,x*E) for x in rat_ax]


plot,axx =plt.subplots(figsize = (7,6),gridspec_kw=margins)
axx.plot(rat_ax,prob_PW, label = 'Plane wave', color = color2)
axx.plot(rat_ax,prob_WP, label = 'Wave package', color = color3)

axx.grid(linestyle = '--')
axx.tick_params(axis='x')
axx.tick_params(axis='y')
axx.set_ylabel(r"$P_{3+1}$", fontsize = 24)
axx.set_xlabel(r"$L/E\ (\text{m}/\text{MeV})$", fontsize = 24)
axx.set_xlim([0,7])
axx.set_ylim([0.89,1.01])

axx.legend(bbox_to_anchor=(0,-0.31, 1, 0), loc="lower center", ncol = 2, fontsize = 20)
axx.vlines(Lcoh, ymin = 0.9, ymax = 1., linestyle = '--', color = 'gray')
axx.annotate(r'$L = L^{\text{coh}}$',(Lcoh*1.05,0.99), size = 20, color = 'k')

plot.suptitle(r'$E = %.1f\text{ MeV},\ \Delta m^2_{41} = 1\text{eV}^2,\ \sin^2 2\theta_{14} = 0.1$'%(E))
plot.savefig('Probability_IsoDAR_E%.2f.pdf'%(E))
