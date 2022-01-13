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

dm_BEST = 10.
angl = 0.4
L_BEST_in = 0.5203
L_BEST_out = 0.5441
E_BEST_1 = 0.747
E_BEST_2 = 0.427
BEST_PW = Models.PlaneWaveSterile(DM2_41 = dm_BEST, Sin22Th14 = angl)
BEST_WP = Models.WavePacketSterile(DM2_41 = dm_BEST, Sin22Th14 = angl)
axis = [0.4,0.75,0.85,1]
axis = [0.3,0.7,0.85,1]
datx = np.arange(axis[0],axis[1],0.001)

# prob_BEST_PW = [BEST_PW.oscProbability(x,L_BEST_in)/BEST_PW.oscProbability(x,L_BEST_out) for x in datx]
# prob_BEST_WP = [BEST_WP.oscProbability(x,L_BEST_in)/BEST_WP.oscProbability(x,L_BEST_out) for x in datx]
# prob_BEST_PW = [BEST_PW.oscProbability(x,L_BEST_in) for x in datx]
# prob_BEST_WP = [BEST_WP.oscProbability(x,L_BEST_in) for x in datx]
prob_BEST_PW = [BEST_PW.oscProbability(E_BEST_1,x) for x in datx]
prob_BEST_WP = [BEST_WP.oscProbability(E_BEST_1,x) for x in datx]

margins = dict(left=0.14, right=0.96,bottom=0.19, top=0.9)
plot,axx =plt.subplots(figsize = (7,6),gridspec_kw=margins)

print(np.sqrt(L_NEOS*dm_NEOS/(4*np.sqrt(2)*2.1e-13))*1e-6)

# axx.plot(datx,prob_DB_PW)
# axx.plot(datx,prob_DB_WP)
axx.grid(linestyle = '--')
axx.tick_params(axis='x')
axx.tick_params(axis='y')
axx.set_xlim([axis[0],axis[1]])
# axx.set_ylim([0.59,1.])
axx.set_ylabel(r"$P_{\text{in}}/P_{\text{out}}$", fontsize = 24)
axx.set_ylabel(r"$P_{ee}$", fontsize = 24)
axx.set_xlabel(r"$E (\text{MeV})$", fontsize = 24)
axx.set_xlabel(r"$L (\text{m})$", fontsize = 24)
axx.plot(datx,prob_BEST_PW, label = 'Plane wave', color = color2)
axx.plot(datx,prob_BEST_WP, label = 'Wave package', color = color3)
# axx.annotate(r'$E = E^{\text{coh}}$',(8.3,0.97), size = 20, color = 'k')
# axx.vlines(np.sqrt(L_NEOS*dm_NEOS/(4*np.sqrt(2)*2.1e-13))*1e-6, ymin = 0.6, ymax = 1., linestyle = '--', color = 'gray')
axx.vlines(0.52, ymin = 0.6,ymax = 1.,linestyle = '--', color = 'gray')
axx.vlines(0.544, ymin = 0.6, ymax = 1.,linestyle = '--', color = 'gray')
axx.legend(loc="lower right", fontsize = 20)
plot.suptitle(r'$E = 747\text{ keV},\ \Delta m^2 = 10 \text{ eV}^2,\ \sin^2 2\theta_{14} = 0.4$')
plot.savefig('Probability_BEST_10.pdf')
