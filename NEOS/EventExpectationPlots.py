import time
import NEOS
import Models

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#import pandas as pd

from scipy import interpolate
import os
cwd = os.getcwd()
path_to_style=cwd+'/Figures'
# plt.style.use(path_to_style+r"/paper.mplstyle")


# matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]


NEOS_test = NEOS.Neos()
Model_noosc = Models.NoOscillations()
Model_osc = Models.PlaneWaveSM()
Model_ste = Models.PlaneWaveSterile(DM2_41 = 1.73,Sin22Th14 = 0.05)
Model_ste2 = Models.PlaneWaveSterile(DM2_41 = 2.32,Sin22Th14 = 0.142)
# Model_osc = Models.PlaneWaveSM(Sin22Th13 = 0.07821,DM2_31 = 2.5e-3)
# Model_coh = Models.WavePacketSM()

# -------------------------------------------------------
# Event expectations
# -------------------------------------------------------

x_ax = (NEOS_test.DataLowerBinEdges+NEOS_test.DataUpperBinEdges)/2
deltaE = (NEOS_test.DataUpperBinEdges-NEOS_test.DataLowerBinEdges)

figev,axev = plt.subplots(1,1,figsize = (12,8),gridspec_kw=dict(left=0.1, right=0.98,bottom=0.1, top=0.93))

begin_time = time.time()
predDB = NEOS_test.get_expectation(Model_osc)
pred = NEOS_test.get_expectation(Model_ste)
pred2 = NEOS_test.get_expectation(Model_ste2)
end_time = time.time()
print(end_time-begin_time)

pred = pred['NEOS'][:,0]
predDB = predDB['NEOS'][:,0]
pred2 = pred2['NEOS'][:,0]
# pred = np.array([np.float(np.str(pred[i])[:-4]) for i in range(len(pred))])

axev.scatter(x_ax,pred, label = "Our prediction", marker = "_")
axev.scatter(x_ax,NEOS_test.AllData['NEOS'][:-1,0], label = "NEOS data")
axev.scatter(x_ax,NEOS_test.AllData['NEOS'][:-1,1], label = "NEOS prediction", marker = "_")
axev.set_xlabel("Energy (MeV)", fontsize = 16)
axev.set_ylabel("Events/(MeV$\ \cdot 10^{5}$)", fontsize = 16)
axev.tick_params(axis='x', labelsize=13)
axev.tick_params(axis='y', labelsize=13)
axev.axis([1.,9.,0.,60.])
axev.grid(linestyle="--")
axev.legend(loc="upper right",fontsize=16)

# figev.suptitle(r'Our best fit: $\Delta m^2_{13} = 2.5·10^{-3} eV^2$, $\sin^2 2\theta_{13} = 0.07821$', fontsize = 17)
# figev.suptitle(r'DB best fit: $\Delta m^2_{13} = 2.4·10^{-3} eV^2$, $\sin^2 2\theta_{13} = 0.0841$', fontsize = 17)
figev.savefig("Figures/Event expectation_SM_PW_vegas.png")
# As we can see, both ways of computing the event expectations give the same result.



# -------------------------------------------------------
# Event expectations - ratio to DB
# -------------------------------------------------------

x_ax = (NEOS_test.DataLowerBinEdges+NEOS_test.DataUpperBinEdges)/2
deltaE = (NEOS_test.DataUpperBinEdges-NEOS_test.DataLowerBinEdges)

figev,axev = plt.subplots(1,1,figsize = (12,8),gridspec_kw=dict(left=0.1, right=0.98,bottom=0.1, top=0.93))

axev.errorbar(x_ax,pred/predDB, xerr = 0.05, label = r'$\Delta m^2_{41} = 1.73 eV^2, \sin^2 2\theta_{14} = 0.05$', fmt = "_g", elinewidth = 2)
axev.errorbar(x_ax,pred2/predDB, xerr = 0.05, label = r'$\Delta m^2_{41} = 2.32 eV^2, \sin^2 2\theta_{14} = 0.142$', fmt = "_r", elinewidth = 2)
axev.errorbar(x_ax,NEOS_test.AllData['NEOS'][:-1,3], yerr = NEOS_test.AllData['NEOS'][:-1,4], label = "NEOS data", fmt = "ok")
# axev.scatter(x_ax,NEOS_test.AllData['NEOS'][:-1,1]/predDB, label = "NEOS prediction", marker = "_")
axev.plot(x_ax,[1 for x in x_ax], linestyle = 'dashed', color = 'yellow')
axev.set_xlabel("Energy (MeV)", fontsize = 16)
axev.set_ylabel("Events NEOS/DB", fontsize = 16)
axev.tick_params(axis='x', labelsize=13)
axev.tick_params(axis='y', labelsize=13)
axev.axis([1.,7.,0.88,1.12])
axev.grid(linestyle="--")
axev.legend(loc="lower left",fontsize=16)

# figev.suptitle(r'Our best fit: $\Delta m^2_{13} = 2.5·10^{-3} eV^2$, $\sin^2 2\theta_{13} = 0.07821$', fontsize = 17)
# figev.suptitle(r'DB best fit: $\Delta m^2_{13} = 2.4·10^{-3} eV^2$, $\sin^2 2\theta_{13} = 0.0841$', fontsize = 17)
figev.savefig("Figures/EventRatioDB_Ste_PW.png")
# As we can see, both ways of computing the event expectations give the same result.






# ----------------------------------------------
# CHI2 per bin per experimental hall
# ----------------------------------------------

# figchi,axchi = plt.subplots(1,3,figsize = (20,7),gridspec_kw=dict(left=0.05, right=0.98,bottom=0.1, top=0.96))
# for i in range(3):
#     data = DB_test.AllData[DB_test.sets_names[i]][:,3]
#     evex = pred[DB_test.sets_names[i]][:,0]
#     axchi[i].scatter(x_ax,-2*(data-evex+data*np.log(evex/data)), label = "Chi2")
#     # axev[i].scatter(x_ax,pred[1][DB_test.sets_names[i]][:,0]/deltaE/1.e5,marker="+",color = "blue", label = "Our no oscillations")
#     # axev[i].scatter(x_ax,DB_test.AllData[DB_test.sets_names[i]][:,5]/deltaE/1.e5,marker="+",color = "red", label = "DB no oscillations")
#     axchi[i].set_xlabel("Energy (MeV)", fontsize = 16)
#     axchi[i].set_ylabel("Arbitrary units", fontsize = 16)
#     axchi[i].tick_params(axis='x', labelsize=13)
#     axchi[i].tick_params(axis='y', labelsize=13)
#     # axev.axis([0.7,12,0.,2.5])
#     axchi[i].grid(linestyle="--")
#     axchi[i].legend(loc="upper right",fontsize=16)
#
# figchi.savefig("Figures/Chi2.png")
