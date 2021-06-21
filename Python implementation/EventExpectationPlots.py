import time
import InverseBetaDecayCrossSection as IBD
import HuberMullerFlux as HMF
import DayaBay as DB
import DayaBayParameters as DBpars
import DayaBayData as DBdata
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


DB_test = DB.DayaBay()
Model_noosc = Models.NoOscillations()
Model_osc = Models.PlaneWaveSM()
# Model_osc = Models.PlaneWaveSM(Sin22Th13 = 0.07821,DM2_31 = 2.5e-3)
# Model_full = Models.PlaneWaveSM_full()
Model_full = Models.PlaneWaveSM_full()
Model_coh = Models.WavePacketSM()

# -------------------------------------------------------
# Event expectations
# -------------------------------------------------------

x_ax = (DB_test.DataLowerBinEdges+DB_test.DataUpperBinEdges)/2
deltaE = (DB_test.DataUpperBinEdges-DB_test.DataLowerBinEdges)

figev,axev = plt.subplots(1,3,figsize = (20,8),gridspec_kw=dict(left=0.05, right=0.98,bottom=0.1, top=0.93))

begin_time = time.time()
pred = DB_test.get_expectation(Model_coh)
end_time = time.time()
print(begin_time-end_time)

print(pred['EH1'][:,0][0],pred['EH1'][:,0][1])

for i in range(3):
    axev[i].scatter(x_ax,pred[DB_test.sets_names[i]][:,0]/deltaE/1.e5, label = "Our prediction")
    axev[i].scatter(x_ax,DB_test.AllData[DB_test.sets_names[i]][:,3]/deltaE/1.e5, label = "DB data")
    # axev[i].scatter(x_ax,pred[1][DB_test.sets_names[i]][:,0]/deltaE/1.e5,marker="+",color = "blue", label = "Our no oscillations")
    # axev[i].scatter(x_ax,DB_test.AllData[DB_test.sets_names[i]][:,5]/deltaE/1.e5,marker="+",color = "red", label = "DB no oscillations")
    axev[i].set_xlabel("Energy (MeV)", fontsize = 16)
    axev[i].set_ylabel("Events/(MeV$\ \cdot 10^{5}$)", fontsize = 16)
    axev[i].tick_params(axis='x', labelsize=13)
    axev[i].tick_params(axis='y', labelsize=13)
    # axev.axis([0.7,12,0.,2.5])
    axev[i].grid(linestyle="--")
    axev[i].legend(loc="upper right",fontsize=16)

# figev.suptitle(r'Our best fit: $\Delta m^2_{13} = 2.5·10^{-3} eV^2$, $\sin^2 2\theta_{13} = 0.07821$', fontsize = 17)
figev.suptitle(r'DB best fit: $\Delta m^2_{13} = 2.4·10^{-3} eV^2$, $\sin^2 2\theta_{13} = 0.0841$', fontsize = 17)
figev.savefig("Figures/Event expectation_decoh.png")
# As we can see, both ways of computing the event expectations give the same result.


# ----------------------------------------------
# CHI2 per bin per experimental hall
# ----------------------------------------------

figchi,axchi = plt.subplots(1,3,figsize = (20,7),gridspec_kw=dict(left=0.05, right=0.98,bottom=0.1, top=0.96))
for i in range(3):
    data = DB_test.AllData[DB_test.sets_names[i]][:,3]
    evex = pred[DB_test.sets_names[i]][:,0]
    axchi[i].scatter(x_ax,-2*(data-evex+data*np.log(evex/data)), label = "Chi2")
    # axev[i].scatter(x_ax,pred[1][DB_test.sets_names[i]][:,0]/deltaE/1.e5,marker="+",color = "blue", label = "Our no oscillations")
    # axev[i].scatter(x_ax,DB_test.AllData[DB_test.sets_names[i]][:,5]/deltaE/1.e5,marker="+",color = "red", label = "DB no oscillations")
    axchi[i].set_xlabel("Energy (MeV)", fontsize = 16)
    axchi[i].set_ylabel("Arbitrary units", fontsize = 16)
    axchi[i].tick_params(axis='x', labelsize=13)
    axchi[i].tick_params(axis='y', labelsize=13)
    # axev.axis([0.7,12,0.,2.5])
    axchi[i].grid(linestyle="--")
    axchi[i].legend(loc="upper right",fontsize=16)

figchi.savefig("Figures/Chi2.png")
