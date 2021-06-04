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
plt.style.use(path_to_style+r"/paper.mplstyle")


matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]


DB_test = DB.DayaBay()
Model_noosc = Models.NoOscillations()
Model_osc = Models.PlaneWaveSM()
Model_full = Models.PlaneWaveSM_full()

# -------------------------------------------------------
# Event expectations
# -------------------------------------------------------

x_ax = (DB_test.DataLowerBinEdges+DB_test.DataUpperBinEdges)/2
deltaE = (DB_test.DataUpperBinEdges-DB_test.DataLowerBinEdges)

figev,axev = plt.subplots(1,3,figsize = (20,7),gridspec_kw=dict(left=0.05, right=0.98,bottom=0.1, top=0.96))

begin_time = time.time()
pred = DB_test.get_expectation(Model_osc)
end_time = time.time()
print(begin_time-end_time)

for i in range(3):
    axev[i].scatter(x_ax,pred[DB_test.sets_names[i]][:,0]/deltaE/1.e5, label = "Our prediction")
    axev[i].scatter(x_ax,DB_test.AllData[DB_test.sets_names[i]][:,4]/deltaE/1.e5, label = "DB prediction")
    # axev[i].scatter(x_ax,pred[1][DB_test.sets_names[i]][:,0]/deltaE/1.e5,marker="+",color = "blue", label = "Our no oscillations")
    # axev[i].scatter(x_ax,DB_test.AllData[DB_test.sets_names[i]][:,5]/deltaE/1.e5,marker="+",color = "red", label = "DB no oscillations")
    axev[i].set_xlabel("Energy (MeV)", fontsize = 16)
    axev[i].set_ylabel("Events/(MeV$\ \cdot 10^{5}$)", fontsize = 16)
    axev[i].tick_params(axis='x', labelsize=13)
    axev[i].tick_params(axis='y', labelsize=13)
    # axev.axis([0.7,12,0.,2.5])
    axev[i].grid(linestyle="--")
    axev[i].legend(loc="upper right",fontsize=16)

figev.savefig("Figures/Event expectation.png")
# As we can see, both ways of computing the event expectations give the same result.
