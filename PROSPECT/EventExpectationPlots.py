import sys
import os
common_dir = '/Common_cython'
sys.path.append(os.getcwd()[:-9]+common_dir)

import time
import PROSPECT as PS
import Models

import numpy as np
import matplotlib.pyplot as plt
import matplotlib


fitter = PS.Prospect()
Model_noosc = Models.NoOscillations()
Model_osc = Models.PlaneWaveSM()
# Model_coh = Models.WavePacketSM()


# # Sterile data
# # ------------
sin2 = 0.5
dm2 = 6
Model_ste = Models.PlaneWaveSterile(DM2_41 = dm2,Sin22Th14 = sin2)
# # Model_ste = Models.WavePacketSterile(DM2_41 = dm2,Sin22Th14 = sin2)



# -----------------------------------------------------
# PRELIMINAR COMPUTATIONS
# -----------------------------------------------------

begin_time = time.time()

predSM = fitter.get_expectation(Model_osc, do_we_integrate = False)
pred = fitter.get_expectation(Model_ste, do_we_integrate = True)

end_time = time.time()
print(end_time-begin_time)


data = fitter.get_data_per_baseline()
Me = 0.0
Pe = 0.0
PeSM = 0.0
for bl in fitter.Baselines:
    Me += np.sum(data[bl])
    Pe += np.sum(pred[bl])
    PeSM += np.sum(predSM[bl])

bkg = fitter.get_bkg_per_baseline()


# -------------------------------------------------------
# Event expectations in SM - total
# -------------------------------------------------------

x_ax = (fitter.DataLowerBinEdges+fitter.DataUpperBinEdges)/2
deltaE = (fitter.DataUpperBinEdges -fitter.DataLowerBinEdges)

figSM,axSM = plt.subplots(2,5,figsize = (21,10),gridspec_kw=dict(left=0.05, right=0.98,bottom=0.1, top=0.93))
axSM = axSM.flatten()

for bl in fitter.Baselines:
    i = bl-1
    axSM[i].step(x_ax,predSM[bl],where = 'mid', label = "Our prediction" )
    axSM[i].step(x_ax,fitter.PredictedData[bl], where = 'mid', label = "No oscillation prediction")
    axSM[i].errorbar(x_ax,data[bl], fmt = 'ok', label = "PROSPECT data")
    axSM[i].errorbar(x_ax,bkg[bl], xerr = 0.1, label = "NEOS background", fmt = "_", elinewidth = 2)
    axSM[i].set_xlabel("Energy (MeV)", fontsize = 16)
    axSM[i].set_ylabel("Events/(0.1 MeV)", fontsize = 16)
    axSM[i].tick_params(axis='x', labelsize=13)
    axSM[i].tick_params(axis='y', labelsize=13)
    # axSM.axis([1.,7.,0.,60.])
    axSM[i].grid(linestyle="--")
    # axSM[i].legend(loc="upper right",fontsize=16)

figSM.suptitle(r'SM fit: $\Delta m^2_{31} = 2.5·10^{-3} eV^2$, $\sin^2 2\theta_{13} = 0.0841$', fontsize = 17)
figSM.savefig("Figures/EventExpectation/EventExpectation_SM.png")


# -------------------------------------------------------
# Event expectations - total
# -------------------------------------------------------

figev,axev = plt.subplots(2,5,figsize = (21,10),gridspec_kw=dict(left=0.05, right=0.98,bottom=0.1, top=0.93))
axev = axev.flatten()

for bl in fitter.Baselines:
    i = bl-1
    axev[i].step(x_ax,pred[bl], where = 'mid', label = "Our prediction", linewidth = 2 )
    axev[i].step(x_ax,predSM[bl], where = 'mid', label = "No oscillation prediction", linewidth = 2)
    axev[i].errorbar(x_ax,data[bl], fmt = 'ok', label = "PROSPECT data")
    axev[i].set_xlabel("Energy (MeV)", fontsize = 16)
    axev[i].set_ylabel("Events/(0.1 MeV)", fontsize = 16)
    axev[i].tick_params(axis='x', labelsize=13)
    axev[i].tick_params(axis='y', labelsize=13)
    # axev.axis([1.,7.,0.,60.])
    axev[i].grid(linestyle="--")
    # axev[i].legend(loc="upper right",fontsize=16)

figev.suptitle(r'Sterile with $\Delta m^2_{41} = %.2f eV^2$, $\sin^2 2\theta_{14} = %.2f$'%(dm2,sin2), fontsize = 17)
figev.savefig("Figures/EventExpectation/EventExpectation_%.2f_%.3f_ste.png"%(dm2,sin2))


# -------------------------------------------------------
# Event expectations - ratio to SM
# -------------------------------------------------------

figev,axev = plt.subplots(2,5,figsize = (21,10),gridspec_kw=dict(left=0.05, right=0.98,bottom=0.1, top=0.93))
axev = axev.flatten()

for bl in fitter.Baselines:
    i = bl-1

    axev[i].step(x_ax,pred[bl]*PeSM/Pe/predSM[bl], where = 'mid', label = "Our prediction", linewidth = 2 )
    axev[i].errorbar(x_ax,data[bl]/Me*Pe/pred[bl], fmt = 'ok', label = "PROSPECT data")

    axev[i].plot(x_ax,[1 for x in x_ax], linestyle = 'dashed', color = 'k', zorder = 0.01)

    axev[i].set_xlabel("Energy (MeV)", fontsize = 16)
    axev[i].set_ylabel("Events/EventsSM", fontsize = 16)
    axev[i].tick_params(axis='x', labelsize=13)
    axev[i].tick_params(axis='y', labelsize=13)
    axev[i].axis([0.8,7.2,0.0,2.0])
    axev[i].grid(linestyle="--")
    # axev[i].legend(loc="upper right",fontsize=16)


figev.suptitle(r'Sterile with $\Delta m^2_{41} = %.2f eV^2$, $\sin^2 2\theta_{14} = %.2f$'%(dm2,sin2), fontsize = 17)
figev.savefig("Figures/EventRatio/EventRatio_%.2f_%.3f_ste.png"%(dm2,sin2))
