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

# cwd = os.getcwd()
# path_to_style=cwd+'/Figures'
# plt.style.use(path_to_style+r"/paper.mplstyle")
# matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

fitter = PS.Prospect()
Model_noosc = Models.NoOscillations()
Model_osc = Models.PlaneWaveSM()
# Model_osc = Models.WavePacketSM()

# # Sterile data
# # ------------
sin2 = 0.079
dm2 = 1.74
Model_ste = Models.PlaneWaveSterile(DM2_41 = dm2,Sin22Th14 = sin2)
# # Model_ste = Models.WavePacketSterile(DM2_41 = dm2,Sin22Th14 = sin2)
#
#
sin2_2 = 0.142
dm2_2 = 2.32
# Model_ste2 = Models.PlaneWaveSterile(DM2_41 = dm2_2,Sin22Th14 = sin2_2)
# # Model_ste2 = Models.WavePacketSterile(DM2_41 = dm2_2,Sin22Th14 = sin2_2)




# -----------------------------------------------------
# PRELIMINAR COMPUTATIONS
# -----------------------------------------------------

begin_time = time.time()
predSM = fitter.get_expectation(Model_osc, do_we_integrate = False)
# pred = predSM
pred = fitter.get_expectation(Model_ste, do_we_integrate = True)
# pred2 = NEOS_test.get_expectation(Model_ste2, do_we_integrate = True)
end_time = time.time()
print(end_time-begin_time)

# def chi2_ratio_cov(Exp,ExpSM):
#     teo = Exp[:,0]/ExpSM[:,0]
#     ratio = NEOS_test.RatioData['NEOS']
#     Vinv = NEOS_test.get_inverse_covariance_matrix()
#     # Vinv *= np.tile(np.sqrt(ExpSM[:,0]),(len(ExpSM[:,0]),1))*(np.tile(np.sqrt(ExpSM[:,0]),(len(ExpSM[:,0]),1)).transpose())
#     return (teo-ratio).dot(Vinv.dot(teo-ratio))

data = fitter.get_data_per_baseline()
Me = 0.0
Pe = 0.0
PeSM = 0.0
for bl in fitter.Baselines:
    Me += np.sum(data[bl])
    Pe += np.sum(pred[bl])
    PeSM += np.sum(predSM[bl])

bkg = fitter.get_bkg_per_baseline()
# print(data)
# chi2 = chi2_ratio_cov(pred,predSM_DB)
# chi2_ratio = np.sum(chi2,axis=0)
chi2_per_exp = 1. #np.sum(chi2)


# -------------------------------------------------------
# Event expectations in SM, comparison with NEOS prediction
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
# axSM.errorbar(x_ax,NEOS_test.PredictedBackground['NEOS'], xerr = 0.05, label = "NEOS background", fmt = "_", elinewidth = 2)

figSM.suptitle(r'SM fit: $\Delta m^2_{31} = 2.5·10^{-3} eV^2$, $\sin^2 2\theta_{13} = 0.0841$', fontsize = 17)
figSM.savefig("Figures/EventExpectation/EventExpectation_SM.png")


# -------------------------------------------------------
# Event expectations
# -------------------------------------------------------

figev,axev = plt.subplots(2,5,figsize = (21,10),gridspec_kw=dict(left=0.05, right=0.98,bottom=0.1, top=0.93))
axev = axev.flatten()

for bl in fitter.Baselines:
    i = bl-1
    axev[i].errorbar(x_ax,pred[bl], xerr = 0.2, label = "Our prediction", fmt = "_", elinewidth = 2 )
    axev[i].errorbar(x_ax,predSM[bl], xerr  = 0.2, label = "No oscillation prediction", fmt = "_", elinewidth = 2)
    axev[i].errorbar(x_ax,data[bl], fmt = 'ok', label = "PROSPECT data")
    axev[i].set_xlabel("Energy (MeV)", fontsize = 16)
    axev[i].set_ylabel("Events/(0.1 MeV)", fontsize = 16)
    axev[i].tick_params(axis='x', labelsize=13)
    axev[i].tick_params(axis='y', labelsize=13)
    # axev.axis([1.,7.,0.,60.])
    axev[i].grid(linestyle="--")
    # axev[i].legend(loc="upper right",fontsize=16)
# axev.errorbar(x_ax,NEOS_test.PredictedBackground['NEOS'], xerr = 0.05, label = "NEOS background", fmt = "_", elinewidth = 2)

# figev.suptitle(r'Our best fit: $\Delta m^2_{13} = 2.5·10^{-3} eV^2$, $\sin^2 2\theta_{13} = 0.07821$', fontsize = 17)
# figev.suptitle(r'DB best fit: $\Delta m^2_{13} = 2.4·10^{-3} eV^2$, $\sin^2 2\theta_{13} = 0.0841$', fontsize = 17)
figev.suptitle(r'Sterile with $\Delta m^2_{41} = %.2f eV^2$, $\sin^2 2\theta_{14} = %.2f$. Total $\chi^2 = %.2f$'%(dm2,sin2,np.sum(chi2_per_exp)), fontsize = 17)
figev.savefig("Figures/EventExpectation/EventExpectation_%.2f_%.3f_ste.png"%(dm2,sin2))
# As we can see, both ways of computing the event expectations give the same result.



# -------------------------------------------------------
# Event expectations - ratio to DB
# -------------------------------------------------------

figev,axev = plt.subplots(2,5,figsize = (21,10),gridspec_kw=dict(left=0.05, right=0.98,bottom=0.1, top=0.93))
axev = axev.flatten()

for bl in fitter.Baselines:
    i = bl-1
    # exerr = np.sqrt(pred[bl])/predSM[bl]
    # axev[i].errorbar(x_ax,pred[bl]/predSM[bl], xerr = 0.2, label = "Our prediction", fmt = "_", elinewidth = 2 )
    # axev[i].errorbar(x_ax,pred[bl]/fitter.PredictedData[bl], xerr  = 0.2, label = "No oscillation prediction", fmt = "_", elinewidth = 2)
    # axev[i].errorbar(x_ax,data[bl]/predSM[bl], fmt = 'ok', label = "PROSPECT data")
    axev[i].errorbar(x_ax,pred[bl]*PeSM/Pe/predSM[bl], xerr = 0.2, label = "Our prediction", fmt = "_", elinewidth = 2 )
    # axev[i].errorbar(x_ax,pred[bl]/fitter.PredictedData[bl], xerr  = 0.2, label = "No oscillation prediction", fmt = "_", elinewidth = 2)
    axev[i].errorbar(x_ax,data[bl]/Me*Pe/pred[bl], fmt = 'ok', label = "PROSPECT data")
    axev[i].plot(x_ax,[1 for x in x_ax], linestyle = 'dashed', color = 'yellow')
    axev[i].set_xlabel("Energy (MeV)", fontsize = 16)
    axev[i].set_ylabel("Events/EventsSM", fontsize = 16)
    axev[i].tick_params(axis='x', labelsize=13)
    axev[i].tick_params(axis='y', labelsize=13)
    axev[i].axis([0.8,7.2,0.0,2.0])
    axev[i].grid(linestyle="--")
    # axev[i].legend(loc="upper right",fontsize=16)
# axev.errorbar(x_ax,NEOS_test.PredictedBackground['NEOS'], xerr = 0.05, label = "NEOS background", fmt = "_", elinewidth = 2)

# figev.suptitle(r'Our best fit: $\Delta m^2_{13} = 2.5·10^{-3} eV^2$, $\sin^2 2\theta_{13} = 0.07821$', fontsize = 17)
# figev.suptitle(r'DB best fit: $\Delta m^2_{13} = 2.4·10^{-3} eV^2$, $\sin^2 2\theta_{13} = 0.0841$', fontsize = 17)
figev.suptitle(r'Sterile with $\Delta m^2_{41} = %.2f eV^2$, $\sin^2 2\theta_{14} = %.2f$. Total $\chi^2 = %.2f$'%(dm2,sin2,np.sum(chi2_per_exp)), fontsize = 17)
figev.savefig("Figures/EventRatio/EventRatio_%.2f_%.3f_ste.png"%(dm2,sin2))
# As we can see, both ways of computing the event expectations give the same result.




# ----------------------------------------------
# CHI2 per bin per experimental hall
# ----------------------------------------------


# figchi,axchi = plt.subplots(1,1,figsize = (12,8),gridspec_kw=dict(left=0.1, right=0.98,bottom=0.1, top=0.93))
# axchi.bar(x_ax,-2*(data-evex+data*np.log(evex/data)),width = 3/4*deltaE, label = "Poisson")
# axchi.bar(x_ax,chi2_ratio,width = 3/4*deltaE, label = "Ratio 3(c)", alpha = 0.5)
# # axev[i].scatter(x_ax,pred[1][DB_test.sets_names[i]][:,0]/deltaE/1.e5,marker="+",color = "blue", label = "Our no oscillations")
# # axev[i].scatter(x_ax,DB_test.AllData[DB_test.sets_names[i]][:,5]/deltaE/1.e5,marker="+",color = "red", label = "DB no oscillations")
# axchi.set_xlabel("Energy (MeV)", fontsize = 16)
# axchi.set_ylabel("NEOS $\chi^2$ per bin", fontsize = 16)
# axchi.tick_params(axis='x', labelsize=13)
# axchi.tick_params(axis='y', labelsize=13)
# # axchi.set_xlim([1.,7.])
# axchi.axis([1.,7.,0.,40.])
# axchi.grid(linestyle="--")
# axchi.legend(loc="upper left",fontsize=16)
#
# figchi.suptitle(r'Sterile with $\Delta m^2_{41} = %.2f eV^2$, $\sin^2 2\theta_{13} = %.2f$. Total $\chi^2 = %.2f$'%(dm2,sin2,np.sum(chi2_per_exp)), fontsize = 17)
# figchi.savefig("Figures/Chi2/Chi2_%.2f_%.3f_ste.png"%(dm2,sin2))
