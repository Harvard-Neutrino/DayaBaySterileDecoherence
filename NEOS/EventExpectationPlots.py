import sys
import os
homedir = os.path.realpath(__file__)[:-len('NEOS/EventExpectationPlots.py')]
common_dir = 'Common_cython/'
plotdir = homedir + 'NEOS/Figures/'
sys.path.append(homedir+common_dir)

import time
import NEOS
import Models

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# cwd = os.getcwd()
# path_to_style=cwd+'/Figures'
# plt.style.use(path_to_style+r"/paper.mplstyle")
# matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

NEOS_test = NEOS.Neos()
Model_osc = Models.PlaneWaveSM()
# Model_osc = Models.WavePacketSM()

# Sterile data
# ------------
sin2 = 0.8
dm2 = 9.5
# Model_ste = Models.PlaneWaveSterile(DM2_41 = dm2,Sin22Th14 = sin2)
Model_ste = Models.WavePacketSterile(DM2_41 = dm2,Sin22Th14 = sin2)


sin2_2 = 0.05
dm2_2 = 1.73
# Model_ste2 = Models.PlaneWaveSterile(DM2_41 = dm2_2,Sin22Th14 = sin2_2)
Model_ste2 = Models.WavePacketSterile(DM2_41 = dm2_2,Sin22Th14 = sin2_2)




# -----------------------------------------------------
# PRELIMINAR COMPUTATIONS
# -----------------------------------------------------

begin_time = time.time()
predDB_DB = NEOS_test.get_expectation(Model_osc, integrate = False, use_HM = False)
predDB_HM =  NEOS_test.get_expectation(Model_osc, integrate = False, use_HM = True)
pred = NEOS_test.get_expectation(Model_ste, integrate = True, use_HM = False)
pred2 = NEOS_test.get_expectation(Model_ste2, integrate = True, use_HM = False)
end_time = time.time()
print(end_time-begin_time)

pred = pred['NEOS']
predDB_DB = predDB_DB['NEOS']
predDB_HM = predDB_HM['NEOS']
pred2 = pred2['NEOS']


def chi2_ratio_cov(Exp,ExpSM):
    teo = Exp[:,0]/ExpSM[:,0]
    ratio = NEOS_test.RatioData['NEOS']
    Vinv = NEOS_test.get_inverse_covariance_matrix()

    return (teo-ratio).dot(Vinv.dot(teo-ratio))


evex = pred[:,0]
data = NEOS_test.ObservedData['NEOS']
ratio_pred = pred[:,0]/predDB_DB[:,0]

chi2 = chi2_ratio_cov(pred,predDB_DB)


# -------------------------------------------------------
# Event expectations in SM, comparison with NEOS prediction
# -------------------------------------------------------

x_ax = (NEOS_test.DataLowerBinEdges+NEOS_test.DataUpperBinEdges)/2
deltaE = (NEOS_test.DataUpperBinEdges -NEOS_test.DataLowerBinEdges)

# CAUTION! The following expectation is computed according to the HM flux.
# Therefore, it needn't be equal to the NEOS expectation, computed according
# to the flux from DayaBay.

figSM,axSM = plt.subplots(1,1,figsize = (12,8),gridspec_kw=dict(left=0.1, right=0.98,bottom=0.1, top=0.93))
axSM.errorbar(x_ax,predDB_HM[:,0], yerr = predDB_HM[:,1], xerr = 0.05, label = "Our prediction", fmt = "_", elinewidth = 2)
axSM.errorbar(x_ax,NEOS_test.PredictedData['NEOS'], xerr  = 0.05, label = "NEOS prediction", fmt = "_", elinewidth = 2)
axSM.errorbar(x_ax,NEOS_test.PredictedBackground['NEOS'], xerr = 0.05, label = "NEOS background", fmt = "_", elinewidth = 2)
axSM.errorbar(x_ax,data, fmt = 'ok', label = "NEOS data")

axSM.set_xlabel("Energy (MeV)", fontsize = 16)
axSM.set_ylabel("Events/(0.1 MeV)", fontsize = 16)
axSM.tick_params(axis='x', labelsize=13)
axSM.tick_params(axis='y', labelsize=13)
axSM.grid(linestyle="--")
axSM.legend(loc="upper right",fontsize=16)

figSM.suptitle(r'SM fit: $\Delta m^2_{31} = 2.5·10^{-3} eV^2$, $\sin^2 2\theta_{13} = 0.0841$', fontsize = 17)
figSM.savefig(plotdir+"EventExpectation/EventExpectation_SM.png")


# -------------------------------------------------------
# Event expectations
# -------------------------------------------------------

figev,axev = plt.subplots(1,1,figsize = (12,8),gridspec_kw=dict(left=0.1, right=0.98,bottom=0.1, top=0.93))

axev.errorbar(x_ax,pred[:,0], yerr = pred[:,1], xerr = 0.1, label = "Sterile", fmt = "_", elinewidth = 2)
axev.errorbar(x_ax,predDB_DB[:,0], yerr = predDB_DB[:,1], xerr = 0.05, label = "Standard Model", fmt = "_", elinewidth = 2)
axev.errorbar(x_ax,data, fmt = 'ok', label = "NEOS data")

axev.set_xlabel("Energy (MeV)", fontsize = 16)
axev.set_ylabel("Events/(0.1 MeV)", fontsize = 16)
axev.tick_params(axis='x', labelsize=13)
axev.tick_params(axis='y', labelsize=13)
axev.grid(linestyle="--")
axev.legend(loc="upper right",fontsize=16)

# figev.suptitle(r'Our best fit: $\Delta m^2_{13} = 2.5·10^{-3} eV^2$, $\sin^2 2\theta_{13} = 0.07821$', fontsize = 17)
# figev.suptitle(r'DB best fit: $\Delta m^2_{13} = 2.4·10^{-3} eV^2$, $\sin^2 2\theta_{13} = 0.0841$', fontsize = 17)
figev.suptitle(r'Sterile with $\Delta m^2_{41} = %.2f eV^2$, $\sin^2 2\theta_{14} = %.2f$. Total $\chi^2 = %.2f$'%(dm2,sin2,chi2), fontsize = 17)
figev.savefig(plotdir+"EventExpectation/EventExpectation_%.2f_%.3f_ste.png"%(dm2,sin2))
# As we can see, both ways of computing the event expectations give the same result.



# -------------------------------------------------------
# Event expectations - ratio to DB
# -------------------------------------------------------

figev,axev = plt.subplots(1,1,figsize = (12,8),gridspec_kw=dict(left=0.1, right=0.98,bottom=0.1, top=0.93))

exerr = np.sqrt(pred[:,0])/predDB_DB[:,0]

axev.step(x_ax,pred[:,0]/predDB_DB[:,0], where = 'mid', label = r'$\Delta m^2_{41} = %.2f eV^2$, $\sin^2 2\theta_{14} = %.2f$'%(dm2,sin2), color = 'green', linewidth = 2)
axev.step(x_ax,pred2[:,0]/predDB_DB[:,0], where = 'mid', label = r'$\Delta m^2_{41} = %.2f eV^2$, $\sin^2 2\theta_{14} = %.2f$'%(dm2_2,sin2_2), color = 'red', linewidth = 2)
axev.errorbar(x_ax,NEOS_test.RatioData['NEOS'], yerr = NEOS_test.RatioStatError['NEOS'], label = "NEOS data", fmt = "ok")
# axev.scatter(x_ax,NEOS_test.AllData['NEOS'][:,1]/predDB, label = "NEOS prediction", marker = "_")

axev.plot(x_ax,[1 for x in x_ax], linestyle = 'dashed', color = 'yellow')
axev.set_xlabel("Energy (MeV)", fontsize = 16)
axev.set_ylabel("Events NEOS/DB", fontsize = 16)
axev.tick_params(axis='x', labelsize=13)
axev.tick_params(axis='y', labelsize=13)
axev.axis([1.,7.,0.88,1.12])
axev.grid(linestyle="--")
axev.legend(loc="lower left",fontsize=16)

figev.suptitle(r'Sterile with $\Delta m^2_{41} = %.2f eV^2$, $\sin^2 2\theta_{14} = %.2f$. Total $\chi^2 = %.2f$'%(dm2,sin2,chi2), fontsize = 17)
figev.savefig(plotdir+"EventRatio/EventRatio_%.2f_%.3f_ste.png"%(dm2,sin2))
