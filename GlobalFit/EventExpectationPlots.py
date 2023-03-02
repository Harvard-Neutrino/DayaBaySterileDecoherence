#!/usr/bin/python3
import time
import sys
import os
homedir = os.path.realpath(__file__)[:-len('GlobalFit/EventExpectationPlots.py')]
common_dir = 'Common_cython'
sys.path.append(homedir+common_dir)
sys.path.append(homedir+"/NEOS")
sys.path.append(homedir+"/DayaBay")


import GlobalFit as GF
import Models

import numpy as np
import matplotlib.pyplot as plt

# We load the models we want to compute the expectations for
fitter = GF.GlobalFit()
Model_noosc = Models.NoOscillations()
Model_osc = Models.PlaneWaveSM()
# Model_osc = Models.WavePacketSM()

# Sterile parameters
sin2 = 0.01
dm2 = 0.25
Model_ste = Models.PlaneWaveSterile(Sin22Th14 = sin2, DM2_41 = dm2)
# Model_ste = Models.WavePacketSterile(Sin22Th14 = sin2, DM2_41 = dm2)

plotdir = homedir + 'GlobalFit/Figures/'

# -------------------------------------------------------------
# INITIAL COMPUTATIONS
# ------------------------------------------------------------


def what_do_we_do(mass):
    """Mass must be in eV^2.

    This functions tells us whether to integrate or to average
    the oscillations for the given mass squared.
    The limits of the regimes are approximate.
    """
    if mass <= 0.15:
        return {'DB':{'integrate':False,'average':False},'NEOS':{'integrate':False,'average':False}}
    elif (mass > 0.15) and (mass <= 1.):
        return {'DB':{'integrate':True,'average':False},'NEOS':{'integrate':False,'average':False}}
    elif (mass > 1.) and (mass <= 2.):
        return {'DB':{'integrate':True,'average':False},'NEOS':{'integrate':True,'average':False}}
    elif (mass > 2.) and (mass <= 10.):
        return {'DB':{'integrate':False,'average':True},'NEOS':{'integrate':True,'average':False}}
    elif (mass > 10.):
        return {'DB':{'integrate':False,'average':True},'NEOS':{'integrate':False,'average':True}}

wdwd = what_do_we_do(dm2)
begin_time = time.time()
predDB = fitter.get_expectation(Model_osc)
all = fitter.get_expectation_ratio_and_chi2(Model_ste,integrate_DB = wdwd['DB']['integrate'],  average_DB = wdwd['DB']['average'],
                                                    integrate_NEOS = wdwd['NEOS']['integrate'],average_NEOS = wdwd['NEOS']['integrate'])
end_time = time.time()
print(begin_time-end_time)

pred = all[0]
chi2_per_exp = all[2]
ratio = all[1]
print(chi2_per_exp)

# We define the prompt energy bins
x_ax = dict([(set_name,(fitter.DataLowerBinEdges[set_name]+fitter.DataUpperBinEdges[set_name])/2.) for set_name in fitter.sets_names])
deltaE = dict([(set_name,(-fitter.DataLowerBinEdges[set_name]+fitter.DataUpperBinEdges[set_name])) for set_name in fitter.sets_names])

# -------------------------------------------------------
# Event expectations
# -------------------------------------------------------

figev,axev = plt.subplots(1,4,figsize = (25,8),gridspec_kw=dict(left=0.05, right=0.98,bottom=0.1, top=0.91))

axis = [[1.3,6.9,0.,3.5],[1.3,6.9,0.,3.],[1.3,6.9,0.,0.9],[1.3,6.9,0.,0.81]]
norm = [1e5,1e5,1e5,1e5]
xerror = [0.1,0.1,0.1,0.05]

for i in range(4):
    set = fitter.sets_names[i]
    axev[i].errorbar(x_ax[set],pred[set]/deltaE[set]/norm[i], yerr = np.sqrt(pred[set])/deltaE[set]/norm[i], xerr = xerror[i], label = "Our prediction", fmt = "_", elinewidth = 2)
    axev[i].scatter(x_ax[set],fitter.ObservedData[set]/deltaE[set]/norm[i], label = "{} data".format(fitter.sets_names[i]),color = "black")

    # Other things to plot: DB prediction of no oscillations (only for i = 0,1,2)
    # axev[i].scatter(x_ax,DB_test.AllData[DB_test.sets_names[i]][:,5]/deltaE/1.e5,marker="+",color = "red", label = "DB no oscillations")

    axev[i].set_xlabel("Energy (MeV)", fontsize = 16)
    axev[i].set_ylabel("Events/(MeV$\ \cdot 10^{%i}$)"%(np.log10(norm[i])), fontsize = 16)
    axev[i].tick_params(axis='x', labelsize=13)
    axev[i].tick_params(axis='y', labelsize=13)
    axev[i].axis(axis[i])
    axev[i].grid(linestyle="--")
    axev[i].title.set_text(fitter.sets_names[i]+r' total $\chi^2 = %.2f $'%(chi2_per_exp[i]))
    axev[i].legend(loc="upper right",fontsize=16)

figev.suptitle(r'Sterile with $\Delta m^2_{41} = %.2f eV^2$, $\sin^2 2\theta_{14} = %.3f$. Total $\chi^2 = %.2f$'%(dm2,sin2,np.sum(chi2_per_exp)), fontsize = 17)
figev.savefig(plotdir+"EventExpectation/EventExpectation_%.2f_%.3f_ste.png"%(dm2,sin2))


# ----------------------------------------------
# EVENT EXPECTATIONS RATIO, HEAVY STERILE VS SM
# -----------------------------------------------

figev,axev = plt.subplots(1,4,figsize = (25,8),gridspec_kw=dict(left=0.05, right=0.98,bottom=0.1, top=0.91))

for i in range(4):
    set = fitter.sets_names[i]

    # We compute the error from the ratio for DB
    ste_dat = pred[fitter.sets_names[i]]
    SM_dat = predDB[fitter.sets_names[i]][:,0]
    ste_err = np.sqrt(ste_dat)
    ratio_err = ste_err/SM_dat

    axev[i].step(x_ax[set],ste_dat/SM_dat, where = 'mid', label = "Heavy sterile/SM", linewidth = 2)
    axev[i].plot(x_ax[set],np.ones([fitter.n_bins[set]]),linestyle = 'dashed', color = 'k', zorder = 0.1)
    if set == 'NEOS':
        axev[i].errorbar(x_ax[set],fitter.NEOSRatioData, yerr = fitter.NEOSRatioStatError, label = "NEOS data", fmt = "ok")
    else:
        axev[i].errorbar(x_ax[set],fitter.ObservedData[set]/SM_dat, yerr = np.sqrt(fitter.ObservedData[set])/SM_dat, label = "{} data".format(set), fmt = "ok")

    axev[i].set_xlabel("Energy (MeV)", fontsize = 16)
    axev[i].set_ylabel("Ratio ste/DB", fontsize = 16)
    axev[i].tick_params(axis='x', labelsize=13)
    axev[i].tick_params(axis='y', labelsize=13)
    # axev[i].axis(axis[i])
    axev[i].grid(linestyle="--")
    axev[i].title.set_text(fitter.sets_names[i]+r' total $\chi^2 = %.2f $'%(chi2_per_exp[i]))
    axev[i].legend(loc="upper right",fontsize=16)

figev.suptitle(r'Sterile with $\Delta m^2_{41} = %.2f eV^2$, $\sin^2 2\theta_{14} = %.3f$. Total $\chi^2 = %.2f$'%(dm2,sin2,np.sum(chi2_per_exp)), fontsize = 17)
figev.savefig(plotdir+"EventRatio/EventRatio_%.2f_%.3f_ste.png"%(dm2,sin2))


# ----------------------------------------------
# CHI2 per bin per experimental hall
# ----------------------------------------------

axis = [[1.3,6.9,0.,1.5],[1.3,6.9,0.,2.],[1.3,6.9,0.,5.5],[1.3,6.9,0.,0.6]]

# For the NEOS experiment, we plot the diagonal elements.
Vinv = fitter.get_inverse_covariance_matrix()['NEOS']
NEOSchi2 = (ratio-fitter.NEOSRatioData)**2*np.diag(Vinv)

figchi,axchi = plt.subplots(1,4,figsize = (25,8),gridspec_kw=dict(left=0.05, right=0.98,bottom=0.1, top=0.91))
for i in range(4):
    set = fitter.sets_names[i]
    data = fitter.ObservedData[fitter.sets_names[i]]
    evex = pred[fitter.sets_names[i]]

    if set == 'NEOS':
        axchi[i].bar(x_ax[set],NEOSchi2, width = 3/4*deltaE[set]) #doesn't include correlations
    else:
        axchi[i].bar(x_ax[set],-2*(data-evex+data*np.log(evex/data)),width = 3/4*deltaE[set])

    axchi[i].set_xlabel("Energy (MeV)", fontsize = 16)
    axchi[i].set_ylabel(r"%s $\chi^2$ per bin"%(fitter.sets_names[i]), fontsize = 16)
    axchi[i].tick_params(axis='x', labelsize=13)
    axchi[i].tick_params(axis='y', labelsize=13)
    # axchi[i].axis(axis[i])
    axchi[i].grid(linestyle="--")
    axchi[i].title.set_text(fitter.sets_names[i]+r' total $\chi^2 = %.2f $'%(chi2_per_exp[i]))
    # axchi[i].legend(loc="upper right",fontsize=16)

figchi.suptitle(r'Sterile with $\Delta m^2_{41} = %.2f eV^2$, $\sin^2 2\theta_{13} = %.2f$. Total $\chi^2 = %.2f$'%(dm2,sin2,np.sum(chi2_per_exp)), fontsize = 17)
figchi.savefig(plotdir+"Chi2/Chi2_%.2f_%.3f_ste.png"%(dm2,sin2))
