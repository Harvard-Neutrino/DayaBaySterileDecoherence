#!/usr/bin/python3
import sys
import os
common_dir = '/Common_cython'
sys.path.append(os.getcwd()[:-10]+common_dir)
sys.path.append(os.getcwd()[:-10]+"/NEOS")
sys.path.append(os.getcwd()[:-10]+"/DayaBay")


import numpy as np
import time
# import GlobalFit as GF
import GlobalFitNuissances as GF
import Models

import matplotlib.pyplot as plt

# cwd = os.getcwd()
# path_to_style=cwd+'/Figures'
# plt.style.use(path_to_style+r"/paper.mplstyle")
# matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

fitter = GF.GlobalFit()
Model_noosc = Models.NoOscillations()
Model_osc = Models.PlaneWaveSM()
# Model_osc = Models.WavePacketSM()

# Sterile parameters
sin2 = 0.8
dm2 = 9.5
Model_ste = Models.PlaneWaveSterile(Sin22Th14 = sin2, DM2_41 = dm2)
# Model_ste = Models.WavePacketSterile(Sin22Th14 = sin2, DM2_41 = dm2)

# -------------------------------------------------------------
# INITIAL COMPUTATIONS
# ------------------------------------------------------------


def what_do_we_do(mass):
    """Mass must be in eV^2. """
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


# chi2_per_exp = []
# for exp in fitter.sets_names[-1]:
#     evex = pred[exp][:,0]
#     data = fitter.AllData[exp][:,0]
#     chi2_per_exp.append(np.sum(-2*(data-evex+data*np.log(evex/data))))

chi2_per_exp = all[2]
ratio = all[1]
print(chi2_per_exp)


# -------------------------------------------------------
# Event expectations
# -------------------------------------------------------

x_ax = dict([(set_name,(fitter.DataLowerBinEdges[set_name]+fitter.DataUpperBinEdges[set_name])/2.) for set_name in fitter.sets_names])
deltaE = dict([(set_name,(-fitter.DataLowerBinEdges[set_name]+fitter.DataUpperBinEdges[set_name])) for set_name in fitter.sets_names])

figev,axev = plt.subplots(1,4,figsize = (25,8),gridspec_kw=dict(left=0.05, right=0.98,bottom=0.1, top=0.91))

axis = [[1.3,6.9,0.,3.5],[1.3,6.9,0.,3.],[1.3,6.9,0.,0.9],[1.3,6.9,0.,0.81]]
norm = [1e5,1e5,1e5,1e5]
xerror = [0.1,0.1,0.1,0.05]

for i in range(4):
    set = fitter.sets_names[i]
    axev[i].errorbar(x_ax[set],pred[set]/deltaE[set]/norm[i], yerr = np.sqrt(pred[set])/deltaE[set]/norm[i], xerr = xerror[i], label = "Our prediction", fmt = "_", elinewidth = 2)
    # axev[i].errorbar(x_ax,pred[fitter.sets_names[i]]/deltaE/norm[i], yerr = err[fitter.sets_names[i]]/deltaE/norm[i], xerr = 0.1, label = "Our prediction", fmt = "_", elinewidth = 2)
    axev[i].scatter(x_ax[set],fitter.ObservedData[set]/deltaE[set]/norm[i], label = "{} data".format(fitter.sets_names[i]),color = "black")
    # axev[i].scatter(x_ax,pred[1][DB_test.sets_names[i]][:,0]/deltaE/1.e5,marker="+",color = "blue", label = "Our no oscillations")
    # axev[i].scatter(x_ax,DB_test.AllData[DB_test.sets_names[i]][:,5]/deltaE/1.e5,marker="+",color = "red", label = "DB no oscillations")
    axev[i].set_xlabel("Energy (MeV)", fontsize = 16)
    axev[i].set_ylabel("Events/(MeV$\ \cdot 10^{%i}$)"%(np.log10(norm[i])), fontsize = 16)
    axev[i].tick_params(axis='x', labelsize=13)
    axev[i].tick_params(axis='y', labelsize=13)
    axev[i].axis(axis[i])
    axev[i].grid(linestyle="--")
    axev[i].title.set_text(fitter.sets_names[i]+r' total $\chi^2 = %.2f $'%(chi2_per_exp[i]))
    axev[i].legend(loc="upper right",fontsize=16)

# figev.suptitle(r'Our best fit: $\Delta m^2_{13} = 2.5·10^{-3} eV^2$, $\sin^2 2\theta_{13} = 0.07821$', fontsize = 17)
# figev.suptitle(r'DB best fit: $\Delta m^2_{13} = 2.4·10^{-3} eV^2$, $\sin^2 2\theta_{13} = 0.0841$', fontsize = 17)
figev.suptitle(r'Sterile with $\Delta m^2_{41} = %.2f eV^2$, $\sin^2 2\theta_{14} = %.3f$. Total $\chi^2 = %.2f$'%(dm2,sin2,np.sum(chi2_per_exp)), fontsize = 17)
figev.savefig("Figures/EventExpectation/EventExpectation_%.2f_%.3f_ste.png"%(dm2,sin2))
# As we can see, both ways of computing the event expectations give the same result.


# ----------------------------------------------
# EVENT EXPECTATIONS RATIO, HEAVY STERILE VS SM
# -----------------------------------------------

# Note that this is not exactly what we're fitting.
# Here the nuissances for the SM expectations are different than for the sterile expectations,
# because they are built in different calls of get_expectation.
# Therefore, the flux and total number of targets is not the same, which makes this ratio not rigorous.
figev,axev = plt.subplots(1,4,figsize = (25,8),gridspec_kw=dict(left=0.05, right=0.98,bottom=0.1, top=0.91))

for i in range(4):
    set = fitter.sets_names[i]
    ste_dat = pred[fitter.sets_names[i]]
    SM_dat = predDB[fitter.sets_names[i]][:,0]
    ste_err = np.sqrt(ste_dat)
    SM_err = np.sqrt(SM_dat)
    # ratio_err = ste_err/SM_dat + ste_dat*SM_err/SM_dat**2
    ratio_err = ste_err/SM_dat
    axev[i].errorbar(x_ax[set],ste_dat/SM_dat, yerr = ratio_err, xerr = 0.1, label = "Heavy sterile/SM", fmt = "_", elinewidth = 2)
    axev[i].plot(x_ax[set],np.ones([fitter.n_bins[set]]),linestyle = 'dashed')
    if set == 'NEOS':
        axev[i].errorbar(x_ax[set],fitter.NEOSRatioData, yerr = fitter.NEOSRatioStatError, label = "NEOS data", fmt = "ok")
    # axev[i].scatter(x_ax,DB_test.AllData[DB_test.sets_names[i]][:,5]/deltaE/1.e5,marker="+",color = "red", label = "DB no oscillations")
    axev[i].set_xlabel("Energy (MeV)", fontsize = 16)
    axev[i].set_ylabel("Ratio ste/DB", fontsize = 16)
    axev[i].tick_params(axis='x', labelsize=13)
    axev[i].tick_params(axis='y', labelsize=13)
    # axev[i].axis(axis[i])
    axev[i].grid(linestyle="--")
    axev[i].title.set_text(fitter.sets_names[i]+r' total $\chi^2 = %.2f $'%(chi2_per_exp[i]))
    axev[i].legend(loc="upper right",fontsize=16)

# figev.suptitle(r'Our best fit: $\Delta m^2_{13} = 2.5·10^{-3} eV^2$, $\sin^2 2\theta_{13} = 0.07821$', fontsize = 17)
# figev.suptitle(r'DB best fit: $\Delta m^2_{13} = 2.4·10^{-3} eV^2$, $\sin^2 2\theta_{13} = 0.0841$', fontsize = 17)
figev.suptitle(r'Sterile with $\Delta m^2_{41} = %.2f eV^2$, $\sin^2 2\theta_{14} = %.3f$. Total $\chi^2 = %.2f$'%(dm2,sin2,np.sum(chi2_per_exp)), fontsize = 17)
figev.savefig("Figures/EventRatio/EventRatio_%.2f_%.3f_ste.png"%(dm2,sin2))

# -----------------------------------------------------
# ONLY NEOS

figNEOS,axNEOS = plt.subplots(1,2,figsize = (14,8),gridspec_kw=dict(left=0.06, right=0.98,bottom=0.1, top=0.91))
ste_dat = pred['NEOS']
ste_err = np.sqrt(ste_dat)
SM_dat = predDB['NEOS'][:,0]

set = 'NEOS'
axNEOS[1].errorbar(x_ax[set],ratio, xerr = 0.05, label = "Our prediction w/out error", fmt = "_", elinewidth = 2.5)
axNEOS[1].errorbar(x_ax[set],fitter.NEOSRatioData, yerr = fitter.NEOSRatioStatError, label = "NEOS data", fmt = "ok")
axNEOS[0].errorbar(x_ax[set],ratio, yerr = ste_err/SM_dat, xerr = 0.05, label = "Our prediction w/error", fmt = "_", elinewidth = 2.5)
axNEOS[0].errorbar(x_ax[set],fitter.NEOSRatioData, label = "NEOS data", fmt = "ok")

for i in range(2):
    axNEOS[i].set_xlabel("Energy (MeV)", fontsize = 16)
    axNEOS[i].set_ylabel("Ratio ste/DB", fontsize = 16)
    axNEOS[i].tick_params(axis='x', labelsize=13)
    axNEOS[i].tick_params(axis='y', labelsize=13)
    axNEOS[i].grid(linestyle="--")
    axNEOS[i].axis([1.2,7.0,0.85,1.25])
    axNEOS[i].legend(loc="lower left",fontsize=16)
    axNEOS[i].plot(x_ax[set],np.ones([fitter.n_bins[set]]),linestyle = 'dashed')

figNEOS.suptitle(r'Sterile with $\Delta m^2_{41} = %.2f eV^2$, $\sin^2 2\theta_{14} = %.3f$. Total $\chi^2 = %.2f$'%(dm2,sin2,np.sum(chi2_per_exp)), fontsize = 17)
figNEOS.savefig("Figures/NEOSRatio/NEOSRatio_%.2f_%.3f_ste.png"%(dm2,sin2))

# ----------------------------------------------
# CHI2 per bin per experimental hall
# ----------------------------------------------

axis = [[1.3,6.9,0.,1.5],[1.3,6.9,0.,2.],[1.3,6.9,0.,5.5],[1.3,6.9,0.,0.6]]

Vinv = fitter.get_inverse_covariance_matrix()['NEOS']
# NEOSchi2 = np.array([(ratio-fitter.NEOSRatioData)[i]*Vinv[:,i].dot(ratio-fitter.NEOSRatioData) for i in range(fitter.n_bins['NEOS'])])
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
    # axev[i].scatter(x_ax,pred[1][DB_test.sets_names[i]][:,0]/deltaE/1.e5,marker="+",color = "blue", label = "Our no oscillations")
    # axev[i].scatter(x_ax,DB_test.AllData[DB_test.sets_names[i]][:,5]/deltaE/1.e5,marker="+",color = "red", label = "DB no oscillations")
    axchi[i].set_xlabel("Energy (MeV)", fontsize = 16)
    axchi[i].set_ylabel(r"%s $\chi^2$ per bin"%(fitter.sets_names[i]), fontsize = 16)
    axchi[i].tick_params(axis='x', labelsize=13)
    axchi[i].tick_params(axis='y', labelsize=13)
    # axchi[i].axis(axis[i])
    axchi[i].grid(linestyle="--")
    axchi[i].title.set_text(fitter.sets_names[i]+r' total $\chi^2 = %.2f $'%(chi2_per_exp[i]))
    # axchi[i].legend(loc="upper right",fontsize=16)

# figchi.suptitle(r'DayaBay best fit (3 neutrino): $\Delta m^2_{ee} = 2.5\times 10^{-3} eV^2$, $\sin^2 2\theta_{13} = 0.0841$. Total $\chi^2 = 41.98$', fontsize = 17)
# figchi.suptitle(r'Sterile best fit (3+1): $\Delta m^2_{41} = 0.067 eV^2$, $\sin^2 2\theta_{13} = 8.29\times 10^{-3}$. Total $\chi^2 = 39.16$', fontsize = 17)
figchi.suptitle(r'Sterile with $\Delta m^2_{41} = %.2f eV^2$, $\sin^2 2\theta_{13} = %.2f$. Total $\chi^2 = %.2f$'%(dm2,sin2,np.sum(chi2_per_exp)), fontsize = 17)
figchi.savefig("Figures/Chi2/Chi2_%.2f_%.3f_ste.png"%(dm2,sin2))
