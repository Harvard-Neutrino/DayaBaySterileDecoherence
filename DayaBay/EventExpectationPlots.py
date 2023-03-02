#!/usr/bin/python3
import time
import os

import DayaBay as DB
import Models

dir = os.path.realpath(__file__)[:-len('EventExpectationPlots.py')]
plotdir = dir + 'Figures/'

import numpy as np
import matplotlib.pyplot as plt

# We load the models we want to compute the expectations for
fitter = DB.DayaBay()
Model_noosc = Models.NoOscillations()
Model_osc = Models.PlaneWaveSM()
# Model_coh = Models.WavePacketSM()

# Sterile parameters
sin2 = 0.06 # sin^2(theta_{14})
dm2 = 0.5   # Delta m_{14}^2
Model_ste = Models.PlaneWaveSterile(Sin22Th14 = sin2, DM2_41 = dm2)


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
        return {'integrate':False,'average':False}
    elif (mass > 0.15) and (mass <= 2.):
        return {'integrate':True, 'average':False}
    elif (mass > 2.):
        return {'integrate':False,'average':True}

wdwd = what_do_we_do(dm2)
begin_time = time.time()
predDB = fitter.get_expectation(Model_osc)
pred = fitter.get_expectation(Model_ste, integrate = wdwd['integrate'], average = wdwd['average'])
end_time = time.time()
print(begin_time-end_time) # Prints the time the computation has taken


# We compute the chi^2 to each experimental hall, separately
chi2_per_exp = []
for exp in fitter.sets_names:
    evex = pred[exp][:,0]
    data = fitter.ObservedData[exp]
    chi2_per_exp.append(np.sum(-2*(data-evex+data*np.log(evex/data))))

# We define the prompt energy bins
x_ax = (fitter.DataLowerBinEdges+fitter.DataUpperBinEdges)/2
deltaE = (fitter.DataUpperBinEdges-fitter.DataLowerBinEdges)

# -------------------------------------------------------
# Event expectations
# -------------------------------------------------------

figev,axev = plt.subplots(1,len(fitter.sets_names),figsize = (20,8),gridspec_kw=dict(left=0.05, right=0.98,bottom=0.1, top=0.91))

axis = [[1.3,6.9,0.,3.5],[1.3,6.9,0.,3.],[1.3,6.9,0.,0.9]]
norm = [1e5,1e5,1e5]


for i in range(len(fitter.sets_names)):
    set = fitter.sets_names[i]
    axev[i].errorbar(x_ax,pred[set][:,0]/deltaE/norm[i], yerr = pred[set][:,1]/deltaE/norm[i], xerr = 0.1, label = "Our prediction", fmt = "_", elinewidth = 2)
    axev[i].scatter(x_ax,fitter.ObservedData[set]/deltaE/norm[i], label = "{} data".format(fitter.sets_names[i]),color = "black")

    # Other things to plot: DB prediction of no oscillations,
    # axev[i].errorbar(x_ax,fitter.AllData[DB_test.sets_names[i]][:,5]/deltaE/norm[i], fmt = '_', elinewidth = 2, color = "red", label = "DB no oscillations")

    axev[i].set_xlabel("Energy (MeV)", fontsize = 16)
    axev[i].set_ylabel("Events/(MeV$\ \cdot 10^{%i}$)"%(np.log10(norm[i])), fontsize = 16)
    axev[i].tick_params(axis='x', labelsize=13)
    axev[i].tick_params(axis='y', labelsize=13)
    axev[i].axis(axis[i])
    axev[i].grid(linestyle="--")
    axev[i].title.set_text(fitter.sets_names[i])
    axev[i].legend(loc="upper right",fontsize=16)

figev.suptitle(r'Sterile with $\Delta m^2_{41} = %.2f eV^2$, $\sin^2 2\theta_{13} = %.2f$. Total $\chi^2 = %.2f$'%(dm2,sin2,np.sum(chi2_per_exp)), fontsize = 17)
figev.savefig(plotdir+"EventExpectation/EventExpectation_%.2f_%.3f_ste.png"%(dm2,sin2))



# ----------------------------------------------
# EVENT EXPECTATIONS RATIO, HEAVY STERILE VS SM
# -----------------------------------------------

figev,axev = plt.subplots(1,len(fitter.sets_names),figsize = (20,8),gridspec_kw=dict(left=0.05, right=0.98,bottom=0.1, top=0.91))

for i in range(len(fitter.sets_names)):
    set = fitter.sets_names[i]
    ste_dat = pred[set][:,0]
    SM_dat = predDB[set][:,0]

    axev[i].errorbar(x_ax,ste_dat/SM_dat, xerr = 0.1, label = "Heavy sterile/SM", fmt = "_", elinewidth = 2)
    axev[i].plot(x_ax,np.ones([fitter.n_bins]),linestyle = 'dashed')
    axev[i].errorbar(x_ax,fitter.ObservedData[set]/SM_dat, yerr = np.sqrt(fitter.ObservedData[set])/SM_dat, label = "{} data".format(fitter.sets_names[i]), fmt = "ok")

    axev[i].set_xlabel("Energy (MeV)", fontsize = 16)
    axev[i].set_ylabel("Ratio ste/DB", fontsize = 16)
    axev[i].tick_params(axis='x', labelsize=13)
    axev[i].tick_params(axis='y', labelsize=13)
    # axev[i].axis(axis[i])
    axev[i].grid(linestyle="--")
    axev[i].title.set_text(fitter.sets_names[i])
    axev[i].legend(loc="upper right",fontsize=16)

# figev.suptitle(r'Our best fit: $\Delta m^2_{13} = 2.5·10^{-3} eV^2$, $\sin^2 2\theta_{13} = 0.07821$', fontsize = 17)
# figev.suptitle(r'DB best fit: $\Delta m^2_{13} = 2.4·10^{-3} eV^2$, $\sin^2 2\theta_{13} = 0.0841$', fontsize = 17)
figev.suptitle(r'Sterile with $\Delta m^2_{41} = %.2f eV^2$, $\sin^2 2\theta_{13} = %.2f$. Total $\chi^2 = %.2f$'%(dm2,sin2,np.sum(chi2_per_exp)), fontsize = 17)
figev.savefig(plotdir+"EventRatio/EventRatio_%.2f_%.3f_ste.png"%(dm2,sin2))



# ----------------------------------------------
# CHI2 per bin per experimental hall
# ----------------------------------------------

axis = [[1.3,6.9,0.,1.5],[1.3,6.9,0.,2.],[1.3,6.9,0.,5.5]]


figchi,axchi = plt.subplots(1,len(fitter.sets_names),figsize = (20,8),gridspec_kw=dict(left=0.05, right=0.98,bottom=0.1, top=0.91))

for i in range(len(fitter.sets_names)):
    set = fitter.sets_names[i]
    data = fitter.ObservedData[set]
    evex = pred[set][:,0]
    axchi[i].bar(x_ax,-2*(data-evex+data*np.log(evex/data)),width = 3/4*deltaE)

    axchi[i].set_xlabel("Energy (MeV)", fontsize = 16)
    axchi[i].set_ylabel(r"%s $\chi^2$ per bin"%(set), fontsize = 16)
    axchi[i].tick_params(axis='x', labelsize=13)
    axchi[i].tick_params(axis='y', labelsize=13)
    # axchi[i].axis(axis[i])
    axchi[i].grid(linestyle="--")
    axchi[i].title.set_text(set+r' total $\chi^2 = %.2f $'%(chi2_per_exp[i]))
    # axchi[i].legend(loc="upper right",fontsize=16)

figchi.suptitle(r'Sterile with $\Delta m^2_{41} = %.2f eV^2$, $\sin^2 2\theta_{13} = %.2f$. Total $\chi^2 = %.2f$'%(dm2,sin2,np.sum(chi2_per_exp)), fontsize = 17)
figchi.savefig(plotdir+"Chi2/Chi2_%.2f_%.3f_ste.png"%(dm2,sin2))
