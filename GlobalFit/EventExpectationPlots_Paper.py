#!/usr/bin/python3
import sys
import os
homedir = os.path.realpath(__file__)[:-len('GlobalFit/EventExpectationPlots_Paper.py')]
common_dir = 'Common_cython'
sys.path.append(homedir+common_dir)
sys.path.append(homedir+"/NEOS")
sys.path.append(homedir+"/DayaBay")


import numpy as np
import time
import GlobalFit as GF
import Models

import matplotlib.pyplot as plt
import matplotlib

path_to_style = homedir + common_dir
plotdir = homedir + 'GlobalFit/Figures/'
dir = 'PlotData/'
plt.style.use(path_to_style+r"/paper.mplstyle")
matplotlib.rcParams.update({'text.usetex': True})
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

color1 = '#FFB14E'
color2 = '#EA5F94'
color3 = '#0000FF'

fitter = GF.GlobalFit()
Model_noosc = Models.NoOscillations()
Model_osc = Models.PlaneWaveSM()
# Model_osc = Models.WavePacketSM()

# Sterile parameters
sin2 = 0.142
dm2 = 2.32
Model_ste = Models.PlaneWaveSterile(Sin22Th14 = sin2, DM2_41 = dm2)
Model_ste2 = Models.WavePacketSterile(Sin22Th14 = sin2, DM2_41 = dm2)


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
all2 = fitter.get_expectation_ratio_and_chi2(Model_ste2,integrate_DB = wdwd['DB']['integrate'],  average_DB = wdwd['DB']['average'],
                                                    integrate_NEOS = wdwd['NEOS']['integrate'],average_NEOS = wdwd['NEOS']['integrate'])
end_time = time.time()
print(begin_time-end_time)

pred = all[0]
ratio = all[1]
pred2 = all2[0]
ratio2 = all2[1]

x_ax = dict([(set_name,(fitter.DataLowerBinEdges[set_name]+fitter.DataUpperBinEdges[set_name])/2.) for set_name in fitter.sets_names])
deltaE = dict([(set_name,(-fitter.DataLowerBinEdges[set_name]+fitter.DataUpperBinEdges[set_name])) for set_name in fitter.sets_names])


# ----------------------------------------------
# EVENT EXPECTATIONS RATIO, HEAVY STERILE VS SM
# -----------------------------------------------------
# ONLY NEOS
# ---------

margins = dict(left=0.15, right=0.97,bottom=0.15, top=0.9)
figNEOS,axNEOS = plt.subplots(1,1,figsize = (8,6),gridspec_kw=margins)


set = 'NEOS'
axNEOS.step(x_ax[set],ratio, where = 'mid', label = 'Plane wave', zorder = 3, color = color2)
axNEOS.step(x_ax[set],ratio2, where = 'mid', label = 'Wave packet', zorder = 2, color = color3)
axNEOS.errorbar(x_ax[set],fitter.NEOSRatioData, yerr = fitter.NEOSRatioStatError, label = "NEOS data", fmt = "ok", zorder = 1, elinewidth = 1)
axNEOS.plot(x_ax[set],np.ones([fitter.n_bins[set]]),linestyle = 'dashed', color = 'k', zorder = 0.1)

axNEOS.set_xlabel(r"$E (\textrm{MeV})$", fontsize = 24)
axNEOS.set_ylabel(r"$P_{ee}^{\textrm{ste}}/P_{ee}^{\textrm{SM}}$", fontsize = 24)
# axNEOS.tick_params(axis='x', labelsize=13)
# axNEOS.tick_params(axis='y', labelsize=13)
axNEOS.grid(linestyle="--")
axNEOS.axis([1.2,7.0,0.9,1.1])
axNEOS.legend(loc="lower left",ncol = 2, fontsize=18, frameon = True, framealpha = 0.8)

figNEOS.suptitle(r'$\textrm{NEOS},\ \Delta m^2_{41} = %.2f \textrm{ eV}^2$, $\sin^2 2\theta_{14} = %.2f$'%(dm2,sin2))
figNEOS.savefig(plotdir+"NEOSRatio/NEOSRatio_Paper.png")
figNEOS.savefig(plotdir+"NEOSRatio/NEOSRatio_Paper.pdf")



# ----------------------------------------------
# EVENT EXPECTATIONS RATIO, HEAVY STERILE VS SM
# -----------------------------------------------------
# ONLY NEOS (FOR POSTER)
# ---------

margins = dict(left=0.15, right=0.97,bottom=0.15, top=0.9)
figNEOS,axNEOS = plt.subplots(1,1,figsize = (8,6),gridspec_kw=margins)


set = 'NEOS'
axNEOS.step(x_ax[set],ratio, where = 'mid', lw = 3, label = 'Plane wave', zorder = 3, color = color2)
axNEOS.step(x_ax[set],ratio2, where = 'mid', lw = 3, label = 'Wave packet', zorder = 2, color = color3)
axNEOS.errorbar(x_ax[set],fitter.NEOSRatioData, yerr = fitter.NEOSRatioStatError, label = "NEOS data", fmt = "ok", zorder = 1, elinewidth = 1)
axNEOS.plot(x_ax[set],np.ones([fitter.n_bins[set]]),linestyle = 'dashed', color = 'k', zorder = 0.1)

axNEOS.set_xlabel(r"$E (\textrm{MeV})$", fontsize = 24)
axNEOS.set_ylabel(r"$P_{ee}^{\textrm{ste}}/P_{ee}^{\textrm{SM}}$", fontsize = 24)
# axNEOS.tick_params(axis='x', labelsize=13)
# axNEOS.tick_params(axis='y', labelsize=13)
# axNEOS.grid(linestyle="--")
axNEOS.axis([1.2,7.0,0.9,1.1])
axNEOS.set_yticks([0.9,0.95,1.,1.05,1.1])
axNEOS.legend(loc="lower left",ncol = 2, fontsize=18, frameon = True, framealpha = 0.8)

figNEOS.suptitle(r'$\textrm{NEOS},\ \Delta m^2_{41} = %.2f \textrm{ eV}^2$, $\sin^2 2\theta_{14} = %.2f$'%(dm2,sin2))
figNEOS.savefig(plotdir+"NEOSRatio/NEOSRatio_Poster.png")
# figNEOS.savefig("Figures/NEOSRatio/NEOSRatio_Paper.pdf")
