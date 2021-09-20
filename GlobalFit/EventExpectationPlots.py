import numpy as np
import time
import GlobalFit as GF
import Models

import matplotlib.pyplot as plt

# cwd = os.getcwd()
# path_to_style=cwd+'/Figures'
# plt.style.use(path_to_style+r"/paper.mplstyle")
# matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

fitter = GF.GlobalFit()
Model_noosc = Models.NoOscillations()
Model_osc = Models.PlaneWaveSM()
Model_coh = Models.WavePacketSM()
Model_ste = Models.PlaneWaveSterile(Sin22Th14 = 8.29000e-03, DM2_41 = 6.70700e-02)
Model_ste = Models.PlaneWaveSterile(Sin22Th14 = 0.6, DM2_41 = 6.)

# -------------------------------------------------------
# Event expectations
# -------------------------------------------------------

x_ax = (fitter.DataLowerBinEdges+fitter.DataUpperBinEdges)/2
deltaE = (fitter.DataUpperBinEdges-fitter.DataLowerBinEdges)

figev,axev = plt.subplots(1,4,figsize = (25,8),gridspec_kw=dict(left=0.05, right=0.98,bottom=0.1, top=0.91))


begin_time = time.time()
predDB = fitter.get_expectation(Model_osc)
pred = fitter.get_expectation(Model_ste,do_we_integrate_DB = True,do_we_average_DB = False, do_we_integrate_NEOS = True,do_we_average_NEOS = False)
end_time = time.time()
print(begin_time-end_time)

# print(pred['EH1'][:,0][0],pred['EH1'][:,0][1])
axis = [[1.3,6.9,0.,3.5],[1.3,6.9,0.,3.],[1.3,6.9,0.,0.9],[1.3,6.9,0.,0.6]]
norm = [1e5,1e5,1e5,1e3]

# print(pred[fitter.sets_names[3]][:,1]/deltaE/norm[3])

for i in range(4):
    axev[i].errorbar(x_ax,pred[fitter.sets_names[i]][:,0]/deltaE/norm[i], yerr = pred[fitter.sets_names[i]][:,1]/deltaE/norm[i], xerr = 0.1, label = "Our prediction", fmt = "_", elinewidth = 2)
    axev[i].scatter(x_ax,fitter.AllData[fitter.sets_names[i]][:,0]/deltaE/norm[i], label = "{} data".format(fitter.sets_names[i]),color = "black")
    # axev[i].scatter(x_ax,pred[1][DB_test.sets_names[i]][:,0]/deltaE/1.e5,marker="+",color = "blue", label = "Our no oscillations")
    # axev[i].scatter(x_ax,DB_test.AllData[DB_test.sets_names[i]][:,5]/deltaE/1.e5,marker="+",color = "red", label = "DB no oscillations")
    axev[i].set_xlabel("Energy (MeV)", fontsize = 16)
    axev[i].set_ylabel("Events/(MeV$\ \cdot 10^{%i}$)"%(np.log10(norm[i])), fontsize = 16)
    axev[i].tick_params(axis='x', labelsize=13)
    axev[i].tick_params(axis='y', labelsize=13)
    axev[i].axis(axis[i])
    axev[i].grid(linestyle="--")
    axev[i].title.set_text(fitter.sets_names[i])
    axev[i].legend(loc="upper right",fontsize=16)

# figev.suptitle(r'Our best fit: $\Delta m^2_{13} = 2.5·10^{-3} eV^2$, $\sin^2 2\theta_{13} = 0.07821$', fontsize = 17)
# figev.suptitle(r'DB best fit: $\Delta m^2_{13} = 2.4·10^{-3} eV^2$, $\sin^2 2\theta_{13} = 0.0841$', fontsize = 17)
figev.suptitle(r'Sterile with $\Delta m^2_{41} = 6 eV^2$, $\sin^2 2\theta_{13} = 0.6$.', fontsize = 17)
figev.savefig("Figures/EventExpectation_big_ste_vegas.png")
# As we can see, both ways of computing the event expectations give the same result.


# ----------------------------------------------
# EVENT EXPECTATIONS RATIO, HEAVY STERILE VS SM
# -----------------------------------------------

x_ax = (fitter.DataLowerBinEdges+fitter.DataUpperBinEdges)/2
deltaE = (fitter.DataUpperBinEdges-fitter.DataLowerBinEdges)

figev,axev = plt.subplots(1,4,figsize = (25,8),gridspec_kw=dict(left=0.05, right=0.98,bottom=0.1, top=0.91))

# print(pred['EH1'][:,0][0],pred['EH1'][:,0][1])
axis = [[1.3,6.9,0.,3.5],[1.3,6.9,0.,3.],[1.3,6.9,0.,0.9],[1.3,6.9,0.,0.6]]
norm = [1e5,1e5,1e5,1e3]

# print(pred[fitter.sets_names[3]][:,1]/deltaE/norm[3])

for i in range(4):
    ste_dat = pred[fitter.sets_names[i]][:,0]
    SM_dat = predDB[fitter.sets_names[i]][:,0]
    ste_err = pred[fitter.sets_names[i]][:,1]
    SM_err = predDB[fitter.sets_names[i]][:,1]
    axev[i].errorbar(x_ax,ste_dat/SM_dat, yerr = ste_err/SM_dat + ste_dat*SM_err/SM_dat**2, xerr = 0.1, label = "Heavy sterile/SM", fmt = "_", elinewidth = 2)
    axev[i].plot(x_ax,np.ones([28]),linestyle = 'dashed')
    # axev[i].scatter(x_ax,pred[1][DB_test.sets_names[i]][:,0]/deltaE/1.e5,marker="+",color = "blue", label = "Our no oscillations")
    # axev[i].scatter(x_ax,DB_test.AllData[DB_test.sets_names[i]][:,5]/deltaE/1.e5,marker="+",color = "red", label = "DB no oscillations")
    axev[i].set_xlabel("Energy (MeV)", fontsize = 16)
    axev[i].set_ylabel("Events/(MeV$\ \cdot 10^{%i}$)"%(np.log10(norm[i])), fontsize = 16)
    axev[i].tick_params(axis='x', labelsize=13)
    axev[i].tick_params(axis='y', labelsize=13)
    # axev[i].axis(axis[i])
    axev[i].grid(linestyle="--")
    axev[i].title.set_text(fitter.sets_names[i])
    axev[i].legend(loc="upper right",fontsize=16)

# figev.suptitle(r'Our best fit: $\Delta m^2_{13} = 2.5·10^{-3} eV^2$, $\sin^2 2\theta_{13} = 0.07821$', fontsize = 17)
# figev.suptitle(r'DB best fit: $\Delta m^2_{13} = 2.4·10^{-3} eV^2$, $\sin^2 2\theta_{13} = 0.0841$', fontsize = 17)
figev.suptitle(r'SM vs Sterile with $\Delta m^2_{41} = 6 eV^2$, $\sin^2 2\theta_{13} = 0.6$.', fontsize = 17)
figev.savefig("Figures/EventRatio_big_ste_vegas.png")
# As we can see, both ways of computing the event expectations give the same result.



# ----------------------------------------------
# CHI2 per bin per experimental hall
# ----------------------------------------------

axis = [[1.3,6.9,0.,1.5],[1.3,6.9,0.,2.],[1.3,6.9,0.,5.5],[1.3,6.9,0.,0.6]]
chi2_per_exp = []
for exp in fitter.sets_names:
    evex = pred[exp][:,0]
    data = fitter.AllData[exp][:,0]
    chi2_per_exp.append(np.sum(-2*(data-evex+data*np.log(evex/data))))


figchi,axchi = plt.subplots(1,4,figsize = (25,8),gridspec_kw=dict(left=0.05, right=0.98,bottom=0.1, top=0.91))
for i in range(4):
    data = fitter.AllData[fitter.sets_names[i]][:,0]
    evex = pred[fitter.sets_names[i]][:,0]
    axchi[i].bar(x_ax,-2*(data-evex+data*np.log(evex/data)),width = 0.15)
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
#
# figchi.suptitle(r'DayaBay best fit (3 neutrino): $\Delta m^2_{ee} = 2.5\times 10^{-3} eV^2$, $\sin^2 2\theta_{13} = 0.0841$. Total $\chi^2 = 41.98$', fontsize = 17)
# figchi.suptitle(r'Sterile best fit (3+1): $\Delta m^2_{41} = 0.067 eV^2$, $\sin^2 2\theta_{13} = 8.29\times 10^{-3}$. Total $\chi^2 = 39.16$', fontsize = 17)
figchi.suptitle(r'Sterile with $\Delta m^2_{41} = 6 eV^2$, $\sin^2 2\theta_{13} = 0.6$. Total $\chi^2 = %.2f $'%(np.sum(chi2_per_exp)), fontsize = 17)
figchi.savefig("Figures/Chi2_big_ste_vegas.png")
