import time
import InverseBetaDecayCrossSection as IBD
import HuberMullerFlux as HMF
import DayaBay as DB
import DayaBayParameters as DBpars
import DayaBayData as DBdata
import Models
import numpy as np
import matplotlib.pyplot as plt


DB_test = DB.DayaBay()
Model_noosc = Models.NoOscillations()
Model_osc = Models.PlaneWaveSM()
Model_full = Models.PlaneWaveSM_full()

# ------------------------------------------------
# PROBABILITIES
# Makes plot comparing the survival probability under different models
# ------------------------------------------------

datx = np.arange(0.7,12,0.01)
DBosc = [Model_osc.oscProbability(x,1265.61) for x in datx]
SMosc = [Model_full.oscProbability(x,1265.61) for x in datx]

figprob, axprob = plt.subplots(figsize=(10,7))
axprob.plot(datx,DBosc,label="DB")
axprob.plot(datx,SMosc,label="SM")
axprob.legend(loc="upper right",fontsize=16)
figprob.savefig("Probabilities.png")


# ------------------------------------------------
# PROBABILITIES
# Makes plot comparing the survival probability under different models
# ------------------------------------------------

datx = np.arange(0.7,12,0.01)
Long = [Model_osc.oscProbability(x,1265.61) for x in datx]
Short = [Model_osc.oscProbability(x,500.141) for x in datx]
Mid = [Model_osc.oscProbability(x,1000.141) for x in datx]

figprob, axprob = plt.subplots(figsize=(10,7))
axprob.plot(datx,Long,label="Long")
axprob.plot(datx,Short,label="Short")
axprob.plot(datx,Mid,label="Mid")

axprob.legend(loc="upper right",fontsize=16)
figprob.savefig("Probabilities_dif_baselines.png")

# ------------------------------------------------
# FLUXES AND CROSS-SECTIONS
# Allows to plot the comparison between the IBD cross-section, and emitted and detected fluxes.
# Also allows to compare between different isotopes.
# ------------------------------------------------

def flux_detected(enu):
    flux = 0.0
    for isotope in DB_test.isotopes_to_consider:
        flux += DB_test.mean_fission_fractions[isotope]*DB_test.get_flux(enu,isotope)*DB_test.get_cross_section(enu)
    return flux

def flux_emitted(enu):
    flux = 0.0
    for isotope in DB_test.isotopes_to_consider:
        flux += DB_test.mean_fission_fractions[isotope]*DB_test.get_flux(enu,isotope)
    return flux

def flux_isotope(enu,i):
    flux = DB_test.get_flux(enu,i)
    #flux *= DB_test.mean_fission_fractions[i]
    return flux


datx = np.arange(1.,8,0.01)
datflux = [flux_detected(x)*1e43 for x in datx]
datfluxe = [flux_emitted(x) for x in datx]
datU235 = [flux_isotope(x,'U235') for x in datx]
datU238 = [flux_isotope(x,'U238') for x in datx]
datP239 = [flux_isotope(x,'PU239') for x in datx]
datP241 = [flux_isotope(x,'PU241') for x in datx]
datsig = [DB_test.get_cross_section(x)*1e41 for x in datx]

figflux, axflux = plt.subplots(figsize=(10,7))
#axflux.plot(datx,datflux, label = "Detected flux ",linewidth = 4)
axflux.plot(datx,datfluxe, label = "Emitted flux", linewidth =3, linestyle = "--",color = "orange")
#axflux.plot(datx,datsig, label = "Cross-section", linewidth =3, linestyle = "--",color = "green")
axflux.plot(datx,datU235, label = "U235", linewidth =3, linestyle = "--",color = "blue")
axflux.plot(datx,datU238, label = "U238", linewidth =3, linestyle = "--",color = "green")
axflux.plot(datx,datP239, label = "PU239", linewidth =3, linestyle = "--",color = "red")
axflux.plot(datx,datP241, label = "PU241", linewidth =3, linestyle = "--",color = "black")
plt.yscale("log")
axflux.set_xlabel("Energy (MeV)", fontsize = 16)
axflux.set_ylabel("Arbitrary units", fontsize = 16)
axflux.tick_params(axis='x', labelsize=13)
axflux.tick_params(axis='y', labelsize=13)
axflux.axis([1.,8,0.0025,2.5])
axflux.grid(linestyle="--")
axflux.legend(loc="upper right",fontsize=16)
figflux.savefig("Fluxes and cross-section.png")
