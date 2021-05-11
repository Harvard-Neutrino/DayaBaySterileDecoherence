import InverseBetaDecayCrossSection as IBD
import HuberMullerFlux as HMF
import DayaBay as DB
import Models
import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------
# Emitted flux, cross-section and measured flux
# ------------------------------------------------

DB_test = DB.DayaBay()
Model_test = Models.PlaneWaveSM()

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

datx = np.arange(1.8,12,0.01)
datflux = [flux_detected(x)*1.e44 for x in datx]
datfluxe = [flux_emitted(x)*3 for x in datx]
datsig = [DB_test.get_cross_section(x)*1e41 for x in datx]

figflux, axflux = plt.subplots(figsize=(10,7))
axflux.plot(datx,datflux, label = "Detected flux ",linewidth = 4)
axflux.plot(datx,datfluxe, label = "Emitted flux", linewidth =3, linestyle = "--",color = "orange")
axflux.plot(datx,datsig, label = "Cross-section", linewidth =3, linestyle = "--",color = "green")
axflux.set_xlabel("Energy (MeV)", fontsize = 16)
axflux.set_ylabel("Arbitrary units", fontsize = 16)
axflux.tick_params(axis='x', labelsize=13)
axflux.tick_params(axis='y', labelsize=13)
axflux.axis([1.8,12,0.,2.5])
axflux.grid(linestyle="--")
axflux.legend(loc="upper right",fontsize=16)
figflux.savefig("Fluxes and cross-section.png")

# -------------------------------------------------------
# Event expectations
# -------------------------------------------------------

x_ax = (DB_test.DataLowerBinEdges+DB_test.DataUpperBinEdges)/2
ev_simp = [DB_test.calculate_naked_event_expectation_simple(Model_test,'EH1',i) for i in range(0,DB_test.n_bins)]
ev_inte = [DB_test.calculate_naked_event_expectation_integr(Model_test,'EH1',i) for i in range(0,DB_test.n_bins)]

figev,axev = plt.subplots(figsize = (10,7))
axev.scatter(x_ax,ev_simp,marker="+",color = "black", label = "Simple calculus")
axev.scatter(x_ax,ev_inte,marker="+",color = "red", label = "Integrated calculus")
axev.set_xlabel("Energy (MeV)", fontsize = 16)
axev.set_ylabel("Total number of events", fontsize = 16)
axev.tick_params(axis='x', labelsize=13)
axev.tick_params(axis='y', labelsize=13)
# axev.axis([0.7,12,0.,2.5])
axev.grid(linestyle="--")
axev.legend(loc="upper right",fontsize=16)
figev.savefig("Event expectation.png")
# As we can see, both ways of computing the event expectations give the same result.
