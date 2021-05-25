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
datflux = [flux_detected(x)*1e43 for x in datx]
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
deltaE = (DB_test.DataUpperBinEdges-DB_test.DataLowerBinEdges)
print(deltaE)
ev_simp = [DB_test.calculate_naked_event_expectation_simple(Model_test,'EH1',i) for i in range(0,DB_test.n_bins)]
#ev_inte = [DB_test.calculate_naked_event_expectation_integr(Model_test,'EH1',i) for i in range(0,DB_test.n_bins)]
data = np.array([43781,31619,40024,47052,53432,58628,63823,66905,67597,67761,66409,
                         63852,60164,55403,51728,47916,44952,41353,37622,33677,29657,25660,
                         21537,18159,15236,12603,9792,7447,5698,4250,3031,2154,1308,862,2873])

figev,axev = plt.subplots(figsize = (10,7))
axev.scatter(x_ax,ev_simp/deltaE/1.e5,marker="+",color = "black", label = "Simple calculus")
#axev.scatter(x_ax,ev_inte,marker="+",color = "red", label = "Integrated calculus")
#axev.scatter(x_ax,data,marker="+",color = "red", label = "Data")
axev.scatter(x_ax,data/deltaE/1.e5,marker="+",color = "red", label = "Data")
axev.set_xlabel("Energy (MeV)", fontsize = 16)
axev.set_ylabel("Total number of events", fontsize = 16)
axev.tick_params(axis='x', labelsize=13)
axev.tick_params(axis='y', labelsize=13)
# axev.axis([0.7,12,0.,2.5])
axev.grid(linestyle="--")
axev.legend(loc="upper right",fontsize=16)
figev.savefig("Event expectation.png")
# As we can see, both ways of computing the event expectations give the same result.
