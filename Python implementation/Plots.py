import InverseBetaDecayCrossSection as IBD
import HuberMullerFlux as HMF
import DayaBay as DB
import DayaBayParameters as DBpars
import Models
import numpy as np
import matplotlib.pyplot as plt


huber_U235 = DBpars.txt_to_array("Huber_U235.dat")
huber_U235 = huber_U235[:,0:2]

DB_data = {'EH1': DBpars.txt_to_array("DataEH1.dat"),
           'EH2': DBpars.txt_to_array("DataEH2.dat"),
           'EH3': DBpars.txt_to_array("DataEH3.dat")}

# ------------------------------------------------
# Emitted flux, cross-section and measured flux
# ------------------------------------------------

DB_test = DB.DayaBay()
Model_test = Models.NoOscillations()

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
axflux.plot(huber_U235[:,0],huber_U235[:,1], label = "Huber table", linewidth =4, linestyle = "-",color = "purple")
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

# -------------------------------------------------------
# Event expectations
# -------------------------------------------------------

x_ax = (DB_test.DataLowerBinEdges+DB_test.DataUpperBinEdges)/2
deltaE = (DB_test.DataUpperBinEdges-DB_test.DataLowerBinEdges)

data = np.array([43781,31619,40024,47052,53432,58628,63823,66905,67597,67761,66409,
                         63852,60164,55403,51728,47916,44952,41353,37622,33677,29657,25660,
                         21537,18159,15236,12603,9792,7447,5698,4250,3031,2154,1308,862,2873])


prediction = []
for set_name in DB_test.sets_names:
    ev_simp = np.array([DB_test.calculate_naked_event_expectation_simple(Model_test,set_name,i) for i in range(0,DB_test.n_bins)])
    sum = np.sum(ev_simp)
    DB_sum = np.sum(DB_data[set_name][:,5])
    bkg_sum = np.sum(DB_test.PredictedBackground[set_name])
    norm = (DB_sum-bkg_sum)/sum
    ev_simp *= norm
    ev_simp += np.array(DB_test.PredictedBackground[set_name])
    prediction.append(ev_simp)

prediction = np.array(prediction)


figev,axev = plt.subplots(1,3,figsize = (20,7),gridspec_kw=dict(left=0.05, right=0.98,bottom=0.1, top=0.96))

for i in range(3):
    axev[i].scatter(x_ax,prediction[i]/deltaE/1.e5,marker="+",color = "black", label = "Our prediction")
    axev[i].scatter(x_ax,DB_data[DB_test.sets_names[i]][:,5]/deltaE/1.e5,marker="+",color = "blue", label = "DB prediction")
    axev[i].set_xlabel("Energy (MeV)", fontsize = 16)
    axev[i].set_ylabel("Total number of events", fontsize = 16)
    axev[i].tick_params(axis='x', labelsize=13)
    axev[i].tick_params(axis='y', labelsize=13)
    # axev.axis([0.7,12,0.,2.5])
    axev[i].grid(linestyle="--")
    axev[i].legend(loc="upper right",fontsize=16)

figev.savefig("Event expectation.png")
# As we can see, both ways of computing the event expectations give the same result.
