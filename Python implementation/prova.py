import InverseBetaDecayCrossSection as IBD
import HuberMullerFlux as HMF
import DayaBay as DB
import Models

#flux_test = HMF.reactor_isotope_flux('U235',huber_muller)
#print(flux_test.isotope_name)
#print(flux_test.flux_parameters[flux_test.isotope_name])

#print(HMF.reactor_isotope_flux('U235',huber_muller).isotope_name)
#print(HMF.reactor_isotope_flux('U238',muller).flux_parameters)

#print(flux_test.GetFlux(0.1))

DB_test = DB.DayaBay()
#print(DB.DayaBay().num_bins())
#print(DB_test.set_ignore_oscillations(True))
#print(DB_test.are_we_ignoring_oscillations())
#print(DB_test.get_flux(0.1,'U235'))


#print(len(DB_test.get_resolution_matrix()))
#print(len(DB_test.get_resolution_matrix()[1]))


#print(DB_test.FindFineBinIndex(0.7))
#print(DB_test.FindFineBinIndex(1.1))
#print(len(DB_test.get_lower_neutrino_bin_edges()))
#print(DB_test.get_distance2('EH1','D1'))

def get_osc(model):
    return model.oscProbability(10,10)

#Model_test = Models.PlaneWaveSM()
Model_test = Models.NoOscillations()
#print(get_osc(Model_test))
print(DB_test.calculate_naked_event_expectation_integr(Model_test,'EH1',1))
print(DB_test.calculate_naked_event_expectation_simple(Model_test,'EH1',1))
# Petit problema, i = 0 dona ~21000, mentre que i = 1 dona ~17500. Molt menys,
# és raonable?
