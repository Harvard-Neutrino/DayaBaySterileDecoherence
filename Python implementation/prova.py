import InverseBetaDecayCrossSection as IBD
import HuberMullerFlux as HMF
import DayaBay as DB
import DayaBayData as DBdata
import Models
import numpy as np
import matplotlib.pyplot as plt
import time

flux_test = HMF.reactor_isotope_flux('U235',HMF.huber_muller)
#print(flux_test.isotope_name)
#print(flux_test.flux_parameters[flux_test.isotope_name])

#print(HMF.reactor_isotope_flux('U235',huber_muller).isotope_name)
#print(HMF.reactor_isotope_flux('U238',muller).flux_parameters)

#print(flux_test.GetFlux(1.))

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

#print(DBdata.all_data['EH1'][:,0:7])

def get_osc(model):
    return model.oscProbability(10,10)

DB_test = DB.DayaBay()
Model_noosc = Models.NoOscillations()
Model_osc = Models.PlaneWaveSM()
Model_full = Models.PlaneWaveSM_full()
#Model_test = Models.WavePacketSterile_full()
#print(get_osc(Model_test))



# exp_events_noosc = DB_test.get_expectation_unnorm_nobkg(Model_noosc)
# norm_factor = DB_test.get_normalization(exp_events_noosc)
# exp_events_noosc = dict([(set_name, exp_events_noosc[set_name]*norm_factor[set_name]+DB_test.PredictedBackground[set_name]) for set_name in DB_test.sets_names])
# print(exp_events_noosc)

begin_time = time.time()
print(DB_test.get_survival(Model_osc))
print(DB_test.get_data_survival())
#data = DB_test.get_expectation(Model_osc)
end_time = time.time()
print(begin_time-end_time)


# print(DB_test.calculate_naked_event_expectation_simple(Model_test,'EH1',1))
# Petit problema, i = 0 dona ~21000, mentre que i = 1 dona ~17500. Molt menys,
# és raonable?

# print(DB_test.get_data()[:,:,0])
# print(DB_test.get_expectation(Model_test))
#print(DB_test.get_chi2(Model_test))
