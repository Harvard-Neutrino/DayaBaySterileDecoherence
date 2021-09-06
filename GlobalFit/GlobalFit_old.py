import sys
import os
import numpy as np
sys.path.append(os.getcwd()+"/NEOS")
sys.path.append(os.getcwd()+"/Python implementation")

import NEOS
import DayaBay as DB
import Models
import GlobalFitData as GFD

class GlobalFit:
    def __init__(self):
        self.sets_names = ['EH1','EH2','EH3','NEOS']
        self.n_bins = GFD.number_of_bins
        self.DataLowerBinEdges = GFD.datlowerbin
        self.DataUpperBinEdges = GFD.datupperbin
        self.AllData = GFD.all_data

        self.DBexp = DB.DayaBay()
        self.NEOSexp = NEOS.Neos()

    def get_data_lower_bin_edges(self):
        """ Deprecated. """
        data_bins = dict([(set_name,self.DBexp.DataLowerBinEdges)
                        for set_name in self.DBexp.sets_names])
        data_bins.update({'NEOS':self.NEOSexp.DataLowerBinEdges})
        return data_bins

    def get_data_upper_bin_edges(self):
        """ Deprecated. """
        data_bins = dict([(set_name,self.DBexp.DataUpperBinEdges)
                        for set_name in self.DBexp.sets_names])
        data_bins.update({'NEOS':self.NEOSexp.DataUpperBinEdges})
        return data_bins

    def get_deltaE(self):
        """ Deprecated. """
        deltaE = dict([(set_name,self.DBexp.DataUpperBinEdges-self.DBexp.DataLowerBinEdges)
                        for set_name in self.DBexp.sets_names])
        deltaE.update({'NEOS':self.NEOSexp.DataUpperBinEdges-self.NEOSexp.DataLowerBinEdges})
        return deltaE

    def get_predicted_obs(self,model):
        obs_pred = self.DBexp.get_expectation_unnorm_nobkg(model,do_we_integrate = False)
        obs_pred.update(self.NEOSexp.get_expectation_unnorm_nobkg(model,do_we_integrate = False))
        return obs_pred

    def get_observed_data(self):
        """ Deprecated. """
        obs_data = dict([(set_name,self.DBexp.ObservedData[set_name])
                        for set_name in self.DBexp.sets_names])
        obs_data.update({'NEOS':self.NEOSexp.ObservedData['NEOS']})
        return obs_data

    def get_predicted_bkg(self):
        """ Deprecated. """
        bkg_data = dict([(set_name,self.DBexp.PredictedBackground[set_name])
                        for set_name in self.DBexp.sets_names])
        bkg_data.update({'NEOS':self.NEOSexp.PredictedBackground['NEOS']})
        return bkg_data

    def normalization_to_data(self,events):
        """
        Input:
        events: a dictionary with a string key for each experimental hall, linking to
        a numpy array for some histogram of events. In principle, it should be
        the number of expected number of events of our model.

        Output:
        norm: a normalisation factor with which to multiply events such that the total
        number of events of "events" is the same as the one from DB and NEOS data.
        """
        deltaE = self.get_deltaE() # Deprecated
        data_obs = self.get_observed_data() # Deprecated
        bkg_pred = self.get_predicted_bkg() # Deprecated


        TotalNumberOfExpEvents = dict([(set_name,np.sum(data_obs[set_name]*deltaE[set_name]))
                                            for set_name in self.sets_names])
        TotalNumberOfBkg = dict([(set_name,np.sum(bkg_pred[set_name]*deltaE[set_name]))
                                for set_name in self.sets_names])
        norm = dict([(set_name,(TotalNumberOfExpEvents[set_name]-TotalNumberOfBkg[set_name])/np.sum(events[set_name]))
                     for set_name in self.sets_names])
        return norm

    def get_nuissance_parameters(self,exp_events,obs_data,bkg_pred):
        """
        Input:
        events: a dictionary with a string key for each experimental hall, linking to
        a numpy array for some histogram of events. In principle, it should be
        the number of expected number of events of the model we want to study.

        Output:
        Returns the nuissance parameters which minimise the Poisson probability at all the
        experimental halls simultaneously (i.e. each nuissance is the same for all halls).
        They are computed as alpha_i = (sum(data_EHi)-sum(bkg_EHi))/(sum(exp_EHi))
        """
        TotalNumberOfEventsPerBin = 0.
        TotalNumberOfBkgPerBin = 0.
        TotalNumberOfExpEvents = 0.
        for set_name in self.sets_names:
            # set_name = 'EH1'
            TotalNumberOfEventsPerBin += obs_data[set_name]
            TotalNumberOfBkgPerBin += bkg_pred[set_name]
            TotalNumberOfExpEvents += exp_events[set_name]
        return (TotalNumberOfEventsPerBin-TotalNumberOfBkgPerBin)/(TotalNumberOfExpEvents)

    def get_expectation(self,model):
        exp_events = self.get_predicted_obs(model)
        norm = self.normalization_to_data(exp_events)
        exp_events = dict([(set_name,exp_events[set_name]*norm[set_name]) for set_name in self.sets_names])

        obs_data = self.get_observed_data()
        bkg_pred = self.get_predicted_bkg()
        nuissances = self.get_nuissance_parameters(exp_events,obs_data,bkg_pred)

        exp_events = dict([(set_name,exp_events[set_name]*nuissances+bkg_pred[set_name])
                                   for set_name in self.sets_names])

        data_lower_bin_edges = self.get_data_lower_bin_edges()
        data_upper_bin_edges = self.get_data_upper_bin_edges()

        model_expectations = dict([(set_name,np.array([(exp_events[set_name][i],np.sqrt(exp_events[set_name][i]),
                                                        data_lower_bin_edges[set_name][i],data_upper_bin_edges[set_name][i])
                                                        for i in range(0,self.n_bins)]))
                                  for set_name in self.sets_names])


GF = GlobalFit()
model = Models.PlaneWaveSM()
print(GF.get_expectation(model))
