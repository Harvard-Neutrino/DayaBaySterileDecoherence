import sys
import os
import numpy as np
sys.path.append(os.getcwd()[:-10]+"/NEOS")
sys.path.append(os.getcwd()[:-10]+"/Python implementation")

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
        self.DataAllBinEdges   = np.append(GFD.datlowerbin,GFD.datupperbin[-1])
        self.DeltaE = self.DataUpperBinEdges - self.DataLowerBinEdges

        # For more information on the necessary format of the data, check GlobalFitData.py
        self.AllData = GFD.all_data
        self.ObservedData = dict([(set_name,self.AllData[set_name][:,0]) for set_name in self.sets_names])
        self.PredictedData = dict([(set_name,self.AllData[set_name][:,1]) for set_name in self.sets_names])
        self.PredictedBackground = dict([(set_name,self.AllData[set_name][:,2]) for set_name in self.sets_names])

        self.DBexp = DB.DayaBay()
        self.NEOSexp = NEOS.Neos()

    def get_predicted_obs(self,model,do_integral_DB = False, do_integral_NEOS = False, do_average_DB = False, do_average_NEOS = False):
        """ Returns the predicted observation (inside a dictionary) according to an oscillatory model. """
        obs_pred = self.DBexp.get_expectation_unnorm_nobkg(model,do_we_integrate = do_integral_DB,imin = 1,imax = self.n_bins+1,do_we_average = do_average_DB)
        obs_pred.update(self.NEOSexp.get_expectation_unnorm_nobkg(model,do_we_integrate = do_integral_NEOS,custom_bins = self.DataAllBinEdges, do_we_average = do_average_NEOS))
        return obs_pred


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
        TotalNumberOfExpEvents = dict([(set_name,np.sum(self.PredictedData[set_name]*self.DeltaE))
                                            for set_name in self.sets_names])
        TotalNumberOfBkg = dict([(set_name,np.sum(self.PredictedBackground[set_name]*self.DeltaE))
                                for set_name in self.sets_names])
        norm = dict([(set_name,(TotalNumberOfExpEvents[set_name]-TotalNumberOfBkg[set_name])/np.sum(events[set_name]))
                     for set_name in self.sets_names])
        return norm

    def get_nuissance_parameters(self,exp_events):
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
            TotalNumberOfEventsPerBin += self.ObservedData[set_name]
            TotalNumberOfBkgPerBin += self.PredictedBackground[set_name]
            TotalNumberOfExpEvents += exp_events[set_name]
        return (TotalNumberOfEventsPerBin-TotalNumberOfBkgPerBin)/(TotalNumberOfExpEvents)

    def get_expectation(self,model,do_we_integrate_DB = False, do_we_integrate_NEOS = False, do_we_average_DB = False, do_we_average_NEOS = False):
        """
        Input:
        model: a model from Models.py for which to compute the expected number of events.

        Output:
        A 2-tuple with the expectation from the model and from a model without oscillations.
        Each element of a tuple is a dictionary, where each key is an experimental hall.
        Such key links to a numpy array which contains: the histogram of expected events,
        the error bars of each bin, the lower limits of each bin, and the upper limits of each bin.
        The error bars are purely statistical, i.e. sqrt(N).
        """
        exp_events = self.get_predicted_obs(model,do_integral_DB = do_we_integrate_DB, do_integral_NEOS = do_we_integrate_NEOS,
                                                  do_average_DB  = do_we_average_DB,   do_average_NEOS =  do_we_average_NEOS)
        norm = self.normalization_to_data(exp_events)
        exp_events = dict([(set_name,exp_events[set_name]*norm[set_name]) for set_name in self.sets_names])

        nuissances = self.get_nuissance_parameters(exp_events)

        exp_events = dict([(set_name,exp_events[set_name]*nuissances+self.PredictedBackground[set_name])
                                   for set_name in self.sets_names])


        model_expectations = dict([(set_name,np.array([(exp_events[set_name][i],np.sqrt(exp_events[set_name][i]),
                                                        self.DataLowerBinEdges[i],self.DataUpperBinEdges[i])
                                                        for i in range(0,self.n_bins)]))
                                  for set_name in self.sets_names])

        return model_expectations

    def get_poisson_chi2(self,model,integrate_DB = False, integrate_NEOS = False,
                                    average_DB = False, average_NEOS = False):
        """
        Computes the chi2 value from the Poisson probability, taking into account
        every bin from every detector in the global fit.

        Input:
        model: a model from Models.py for which to compute the expected number of events.

        Output: (float) the chi2 value.
        """
        Exp = self.get_expectation(model,do_we_integrate_DB = integrate_DB, do_we_integrate_NEOS = integrate_NEOS,
                                         do_we_average_DB  =  average_DB,   do_we_average_NEOS =   average_NEOS)
        Data = self.ObservedData

        TotalLogPoisson = 0.0
        for set_name in self.sets_names:
            lamb = Exp[set_name][:,0]
            k = Data[set_name]
            TotalLogPoisson += (k - lamb + k*np.log(lamb/k))

        return -2*np.sum(TotalLogPoisson)



# GF = GlobalFit()
# model = Models.PlaneWaveSM()
# print(GF.get_expectation(model))
