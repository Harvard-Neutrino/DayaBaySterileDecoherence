import sys
import os
common_dir = '/Common_cython'
sys.path.append(os.getcwd()[:-10]+common_dir)
sys.path.append(os.getcwd()[:-10]+"/NEOS")
sys.path.append(os.getcwd()[:-10]+"/DayaBay")

import NEOS
import DayaBay as DB
import Models
import GlobalFitData as GFD
import numpy as np


class GlobalFit:

    # INITIALISATION OF THE CLASS
    # ---------------------------
    def __init__(self):
        self.sets_names = ['EH1','EH2','EH3','NEOS']
        self.deltaEfine = 0.05 # in MeV. It is the resolution of the Etrue to Erec matrix

        # We allow for different number of bins between different experiments.
        # However, the edges of the bins should coincide.
        # For example, in the present case, there's two NEOS bins for each of DB.
        self.n_bins = GFD.number_of_bins
        self.DataLowerBinEdges = GFD.datlowerbin
        self.DataUpperBinEdges = GFD.datupperbin
        self.DataAllBinEdges   = GFD.datallbin
        self.DeltaE = GFD.deltaE


        self.NeutrinoLowerBinEdges = GFD.neutrino_lower_bin_edges
        self.NeutrinoUpperBinEdges = GFD.neutrino_upper_bin_edges
        self.NeutrinoCovarianceMatrix = GFD.neutrino_cov_mat
        self.FromEtrueToErec = GFD.reconstruct_mat

        # For more information on the necessary format of the data, check GlobalFitData.py
        self.AllData = GFD.all_data
        self.ObservedData = dict([(set_name,self.AllData[set_name][:,0]) for set_name in self.sets_names])
        self.PredictedData = dict([(set_name,self.AllData[set_name][:,1]) for set_name in self.sets_names])
        self.PredictedBackground = dict([(set_name,self.AllData[set_name][:,2]) for set_name in self.sets_names])

        # These are the objects which allow to compute the event expectations for each experiment.
        self.DBexp = DB.DayaBay()
        self.NEOSexp = NEOS.Neos()

    def FindFineBinIndex(self,energy):
        """
        Input:
        energy (float): prompt energy or neutrino true energy.

        Output:
        An integer telling us the index of the true_energy_bin_centers
        bin in which the input energy is found.
        """
        dindex = np.int(np.floor(energy/self.deltaEfine - 0.5))
        if dindex<0:
            return 0
        else:
            return dindex



    def get_predicted_obs(self,model,do_integral_DB = False, do_integral_NEOS = False, do_average_DB = False, do_average_NEOS = False):
        """
        Returns the predicted observation (inside a dictionary)
        according to an oscillatory model. No background, and no normalisation.

        Input:
        model: a class containing the information of the model.
               Must contain a method oscProbability (+info on Models.py)
        do_integral_DB et al: whether to integrate/average for DB/NEOS.
               For more information, check each class DayaBay.py and NEOS.py

        """
        obs_pred = self.DBexp.get_expectation_unnorm_nobkg(model,do_we_integrate = do_integral_DB,imin = 1,imax = self.n_bins['EH1']+1,do_we_average = do_average_DB)
        obs_pred.update(self.NEOSexp.get_expectation_unnorm_nobkg(model,do_we_integrate = do_integral_NEOS,custom_bins = self.DataAllBinEdges['NEOS'], do_we_average = do_average_NEOS))
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
        TotalNumberOfExpEvents = dict([(set_name,np.sum(self.PredictedData[set_name]))
                                            for set_name in self.sets_names])
        TotalNumberOfBkg = dict([(set_name,np.sum(self.PredictedBackground[set_name]))
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

        for set_name in self.sets_names[:-1]: # Not considering NEOS
            # We sum the contribution to the nuissance from DB.
            TotalNumberOfEventsPerBin += self.ObservedData[set_name]
            TotalNumberOfBkgPerBin += self.PredictedBackground[set_name]
            TotalNumberOfExpEvents += exp_events[set_name]

        # One has to consider that NEOS has twice as many bins.
        # We decided that two consecutive bin contributed to and had the same nuissance parameter.
        TotalNumberOfEventsPerBin += self.ObservedData['NEOS'][::2]+self.ObservedData['NEOS'][1::2]
        TotalNumberOfBkgPerBin += self.PredictedBackground['NEOS'][::2]+self.PredictedBackground['NEOS'][1::2]
        TotalNumberOfExpEvents += exp_events['NEOS'][::2]+exp_events['NEOS'][1::2]

        nuissances = (TotalNumberOfEventsPerBin-TotalNumberOfBkgPerBin)/(TotalNumberOfExpEvents)

        # As we said, each nuissance corresponds to two consecutive bins in NEOS.
        nuissancesdict = dict([(set_name,nuissances) for set_name in self.sets_names[:-1]])
        nuissancesdict.update({'NEOS':np.repeat(nuissances,2)})

        return nuissancesdict

    def get_expectation(self,model,do_we_integrate_DB = False, do_we_integrate_NEOS = False, do_we_average_DB = False, do_we_average_NEOS = False):
        """
        Input:
        model: a model from Models.py for which to compute the expected number of events.
        do_we_integrate_DB et al: whether to integrate/average for DB/NEOS.
               For more information, check each class DayaBay.py and NEOS.py

        Output:
        A 2-tuple with the expectation from the model and from a model without oscillations.
        Each element of a tuple is a dictionary, where each key is an experimental hall.
        Such key links to a numpy array which contains: the histogram of expected events,
        the error bars of each bin, the lower limits of each bin, and the upper limits of each bin.
        The error bars are purely statistical, i.e. sqrt(N).
        """
        exp_events = self.get_predicted_obs(model,do_integral_DB = do_we_integrate_DB, do_integral_NEOS = do_we_integrate_NEOS,
                                                  do_average_DB  = do_we_average_DB,   do_average_NEOS =  do_we_average_NEOS)

        # We normalise the data from each experimental hall to its own data.
        # norm = self.normalization_to_data(exp_events) # Huge mistake
        # exp_events = dict([(set_name,exp_events[set_name]*norm[set_name]) for set_name in self.sets_names])

        # We compute and apply the nuissances parameters to all experiments.
        nuissances = self.get_nuissance_parameters(exp_events)
        exp_events = dict([(set_name,exp_events[set_name]*nuissances[set_name]+self.PredictedBackground[set_name])
                                   for set_name in self.sets_names])

        # We build the result inside a dictionary.
        model_expectations = dict([(set_name,np.array([(exp_events[set_name][i],np.sqrt(exp_events[set_name][i]),
                                                        self.DataLowerBinEdges[set_name][i],self.DataUpperBinEdges[set_name][i])
                                                        for i in range(0,self.n_bins[set_name])]))
                                  for set_name in self.sets_names])

        return model_expectations



# ----------------------------------------------------------
# FITTING THE DATA
# ----------------------------------------------------------

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
            TotalLogPoisson += np.sum(k - lamb + k*np.log(lamb/k))

        return -2*TotalLogPoisson


# ----------------------------------------------------------
# FITTING THE DATA WITH THE COVARIANCE MATRIX
# ----------------------------------------------------------

    def get_inverse_flux_covariance(self):
        """
        Returns the inverse of the neutrino covariance matrix, V^-1.
        The output is a dictionary with the different covariances matrices for every experiment.
        """
        vinv = dict([(set_name,np.linalg.inv(np.array(self.NeutrinoCovarianceMatrix[set_name])))
                      for set_name in self.sets_names])

        return vinv

    def get_resolution_matrix_underdim(self):
        """
        Returns an underdimension of the response matrix.
        The output is a dictionary with the different response matrices for every experiment.
        """
        resmats = {}
        for set_name in self.sets_names:
            mat = np.zeros((len(self.NeutrinoLowerBinEdges[set_name]),self.n_bins[set_name]))
            for i in range(0,self.n_bins[set_name]):
                minrec = self.FindFineBinIndex(self.DataLowerBinEdges[set_name][i])
                maxrec = self.FindFineBinIndex(self.DataUpperBinEdges[set_name][i])
                for j in range(0,len(self.NeutrinoLowerBinEdges[set_name])):
                    mintrue = self.FindFineBinIndex(self.NeutrinoLowerBinEdges[set_name][j])
                    maxtrue = self.FindFineBinIndex(self.NeutrinoUpperBinEdges[set_name][j])
                    mat[j,i] = np.mean(self.FromEtrueToErec[set_name][mintrue:maxtrue,minrec:maxrec])
            resmats.update({set_name:mat})
        return resmats

    def get_chi2(self,model, integrate_DB = False, integrate_NEOS = False,
                             average_DB = False, average_NEOS = False):
        """
        Input: a  model with which to compute expectations.
        Output: a chi2 statistic comparing data and expectations.
        """
        U = self.get_resolution_matrix_underdim()
        Vinv = self.get_inverse_flux_covariance()

        Exp = self.get_expectation(model,do_we_integrate_DB = integrate_DB, do_we_integrate_NEOS = integrate_NEOS,
                                         do_we_average_DB  =  average_DB,   do_we_average_NEOS =   average_NEOS)
        Data = self.ObservedData
        chi2 = 0.
        for set_name in self.sets_names:
            exp_i = Exp[set_name][:,0]
            dat_i = Data[set_name]

            chi2 += (dat_i-exp_i).dot((U[set_name].transpose()).dot(Vinv[set_name].dot(U[set_name].dot(dat_i-exp_i))))

        return chi2
