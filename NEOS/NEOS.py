import sys
import os
homedir = os.path.realpath(__file__)[:-len('NEOS/NEOS.py')]
common_dir = 'Common_cython'
sys.path.append(homedir+common_dir)

import InverseBetaDecayCrossSection as IBD
import HuberMullerFlux as HMF
import Models

import NEOSParameters as NEOSP
import NEOSData as NEOSD

import numpy as np
from scipy import integrate as integrate


class Neos:

    # INITIALISATION OF THE CLASS
    # ---------------------------
    def __init__(self):
        """
        Initialises the class.
        No numeric information is found here. All parameters and data are
        gathered in files 'NEOSParameters.py' or 'NEOSData.py',
        in order to have a better organisation and to make tuning easier.
        For more information on these data, check the files.
        """
        # Physical quantities
        self.deltaEfine = 0.05 # in MeV. It is the resolution of the Etrue to Erec matrix.

        # In principle, our analysis is flux-free, i.e. independent of the flux.
        # Therefore, the total normalisation of the flux is not important.
        # However, we have a normalisation factor which normalises our data to the
        # same order than the observed data from NEOS.
        # The normalisation factor depends on the reference flux we are using.
        self.TotalNumberOfProtonsHM = 9.74503e51
        self.TotalNumberOfProtonsDB = 1.06406e06

        self.sets_names = NEOSP.exp_names
        self.reactor_names = NEOSP.reac_names
        self.isotopes_to_consider = NEOSP.isotopes
        self.mean_fission_fractions = NEOSP.mean_fis_frac

        self.EfficiencyOfHall = NEOSP.efficiency # Not quite useful in NEOS (only 1 detector), but yes in general.
        self.DistanceFromReactorToHall = NEOSP.distance
        self.WidthOfHall = NEOSP.width

        self.NeutrinoCovarianceMatrix = NEOSP.neutrino_covariance_matrix
        self.NeutrinoCorrelationMatrix = NEOSP.neutrino_correlation_matrix
        self.NeutrinoLowerBinEdges = NEOSP.nulowerbin
        self.NeutrinoUpperBinEdges = NEOSP.nuupperbin
        self.NeutrinoFlux = NEOSP.spectrum

        self.DataLowerBinEdges = NEOSD.datlowerbin
        self.DataUpperBinEdges = NEOSD.datupperbin
        self.DataCentrBinEdges = (NEOSD.datlowerbin+NEOSD.datupperbin)/2.
        self.DeltaEData = (self.DataUpperBinEdges-self.DataLowerBinEdges)
        self.n_bins = NEOSD.number_of_bins
        self.FromEtrueToErec = NEOSP.reconstruct_mat

        self.AllData = NEOSD.all_data
        self.PredictedData = NEOSD.predicted_data
        self.PredictedDataHM = NEOSD.predicted_data_HM
        self.ObservedData = NEOSD.observed_data
        self.PredictedBackground = NEOSD.predicted_bkg
        self.RatioData = NEOSD.ratio_data
        self.RatioStatError = NEOSD.ratio_stat_error
        self.RatioSystError = NEOSD.ratio_syst_error


    # FUNCTIONS TO MAKE HISTOGRAMS
    # -----------------------------------

    def get_true_energy_bin_centers(self):
        """
        This function divides the whole reconstructed energy spectrum in
        the energy intervals which the experiment is capable to resolve.
        The resolution is given by deltaEfine (here, 0.05 MeV).
        This energy will allow neutrino energies from 0.78 MeV to 12.78 MeV (after
        summing 0.78 MeV in the function), approximately.

        Output:
        returns a numpy array (the same length as the resolution
        matrix) with the center of the bins of the resolution of the experiment.
        """
        enutrue = []
        for i in range(0,len(self.FromEtrueToErec[1])):
            enutrue.append((i+0.5)*self.deltaEfine)
        return enutrue

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


    # FUNCTIONS TO GET FLUX, DISTANCES AND CROSS-SECTION
    # ---------------------------------------------------
    def get_cross_section(self,enu):
        """ Check InverseBetaDecayCrossSection.py for more info."""
        return IBD.CrossSection(enu)

    def get_flux(self,enu):
        """
        Input:
        enu (float): energy of the outgoing antineutrino.

        Output:
        The flux of neutrinos with energy enu coming from reactor,
        according to the DB data from table 12 at https://doi.org/10.1088/1674-1137/41/1/013002
        """
        if enu < 1.806: # MeV
            return 0
        else:
            x = (self.NeutrinoLowerBinEdges+self.NeutrinoUpperBinEdges)/2.
            y = self.NeutrinoFlux
            flux = np.interp(enu,x,y)
            return flux/self.get_cross_section(enu)

    def get_flux_HM(self, enu, isotope_name, flux_parameters = HMF.huber_muller):
        """
        Input:
        enu (double): energy of the outgoing antineutrino.
        isotope_name (str): name of the isotope which produces the nu.
        flux_parameters (dict): check 'HubberMullerFlux.py'.

        Output:
        The flux of neutrinos with energy enu coming from the isotope,
        according to the Huber-Mueller prediction.
        """
        flux = HMF.reactor_isotope_flux(isotope_name, flux_parameters)
        return flux.GetFlux(enu)

    def get_distance(self,experiment,reactor):
        """
        Input:
        experiment (str): name of the experimental hall.
        reactor (str): name of the reactor.

        Output:
        The distance between these EH and reactor (data in meters).
        """
        return self.DistanceFromReactorToHall[experiment][reactor]

    def get_width(self,experiment):
        """
        Input:
        experiment (str): name of the experimental hall.

        Output:
        The width (in meters) of such detector.
        """
        return self.WidthOfHall[experiment]


    # CALCULATION OF EXPECTED EVENTS
    # ------------------------------

    def calculate_naked_event_expectation_simple(self,model,set_name,i, bins = None, average = False, use_HM = True):
        """
        This function implements formula (A13).
        We don't neglect the width of the detector, and integrate over it.

        Input:
        model: a class containing the information of the model.
               Must contain a method oscProbability (+info on Models.py)
        set_name (str): name of the experimental hall studied.
        i (int): the data bin we want to compute the expectation of.
        bins: a numpy array of custom bins to calculate expectations with. Useful for GlobalFit.py
        average (bool): whether to pick oscProbability_av or not from the model.
        use_HM (bool): whether to use the HM or the DB fluxes (check get_flux and get_flux_HM).
        """
        expectation = 0.0

        # We want to know what are the fine reconstructed energies for which
        # we want to make events inside the data bin i.
        # We allow for the possibility of custom bins, using the argument bins
        # This is necessary for the GlobalFit (DB+NEOS) routine.
        if isinstance(bins,np.ndarray): #Check whether the user has introduced a numpy array of custom bins
            min_energy_fine_index = self.FindFineBinIndex(bins[:-1][i])
            max_energy_fine_index = self.FindFineBinIndex(bins[ 1:][i])
        else:
            min_energy_fine_index = self.FindFineBinIndex(self.DataLowerBinEdges[i])
            max_energy_fine_index = self.FindFineBinIndex(self.DataUpperBinEdges[i])

        W = self.get_width(set_name) # in meters, the detector total width is 2W
        ndL = 10 # We integrate through L with ten points. It is probably enough.
        dL = (2*W)/(ndL-1)

        for reactor in self.reactor_names:
            # There is only one reactor, so we will only do this loop once,
            # but this keeps full generality.
            Lmin = self.get_distance(set_name,reactor) - W
            Lmax = self.get_distance(set_name,reactor) + W

            for j in range(ndL):
                L = Lmin + j*dL

                for erf in range(min_energy_fine_index,max_energy_fine_index):

                    for etf in range(0,len(self.FromEtrueToErec[1])):
                        enu = (etf+0.5)*self.deltaEfine# + (
                              #DeltaNeutronToProtonMass - 2*ElectronMass)
                        if average == False:
                            oscprob = model.oscProbability(enu,L)
                        elif average == True:
                            oscprob = model.oscProbability_av(enu,L)

                        if use_HM == True:
                            # For the GlobalFit, it is necessary to use HM flux.
                            flux = np.sum(np.array([self.get_flux_HM(enu,isotope)*self.mean_fission_fractions[isotope]
                                                                             for isotope in self.isotopes_to_consider]))
                        else:
                            # For the NEOS only fit, we use the DB flux.
                            flux = self.get_flux(enu)


                        # Here we perform trapezoidal integration, the extremes contribute 1/2.
                        if ((etf == 0) or (etf == len(self.FromEtrueToErec[1])-1) or
                            (erf == min_energy_fine_index) or (erf == max_energy_fine_index-1) or
                            (j == 0) or (j == ndL - 1)):
                            expectation += (flux * self.get_cross_section(enu) *
                                            self.FromEtrueToErec[erf][etf] * oscprob)/L**2/2.
                        else:
                            expectation += (flux * self.get_cross_section(enu) *
                                        self.FromEtrueToErec[erf][etf] * oscprob)/L**2
                        # isotope loop ends

                    # real antineutrino energies loop ends
                # L distances loop ends
            # reconstructed energies loop ends

            # reactor loop ends
            # the two deltaEfine are to implement a trapezoidal numeric integration in etrue and erec
            # the dL is to implement a trapezoidal numeric integration in L
            # we divide by the total width 2W because we want an intensive quantity! It is an average, not a total sum.
        if use_HM == True:
            TotalNumberOfProtons = self.TotalNumberOfProtonsHM
        else:
            TotalNumberOfProtons = self.TotalNumberOfProtonsDB

        return expectation*self.deltaEfine**2*dL/(2*W)*self.EfficiencyOfHall[set_name]*TotalNumberOfProtons


    def integrand(self,enu,L,model,erf,etf, use_HM = True):
        """
        This function returns the integrand of formula (A13).

        Input:
        erf, etf: indices of reconstructed and true energies, in the response matrix.
        enu: true energy of the neutrino. This will be the parameter of integration.
        L: the length between experimental hall and reactor.
        model: the model with which to compute the oscillation probability.
        use_HM (bool): whether to use the HM or the DB fluxes (check get_flux and get_flux_HM).
        """
        # Computes the HM flux for all isotopes
        if use_HM == True:
            # For the GlobalFit, it is necessary to use HM flux.
            flux = np.sum(np.array([self.get_flux_HM(enu,isotope)*self.mean_fission_fractions[isotope]
                                    for isotope in self.isotopes_to_consider]))
        else:
            flux = self.get_flux(enu) # the flux from DB slows down the program A LOT, use with caution

        return (flux*
                self.get_cross_section(enu) *
                self.FromEtrueToErec[erf][etf] *
                model.oscProbability(enu,L))


    def calculate_naked_event_expectation_integr(self,model,set_name,i, bins = None, use_HM = True):
        """
        This function implements formula (A13).
        Here we don't neglect the width of the detector, and integrate over it.
        We also perform an integral inside the fine energy bins,
        to take into account possible rapid oscillations (e.g., a heavy sterile).

        Input:
        model: a class containing the information of the model.
               Must contain a method oscProbability (+info on Models.py)
        set_name (str): name of the experimental hall studied.
        bins: a numpy array of custom bins to calculate expectations with. Useful for GlobalFit.py
        i (int): the data bin we want to compute the expectation of.
        use_HM (bool): whether to use the HM or the DB fluxes (check get_flux and get_flux_HM).
        """
        expectation = 0.0
        # We want to know what are the fine reconstructed energies for which
        # we want to make events inside the data bin i.
        # We allow for the possibility of custom bins, using the argument bins
        if isinstance(bins,np.ndarray): #Check whether the user has introduced a numpy array of custom bins
            min_energy_fine_index = self.FindFineBinIndex(bins[:-1][i])
            max_energy_fine_index = self.FindFineBinIndex(bins[ 1:][i])
        else:
            min_energy_fine_index = self.FindFineBinIndex(self.DataLowerBinEdges[i])
            max_energy_fine_index = self.FindFineBinIndex(self.DataUpperBinEdges[i])

        W = self.get_width(set_name) # in meters, the detector total width is 2W
        ndL = 10 # We only integrate through L with three points. It is probably enough.
        dL = (2*W)/(ndL-1)


        for reactor in self.reactor_names:
            Lmin = self.get_distance(set_name,reactor) - W
            Lmax = self.get_distance(set_name,reactor) + W

            for j in range(ndL):
                L = Lmin + j*dL

                for erf in range(min_energy_fine_index,max_energy_fine_index):

                    for etf in range(0,len(self.FromEtrueToErec[1])):
                        enu_min = (etf)*self.deltaEfine
                        enu_max = (etf+1)*self.deltaEfine # in MeV

                        if ((erf == min_energy_fine_index) or (erf == max_energy_fine_index-1) or
                            (j == 0) or (j == ndL - 1)):
                            expectation += integrate.quad(self.integrand,enu_min,enu_max,
                                                          args=(L,model,erf,etf,use_HM))[0]/L**2/2
                        else:
                            expectation += integrate.quad(self.integrand,enu_min,enu_max,
                                                          args=(L,model,erf,etf,use_HM))[0]/L**2
                    # isotope loop ends

                # real antineutrino energies loop ends

            # reconstructed energies loop ends
            # only one trapezoidal numeric integration has been done

        # reactor loop ends
        if use_HM == True:
            TotalNumberOfProtons = self.TotalNumberOfProtonsHM
        else:
            TotalNumberOfProtons = self.TotalNumberOfProtonsDB
        return expectation*self.deltaEfine*dL/(2*W)*self.EfficiencyOfHall[set_name]*TotalNumberOfProtons


    def get_expectation_unnorm_nobkg(self,model,do_we_integrate = False,custom_bins = None, do_we_average = False, use_HM = True):
        """
        Computes the histogram of expected number of events without normalisation
        to the real data, and without summing the predicted background.

        Input:
        model: a Models.py class with which to compute oscillation probabilities.
        do_we_integrate: whether to integrate inside each energy bin or not.
        do_we_average: whether to use the oscProbability_av from the model.
        custom_bins: a numpy array of custom bins to calculate expectations with. Useful for GlobalFit.py
        use_HM (bool): whether to use the HM or the DB fluxes (check get_flux and get_flux_HM).

        Output:
        A dictionary with a string key for each experimental hall, linking to a
        numpy array with the expected events for each histogram bin.
        """
        if do_we_integrate == False:
            # Once we know we don't integrate, we check whether the user has introduced a custom binning
            if isinstance(custom_bins,np.ndarray):
                imax = len(custom_bins)-1
                Expectation = dict([(set_name,
                                     np.array([self.calculate_naked_event_expectation_simple(model,set_name,i,bins = custom_bins, average = do_we_average, use_HM = use_HM) for i in range(0,imax)]))
                                    for set_name in self.sets_names])
            else:
                Expectation = dict([(set_name,
                                     np.array([self.calculate_naked_event_expectation_simple(model,set_name,i, use_HM = use_HM) for i in range(0,self.n_bins)]))
                                    for set_name in self.sets_names])

        if do_we_integrate == True:
            # Once we know we do integrate, we check whether the user has introduced a custom binning
            if isinstance(custom_bins,np.ndarray):
                imax = len(custom_bins)-1
                Expectation = dict([(set_name,
                                     np.array([self.calculate_naked_event_expectation_integr(model,set_name,i,bins = custom_bins, use_HM = use_HM) for i in range(0,imax)]))
                                    for set_name in self.sets_names])
            else:
                Expectation = dict([(set_name,
                                     np.array([self.calculate_naked_event_expectation_integr(model,set_name,i, use_HM = use_HM) for i in range(0,self.n_bins)]))
                                    for set_name in self.sets_names])
        return Expectation


    def normalization_to_data(self,events):
        """
        Returns a normalization factor with which to normalise the expected events
        to the observed data from NEOS.

        Input:
        events: a dictionary with a string key for each experimental hall, linking to
        a numpy array for some histogram of events. In principle, it should be
        the number of expected number of events of our model.

        Output:
        norm: a normalisation factor with which to multiply events such that the total
        number of events of "events" is the same as the one from NEOS data.
        """

        TotalNumberOfEvents = dict([(set_name,np.sum(self.ObservedData[set_name]))
                                     for set_name in self.sets_names])
        TotalNumberOfBkg = dict([(set_name,np.sum(self.PredictedBackground[set_name]))
                                  for set_name in self.sets_names])

        norm = dict([(set_name,(TotalNumberOfEvents[set_name]-TotalNumberOfBkg[set_name])/np.sum(events[set_name]))
                     for set_name in self.sets_names])

        # print('norm: ', norm)
        return norm


    def get_expectation(self,model, integrate = False, average = False, use_HM = True):
        """
        This function is only called in the NEOS only fit.

        Input:
        model: a model from Models.py for which to compute the expected number of events.
        integrate: whether to integrate inside each energy bin or not.
        average: whether to use the oscProbability_av from the model.
        use_HM (bool): whether to use the HM or the DB fluxes (check get_flux and get_flux_HM).

        Output:
        A 2-tuple with the expectation from the model and from a model without oscillations.
        Each element of a tuple is a dictionary, where each key is an experimental hall.
        Such key links to a numpy array which contains: the histogram of expected events,
        the error bars of each bin, the lower limits of each bin, and the upper limits of each bin.
        The error bars are purely statistical, i.e. sqrt(N).
        """

        # We build the expected number of events for our model and we roughly normalise so that is of the same order of the data.
        exp_events = self.get_expectation_unnorm_nobkg(model,do_we_integrate = integrate, do_we_average = average, use_HM = use_HM)

        norm = self.normalization_to_data(exp_events) # This should only be allowed for NEOS only fit.
        exp_events = dict([(set_name,exp_events[set_name]*norm[set_name] +self.PredictedBackground[set_name]) for set_name in self.sets_names])

        # For the NEOS single fit, there are no nuissance parameters. We just return the data.
        model_expectations = dict([(set_name,np.array([(exp_events[set_name][i],np.sqrt(exp_events[set_name][i]),
                                                        self.DataLowerBinEdges[i],self.DataUpperBinEdges[i])
                                                        for i in range(0,self.n_bins)]))
                                  for set_name in self.sets_names])

        return model_expectations



# ----------------------------------------------------------
# BUILDING THE COVARIANCE MATRIX
# ----------------------------------------------------------

    def get_total_covariance_matrix(self):
        """
        Returns a numpy array with the total covariance matrix in prompt energy,
        taking into account both systematic and statistical errors,
        normalised to match the systematic errors from figure 3(c) in 1610.05134.
        Therefore, it is not normalised to N.
        For more information on the correlation matrix, check NEOSParameters.py

        This function is not in use, just as a demonstration on how one can
        construct the covariance matrix from the correlation matrix and the stat. errors.
        Instead, we simply use the data from Data/NeutrinoCovMatrix.dat, which has been
        obtained using this same function.
        """
        corr_mat = self.NeutrinoCorrelationMatrix

        # We load σi, the square root of the diagonal elements of the systematic part of the cov matrix.
        syst_err = self.RatioSystError['NEOS']
        # We compute all elements of the covariance matrix as Vij = σi·σj
        syst_err = np.tile(syst_err,(len(syst_err),1))*(np.tile(syst_err,(len(syst_err),1)).transpose())

        # Statistical errors are only in the diagonal
        stat_err = self.RatioStatError['NEOS']**2*np.identity(self.n_bins)

        # np.savetxt('Data/NeutrinoCovMatrix.dat',corr_mat*syst_err+stat_err,delimiter = ',')
        return corr_mat*syst_err+stat_err

    def get_inverse_covariance_matrix(self):
        """
        Returns the inverse total covariance matrix.
        For more information, check the function get_total_covariance_matrix.
        """
        return np.linalg.inv(np.array(self.NeutrinoCovarianceMatrix))




# ----------------------------------------------------------
# FITTING THE DATA
# ----------------------------------------------------------

    def get_chi2(self,model, do_we_integrate = False, do_we_average = False, use_HM = True):
        """
        Computes the "chi2" value according to formula (A12), that is,
        using the ratio of the expected events to DB in figure 3(c) of 1610.05134,
        using the covariance matrix provided in NEOSParameters.py.

        Input:
        model: a model from Models.py for which to compute the expected number of events
        do_we_integrate: whether to integrate inside each energy bin or not.
        do_we_average: whether to use the oscProbability_av from the model.
        use_HM (bool): whether to use the HM or the DB fluxes (check get_flux and get_flux_HM).

        Output: (float) the chi2 value.
        """

        # Computes the expectation according to the model.
        Exp = self.get_expectation(model, integrate = do_we_integrate, average = do_we_average, use_HM = use_HM)['NEOS']

        # Computes the expectation according to the the SM.
        # For full rigorosity, this should be PlaneWaveSM or WavePacketSM depending on the introduced model.
        # However, the difference between both probabilities is very small and we can neglect it.
        modelSM = Models.PlaneWaveSM()
        ExpSM = self.get_expectation(modelSM, use_HM = use_HM)['NEOS']

        # Computes the theoretical ratio and compares it to the data in 3(c).
        teo = Exp[:,0]/ExpSM[:,0]
        ratio = self.RatioData['NEOS']
        Vinv = self.get_inverse_covariance_matrix()

        return (teo-ratio).dot(Vinv.dot(teo-ratio))
