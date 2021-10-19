import sys
import os
common_dir = '/Common_cython'
sys.path.append(os.getcwd()[:-5]+common_dir)

import InverseBetaDecayCrossSection as IBD
import HuberMullerFlux as HMF
import NEOSParameters as NEOSP
import NEOSData as NEOSD
import Models
import numpy as np
from scipy import integrate as integrate
from scipy import interpolate as interpolate


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
        # However, we consider an arbitrary large number of targets to prevent very small event expectations.
        self.TotalNumberOfProtonsHM = 9.74503e51#4.76567e50
        self.TotalNumberOfProtonsDB = 1.06406e06

        self.sets_names = NEOSP.exp_names
        self.reactor_names = NEOSP.reac_names
        self.isotopes_to_consider = NEOSP.isotopes

        self.mean_fission_fractions = NEOSP.mean_fis_frac
        self.EfficiencyOfHall = NEOSP.efficiency # Not quite useful in NEOS, but yes in general.
        self.DistanceFromReactorToHall = NEOSP.distance
        self.WidthOfHall = NEOSP.width

        self.NeutrinoCovarianceMatrix = NEOSP.neutrino_covariance_matrix
        self.NeutrinoCovarianceMatrixPrompt = NEOSP.neutrino_covariance_matrix_prompt
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
        self.RatioError = NEOSD.ratio_error

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
        The flux of neutrinos with energy enu coming from the isotope.
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
        The width (in meters) of such detector
        """
        return self.WidthOfHall[experiment]


    # CALCULATION OF EXPECTED EVENTS
    # ------------------------------

    def calculate_naked_event_expectation_simple(self,model,set_name,i, bins = None, average = False, use_HM = True):
        """
        This function implements formula (A.2) from 1709.04294, adapted to NEOS.
        Here we don't neglect the width of the detector, and integrate over it.

        Input:
        model: a class containing the information of the model.
               Must contain a method oscProbability (+info on Models.py)
        set_name (str): name of the experimental hall studied.
        i (int): the data bin we want to compute the expectation of.
        bins: a numpy array of custom bins to calculate expectations with. Useful for GlobalFit.py
        average (bool): whether to pick oscProbability_av or not from the model.
        """
        DeltaNeutronToProtonMass = 1.29322 # MeV from PDG2018 mass differences
        ElectronMass = 0.511 # MeV

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
        ndL = 3 # We only integrate through L with three points. It is probably enough.
        dL = (2*W)/(ndL-1)

        for reactor in self.reactor_names:
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
                            flux = np.sum(np.array([self.get_flux_HM(enu,isotope)*self.mean_fission_fractions[isotope] for isotope in self.isotopes_to_consider]))
                        else:
                            flux = self.get_flux(enu) # the flux from DB slows down the program A LOT, use with caution


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
            #expectation /= L**2 # this is an error, and the root of all evil in the world

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
        This function returns the integrand of formula (A.2) from 1709.04294.

        Input:
        erf, etf: indices of reconstructed and true energies, in the response matrix.
        enu: true energy of the neutrino. This will be the parameter of integration.
        L: the length between experimental hall and reactor.
        model: the model with which to compute the oscillation probability.
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
        This function implements formula (A.2) from 1709.04294.
        Here we don't neglect the width of the detector, and integrate over it.
        In this case, however, we perform an integral inside the fine energy
        bins, to take into account possible rapid oscillations (e.g., a heavy sterile).

        Input:
        model: a class containing the information of the model.
               Must contain a method oscProbability (+info on Models.py)
        set_name (str): name of the experimental hall studied.
        bins: a numpy array of custom bins to calculate expectations with. Useful for GlobalFit.py
        i (int): the data bin we want to compute the expectation of.
        """
        DeltaNeutronToProtonMass = 1.29322 # MeV from PDG2018 mass differences
        ElectronMass = 0.511 # MeV

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
        ndL = 3 # We only integrate through L with three points. It is probably enough.
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
            # Once we know we don't integrate, we check whether the user has introduced a custom binning
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
        to the predicated data according to SM.

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
        Input:
        model: a model from Models.py for which to compute the expected number of events.

        Output:
        A 2-tuple with the expectation from the model and from a model without oscillations.
        Each element of a tuple is a dictionary, where each key is an experimental hall.
        Such key links to a numpy array which contains: the histogram of expected events,
        the error bars of each bin, the lower limits of each bin, and the upper limits of each bin.
        The error bars are purely statistical, i.e. sqrt(N).
        """

        # We build the expected number of events for our model and we roughly normalise so that is of the same order of the data.
        exp_events = self.get_expectation_unnorm_nobkg(model,do_we_integrate = integrate, do_we_average = False, use_HM = use_HM)

        norm = self.normalization_to_data(exp_events) # This should only be allowed for NEOS only fit.
        exp_events = dict([(set_name,exp_events[set_name]*norm[set_name] +self.PredictedBackground[set_name]) for set_name in self.sets_names])

        # For the NEOS single fit, there are no nuissance parameters. We just return the data.
        model_expectations = dict([(set_name,np.array([(exp_events[set_name][i],np.sqrt(exp_events[set_name][i]),
                                                        self.DataLowerBinEdges[i],self.DataUpperBinEdges[i])
                                                        for i in range(0,self.n_bins)]))
                                  for set_name in self.sets_names])

        return model_expectations


# ----------------------------------------------------------
# FITTING THE DATA
# ----------------------------------------------------------

    def get_poisson_chi2(self,model, integrate = False, average = False, use_HM = True):
        """
        Computes the "chi2" value from the Poisson probability, taking into account
        every bin from every DayaBay detector.

        Input:
        model: a model from Models.py for which to compute the expected number of events.

        Output: (float) the log Poisson "chi2" value.
        """
        # Honestly, have never tried this. Probably, it should work.
        Exp = self.get_expectation(model, integrate = integrate, average = average, use_HM = use_HM)
        Data = self.ObservedData

        TotalLogPoisson = 0.0
        for set_name in self.sets_names:
            lamb = Exp[set_name][:,0]#+Bkg[set_name]
            k = Data[set_name]
            TotalLogPoisson += (k - lamb + k*np.log(lamb/k))#*fudge[set_name]

        return -2*np.sum(TotalLogPoisson)

    def get_chi2_ratio(self, model, integrate = False, average = False, use_HM = True):
        # Honestly, have never tried this. Probably, it should work.
        Exp = self.get_expectation(model, integrate = integrate, average = average, use_HM = use_HM)['NEOS']
        modelSM = Models.PlaneWaveSM()
        ExpSM = self.get_expectation(modelSM, use_HM = use_HM)['NEOS']
        teo = Exp/ExpSM
        ratio = self.RatioData['NEOS']
        ratio_err = self.RatioError['NEOS']

        chi2 = (teo[:,0]-ratio)**2/ratio_err**2
        chi2 = np.sum(chi2)
        return chi2

    def get_both_chi2(self, model, integrate = False, average = False, use_HM = True):
        Exp = self.get_expectation(model, integrate = integrate, average = average, use_HM = use_HM)
        Data = self.ObservedData

        TotalLogPoisson = 0.0
        for set_name in self.sets_names:
            lamb = Exp[set_name][:,0]#+Bkg[set_name]
            k = Data[set_name]
            TotalLogPoisson += (k - lamb + k*np.log(lamb/k))#*fudge[set_name]
        poisson_chi2 = -2*np.sum(TotalLogPoisson)


        modelSM = Models.PlaneWaveSM()
        ExpSM = self.get_expectation(modelSM, use_HM = use_HM)['NEOS']
        teo = Exp['NEOS']/ExpSM
        ratio = self.RatioData['NEOS']
        ratio_err = self.RatioError['NEOS']

        ratio_chi2 = (teo[:,0]-ratio)**2/ratio_err**2
        ratio_chi2 = np.sum(ratio_chi2)
        return (poisson_chi2,ratio_chi2)


# ----------------------------------------------------------
# BUILDING THE COVARIANCE MATRIX
# ----------------------------------------------------------

    # def get_neutrino_flux_binned(self):
    #     flux_bin = []
    #     for etf in range(0,len(self.NeutrinoLowerBinEdges)):
    #
    #
    # def get_all_flux_prompt_energy(self):
    #     flux_rec = []
    #     for i in range(0,len(self.DataLowerBinEdges)):
    #         erec = (self.DataLowerBinEdges[erf]+self.DataUpperBinEdges[erf])/2.
    #         erf = self.FindFineBinIndex(erec)
    #         for etf in range(0,len(self.FromEtrueToErec[1])):
    #             enu = (etf+0.5)*self.deltaEfine
    #             flux = self.get_flux(enu)
    #             flux_rec.append(flux*self.FromEtrueToErec[erf][etf])
    #
    #     return np.array(flux_rec)

    def get_flux_prompt_energy(self,erec,use_HM = True):
                # For the GlobalFit, it is necessary to use HM flux.
        flux_rec = 0.0
        erf = self.FindFineBinIndex(erec)
        for etf in range(0,len(self.FromEtrueToErec[1])):
            enu = (etf + 0.5)*self.deltaEfine
            if use_HM == True:
                # For the GlobalFit, it is necessary to use HM flux.
                flux = np.sum(np.array([self.get_flux_HM(enu,isotope)*self.mean_fission_fractions[isotope]
                                        for isotope in self.isotopes_to_consider]))
            else:
                flux = self.get_flux(enu) # the flux from DB slows down the program A LOT, use with caution
            if ((etf == 0) or (etf == len(self.FromEtrueToErec[1]-1))):
                flux_rec += flux*self.FromEtrueToErec[erf][etf]/2.
            else:
                flux_rec += flux*self.FromEtrueToErec[erf][etf]
        return flux_rec

    def FindTrueBinIndex(self,enuf):
        if enuf <= 2.125:
            return 0
        elif enuf >= 8.125:
            return len(self.NeutrinoLowerBinEdges)-1
        else:
            return int((enuf-2.125)/0.25)+1

    def get_flux_covariance_prompt_energy(self,erec1,erec2):
        covr = 0.0

        if (erec1 < 1.022) or (erec2 < 1.022):
            return 0.

        erf1 = self.FindFineBinIndex(erec1)
        erf2 = self.FindFineBinIndex(erec2)

        for etf1 in range(self.FindFineBinIndex(1.806)+1,len(self.FromEtrueToErec[1])):
            enu1 = (etf1+0.5)*self.deltaEfine
            et1  = self.FindTrueBinIndex(enu1)
            for etf2 in range(self.FindFineBinIndex(1.806)+1,len(self.FromEtrueToErec[1])):
                enu2 = (etf2+0.5)*self.deltaEfine
                et2  = self.FindTrueBinIndex(enu2)
                covr += (self.NeutrinoCovarianceMatrix[et1][et2]/self.get_cross_section(enu1)/self.get_cross_section(enu2)
                         *self.FromEtrueToErec[erf1][etf1]*self.FromEtrueToErec[erf2][etf2])
        return covr

    def get_table_flux_covariance_matrix_prompt(self):
        covtab = np.array([[self.get_flux_covariance_prompt_energy(self.DataCentrBinEdges[eri1],self.DataCentrBinEdges[eri2])
                         for eri1 in range(0,self.n_bins)] for eri2 in range(0,self.n_bins)])
        return covtab

    def get_total_covariance_matrix(self):
        flux = np.array([self.get_flux_prompt_energy(self.DataCentrBinEdges[eri], use_HM = False)
                         for eri in range(0,self.n_bins)])

        Neff = self.ObservedData['NEOS']/flux
        N = self.ObservedData['NEOS']
        # print(Neff**2)

        flux_cov = self.NeutrinoCovarianceMatrixPrompt
        # flux_cov = self.NeutrinoCovarianceMatrix
        # print(np.diag(flux_cov)/np.mean(self.NeutrinoFlux)**2)
        # print(np.diag(flux_cov)*Neff**2)
        # print(flux)
        # print(self.NeutrinoFlux)

        # print(1/np.sqrt(N))

        stat_err = N*np.identity(self.n_bins)
        # print(N/flux)
        cov_err = np.tile(np.sqrt(N)/flux,(len(N),1))*(np.tile(np.sqrt(N)/flux,(len(N),1)).transpose())
        cov_err = flux_cov*cov_err

        return cov_err+stat_err

    def get_inverse_covariance_matrix(self):
        return np.linalg.inv(np.array(self.get_total_covariance_matrix()))

    def get_chi2_cov(self,model, do_we_integrate = False, do_we_average = False, use_HM = True):
        """
        Input: a  model with which to compute expectations.
        Output: a chi2 statistic comparing data and expectations.
        """
        Vinv = self.get_inverse_covariance_matrix()
        Exp = self.get_expectation(model, integrate = do_we_integrate, average = do_we_average, use_HM = use_HM)
        Data = self.ObservedData
        chi2 = 0.
        for set_name in self.sets_names:
            exp_i = Exp[set_name][:,0]#+Bkg[set_name]
            dat_i = Data[set_name]
            chi2 += (dat_i-exp_i).dot(Vinv.dot(dat_i-exp_i))

        return chi2

    def get_chi2_ratio_cov(self,model, do_we_integrate = False, do_we_average = False, use_HM = True):
        Exp = self.get_expectation(model, integrate = do_we_integrate, average = do_we_average, use_HM = use_HM)['NEOS']
        modelSM = Models.PlaneWaveSM()
        ExpSM = self.get_expectation(modelSM, use_HM = use_HM)['NEOS']
        teo = Exp[:,0]/ExpSM[:,0]
        ratio = self.RatioData['NEOS']
        Vinv = self.get_inverse_covariance_matrix()
        # Vinv *= np.tile(np.sqrt(ExpSM[:,0]),(len(ExpSM[:,0]),1))*(np.tile(np.sqrt(ExpSM[:,0]),(len(ExpSM[:,0]),1)).transpose())
        Vinv *= np.tile(ExpSM[:,0],(len(ExpSM[:,0]),1))*(np.tile(ExpSM[:,0],(len(ExpSM[:,0]),1)).transpose())

        return (teo-ratio).dot(Vinv.dot(teo-ratio))
