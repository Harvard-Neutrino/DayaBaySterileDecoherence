import sys
import os
common_dir = '/Common_cython'
sys.path.append(os.getcwd()[:-9]+common_dir)

import InverseBetaDecayCrossSection as IBD
import HuberMullerFlux as HMF
import PROSPECTParameters as PROSP
import numpy as np
from scipy import integrate as integrate
from scipy import interpolate as interpolate


class Prospect:

    # INITIALISATION OF THE CLASS
    # ---------------------------
    def __init__(self):
        """
        Initialises the class.
        No numeric information is found here. All parameters and data are
        gathered in files 'PROSPECTParameters.py' or 'PROSPECTData.py',
        in order to have a better organisation and to make tuning easier.
        For more information on these data, check the files.
        """
        # Physical quantities

        # THIS MUST BE CHANGED!!! THE RECONSTRUCTION MATRIX HAS DIFFERENT DELTAE IN TRUE/REC
        self.deltaEtfine = 0.1 # true energy in MeV. It is the resolution of the Etrue to Erec matrix.

        # This is the normalisation to match the PROSPECT data of expected events without oscillations (assuming no sterile osc).
        # In principle, we leave that each baseline has its own normalisation factor, because it has a different
        # number and characterstics of segments. However, the values are similar because of the construction of the experiment.
        self.TotalNumberOfProtons = {1: 7.90080e+48, 2: 7.907335e+48, 3: 7.816715e+48, 4: 7.82966e+48, 5: 7.78062e+48,
                                     6: 7.69161e+48, 7: 7.660865e+48, 8: 7.723525e+48, 9: 7.77155e+48, 10: 7.76728e+48}
                # self.TotalNumberOfProtonsDB = 1.06406e06#*180/(180-46)

        self.sets_names = PROSP.exp_names
        self.reactor_names  = PROSP.reac_names
        self.isotopes_to_consider = PROSP.isotopes

        self.act_segments = PROSP.active_segments
        self.n_segments = PROSP.number_of_segments

        self.mean_fission_fractions = PROSP.mean_fis_frac
        self.EfficiencyOfHall = PROSP.efficiency
        self.DistanceFromReactorToHall = PROSP.distance
        self.Baselines = PROSP.baselines
        self.WidthOfHall = PROSP.width

        self.FromEtrueToErec = PROSP.reconstruct_mat
        self.CovarianceMatrix = PROSP.covariance_matrix
        self.NeutrinoLowerBinEdges = PROSP.nulowerbin
        self.NeutrinoUpperBinEdges = PROSP.nuupperbin

        self.DataLowerBinEdges = PROSP.datlowerbin
        self.DataUpperBinEdges = PROSP.datupperbin
        self.DataCentrBinEdges = (PROSP.datlowerbin+PROSP.datupperbin)/2.
        self.DeltaEData = (self.DataUpperBinEdges-self.DataLowerBinEdges)
        self.n_bins = PROSP.number_of_bins

        self.AllData = PROSP.all_data
        self.PredictedData = PROSP.predicted_data
        self.ObservedData = PROSP.observed_data
        self.PredictedBackground = PROSP.background
        self.DataError = PROSP.error_data


    # # FUNCTIONS TO MAKE HISTOGRAMS (Probably not necessary in PROSPECT)
    # # -----------------------------------
    #
    # def get_true_energy_bin_centers(self):
    #     """
    #     This function divides the whole reconstructed energy spectrum in
    #     the energy intervals which the experiment is capable to resolve.
    #     The resolution is given by deltaEfine (here, 0.05 MeV).
    #     This energy will allow neutrino energies from 0.78 MeV to 12.78 MeV (after
    #     summing 0.78 MeV in the function), approximately.
    #
    #     Output:
    #     returns a numpy array (the same length as the resolution
    #     matrix) with the center of the bins of the resolution of the experiment.
    #     """
    #     enutrue = []
    #     for i in range(0,len(self.FromEtrueToErec[1])):
    #         enutrue.append((i+0.5)*self.deltaEfine)
    #     return enutrue
    #
    # def FindFineBinIndex(self,energy):
    #     """
    #     Input:
    #     energy (float): prompt energy or neutrino true energy.
    #
    #     Output:
    #     An integer telling us the index of the true_energy_bin_centers
    #     bin in which the input energy is found.
    #     """
    #     dindex = np.int(np.floor(energy/self.deltaEfine - 0.5))
    #     if dindex<0:
    #         return 0
    #     else:
    #         return dindex


    # FUNCTIONS TO GET FLUX, DISTANCES AND CROSS-SECTION
    # ---------------------------------------------------
    def get_cross_section(self,enu):
        """ Check InverseBetaDecayCrossSection.py for more info."""
        return IBD.CrossSection(enu)

    def get_flux(self, enu, isotope_name, flux_parameters = HMF.huber_muller):
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

    def get_distance(self,segment,reactor):
        """
        Input:
        experiment (str): name of the segment in the detector.
        reactor (str): name of the reactor.

        Output:
        The distance between the segment and reactor (data in meters).
        """
        return self.DistanceFromReactorToHall[segment][reactor]

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

    def calculate_naked_event_expectation_simple(self,model,segment,i, do_we_average = False):
        """
        This function implements formula (A.2) from 1709.04294, adapted to PROSPECT.
        Here we don't neglect the width of the segment, and integrate over it.
        However, the integral is not exhaustive, since we consider the segment small.

        Input:
        model: a class containing the information of the model.
               Must contain a method oscProbability (+info on Models.py)
        segment (str): name of the experimental hall studied.
        i (int): the data bin we want to compute the expectation of.
        bins: a numpy array of custom bins to calculate expectations with. Useful for GlobalFit.py
        average (bool): whether to pick oscProbability_av or not from the model.
        """
        DeltaNeutronToProtonMass = 1.29322 # MeV from PDG2018 mass differences
        ElectronMass = 0.511 # MeV
        ThresholdEnergy = 1.806

        expectation = 0.0

        # We want to know what are the fine reconstructed energies for which
        # we want to make events inside the data bin i.

        W = self.get_width(segment) # in meters, the detector total width is 2W
        ndL = 5 # We only integrate through L with three points. It is probably enough.
        # ndL = 1 #We prefer not to integrate in L yet.
        dL = (2*W)/(ndL-1)

        for reactor in self.reactor_names:
            Lmin = self.get_distance(segment,reactor) - W
            Lmax = self.get_distance(segment,reactor) + W

            for j in range(ndL):
                L = Lmin + j*dL #+W # Comment this +W if you want to perform the integration in L.

                for etf in range(0,len(self.FromEtrueToErec[segment][0])):
                    enu = (etf+0.5)*self.deltaEtfine + ThresholdEnergy

                    if do_we_average == False:
                        oscprob = model.oscProbability(enu,L)
                    else:
                        oscprob = model.oscProbability_av(enu,L)

                    flux = np.sum(np.array([self.get_flux(enu,isotope)*self.mean_fission_fractions[isotope]
                                            for isotope in self.isotopes_to_consider]))

                    # Here we perform trapezoidal integration, the extremes contribute 1/2.
                    if ((etf == 0) or (etf == len(self.FromEtrueToErec[segment][1])-1) or
                        (j == 0) or (j == ndL - 1)):
                        expectation += (flux * self.get_cross_section(enu) *
                                        self.FromEtrueToErec[segment][i][etf] * oscprob)/L**2/2.
                    else:
                        expectation += (flux * self.get_cross_section(enu) *
                                    self.FromEtrueToErec[segment][i][etf] * oscprob)/L**2

                    # real antineutrino energies loop ends
                # L distances loop ends
            #expectation /= L**2 # this is an error, and the root of all evil in the world

            # reactor loop ends
            # the two deltaEfine are to implement a trapezoidal numeric integration in etrue and erec
            # the dL is to implement a trapezoidal numeric integration in L
            # we divide by the total width 2W because we want an intensive quantity! It is an average, not a total sum.
        return expectation*self.deltaEtfine*self.EfficiencyOfHall[segment]*dL/(2*W)


    def integrand(self,enu,L,model,segment,erf,etf):
        """
        This function returns the integrand of formula (A.2) from 1709.04294.

        Input:
        erf, etf: indices of reconstructed and true energies, in the response matrix.
        enu: true energy of the neutrino. This will be the parameter of integration.
        L: the length between experimental hall and reactor.
        model: the model with which to compute the oscillation probability.
        """
        # Computes the HM flux for all isotopes
        flux = np.sum(np.array([self.get_flux(enu,isotope)*self.mean_fission_fractions[isotope]
                                    for isotope in self.isotopes_to_consider]))

        return (flux*
                self.get_cross_section(enu) *
                self.FromEtrueToErec[segment][erf][etf] *
                model.oscProbability(enu,L))


    def calculate_naked_event_expectation_integr(self,model,segment,i):
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
        ThresholdEnergy = 1.806

        expectation = 0.0

        # We want to know what are the fine reconstructed energies for which
        # we want to make events inside the data bin i.

        W = self.get_width(segment) # in meters, the detector total width is 2W
        ndL = 5 # We only integrate through L with three points. It is probably enough.
        dL = (2*W)/(ndL-1)


        for reactor in self.reactor_names:
            Lmin = self.get_distance(segment,reactor) - W
            Lmax = self.get_distance(segment,reactor) + W

            for j in range(ndL):
                L = Lmin + j*dL

                # for erf in range(min_energy_fine_index,max_energy_fine_index):

                for etf in range(0,len(self.FromEtrueToErec[segment][0])):
                    enu_min = (etf)*self.deltaEtfine + ThresholdEnergy
                    enu_max = (etf+1)*self.deltaEtfine + ThresholdEnergy # in MeV

                    if ((j == 0) or (j == ndL - 1)):
                        expectation += integrate.quad(self.integrand,enu_min,enu_max,
                                                      args=(L,model,segment,i,etf))[0]/L**2/2
                    else:
                        expectation += integrate.quad(self.integrand,enu_min,enu_max,
                                                      args=(L,model,segment,i,etf))[0]/L**2
                # isotope loop ends

            # real antineutrino energies loop ends

            # # reconstructed energies loop ends
            # only one trapezoidal numeric integration has been done

        # reactor loop ends
        return expectation*dL/(2*W)*self.EfficiencyOfHall[segment]

    def get_baseline_expectation(self,model,baseline,do_we_integrate = False, do_we_average = False):
        exp = np.zeros(self.n_bins)

        if do_we_integrate == False:
            for segment in self.Baselines[baseline]:
                exp += np.array([self.calculate_naked_event_expectation_simple(model,segment,i, do_we_average = do_we_average) for i in range(0,self.n_bins)])

        else:
            for segment in self.Baselines[baseline]:
                exp += np.array([self.calculate_naked_event_expectation_integr(model,segment,i) for i in range(0,self.n_bins)])

        return exp*self.TotalNumberOfProtons[baseline]

    def get_expectation_unnorm_nobkg(self,model,do_we_integrate = False, do_we_average = False):
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
        NO DICTIONARY HERE, WE DON'T LIKE DICTIONARIES
        """
        Expectation = []
        for baseline in self.Baselines:
            Expectation = np.append(Expectation,self.get_baseline_expectation(model,baseline,do_we_integrate = do_we_integrate, do_we_average = do_we_average))

        # Expectation = dict([(set_name,np.array(Expectation)) for set_name in self.sets_names])
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
        TotalNumberOfEvents = []

        for baseline in self.Baselines:
            TotalNumberOfEvents.append(np.sum(self.PredictedData[baseline]))

        # TotalNumberOfBkg = dict([(set_name,np.sum(self.PredictedBackground[set_name]))
        #                           for set_name in self.sets_names])
        events = events.reshape((10,16))
        # norm = dict([(set_name,(TotalNumberOfEvents-TotalNumberOfBkg[set_name])/np.sum(events[set_name]))
                     # for set_name in self.sets_names])
        norm = np.array(TotalNumberOfEvents)/np.sum(events,axis = 1)
        norm = dict([(bl,norm[bl-1]) for bl in self.Baselines])

        print('norm: ', norm)
        return norm


    def get_expectation(self,model, do_we_integrate = False, do_we_average = False):
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
        exp_events = self.get_expectation_unnorm_nobkg(model,do_we_integrate = do_we_integrate, do_we_average = do_we_average)

        # exp_events = dict([(set_name,exp_events[set_name] +self.PredictedBackground[set_name]) for set_name in self.sets_names])
        self.normalization_to_data(exp_events)

        # For the NEOS single fit, there are no nuissance parameters. We just return the data.
        # model_expectations = dict([(set_name,np.array([(exp_events[set_name][i],np.sqrt(exp_events[set_name][i]),
        #                                                 self.DataLowerBinEdges[i],self.DataUpperBinEdges[i])
        #                                                 for i in range(0,self.n_bins)]))
        #                           for set_name in self.sets_names])
        exp_events = exp_events.reshape((10,16))
        exp_events = dict([(bl,exp_events[bl-1]) for bl in self.Baselines])
        return exp_events


    def get_data_per_baseline(self):
        data = self.ObservedData
        data_per_baseline = {}
        for bl in self.Baselines:
            total_data = 0.0
            for segment in self.Baselines[bl]:
                total_data += data[segment]
            data_per_baseline.update({bl:total_data})
        return data_per_baseline

    def get_bkg_per_baseline(self):
        bkg = self.PredictedBackground
        data_per_baseline = {}
        for bl in self.Baselines:
            total_data = 0.0
            for segment in self.Baselines[bl]:
                total_data += bkg[segment]
            data_per_baseline.update({bl:total_data})
        return data_per_baseline




# ----------------------------------------------------------
# BUILDING THE COVARIANCE MATRIX
# ----------------------------------------------------------

    def get_covariance_matrix(self):
        """
        Returns a numpy array with the total covariance matrix in prompt energy,
        taking into account both systematic and statistical errors.

        This function is not in use.
        """
        return self.CovarianceMatrix

    def get_inverse_covariance_matrix(self):
        """
        Returns the inverse total covariance matrix.
        For more information, check the function get_total_covariance_matrix.
        """
        return np.linalg.inv(np.array(self.CovarianceMatrix))


# ----------------------------------------------------------
# FITTING THE DATA
# ----------------------------------------------------------


    def get_chi2(self,model, do_we_integrate = False, do_we_average = False):
        """
        Computes the "chi2" value from the total number of events in figure 3(a),
        using the covariance matrix provided in NEOSParameters.py

        Input:
        model: a model from Models.py for which to compute the expected number of events.

        Output: (float) the chi2 value.
        """
        Vinv = self.get_inverse_covariance_matrix()

        Data = self.get_data_per_baseline()
        Data = np.array([Data[bl] for bl in self.Baselines])
        Data_sum = np.repeat(np.sum(Data, axis = 1),16)
        Data = Data.flatten()

        # Prediction from PROSPECT, has a chi2 = 129.
        # exp_events = self.PredictedData
        # exp_events = np.array([exp_events[bl] for bl in self.Baselines]).flatten()

        # Our prediction
        exp_events = self.get_expectation_unnorm_nobkg(model,do_we_integrate = do_we_integrate, do_we_average = do_we_average)


        Events_sum = np.repeat(np.sum(exp_events.reshape((10,16)), axis = 1),16)

        x = Data-Data_sum*exp_events/Events_sum

        chi2 = x.dot(Vinv.dot(x))
        return chi2
