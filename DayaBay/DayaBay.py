import sys
import os
common_dir = '/Common_cython'
sys.path.append(os.getcwd()[:-8]+common_dir)

import InverseBetaDecayCrossSection as IBD
import HuberMullerFlux as HMF
import DayaBayParameters as DBP
import DayaBayData as DBD
import numpy as np
from scipy import integrate as integrate

class DayaBay:

    # INITIALISATION OF THE CLASS
    # ---------------------------
    def __init__(self):
        """
        Initialises the class.
        No numeric information is found here. All parameters and data are
        gathered in files 'DayaBayParameters.py' or 'DayaBayData.py',
        in order to have a better organisation and to make tuning easier.
        For more information on these data, check the files.
        """
        # Physical quantities
        DeltaNeutronToProtonMass = 1.29322 # MeV from PDG2018 mass differences
        ElectronMass = 0.511 # MeV
        Na = 6.022140857e23 # Avogadro's number
        FH = 12.02 # hydrogen fraction in GdLS (what is this and what for?)
        IH1 = 0.9998 # H1 isotope abundance
        HidrogenMass = 1.673723e-27 # hidrogen mass in kg
        TotalMass = 20.e3 # in kg

        # Associated objects of the class
        self.deltaEfine = 0.05 # in MeV. It is the resolution of the Etrue to Erec matrix

        self.TotalNumberOfProtons = TotalMass*FH*IH1/HidrogenMass
        # In principle, our analysis is flux-free, i.e. independent of the flux.
        # Therefore, the total normalisation of the flux is not important.
        # However, we consider an arbitrary large number of targets to prevent very small event expectations.
        self.TotalNumberOfProtons = {'EH1': 16.968688565113972e53, 'EH2': 15.783625798930741e53, 'EH3': 16.21670112543282e53}

        self.sets_names = DBP.exp_names
        self.reactor_names = DBP.reac_names
        self.isotopes_to_consider = DBP.isotopes

        self.mean_fission_fractions = DBP.mean_fis_frac
        self.EfficiencyOfHall = DBP.efficiency
        self.DistanceFromReactorToHall = DBP.distance
        self.FudgeFactorPerHall = DBP.fudge_factors

        self.NeutrinoCovarianceMatrix = DBP.neutrino_covariance_matrix
        self.NeutrinoLowerBinEdges = DBP.nulowerbin
        self.NeutrinoUpperBinEdges = DBP.nuupperbin

        self.DataLowerBinEdges = DBD.datlowerbin
        self.DataUpperBinEdges = DBD.datupperbin
        self.n_bins = DBD.number_of_bins
        self.FromEtrueToErec = DBP.reconstruct_mat

        self.AllData = DBD.all_data
        self.ObservedData = DBD.observed_data
        self.PredictedDataNoOsc = DBD.predicted_data_noosc
        self.PredictedBackground = DBD.predicted_bkg


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

    def get_cross_section(self,enu):
        """ Check InverseBetaDecayCrossSection.py for more info."""
        return IBD.CrossSection(enu)

    def get_distance(self,experiment,reactor):
        """
        Input:
        experiment (str): name of the experimental hall.
        reactor (str): name of the reactor.

        Output:
        The distance between these EH and reactor (data in meters).
        """
        return self.DistanceFromReactorToHall[experiment][reactor]


    # CALCULATION OF EXPECTED EVENTS
    # ------------------------------

    def calculate_naked_event_expectation_simple(self,model,set_name,i, average = False):
        """
        This function implements formula (A.2) from 1709.04294.

        Input:
        model: a class containing the information of the model.
               Must contain a method oscProbability (+info on Models.py)
        set_name (str): name of the experimental hall studied.
        i (int): the data bin we want to compute the expectation of.
        average: whether to use the averaged probability, in case we're studying
                 sterile neutrino very-fast oscillations. In such case, the model
                 must contain a method oscProbability_av.

        Output:
        The number of expected events (not normalised, without background)
        at the experimental hall set_name for histogram i.
        """
        DeltaNeutronToProtonMass = 1.29322 # MeV from PDG2018 mass differences
        ElectronMass = 0.511 # MeV

        expectation = 0.0
        # We want to know what are the fine reconstructed energies for which
        # we want to make events inside the data bin i.
        min_energy_fine_index = self.FindFineBinIndex(self.DataLowerBinEdges[i])
        max_energy_fine_index = self.FindFineBinIndex(self.DataUpperBinEdges[i])

        for reactor in self.reactor_names:
            L = self.get_distance(set_name,reactor) # in meters

            for erf in range(min_energy_fine_index,max_energy_fine_index):

                for etf in range(0,len(self.FromEtrueToErec[1])):
                    enu = (etf+0.5)*self.deltaEfine + (
                          DeltaNeutronToProtonMass - ElectronMass)

                    if average == False:
                        # Is the neutrino undergoing fast oscillations?
                        oscprob = model.oscProbability(enu,L)
                    elif average == True:
                        oscprob = model.oscProbability_av(enu,L)

                    flux = 0.0
                    for isotope in self.isotopes_to_consider:
                        # This computes the HubberMullerFlux for all isotopes
                        flux += (self.mean_fission_fractions[isotope]*
                                 self.get_flux(enu,isotope))

                    # Here we perform trapezoidal integration, the extremes contribute 1/2.
                    if ((etf == 0) or (etf == len(self.FromEtrueToErec[1])-1) or
                        (erf == min_energy_fine_index) or (erf == max_energy_fine_index-1)):
                        expectation += (flux * self.get_cross_section(enu) *
                                        self.FromEtrueToErec[erf][etf] * oscprob)/L**2/2.
                    else:
                        expectation += (flux * self.get_cross_section(enu) *
                                        self.FromEtrueToErec[erf][etf] * oscprob)/L**2

                # real antineutrino energies loop ends

            # reconstructed energies loop ends
            #expectation /= L**2 # this is an error, and the root of all evil in the world

        # reactor loop ends
        # the two deltaEfine are to realise a trapezoidal numeric integration
        return expectation*self.deltaEfine**2*self.EfficiencyOfHall[set_name]* self.TotalNumberOfProtons[set_name]

    def integrand(self,enu,L,model,erf,etf):
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
                self.FromEtrueToErec[erf][etf] *
                model.oscProbability(enu,L))


    def calculate_naked_event_expectation_integr(self,model,set_name,i):
        """
        This function implements formula (A.2) from 1709.04294.
        In this case, however, we perform an integral inside the fine energy
        bins, to take into account possible rapid oscillations (e.g., a heavy sterile).

        Input:
        model: a class containing the information of the model.
               Must contain a method oscProbability (+info on Models.py)
        set_name (str): name of the experimental hall studied.
        i (int): the data bin we want to compute the expectation of.
        """
        DeltaNeutronToProtonMass = 1.29322 # MeV from PDG2018 mass differences
        ElectronMass = 0.511 # MeV

        expectation = 0.0
        # We want to know what are the fine reconstructed energies for which
        # we want to make events inside the data bin i.
        min_energy_fine_index = self.FindFineBinIndex(self.DataLowerBinEdges[i])
        max_energy_fine_index = self.FindFineBinIndex(self.DataUpperBinEdges[i])

        for reactor in self.reactor_names:
            L = self.get_distance(set_name,reactor) #in meters

            for erf in range(min_energy_fine_index,max_energy_fine_index):

                for etf in range(0,len(self.FromEtrueToErec[1])):
                    enu_min = (etf)*self.deltaEfine + (
                          DeltaNeutronToProtonMass - ElectronMass)
                    enu_max = (etf+1)*self.deltaEfine + (
                          DeltaNeutronToProtonMass - ElectronMass) # in MeV

                    # print(enu_min,enu_max,f(enu_min),f(enu_max))
                    if ((erf == min_energy_fine_index) or (erf == max_energy_fine_index-1)):
                        expectation += integrate.quad(self.integrand,enu_min,enu_max,
                                                      args=(L,model,erf,etf))[0]/L**2/2.

                    else:
                        expectation += integrate.quad(self.integrand,enu_min,enu_max,
                                                      args=(L,model,erf,etf))[0]/L**2
                    # isotope loop ends

                # real antineutrino energies loop ends

            # reconstructed energies loop ends
            # expectation /= L**2

        # reactor loop ends
        # only one trapezoidal numeric integration has been done
        return expectation*self.deltaEfine*self.EfficiencyOfHall[set_name]*self.TotalNumberOfProtons[set_name]


    def get_expectation_unnorm_nobkg(self,model,do_we_integrate = False,imin = 0,imax = DBD.number_of_bins, do_we_average = False):
        """
        Computes the histogram of expected number of events without normalisation
        to the real data, and without summing the predicted background.

        Input:
        model: a Models.py class with which to compute oscillation probabilities.
        do_we_integrate: whether to integrate inside each energy bin or not.
        do_we_average: whether to use oscProbability_av from the model.
        imin, imax = indexs of the lowest and "uppest" bins to compute. Only useful for GlobalFit.py.

        Output:
        A dictionary with a string key for each experimental hall, linking to a
        numpy array with the expected events for each histogram bin.
        """
        if do_we_integrate == False:
            Expectation = dict([(set_name,
                                 np.array([self.calculate_naked_event_expectation_simple(model,set_name,i, average = do_we_average) for i in range(imin,imax)]))
                                for set_name in self.sets_names])

        elif do_we_integrate == True:
            Expectation = dict([(set_name,
                                 np.array([self.calculate_naked_event_expectation_integr(model,set_name,i) for i in range(imin,imax)]))
                                for set_name in self.sets_names])
        return Expectation


    def normalization_to_data(self,events):
        """
        Input:
        events: a dictionary with a string key for each experimental hall, linking to
        a numpy array for some histogram of events. In principle, it should be
        the number of expected number of events of our model.

        Output:
        norm: a normalisation factor with which to multiply events such that the total
        number of events of "events" is the same as the one from observed data.
        Each experimental hall has a different factor.
        """
        TotalNumberOfExpEvents = dict([(set_name,np.sum(self.PredictedDataNoOsc[set_name]))
                                            for set_name in self.sets_names])
        TotalNumberOfBkg = dict([(set_name,np.sum(self.PredictedBackground[set_name]))
                                 for set_name in self.sets_names])
        norm = dict([(set_name,(TotalNumberOfExpEvents[set_name]-TotalNumberOfBkg[set_name])/np.sum(events[set_name]))
                     for set_name in self.sets_names])

        print('norms: ', norm)
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

    def get_expectation(self,model, integrate = False, average = False):
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

        # We build the expected number of events for our model and we normalise so it has the same number of events of the data.
        exp_events = self.get_expectation_unnorm_nobkg(model,do_we_integrate = integrate, do_we_average = average)
        # norm = self.normalization_to_data(exp_events)
        # exp_events = dict([(set_name,exp_events[set_name]*norm[set_name]) for set_name in self.sets_names])

        # We construct the nuissance parameters which minimise the Poisson probability
        # alpha_i = (dataEH1_i+dataEH2_i+dataEH3_i)/(expEH1_i+expEH2_i+expEH3_i)
        nuissances = self.get_nuissance_parameters(exp_events)

        # We apply the nuissance parameters to the data and obtain
        exp_events = dict([(set_name,exp_events[set_name]*nuissances+self.PredictedBackground[set_name])
                           for set_name in self.sets_names])

        model_expectations = dict([(set_name,np.array([(exp_events[set_name][i],np.sqrt(exp_events[set_name][i]),
                                                        self.DataLowerBinEdges[i],self.DataUpperBinEdges[i])
                                                        for i in range(0,self.n_bins)]))
                                  for set_name in self.sets_names])

        return model_expectations


# ----------------------------------------------------------
# FITTING THE DATA
# ----------------------------------------------------------

    def get_poisson_chi2(self,model, do_we_integrate = False, do_we_average = False):
        """
        Computes the "chi2" value from the Poisson probability, taking into account
        every bin from every DayaBay detector.

        Input:
        model: a model from Models.py for which to compute the expected number of events.

        Output: (float) the log Poisson "chi2" value.
        """
        Exp = self.get_expectation(model, integrate = do_we_integrate, average = do_we_average)
        Data = self.ObservedData
        #Bkg = self.PredictedBackground
        TotalLogPoisson = 0.0
        for set_name in self.sets_names:
            lamb = Exp[set_name][:,0]#+Bkg[set_name]
            k = Data[set_name]
            TotalLogPoisson += (k - lamb + k*np.log(lamb/k))#*fudge[set_name]

        return -2*np.sum(TotalLogPoisson)


# ----------------------------------------------------------
# FITTING THE DATA WITH THE COVARIANCE MATRIX
# ----------------------------------------------------------

    def get_inverse_flux_covariance(self):
        """
        Returns the inverse of the neutrino covariance matrix, V^-1.
        """
        return np.linalg.inv(np.array(self.NeutrinoCovarianceMatrix))

    def get_resolution_matrix_underdim(self):
        """
        Returns an underdimension of the response matrix.
        """
        mat = np.zeros((len(self.NeutrinoLowerBinEdges),self.n_bins))
        for i in range(0,self.n_bins):
            minrec = self.FindFineBinIndex(self.DataLowerBinEdges[i])
            maxrec = self.FindFineBinIndex(self.DataUpperBinEdges[i])
            for j in range(0,len(self.NeutrinoLowerBinEdges)):
                mintrue = self.FindFineBinIndex(self.NeutrinoLowerBinEdges[j])
                maxtrue = self.FindFineBinIndex(self.NeutrinoUpperBinEdges[j])
                mat[j,i] = np.mean(self.FromEtrueToErec[mintrue:maxtrue,minrec:maxrec])
        return mat

    def get_chi2(self,model, do_we_integrate = False, do_we_average = False):
        """
        Input: a  model with which to compute expectations.
        Output: a chi2 statistic comparing data and expectations.
        """
        U = self.get_resolution_matrix_underdim()
        UT = U.transpose()
        Vinv = self.get_inverse_flux_covariance()

        Exp = self.get_expectation(model, integrate = do_we_integrate, average = do_we_average)
        Data = self.ObservedData
        chi2 = 0.
        for set_name in self.sets_names:
            exp_i = Exp[set_name][:,0]#+Bkg[set_name]
            dat_i = Data[set_name]
            chi2 += (dat_i-exp_i).dot(UT.dot(Vinv.dot(U.dot(dat_i-exp_i))))

        return chi2