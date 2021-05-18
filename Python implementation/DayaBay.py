import InverseBetaDecayCrossSection as IBD
import HuberMullerFlux as HMF
import DayaBayParameters as DBP
import DayaBayData as DBD
import Models
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
        FH = 12.02 # hydrogen fraction in GdLS
        IH1 = 0.9998 # H1 isotope abundance
        HidrogenMass = 1.673723e-27 # hidrogen mass in kg
        TotalMass = 20.e3 # in kg

        # Associated objects of the class
        self.deltaEfine = 0.05 # in MeV. It is the resolution of the Etrue to Erec matrix
        self.TotalNumberOfProtons = TotalMass*FH*Na*IH1/HidrogenMass

        self.sets_names = DBP.exp_names
        self.reactor_names = DBP.reac_names
        self.isotopes_to_consider = DBP.isotopes

        self.mean_fission_fractions = DBP.mean_fis_frac
        self.EfficiencyOfHall = DBP.efficiency
        self.DistanceFromReactorToHallSquare = DBP.distance
        self.FudgeFactorPerHall = DBP.fudge_factors

        self.NeutrinoCovarianceMatrix = DBP.neutrino_covariance_matrix
        self.NeutrinoLowerBinEdges = DBP.nulowerbin
        self.NeutrinoUpperBinEdges = DBP.nuupperbin

        self.DataLowerBinEdges = DBD.datlowerbin
        self.DataUpperBinEdges = DBD.datupperbin
        self.n_bins = DBD.number_of_bins
        self.FromEtrueToErec = DBP.reconstruct_mat

        self.ObservedData = DBD.observed_data
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
        The squared distance between these EH and reactor.
        """
        return self.DistanceFromReactorToHallSquare[experiment][reactor]

    # CALCULATION OF EXPECTED EVENTS
    # ------------------------------

    def calculate_naked_event_expectation_simple(self,model,set_name,i):
        """
        This function implements formula (A.2) from 1709.04294.

        Input:
        model: a class containing the information of the model.
               Must contain a method oscProbability (+info on Models.py)
        set_name (str): name of the experimental hall studied.
        i (int): the data bin we want to compute the expectation of.
        """
        DeltaNeutronToProtonMass = 1.29322 # MeV from PDG2018 mass differences
        ElectronMass = 0.511 # MeV

        if (set_name not in self.sets_names):
            print("Cannot calculate naked rate. Invalid set.")
            return None
        elif (i > self.n_bins):
            print("Cannot calculate naked rate. Bin number is invalid.")
            return None

        expectation = 0.0
        # We want to know what are the fine reconstructed energies for which
        # we want to make events inside the data bin i.
        min_energy_fine_index = self.FindFineBinIndex(self.DataLowerBinEdges[i])
        max_energy_fine_index = self.FindFineBinIndex(self.DataUpperBinEdges[i])

        for reactor in self.reactor_names:
            L = self.get_distance(set_name,reactor)

            for erf in range(min_energy_fine_index,max_energy_fine_index):

                for etf in range(0,len(self.FromEtrueToErec[1])):
                    enu = (etf+0.5)*self.deltaEfine + (
                          DeltaNeutronToProtonMass - ElectronMass)
                    oscprob = model.oscProbability(enu,L)
                    flux = 0.0

                    for isotope in self.isotopes_to_consider:
                        flux += (self.mean_fission_fractions[isotope]*
                                 self.get_flux(enu,isotope))
                    expectation += (flux * self.get_cross_section(enu) *
                                    self.FromEtrueToErec[erf][etf] * oscprob)
                    # isotope loop ends

                # real antineutrino energies loop ends

            # reconstructed energies loop ends
            # the two deltaEfine are to realise a trapezoidal numeric integration
            expectation *= self.deltaEfine**2
            expectation *= self.EfficiencyOfHall[set_name]
            expectation /= L**2

        # reactor loop ends
        return expectation * self.TotalNumberOfProtons

    def integrand(self,enu,L,model,isotope,erf,etf):
        return (self.mean_fission_fractions[isotope]*
                self.get_flux(enu,isotope)*
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

        if (set_name not in self.sets_names):
            print("Cannot calculate naked rate. Invalid set.")
            return None
        elif (i > self.n_bins):
            print("Cannot calculate naked rate. Bin number is invalid.")
            return None

        expectation = 0.0
        # We want to know what are the fine reconstructed energies for which
        # we want to make events inside the data bin i.
        min_energy_fine_index = self.FindFineBinIndex(self.DataLowerBinEdges[i])
        max_energy_fine_index = self.FindFineBinIndex(self.DataUpperBinEdges[i])

        for reactor in self.reactor_names:
            L = self.get_distance(set_name,reactor)

            for erf in range(min_energy_fine_index,max_energy_fine_index):

                for etf in range(0,len(self.FromEtrueToErec[1])):
                    enu_min = (etf)*self.deltaEfine + (
                          DeltaNeutronToProtonMass - ElectronMass)
                    enu_max = (etf+1)*self.deltaEfine + (
                          DeltaNeutronToProtonMass - ElectronMass)

                    for isotope in self.isotopes_to_consider:
                        expectation += integrate.quad(self.integrand,enu_min,enu_max,
                                                      args=(L,model,isotope,erf,etf))[0]
                    # isotope loop ends

                # real antineutrino energies loop ends

            # reconstructed energies loop ends
            # only one trapezoidal numeric integration has been done
            expectation *= self.deltaEfine
            expectation *= self.EfficiencyOfHall[set_name]
            expectation /= L**2

        # reactor loop ends
        return expectation * self.TotalNumberOfProtons

    def get_data(self):
        """
        Output:
        Returns a list, with as many elements as experimental halls.
        Each element is a list with as many subelements as data bins.
        Each subelement is a 4-tuple with: the number of counts, the error
        (computed as the sqrt), and the lower and upper bin.

        Note: one should take a better look at how the error is computed.
        """
        all_data = []
        for set_name in self.sets_names:
            observed_data = self.ObservedData[set_name]
            data_list = []
            for i in range(0,self.n_bins):
                data_list.append([observed_data[i],np.sqrt(observed_data[i]),
                                  self.DataLowerBinEdges[i],self.DataUpperBinEdges[i]])
            all_data.append(data_list)
        return np.array(all_data)

    def get_expectation(self,model):
        """
        Input:
        A model with which to compute the oscillation probability.

        Output:
        Returns a list, with as many elements as experimental halls.
        Each element is a list with as many subelements as data bins.
        Each subelement is a 4-tuple with: the number of counts, the error
        (computed as the sqrt), and the lower and upper bin.

        Note: one should take a better look at how the error is computed.
        """
        all_expe = []
        for set_name in self.sets_names:
            predicted_background = self.PredictedBackground[set_name]
            expe_list = []
            for i in range(0,self.n_bins):
                expectation = self.calculate_naked_event_expectation_simple(model,set_name,i)
                expectation += predicted_background[i]
                expe_list.append((expectation,np.sqrt(expectation),
                                  self.DataLowerBinEdges[i],self.DataUpperBinEdges[i]))
            all_expe.append(expe_list)
        return np.array(all_expe)

    def get_inverse_flux_covariance(self):
        return np.linalg.inv(np.array(self.NeutrinoCovarianceMatrix))

    def get_inverse_resolution_matrix(self):
        return np.linalg.inv(np.array(self.FromEtrueToErec))

    def get_inverse_resolution_matrix_underdim(self):
        M = self.get_inverse_resolution_matrix()
        mat = np.zeros((len(self.NeutrinoLowerBinEdges),self.n_bins))
        for i in range(0,self.n_bins):
            minrec = self.FindFineBinIndex(self.DataLowerBinEdges[i])
            maxrec = self.FindFineBinIndex(self.DataUpperBinEdges[i])
            for j in range(0,len(self.NeutrinoLowerBinEdges)):
                mintrue = self.FindFineBinIndex(self.NeutrinoLowerBinEdges[j])
                maxtrue = self.FindFineBinIndex(self.NeutrinoUpperBinEdges[j])
                mat[j,i] = np.mean(self.FromEtrueToErec[mintrue:maxtrue,minrec:maxrec])
        return mat

    def get_chi2(self,model):
        """
        Input: a  model with which to compute expectations.
        Output: a chi2 statistic comparing data and expectations.
        """
        U = self.get_inverse_resolution_matrix_underdim()
        UT = U.transpose()
        data = self.get_data()[0,:,0]
        expectation = self.get_expectation(model)[0,:,0]
        Vinv = self.get_inverse_flux_covariance()
        return (data-expectation).dot(UT.dot(Vinv.dot(U.dot(data-expectation))))
