import InverseBetaDecayCrossSection as IBD
import HuberMullerFlux as HMF
import DayaBayParameters as DBP
import DayaBayData as DBD

# THINGS TO TAKE INTO ACCOUNT
# 1. One should be careful between which variables are public and which are private!
# 2. I am not understanding the function get_true_energy_bin_centers,
#    its result is not coherent with the data bins (which are not of uniform size).
# 3. As a result of 2, I don't know if FindFineBinIndex is correct.

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
        self.deltaEfine = 0.05 # in MeV. What is this???
        self.TotalNumberOfProtons = TotalMass*FH*Na*IH1/HidrogenMass
        self.ignore_oscillations = False

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


    # USEFUL FUNCTIONS TO ACCESS THE CLASS DATA
    # -----------------------------------------
    def name(self):
        return "DayaBay"

    def num_bins(self):
        return self.n_bins

    def set_ignore_oscillations(self,io):
        self.ignore_oscillations = io
        return self.ignore_oscillations

    def are_we_ignoring_oscillations(self):
        return self.ignore_oscillations

    def get_isotope_names(self):
        return self.isotopes_to_consider

    def get_resolution_matrix(self):
        return self.FromEtrueToErec


    # USEFUL FUNCTIONS TO MAKE HISTOGRAMS
    # -----------------------------------
    def get_lower_neutrino_bin_edges(self):
        return self.NeutrinoLowerBinEdges

    def get_upper_neutrino_bin_edges(self):
        return self.NeutrinoUpperBinEdges

    def get_data_upper_bin_edges(self):
        return self.DataUpperBinEdges

    def get_data_lower_bin_edges(self):
        return self.DataLowerBinEdges

    def get_true_energy_bin_centers(self):
        """
        Output: returns a list with the centers of the bins?
        """
        enutrue = []
        for i in range(0,len(self.FromEtrueToErec[1])):
            enutrue.append((i+0.5)*self.deltaEfine)
        return enutrue

    def FindFineBinIndex(self,energy):
        """
        Returns the index of the histogram in which the
        input energy is found?
        """
        dindex = floor(energy/self.deltaEfine - 0.5)
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

    def get_distance2(self,experiment,reactor):
        """
        Input:
        experiment (str): name of the experimental hall.
        reactor (str): name of the reactor.

        Output:
        The squared distance between these EH and reactor.
        """
        return self.DistanceFromReactorToHallSquare[experiment][reactor]**2
