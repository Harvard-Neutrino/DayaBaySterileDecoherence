import InverseBetaDecayCrossSection as IBD
import HuberMullerFlux as HMF
import DayaBayParameters as DBP
import DayaBayData as DBD

# THINGS TO TAKE INTO ACCOUNT
# 1. One should be careful between which variables are public and which are private!
# 2. I am not understanding the function get_true_energy_bin_centers,
#    its result is not coherent with the data bins (which are not of uniform size).
# 3. As a result of 2, I don't know if FindFineBinIndex is correct.
# 4. What to do with the implementation of nusquids?
# 5. In line 131 of DayaBay.h (and lots of other places), one defines a Model?
#    Probably this is nusquids stuff.

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

    def get_true_energy_bin_centers_deprecated(self):
        """
        Output: returns a list with the centers of the bins?
        I want to deprecate this.
        """
        enutrue = []
        for i in range(0,len(self.FromEtrueToErec[1])):
            enutrue.append((i+0.5)*self.deltaEfine)
        return enutrue

    def get_true_energy_bin_centers(self):
        """
        Output:
        Returns a numpy array with the centers of the real neutrino energy
        of the histogram bins.
        """
        return (self.NeutrinoLowerBinEdges+self.NeutrinoUpperBinEdges)/2

    def FindFineBinIndex_deprecated(self,energy):
        """
        Returns the index of the histogram in which the
        input energy is found. I want to deprecate this.
        """
        dindex = floor(energy/self.deltaEfine - 0.5)
        if dindex<0:
            return 0
        else:
            return dindex

    def FindFineBinIndex(self,energy):
        """
        Input:
        energy (float): true energy of the antineutrino.

        Output:
        The index of the bin in which this energy is found (int).
        This function works for an arbitrary distribution of bins
        (even if they are not equidistant).
        """
        if ((energy < self.NeutrinoLowerBinEdges[0]) or (energy > self.NeutrinoUpperBinEdges[-1]):
            print("Energy not in histogram.")
            return None
        else:
            # We create an array with all the edges of the histogram.
            allbins = self.NeutrinoLowerBinEdges[0].append(self.NeutrinoUpperBinEdges[-1])
            # when introduced a value, np.histogram return an array such as (0,...,0,1,0,...,0)
            # only thing left to do is to find in which index is the number one.
            return np.where(np.histogram(energy,bins=allbins)[0]==1)[0][0]

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

    # CALCULATION OF EXPECTED EVENTS
    # ------------------------------

    def oscProbability(self, enu, L):
        """
        Input:
        enu (float): the energy of the electron antineutrino.
        L (float): the length travelled by the antineutrino.

        Output:
        The probability of the antineutrino remaining an antineutrino.
        This is computed according to (find paper!).
        """
        sin22th13 = 0.092
        dm2_31 = 2.494e-3
        x = 1.267*dm2_31*L/enu
        return 1. - sin22th13*sin(x)**2

#    def calculate_naked_event_expectation()
