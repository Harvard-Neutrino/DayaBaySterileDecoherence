import numpy as np
import os

# To understand the information on the bins, we must understand the physical process.
# An electron antineutrino arrives to the detector with an energy which can go between
# 0 and infty. However, only if the antineutrino has an energy > 1.8 MeV, it can
# produce IBD. In such case, a positron will be produced (with kinetic and rest energy).

# This positron will have total energy between 511 keV and infty. It will annihilate with
# an electron at rest. This will produce a flash of light, whose energy can go between
# 1.022 MeV and infty (due to the rest mass energy of the positron+electron).

# The energy of this light is called prompt energy, and is the one from the data bins.
# We can relate the prompt energy with the incoming antineutrino energy through
# Eprompt = Erealnu - 0.78 (MeV).
# For more information, the process is described in 1610.04802.

# -------------------------------------------------------------
# PS: The program begins to take prompt energies from ~0.78 MeV.
#     This is a bit stupid, since the first possible antineutrino energy should be
#     ~1.806 MeV and the first non-null prompt energy will be 1.022 MeV.
#     However, it will bring up no error, since this is taken into account
#     in the IBD cross-section, which is set to be 0 if the antineutrino does not
#     have enough energy.

dir = os.path.dirname(os.path.abspath(__file__))+"/Data/"

# -------------------------------------------------------------
#  RECONSTRUCTION MATRIX FUNCTION
# -------------------------------------------------------------


# The reconstruction matrix has a long tail at low energies, as is
# explained in 1609.03910.
def gaussian(x,mu,sig):
    return 1/(np.sqrt(2*np.pi)*sig)*np.exp(-(x-mu)**2/sig**2)

def reconstruct_matrix_function(self,etrue,erec):
    """
    This function tries to mimic the response matrix in inset from figure 3(a)
    in 1610.05134. More information on the fit in 1609.03910.

    Input:
    etrue, erec (float): the true and reconstructed energies.

    Output:
    value (in arbitrary units) of the response matrix at such energies.
    """
    mu1 = -0.84082 + 0.99172*etrue
    mu2 = -1.48036 + 1.06333*etrue
    sig1 = 0.025*etrue + 0.09
    sig2 = 0.055*etrue + 0.033
    factor_g = (0.055*etrue + 0.035)/0.4
    factor_s = (0.025*etrue + 0.85)
    cut = -0.042*etrue + 1.3206
    if erec > mu1+1.5*sig1:
        return 0.
    elif erec > etrue - 1.022:
        return factor_s*gaussian(erec,mu1,sig1)
    elif erec > etrue -cut:
        return factor_g*gaussian(etrue-cut,mu2,sig2)+0.01
    else:
        return factor_g*gaussian(erec,mu2,sig2)+0.01


# -------------------------------------------------------------
#   HISTOGRAM BINS
# -------------------------------------------------------------

# Histogram bins of the measured data.
number_of_bins = 61-1
datlowerbin = np.array([1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,
               2.8,2.9,3.0,3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9,4.0,4.1,4.2,4.3,4.4,4.5,4.6,4.7,
               4.8,4.9,5.0,5.1,5.2,5.3,5.4,5.5,5.6,5.7,5.8,5.9,6.0,6.1,6.2,6.3,6.4,6.5,6.6,6.7,
               6.8,6.9,7.0][:-1])
datupperbin = np.array([1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,
               2.8,2.9,3.0,3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9,4.0,4.1,4.2,4.3,4.4,4.5,4.6,4.7,
               4.8,4.9,5.0,5.1,5.2,5.3,5.4,5.5,5.6,5.7,5.8,5.9,6.0,6.1,6.2,6.3,6.4,6.5,6.6,6.7,
               6.8,6.9,7.0,10.0][:-1])
deltaE = datupperbin - datlowerbin


# -------------------------------------------------------------
#   DAYABAY FULL DATA
# -------------------------------------------------------------

def txt_to_array(filename, sep = ","):
    """
    Input:
    filename (str): the text file containing a matrix
    which we want to read.

    Output:
    A numpy array of the matrix.
    """
    inputfile = open(filename,'r+')
    file_lines = inputfile.readlines()

    mat = []
    for line in file_lines:
        mat.append(line.strip().split(sep))
    mat = np.array(mat).astype(np.float)
    return mat

# Contains the full data from 1610.05134 for all three experimental halls.
# Columns:
# 0: Number of events, per day, per 100 keV;
# 1: Number of expected events, per day, per 100 keV
# 2: Number of background events, per day, per 100 keV
# 3: Ratio of NEOS data to DayaBay data
# 4: Error of ratio of NEOS data to DayaBay data
# 5: Number of expected events according to HM flux, per day, per 100 keV

norm = 180.
# It is important to say: this normalisation factor must not include the deltaE

fudge_data = 1.
fudge_bkg = 1.
all_data = {'NEOS': txt_to_array(dir+"AllData.dat")}
observed_data  = {'NEOS': norm*fudge_data*txt_to_array(dir+"AllData.dat")[:-1,0]}
predicted_data = {'NEOS': norm*fudge_data*txt_to_array(dir+"AllData.dat")[:-1,1]}
predicted_data_HM = {'NEOS': norm*fudge_data*txt_to_array(dir+"AllData.dat")[:-1,5]}
predicted_bkg = {'NEOS': norm*fudge_bkg*txt_to_array(dir+"AllData.dat")[:-1,2]}
# predicted_bkg = {'NEOS':np.zeros([number_of_bins])}


ratio_data = {'NEOS': txt_to_array(dir+"AllData.dat")[:-1,3]}
ratio_error = {'NEOS': txt_to_array(dir+"AllData.dat")[:-1,4]}


# -------------------------------------------------------------------
#  DAYA BAY DATA
# -------------------------------------------------------------------

# Contains the full data from 1610.04802 for all three experimental halls.
# Columns:
# 0: Emin; 1: Emax, 2: Ecentral
# 3: Nobs  4: Npred (including background)
# 5: Nprednoosc; 6: Nbkg
DB_all_data = {'EH1': txt_to_array(dir+"DataEH1.dat")[:,0:7],
            'EH2': txt_to_array(dir+"DataEH2.dat")[:,0:7],
            'EH3': txt_to_array(dir+"DataEH3.dat")[:,0:7]}


DB_observed_data = {'EH1': txt_to_array(dir+"DataEH1.dat")[:,3],
                 'EH2': txt_to_array(dir+"DataEH2.dat")[:,3],
                 'EH3': txt_to_array(dir+"DataEH3.dat")[:,3]}

DB_predicted_bkg = {'EH1': txt_to_array(dir+"DataEH1.dat")[:,6],
                 'EH2': txt_to_array(dir+"DataEH2.dat")[:,6],
                 'EH3': txt_to_array(dir+"DataEH3.dat")[:,6]}
