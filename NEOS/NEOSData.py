import numpy as np
import os

dir = os.path.realpath(__file__)[:-len('NEOSData.py')]+'Data/'
# common_dir = 'Common_cython'
# sys.path.append(homedir+common_dir)
# dir = os.path.dirname(os.path.abspath(__file__))+"/Data/"



# -------------------------------------------------------------
#  RECONSTRUCTION MATRIX FUNCTION
# -------------------------------------------------------------

# The reconstruction matrix has a long tail at low energies, as is
# explained in 1609.03910. In the following functions, we mimic it.

def gaussian(x,mu,sig):
    return 1/(np.sqrt(2*np.pi)*sig)*np.exp(-(x-mu)**2/sig**2)

def reconstruct_matrix_function(etrue,erec):
    """
    This function tries to mimic the response matrix in inset from figure 3(a)
    in 1610.05134. More information on the fit in 1609.03910.
    This also tries to mimic the plots from references
       Yoon (2021): Search for sterile neutrinos at RENO, XIX International Workshop on Neutrino Telescopes.
       H. Kim (2017): Search for Sterile Neutrino at NEOS Experiment, 13th Recontres du Vietnam.
    This function is not in use in the program, and is only here as a reference
    as to how we have computed the response matrix in Data/ReconstructMatrix.dat

    Input:
    etrue, erec (float): the true and reconstructed energies.

    Output:
    value (in arbitrary units) of the response matrix at such energies.
    """
    norm = 1/20.062665 # This normalises the matrix to the same value as the DB response mat
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
        return factor_s*gaussian(erec,mu1,sig1)*norm
    elif erec > etrue -cut:
        return (factor_g*gaussian(etrue-cut,mu2,sig2)+0.01)*norm
    else:
        return (factor_g*gaussian(erec,mu2,sig2)+0.01)*norm



# -------------------------------------------------------------
#   HISTOGRAM BINS
# -------------------------------------------------------------

# Histogram bins of the measured data.

# The last bin in the NEOS analysis has given some trouble due to the uncertainty
# in digitalizing the data from figure 3(a).
# Therefore, in our analysis we have chosen to ignore it, since its effect should not be very important.

lastbin = 1 # 0 for including the last bin (7-10 MeV), 1 for removing it
number_of_bins = 61-lastbin
datlowerbin = np.array([1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,
               2.8,2.9,3.0,3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9,4.0,4.1,4.2,4.3,4.4,4.5,4.6,4.7,
               4.8,4.9,5.0,5.1,5.2,5.3,5.4,5.5,5.6,5.7,5.8,5.9,6.0,6.1,6.2,6.3,6.4,6.5,6.6,6.7,
               6.8,6.9,7.0][:number_of_bins])
datupperbin = np.array([1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,
               2.8,2.9,3.0,3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9,4.0,4.1,4.2,4.3,4.4,4.5,4.6,4.7,
               4.8,4.9,5.0,5.1,5.2,5.3,5.4,5.5,5.6,5.7,5.8,5.9,6.0,6.1,6.2,6.3,6.4,6.5,6.6,6.7,
               6.8,6.9,7.0,10.0][:number_of_bins])
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

# Contains the full data from 1610.05134 for all three experimental halls, digitalized.
# Columns:
# 0: Number of events, per day, per 100 keV;
# 1: Number of expected events, per day, per 100 keV
# 2: Number of expected events according to HM flux, per day, per 100 keV
# 3: Number of background events, per day, per 100 keV
# 4: Ratio of NEOS data to DayaBay data
# 5: Statistical error of ratio of NEOS data to DayaBay data
# 6: Systematical error of ratio of NEOS data to DayaBay data


norm = 180.-46.
# This is the total number of days that the NEOS experiment has been running.
# This normalisation matches the total number of events computed from the statistical errors in figure 3(c).

all_data = {'NEOS': txt_to_array(dir+"AllData.dat")[:number_of_bins]}
observed_data  = {'NEOS': norm*txt_to_array(dir+"AllData.dat")[:number_of_bins,0]}
predicted_data = {'NEOS': norm*txt_to_array(dir+"AllData.dat")[:number_of_bins,1]}
predicted_data_HM = {'NEOS': norm*txt_to_array(dir+"AllData.dat")[:number_of_bins,2]}
predicted_bkg = {'NEOS': norm*txt_to_array(dir+"AllData.dat")[:number_of_bins,3]}


ratio_data = {'NEOS': txt_to_array(dir+"AllData.dat")[:number_of_bins,4]}
ratio_stat_error = {'NEOS': txt_to_array(dir+"AllData.dat")[:number_of_bins,5]}
ratio_syst_error = {'NEOS': txt_to_array(dir+"AllData.dat")[:number_of_bins,6]}
