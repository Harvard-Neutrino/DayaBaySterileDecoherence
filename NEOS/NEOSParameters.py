import numpy as np

# -----------------------------------------------------
#    EXPERIMENTAL PARAMETERS
# -----------------------------------------------------

# NAMES
# -----
# Here we introduce the names of the experimental halls, of the reactors, and of the isotopes.
exp_names = ['NEOS']
reac_names = ['H5']
isotopes = ['U235','U238','PU239','PU241']



# FUDGE FACTORS:
# These are just some factors Carlos entered in case something didn't work right.
fudge_factors = {'NEOS':1.0}


# AVERAGE OF FISSION FRACTIONS:
# -----------------------------
# This has been taken from Table IX (nueve) on 1607.05378.
# One might want to differentiate between different halls (slight diff.)
mean_fis_frac = {'U235':0.55825,'U238':0.07600,'PU239':0.30975,'PU241':0.05575}


# EFFICIENCY AND QUANTITY OF DETECTORS
# ------------------------------------
# We have used the same efficiency as in EH1 from DB
efficiency = {'NEOS': 0.82*0.97}


# DISTANCE FROM REACTORS TO EXP. HALLS
# ------------------------------------
distance = {'NEOS': {'H5': 23.7}}
width = {'NEOS': 0.3} # this is half-width, i.e. NEOS has L = (23.7 +- 0.3) m.


# Here we define a function to easily access the square of this distance.
def get_distance2(experiment,reactor):
    """
    Input:
    experiment (str): name of the experimental hall.
    reactor (str): name of the reactor.

    Output:
    The square distance between these EH and reactor.
    """
    return distance[experiment][reactor]**2



# -----------------------------------------------------
#   MATRICES (Covariance matrix and reconstruction matrix)
# -----------------------------------------------------

# This information is saved in separate files 'NeutrinoCovMatrix.dat' and
# 'ReconstructMatrix.dat'. We define this function to access their info.
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

# We use the previous function to read the neutrino covariance matrix used in the
# chi2 test and the reconstruction matrix, which is used to transform from real
# energies to reconstruted energies.
dir = "Data/"
reconstruct_mat = txt_to_array(dir+"ReconstructMatrix.dat")


# -------------------------------------------------------------
#   HISTOGRAM BINS
# -------------------------------------------------------------

# Histogram bins of the neutrino real energies
# The histogram bins begin at 1.800 since it is the minimum energy allowed for
# neutrinos obtained from inverse beta decay.
nulowerbin = np.array([1.800, 2.125, 2.375, 2.625, 2.875, 3.125, 3.375, 3.625, 3.875, 4.125, 4.375, 4.625, 4.875,
              5.125, 5.375, 5.625, 5.875, 6.125, 6.375, 6.625, 6.875, 7.125, 7.375, 7.625, 7.875, 8.125])
nuupperbin = np.array([2.125, 2.375, 2.625, 2.875, 3.125, 3.375, 3.625, 3.875, 4.125, 4.375, 4.625, 4.875, 5.125,
              5.375, 5.625, 5.875, 6.125, 6.375, 6.625, 6.875, 7.125, 7.375, 7.625, 7.875, 8.125, 12.00])


# -------------------------------------------------------
# NEUTRINO SPECTRUM FROM DAYA BAY
# -------------------------------------------------------
# The bins are the ones up above.
# This is weighted with the IBD cross-section.
# Units cm^2/fission/MeV × 10^−46
spectrum = np.array([344.19, 770.96, 1080.9, 1348.4, 1528.8, 1687.0, 1746.6, 1760.6,
                     1719.3, 1617.6, 1466.5, 1309.3, 1203.0, 1105.4, 976.50, 852.31,
                     713.19, 573.90, 463.54, 368.70, 274.56, 190.00, 132.08, 92.114,
                     56.689, 4.0214])
