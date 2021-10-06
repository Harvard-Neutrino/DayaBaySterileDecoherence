import numpy as np
import os

# -----------------------------------------------------
#    EXPERIMENTAL PARAMETERS
# -----------------------------------------------------

# NAMES
# -----
# Here we introduce the names of the experimental halls, of the reactors, and of the isotopes.
exp_names = ['NEOS']
reac_names = ['H5']
isotopes = ['U235','U238','PU239','PU241']


# AVERAGE OF FISSION FRACTIONS:
# -----------------------------
# This has been taken from Table IX (nueve) on 1607.05378.
# In principle this only applies to DayaBay, but we have assumed the same works for NEOS.
# It is important to keep this because we must use the same initial flux function for all reactors,
# in order for the nuissance parameters in our global fit to be reasonable.
mean_fis_frac = {'U235':0.655,'U238':0.0720,'PU239':0.235,'PU241':0.038}


# EFFICIENCY AND QUANTITY OF DETECTORS
# ------------------------------------
# We have used the same efficiency as in EH1 from DB.
# In principle, this is not important. Any change on this would be
# absorbed by a normalisation to the observed data.
efficiency = {'NEOS': 0.82*0.97}


# DISTANCE FROM REACTORS TO EXP. HALLS
# ------------------------------------
# NEOS only has one reactor and one detector.
# However, since they are very near, we cannot consider the detector puntual,
# and must integrate over its width. That's why we must consider its width.
distance = {'NEOS': {'H5': 23.7}}
width = {'NEOS': 0.3} # this is half-width, i.e. NEOS has L = (23.7 +- 0.3) m.




# -----------------------------------------------------
#   RECONSTRUCTION/RESPONSE MATRIX
# -----------------------------------------------------

# This information is saved in a separate file 'ReconstructMatrix.dat'.
# We define this function to read and save the info on the file.
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

# We use the previous function to read the reconstruction matrix, which is used
# to transform from real energies to reconstructed energies.
# This response matrix is normalised to the same total sum as DB response matrix.
dir = os.path.dirname(os.path.abspath(__file__))+"/Data/"
reconstruct_mat = txt_to_array(dir+"ReconstructMatrix.dat")

# In order to


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
# From 1607.05378
# The bins are the ones up above.
# This is weighted with the IBD cross-section.
# Units cm^2/fission/MeV × 10^−46
spectrum = np.array([344.19, 770.96, 1080.9, 1348.4, 1528.8, 1687.0, 1746.6, 1760.6,
                     1719.3, 1617.6, 1466.5, 1309.3, 1203.0, 1105.4, 976.50, 852.31,
                     713.19, 573.90, 463.54, 368.70, 274.56, 190.00, 132.08, 92.114,
                     56.689, 4.0214])
