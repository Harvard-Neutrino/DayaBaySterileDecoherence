import numpy as np
import os


# -----------------------------------------------------
#    EXPERIMENTAL PARAMETERS
# -----------------------------------------------------

# NAMES
# -----
# Here we introduce the names of the experimental halls, of the reactors, and of the isotopes.
exp_names = ['EH1','EH2','EH3']
reac_names = ['D1','D2','L1','L2','L3','L4']
isotopes = ['U235','U238','PU239','PU241']



# FUDGE FACTORS:
# These are just some factors Carlos entered in case something didn't work right.
fudge_factors = {'EH1':1.0,'EH2':1.0,'EH3':1.0}


# AVERAGE OF FISSION FRACTIONS:
# -----------------------------
# This has been taken from Table IX (nueve) on 1607.05378.
# One might want to differentiate between different halls (slight diff.)
mean_fis_frac = {'U235':0.55825,'U238':0.07600,'PU239':0.30975,'PU241':0.05575}


# EFFICIENCY AND QUANTITY OF DETECTORS
# ------------------------------------
# In this quantity we have also added the number of detectors per hall.
# EH1 has two ~ 20kT detectors, EH2 has two, and EH3 has 4.
# However, we have taken into account slight variations of nº protons with respecte to AD1.
# The efficiency is computed using \epsilon_e * \epsilon_mu
# All data obtained from table VI of 1610.04802.
efficiency = {'EH1': 0.82*0.97*(1.+1.0013),
              'EH2': 0.85*0.97*(2.-0.0025+0.0002),
              'EH3': 0.98*0.97*(4.-0.0012+0.0024-0.0025-0.0005)}

# I understand that the following implementation is incorrect, since
# table VI gives \Delta N in units of %: 0.23% is 0.0023.
# efficiency = {'EH1': 2.0*0.82*0.97*(1.+0.),'EH2': 2*0.85*0.97*(1.-0.18),
#               'EH3': 4.0*0.98*0.97*(1.-0.18)}



# DISTANCE FROM REACTORS TO EXP. HALLS
# ------------------------------------
# Obtained from arXiv:1610.04802  PHYSICAL REVIEW D 95, 072006 (2017)
# Averaged the distances of the detector in each hall in quadrature (but I don't think so)
# distance = {'EH1': {'D1': 360.167,'D2': 370.089,'L1': 903.41, 'L2': 817.03, 'L3': 1353.93, 'L4': 1265.61},
#             'EH2': {'D1': 1334.96,'D2': 1360.52,'L1': 470.278,'L2': 492.473,'L3': 558.145, 'L4': 500.141},
#             'EH3': {'D1': 1921.39,'D2': 1895.92,'L1': 1536.93,'L2': 1537.25,'L3': 1555.56, 'L4': 1529.06}}

distance = {'EH1': {'D1': 360.167,'D2': 370.089,'L1': 903.41, 'L2': 817.03, 'L3': 1353.93, 'L4': 1265.61},
            'EH2': {'D1': 1334.96,'D2': 1360.52,'L1': 470.278,'L2': 492.473,'L3': 558.145, 'L4': 500.141},
            'EH3': {'D1': 1921.39,'D2': 1895.92,'L1': 1536.93,'L2': 1537.25,'L3': 1555.56, 'L4': 1529.06}}


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
dir = os.path.dirname(os.path.abspath(__file__))+"/Data/"

# The neutrino flux covariance matrix is obtained from 1607.05378 Table 13.
# Units (cm^2/fission/MeV)^2 x 10^{-92}
neutrino_covariance_matrix = txt_to_array(dir+"NeutrinoCovMatrix.dat")
cov_lower_edges = np.array([1.800, 2.125, 2.375, 2.625, 2.875, 3.125, 3.375, 3.625, 3.875, 4.125, 4.375, 4.625, 4.875, 5.125, 5.375, 5.625, 5.875, 6.125, 6.375, 6.625, 6.875, 7.125, 7.375, 7.625, 7.875, 8.125])
cov_upper_edges = np.array([2.125, 2.375, 2.625, 2.875, 3.125, 3.375, 3.625, 3.875, 4.125, 4.375, 4.625, 4.875, 5.125, 5.375, 5.625, 5.875, 6.125, 6.375, 6.625, 6.875, 7.125, 7.375, 7.625, 7.875, 8.125, 12.000])

reconstruct_mat = txt_to_array(dir+"ReconstructMatrix.dat")
for i in range(len(reconstruct_mat)):
    print(np.sum(reconstruct_mat[i]),np.sum(reconstruct_mat[:,i]))
print(np.sum(reconstruct_mat))



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
