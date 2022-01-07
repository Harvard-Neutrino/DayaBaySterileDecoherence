import numpy as np
import os


# -----------------------------------------------------
#    EXPERIMENTAL PARAMETERS
# -----------------------------------------------------

# NAMES
# -----
# Here we introduce the names of the experimental halls, of the reactors, and of the isotopes to consider for the flux.
exp_names = ['EH1','EH2','EH3']
reac_names = ['D1','D2','L1','L2','L3','L4']
isotopes = ['U235','U238','PU239','PU241']


# AVERAGE OF FISSION FRACTIONS:
# -----------------------------
# This has been taken from Table IX (nine) on 1607.05378.
# We neglect differences between different halls (they are small)
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


# DISTANCE FROM REACTORS TO EXP. HALLS
# ------------------------------------
# Obtained from arXiv:1610.04802  PHYSICAL REVIEW D 95, 072006 (2017)
# Averaged the distances of the detector in each hall in quadrature (in meters)
distance = {'EH1': {'D1': 360.167,'D2': 370.089,'L1': 903.41, 'L2': 817.03, 'L3': 1353.93, 'L4': 1265.61},
            'EH2': {'D1': 1334.96,'D2': 1360.52,'L1': 470.278,'L2': 492.473,'L3': 558.145, 'L4': 500.141},
            'EH3': {'D1': 1921.39,'D2': 1895.92,'L1': 1536.93,'L2': 1537.25,'L3': 1555.56, 'L4': 1529.06}}




# -----------------------------------------------------
#   RECONSTRUCTION MATRIX
# -----------------------------------------------------

# This information is saved in separate file 'ReconstructMatrix.dat'.
# We define this function to access its info.
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

# We use the previous function to read the reconstruction matrix,
# which is used to transform from real energies to reconstructed energies.
dir = os.path.dirname(os.path.abspath(__file__))+"/Data/"

reconstruct_mat = txt_to_array(dir+"ReconstructMatrix.dat")
