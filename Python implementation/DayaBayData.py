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
#   HISTOGRAM BINS
# -------------------------------------------------------------

# Histogram bins of the measured data.
number_of_bins = 35
datlowerbin = np.array([0.7,1.3,1.5,1.7,1.9,2.1,2.3,2.5,2.7,2.9,3.1,3.3,3.5,3.7,3.9,4.1,4.3,4.5,
               4.7,4.9,5.1,5.3,5.5,5.7,5.9,6.1,6.3,6.5,6.7,6.9,7.1,7.3,7.5,7.7,7.9])
datupperbin = np.array([1.3,1.5,1.7,1.9,2.1,2.3,2.5,2.7,2.9,3.1,3.3,3.5,3.7,3.9,4.1,4.3,4.5,4.7,
               4.9,5.1,5.3,5.5,5.7,5.9,6.1,6.3,6.5,6.7,6.9,7.1,7.3,7.5,7.7,7.9,12.0])


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

# Contains the full data from 1610.04802 for all three experimental halls.
# Columns:
# 0: Emin; 1: Emax, 2: Ecentral
# 3: Nobs  4: Npred (including background)
# 5: Nprednoosc; 6: Nbkg
all_data = {'EH1': txt_to_array(dir+"DataEH1.dat")[:,0:7],
            'EH2': txt_to_array(dir+"DataEH2.dat")[:,0:7],
            'EH3': txt_to_array(dir+"DataEH3.dat")[:,0:7]}


observed_data = {'EH1': txt_to_array(dir+"DataEH1.dat")[:,3],
                 'EH2': txt_to_array(dir+"DataEH2.dat")[:,3],
                 'EH3': txt_to_array(dir+"DataEH3.dat")[:,3]}

predicted_bkg = {'EH1': txt_to_array(dir+"DataEH1.dat")[:,6],
                 'EH2': txt_to_array(dir+"DataEH2.dat")[:,6],
                 'EH3': txt_to_array(dir+"DataEH3.dat")[:,6]}
