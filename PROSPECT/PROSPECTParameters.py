import numpy as np
import os


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


dir = os.path.dirname(os.path.abspath(__file__))+"/Data/"



# -----------------------------------------------------
#    EXPERIMENTAL PARAMETERS
# -----------------------------------------------------

# NAMES
# -----
# Here we introduce the names of the experimental halls, of the reactors, and of the isotopes.
exp_names = ['PROSPECT']
reac_names = ['HFIR']

active_segments = np.array([15,16,19,20,22,30,33,35,37,38,39,45,49,51,53,54,57,58,59,61,62,64,65,66,67,71,72,74,75,76,77,78,80,81,82,85,
          88,89,90,91,92,93,95,96,99,100,101,103,104,105,106,108,109,110,113,114,116,117,118,119,120,123,124,129,131,132,134,135,137,138])
number_of_segments = active_segments.shape[0]

isotopes = ['U235']#,'U238','PU239','PU241']



# DISTANCE FROM SEGMENTS TO REACTOR
# ------------------------------------

# PROSPECT has 70 different active segments, each of them can be interpreted as an independent detector.
# The distance of each segment to the reactor core is written in '1.1_Osc_SegmentMap.txt'.
segment_map = txt_to_array(dir+'1.1_Osc_SegmentMap.txt')
distance = dict([(segment,{'HFIR':segment_map[segment,1]}) for segment in active_segments])

# All segments which have a similar distance to the reactor are put into a common group, the "baseline".
# The information on which baseline corresponds to each detector is on the same file.
# Here we define a dictionary which tells us which segments belong to each baseline.
active_segment_map = segment_map[active_segments]
baselines = dict([(bl,active_segment_map[active_segment_map[:,0] == bl][:,2].astype(int)) for bl in range(1,11)])

# Each segment has dimensions 14.5cm x 14.5cm x 117.6 cm.
# Since the detector is quite near, in principle we cannot consider the segments puntual,
# we must integrate over their width.
# We can approximate each segment by a 30cm cube. Physics in a nutshell!
segment_width = 0.50/2 # this is half-width of HFIR reactor.
width = dict([(segment,segment_width) for segment in active_segments])
# We still need to check if this will be of any use



# AVERAGE OF FISSION FRACTIONS:
# -----------------------------
# This has been taken from Table IX (nueve) on 1607.05378.
# In principle this only applies to DayaBay, but we have assumed the same works for PROSPECT.
# However, since HFIR from PROSPECT only uses U235, this should not worry us, it might be absorbed in
# a renormalisation of the flux.
mean_fis_frac = {'U235':0.655}#,'U238':0.0720,'PU239':0.235,'PU241':0.038}




# EFFICIENCY AND QUANTITY OF DETECTORS
# ------------------------------------

# Each segment has its own relative efficiency, which is saved in the file '1.5_Osc_RelativeEfficiencies.txt'
efficiency_data = txt_to_array(dir+'1.5_Osc_RelativeEfficiencies.txt')
efficiency = dict([(segment,efficiency_data[segment,2]) for segment in active_segments])
# print(efficiency)



#   RECONSTRUCTION/RESPONSE MATRIX
# -----------------------------------------------------

# Every active segment has its own response matrix, which is saved in '1.3_Osc_DetResponseXX.txt'
reconstruct_mat = dict([(segment,txt_to_array(dir+'1.3_Osc_DetResponse{}.txt'.format(segment))) for segment in active_segments])
# We use the previous function to read the reconstruction matrix, which is used
# to transform from real energies to reconstructed energies.
# Each reconstruction matrix is a 82 (Etrue) x 16 (Erec) matrix.


# COVARIANCE MATRIX
# -----------------------------------------------------------

# The PROSPECT Covariance Matrix is a 160x160 matrix (10 baselines x 16 energy bins)
# in Eprompt energy. It should only be used in the chi2 implementation.
covariance_matrix = txt_to_array(dir+"1.2_Osc_CovarianceMatrix.txt")


# -------------------------------------------------------------
#   EXPERIMENTAL DATA
# -------------------------------------------------------------



# TRUE ENERGY BINS
# -----------------------------------------------------------

# Histogram bins of the neutrino real energies
# The histogram bins begin at 1.800 since it is the minimum energy allowed for
# neutrinos obtained from inverse beta decay. Up to 10 MeV, every 100 keV.
nulowerbin = np.arange(1.8,10,0.1)
nuupperbin = np.arange(1.9,10.1,0.1)


# PROMPT ENERGY (DATA) BINS
# -------------------------------------------------------------

# Histogram bins of the measured data.
number_of_bins = 16
datlowerbin = np.arange(0.8,7.2,0.4)
datupperbin = np.arange(1.2,7.6,0.4)
deltaE = datupperbin - datlowerbin
# Maybe you can consider to repeat this array 10 times.



# PROSPECT DATA
# -------------------------------------------------------------

# Contains the full data from PROSPECT for every segment. The columns are the following:
#  0: Bin Center
#  1: Background subtracted IBD counts # LOOK OUT! THIS DOES NOT INCLUDE BACKGROUND
#  2: Total Stats Error
#  3: Background Spectrum counts
#  4: Background Stats Error
# The errors in this data are computed according to the covariance matrix.
all_data = dict([(segment,txt_to_array(dir+'1.4_Osc_Prompt{}.txt'.format(segment))) for segment in active_segments])
observed_data = dict([(segment,txt_to_array(dir+'1.4_Osc_Prompt{}.txt'.format(segment))[:,1]) for segment in active_segments])
background =  dict([(segment,txt_to_array(dir+'1.4_Osc_Prompt{}.txt'.format(segment))[:,3]) for segment in active_segments])
error_data =  dict([(segment,txt_to_array(dir+'1.4_Osc_Prompt{}.txt'.format(segment))[:,2]) for segment in active_segments])


# PROSPECT EXPECTED EVENTS (NO OSCILLATIONS)
# -------------------------------------------------------------

# The files 1.6_Osc_NullOscPredXX.txt contain the prediction of events without oscillations.
# This will be useful to normalize the data.
# I don't know yet what does "No oscillations" mean. Does it mean no sterile oscillations, or absolutely no oscillations? It is not the same.
predicted_data = dict([(bl,txt_to_array(dir+'1.6_Osc_NullOscPred{}.txt'.format(bl))[:,1]) for bl in range(1,11)])
