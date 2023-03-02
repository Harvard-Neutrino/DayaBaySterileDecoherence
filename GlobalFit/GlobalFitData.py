import numpy as np
import os


# DATA BINS
# -------------------------

# The bins presented here are the only ones which correctly overlap between DayaBay and NEOS.
# Then, these are the only ones which can be used without introducing any additional information.

# The bins from NEOS go from 1.3 MeV to 6.9 MeV, in steps of 0.2 MeV.
number_of_bins_DB = 28
datlowerbinDB = np.array([1.3,1.5,1.7,1.9,2.1,2.3,2.5,2.7,2.9,3.1,3.3,3.5,3.7,3.9,4.1,4.3,4.5,
               4.7,4.9,5.1,5.3,5.5,5.7,5.9,6.1,6.3,6.5,6.7])
datupperbinDB = np.array([1.5,1.7,1.9,2.1,2.3,2.5,2.7,2.9,3.1,3.3,3.5,3.7,3.9,4.1,4.3,4.5,4.7,
               4.9,5.1,5.3,5.5,5.7,5.9,6.1,6.3,6.5,6.7,6.9])

# The bins from NEOS go from 1.3 MeV to 6.9 MeV, in steps of 0.1 MeV.
number_of_bins_NEOS = 56
datlowerbinNEOS = np.array([1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,
                            2.8,2.9,3.0,3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9,4.0,4.1,4.2,4.3,4.4,4.5,4.6,4.7,
                            4.8,4.9,5.0,5.1,5.2,5.3,5.4,5.5,5.6,5.7,5.8,5.9,6.0,6.1,6.2,6.3,6.4,6.5,6.6,6.7,6.8])
datupperbinNEOS = np.array([1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,
                            2.8,2.9,3.0,3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9,4.0,4.1,4.2,4.3,4.4,4.5,4.6,4.7,
                            4.8,4.9,5.0,5.1,5.2,5.3,5.4,5.5,5.6,5.7,5.8,5.9,6.0,6.1,6.2,6.3,6.4,6.5,6.6,6.7,6.8,6.9])

# Therefore, for every bin of DayaBay, there are two bins of NEOS.
# We are losing some bins from NEOS and DayaBay. However, we expect that such bins
# are not the most significant and don't modify largely our result.


# We save this information so it can be accessed per experimental hall.
number_of_bins = {'EH1': number_of_bins_DB, 'EH2': number_of_bins_DB,
                  'EH3': number_of_bins_DB, 'NEOS': number_of_bins_NEOS}

# Only the upper and lower edges from the bins, respectively:
datlowerbin = {'EH1': datlowerbinDB, 'EH2': datlowerbinDB,
              'EH3': datlowerbinDB, 'NEOS': datlowerbinNEOS}
datupperbin = {'EH1': datupperbinDB, 'EH2': datupperbinDB,
              'EH3': datupperbinDB, 'NEOS': datupperbinNEOS}

# All bin edges, including the first and the last one.
datallbin = {'EH1': np.append(datlowerbinDB,datupperbinDB[-1]), 'EH2': np.append(datlowerbinDB,datupperbinDB[-1]),
              'EH3': np.append(datlowerbinDB,datupperbinDB[-1]), 'NEOS': np.append(datlowerbinNEOS,datupperbinNEOS[-1])}

# The spacing of the bins
deltaE = {'EH1': datupperbinDB - datlowerbinDB, 'EH2': datupperbinDB - datlowerbinDB,
              'EH3': datupperbinDB - datlowerbinDB, 'NEOS': datupperbinNEOS - datlowerbinNEOS}



# DATA FILES
# ----------------------------------

# We define the following function to read the files in DB and NEOS Data directories.
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


# The data document should be separated by comma (,). For other delimiters, check the above function txt_to_array.
# The data must contain this information:
    # Number of observed events
    # Number of expected events
    # Number of background events
# Modify all_data such that the observed events is in column 0, expected events in 1 and background events in 2.

# Here we read the data files from DayaBay (1610.04802) and the digitised data from NEOS results (1610.05134).
dir = os.path.realpath(__file__)[:-len('GlobalFit/GlobalFitData.py')]
dirDB = dir+"/DayaBay/Data/"
dirNEOS = dir+"/NEOS/Data/"

fudge = 180.-46. # Total of days the NEOS detector was on. Normalised from the statistical error of figure 3(c).
# Note that the data digitised from figure 3(a) in 1610.05134 is the number of events, PER DAY.
all_data = {'EH1': np.concatenate((txt_to_array(dirDB+"DataEH1.dat")[1:29,3:5],txt_to_array(dirDB+"DataEH1.dat")[1:29,6:7]),axis=1),
            'EH2': np.concatenate((txt_to_array(dirDB+"DataEH2.dat")[1:29,3:5],txt_to_array(dirDB+"DataEH2.dat")[1:29,6:7]),axis=1),
            'EH3': np.concatenate((txt_to_array(dirDB+"DataEH3.dat")[1:29,3:5],txt_to_array(dirDB+"DataEH3.dat")[1:29,6:7]),axis=1),
            'NEOS':fudge*np.concatenate((txt_to_array(dirNEOS+'AllData.dat')[3:-2,0:2],txt_to_array(dirNEOS+'AllData.dat')[3:-2,3:4]), axis = 1)}

# We will also need the ratio data, digitised from figure 3(c) in 1610.05134
ratio_data = {'NEOS': txt_to_array(dirNEOS+'AllData.dat')[3:-2,4:]}

# The following data is the prediction of the 3+0 model (with standard oscillation parameters from nu-fit.org)
# For more information on how these data is obtained, check get_expectation in GlobalFit.py
neos_data_SM_PW = {'NEOS': np.array([4517.36176096,5059.54129615,5344.43875083,5875.71307814,6129.76758459,
                                  6547.40771313,6805.87475128,7154.26046943,7270.03639783,7482.72105086,
                                  7696.24175979,7713.68661912,7792.59449109,7863.56831131,7596.01409129,
                                  7546.82407334,7372.96447092,7306.94331942,7218.13638212,7034.82318852,
                                  6773.52857204,6584.66521499,6239.13519168,5996.98158414,5638.49164146,
                                  5414.88801142,5210.29025716,4866.41747015,4723.54608369,4466.82841058,
                                  4346.31194421,4101.88594656,3939.30305093,3726.62569052,3582.71286085,
                                  3373.86474197,3147.8437743,2962.89045478,2758.50707125,2533.57179556,
                                  2324.68323806,2145.56683505,1938.97420276,1780.84130096,1595.28943724,
                                  1425.20551605,1283.78000348,1156.44937347,995.62217022,865.76218956,
                                  764.20402218,670.08816731,528.22701962,492.46815623,421.16728167,
                                  362.07369061])}

neos_data_SM_WP = {'NEOS': np.array([4514.40653482,5056.13286082,5339.88464703,5870.6753701,6124.55711664,
                                     6541.79683228,6800.79145733,7148.90475879,7265.58052072,7478.11105342,
                                     7692.58017417,7709.95702358,7789.78859749,7860.74892697,7593.99088245,
                                     7544.81530675,7371.53700604,7305.5405309,7217.15959567,7033.87142239,
                                     6772.91127533,6584.06863347,6238.77171069,5996.6327964,5638.29800077,
                                     5414.70335219,5210.20381233,4866.33549414,4723.52560676,4466.80911281,
                                     4346.3312742,4101.90416653,3939.34434219,3726.66440575,3582.76443451,
                                     3373.91319741,3147.89779405,2962.94068697,2758.55786171,2533.61878546,
                                     2324.72800038,2145.60799765,1939.01150527,1780.87535049,1595.31943769,
                                     1425.23264815,1283.80371287,1156.47057287,995.64027169,865.77815097,
                                     764.21697912,670.09940263,528.23575122,492.47571455,421.17332215,
                                     362.07871724])}


# RECONSTRUCTION MATRIX
# ---------------------

# The response matrices are read from the DayaBay&NEOS Data directories.
reconstruct_mat = {'EH1': txt_to_array(dirDB+"ReconstructMatrix.dat"),
                   'EH2': txt_to_array(dirDB+"ReconstructMatrix.dat"),
                   'EH3': txt_to_array(dirDB+"ReconstructMatrix.dat"),
                   'NEOS': txt_to_array(dirNEOS+"ReconstructMatrix.dat")}


# NEUTRINO BIN EDGES
# ---------------------

# Histogram bins of the neutrino real energies
# The histogram bins begin at 1.800 since it is the minimum energy allowed for
# neutrinos obtained from inverse beta decay.
nulowerbin = np.array([1.800, 2.125, 2.375, 2.625, 2.875, 3.125, 3.375, 3.625, 3.875, 4.125, 4.375, 4.625, 4.875,
              5.125, 5.375, 5.625, 5.875, 6.125, 6.375, 6.625, 6.875, 7.125, 7.375, 7.625, 7.875, 8.125])
nuupperbin = np.array([2.125, 2.375, 2.625, 2.875, 3.125, 3.375, 3.625, 3.875, 4.125, 4.375, 4.625, 4.875, 5.125,
              5.375, 5.625, 5.875, 6.125, 6.375, 6.625, 6.875, 7.125, 7.375, 7.625, 7.875, 8.125, 12.00])

neutrino_lower_bin_edges = {'EH1': nulowerbin,
                            'EH2': nulowerbin,
                            'EH3': nulowerbin,
                            'NEOS': nulowerbin}

neutrino_upper_bin_edges = {'EH1': nuupperbin,
                            'EH2': nuupperbin,
                            'EH3': nuupperbin,
                            'NEOS': nuupperbin}


# NEUTRINO COVARIANCE MATRIX
# --------------------------

# For more information in the covariance matrices from each experiment,
# check the corresponding XXXXParameters.py files.
# There are some inhomogeneities between the covariance matrices, which we most work through.
# For example, the NEOS one is in prompt energy, and we must trim it to the GlobalFit number of bins.
# In principle, the DB ones should not be used, we include them here for easier generalisation.
neutrino_cov_mat = {'EH1': txt_to_array(dirDB+"NeutrinoCovMatrix.dat"),
                    'EH2': txt_to_array(dirDB+"NeutrinoCovMatrix.dat"),
                    'EH3': txt_to_array(dirDB+"NeutrinoCovMatrix.dat"),
                    'NEOS': txt_to_array(dirNEOS+"NeutrinoCovMatrix.dat")[3:-1,3:-1]}

# In principle, the NEOS covariance matrix is computed using the DB antineutrino flux, which
# relies on the assumption of 3+0 oscillations. Therefore, this should not be entirely
# self-consistent. However, we neglect the effect of this wrong assumption in the covariance matrix.
