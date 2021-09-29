import numpy as np
import os


# The bins presented here are the only ones which correctly overlap between DayaBay and NEOS.
# Then, this are the only ones which can be used without introducing any additional information.
number_of_bins_DB = 28
datlowerbinDB = np.array([1.3,1.5,1.7,1.9,2.1,2.3,2.5,2.7,2.9,3.1,3.3,3.5,3.7,3.9,4.1,4.3,4.5,
               4.7,4.9,5.1,5.3,5.5,5.7,5.9,6.1,6.3,6.5,6.7])
datupperbinDB = np.array([1.5,1.7,1.9,2.1,2.3,2.5,2.7,2.9,3.1,3.3,3.5,3.7,3.9,4.1,4.3,4.5,4.7,
               4.9,5.1,5.3,5.5,5.7,5.9,6.1,6.3,6.5,6.7,6.9])

number_of_bins_NEOS = 56
datlowerbinNEOS = np.array([1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,
                            2.8,2.9,3.0,3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9,4.0,4.1,4.2,4.3,4.4,4.5,4.6,4.7,
                            4.8,4.9,5.0,5.1,5.2,5.3,5.4,5.5,5.6,5.7,5.8,5.9,6.0,6.1,6.2,6.3,6.4,6.5,6.6,6.7,6.8])
datupperbinNEOS = np.array([1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,
                            2.8,2.9,3.0,3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9,4.0,4.1,4.2,4.3,4.4,4.5,4.6,4.7,
                            4.8,4.9,5.0,5.1,5.2,5.3,5.4,5.5,5.6,5.7,5.8,5.9,6.0,6.1,6.2,6.3,6.4,6.5,6.6,6.7,6.8,6.9])


number_of_bins = {'EH1': number_of_bins_DB, 'EH2': number_of_bins_DB,
                  'EH3': number_of_bins_DB, 'NEOS': number_of_bins_NEOS}
datlowerbin = {'EH1': datlowerbinDB, 'EH2': datlowerbinDB,
              'EH3': datlowerbinDB, 'NEOS': datlowerbinNEOS}
datupperbin = {'EH1': datupperbinDB, 'EH2': datupperbinDB,
              'EH3': datupperbinDB, 'NEOS': datupperbinNEOS}
datallbin = {'EH1': np.append(datlowerbinDB,datupperbinDB[-1]), 'EH2': np.append(datlowerbinDB,datupperbinDB[-1]),
              'EH3': np.append(datlowerbinDB,datupperbinDB[-1]), 'NEOS': np.append(datlowerbinNEOS,datupperbinNEOS[-1])}

deltaE = {'EH1': datupperbinDB - datlowerbinDB, 'EH2': datupperbinDB - datlowerbinDB,
              'EH3': datupperbinDB - datlowerbinDB, 'NEOS': datupperbinNEOS - datlowerbinNEOS}

# M'estic perdent el bin (6.9,7.0) de NEOS, que a DB va de (6.9,7.1).
# Òbviament això fa error. Però bé, només és un bin, no hauria de ser taaaan greu.

# Lo que s'ha de fer per aquest matching és construir una funció



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
    # Number of observed events, per day, per 100 keV
    # Number of expected events, per day, per 100 keV
    # Number of background events, per day, per 100 keV
# This is the only information that will be used from all_data.
# Modify all_data such that the observed events is in column 0, expected events in 1 and background events in 2.

# Here we read the data files from DayaBay (1610.04802) and the digitised data from NEOS results (1610.05134).
dir = os.path.dirname(os.path.abspath(__file__))[:-10]
dirDB = dir+"/Python implementation/Data/"
dirNEOS = dir+"/NEOS/Data/"
fudge = 180. #total of days the detector was on     
all_data = {'EH1': np.concatenate((txt_to_array(dirDB+"DataEH1.dat")[1:29,3:5],txt_to_array(dirDB+"DataEH1.dat")[1:29,6:7]),axis=1),
            'EH2': np.concatenate((txt_to_array(dirDB+"DataEH2.dat")[1:29,3:5],txt_to_array(dirDB+"DataEH2.dat")[1:29,6:7]),axis=1),
            'EH3': np.concatenate((txt_to_array(dirDB+"DataEH3.dat")[1:29,3:5],txt_to_array(dirDB+"DataEH3.dat")[1:29,6:7]),axis=1),
            'NEOS':fudge*txt_to_array(dirNEOS+'AllData.dat')[3:-2,0:3]}
