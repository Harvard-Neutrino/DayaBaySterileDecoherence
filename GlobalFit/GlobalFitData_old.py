import numpy as np
import os


# The bins presented here are the only ones which correctly overlap between DayaBay and NEOS.
# Then, this are the only ones which can be used without introducing any additional information.
number_of_bins = 28
datlowerbin = np.array([1.3,1.5,1.7,1.9,2.1,2.3,2.5,2.7,2.9,3.1,3.3,3.5,3.7,3.9,4.1,4.3,4.5,
               4.7,4.9,5.1,5.3,5.5,5.7,5.9,6.1,6.3,6.5,6.7])
datupperbin = np.array([1.5,1.7,1.9,2.1,2.3,2.5,2.7,2.9,3.1,3.3,3.5,3.7,3.9,4.1,4.3,4.5,4.7,
               4.9,5.1,5.3,5.5,5.7,5.9,6.1,6.3,6.5,6.7,6.9])

# Lo que s'ha de fer per aquest matching és construir una funció
# Repetir plot ratio NEOS/DB per mirar si el problema és haver fet un 2x1 a NEOS.



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
fudge = 161.1
fudge = 1.
all_data = {'EH1': np.concatenate((txt_to_array(dirDB+"DataEH1.dat")[1:29,3:5],txt_to_array(dirDB+"DataEH1.dat")[1:29,6:7]),axis=1),
            'EH2': np.concatenate((txt_to_array(dirDB+"DataEH2.dat")[1:29,3:5],txt_to_array(dirDB+"DataEH2.dat")[1:29,6:7]),axis=1),
            'EH3': np.concatenate((txt_to_array(dirDB+"DataEH3.dat")[1:29,3:5],txt_to_array(dirDB+"DataEH3.dat")[1:29,6:7]),axis=1),
            'NEOS':fudge*txt_to_array(dirNEOS+'AllData.dat')[3:-2:2,0:3]+fudge*txt_to_array(dirNEOS+'AllData.dat')[4:-2:2,0:3]}
