import matplotlib.pyplot as plt
import numpy as np

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

data = txt_to_array('SMChi2.dat')
print(data[:,0:2])
print(data[:,2]-5000)
plt.contour(data[:,0].reshape(6,7),data[:,1].reshape(6,7),(data[:,2]-5000).reshape(6,7))
plt.savefig('Figures/SMContour.png')
