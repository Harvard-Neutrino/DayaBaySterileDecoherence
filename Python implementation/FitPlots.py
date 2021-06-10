import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib import rc

import os
cwd = os.getcwd()
path_to_style=cwd+'/Figures'
# plt.style.use(path_to_style+r"/paper.mplstyle")

# matplotlib.rcParams.update({'text.usetex': True})

# matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]


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

data = txt_to_array('SMChi.dat')
# data[:,2] = data[:,2]-np.min(data[:,2])
data[:,0] *= 1e3
min_index = np.where(data[:,2] == np.min(data[:,2]))[0][0]
bestfit = data[min_index]
print(bestfit)
nang = 40
nmas = 40

figSM,axSM = plt.subplots(figsize = (7,7))
axSM.contour(data[:,0].reshape(nmas,nang),data[:,1].reshape(nmas,nang),(data[:,2]-np.min(data[:,2])).reshape(nmas,nang),levels = [2.30,6.18,11.83])
axSM.scatter(bestfit[0],bestfit[1],marker = '+', label = 'Our best fit')
axSM.scatter(2.4,0.0841, marker = '+', label = 'DB best fit')
axSM.grid(linestyle = '--')
axSM.legend(loc = 'upper right', fontsize = 16)
axSM.tick_params(axis='x', labelsize=13)
axSM.tick_params(axis='y', labelsize=13)
axSM.set_xlabel(r"$\Delta m^2_{13} (eV^2)$", fontsize = 16)
axSM.set_ylabel(r"$\sin^2 2 \theta_{13}$", fontsize = 16)
figSM.suptitle(r'Our best fit: $\Delta m^2_{13} = 2.5·10^{-3} eV^2$, $\sin^2 2\theta_{13} = 0.07821$', fontsize = 17)
figSM.savefig('Figures/SMContour.png')
