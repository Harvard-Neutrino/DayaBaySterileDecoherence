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

# -------------------------------------------
# Sterile stuff - plane wave
# -------------------------------------------

data = txt_to_array('SMSterileChi2.dat')
# data[:,2] = data[:,2]-np.min(data[:,2])
min_index = np.where(data[:,2] == np.min(data[:,2]))[0][0]
bestfit = data[min_index]
print(bestfit)
nang = 9
nmas = 10

figSt,axSt = plt.subplots(figsize = (7,7))
axSt.contour(data[:,1].reshape(nmas,nang),data[:,0].reshape(nmas,nang),(data[:,2]-np.min(data[:,2])).reshape(nmas,nang),levels = [2.30,6.18,11.83])
# axSM.scatter(,0.07821,marker = '+', label = 'Coherent best fit')
axSt.scatter(bestfit[1],bestfit[0],marker = '+', label = 'Our best fit')
# axSM.scatter(2.471,0.0841, marker = '+', label = 'DB best fit')
axSt.grid(linestyle = '--')
axSt.legend(loc = 'upper right', fontsize = 16)
axSt.tick_params(axis='x', labelsize=13)
axSt.tick_params(axis='y', labelsize=13)
axSt.set_xscale('log')
axSt.set_yscale('log')
axSt.set_ylabel(r"$\Delta m^2_{14} (eV^2)$", fontsize = 16)
axSt.set_xlabel(r"$\sin^2 2 \theta_{14}$", fontsize = 16)
axSt.set_xlim([1e-3,1])
axSt.set_ylim([1e-2,10])
# figSt.suptitle(r'Sterile PW best fit: $\Delta m^2_{14} = 1.8·10^{-2} eV^2$, $\sin^2 2\theta_{14} = 0.00346$', fontsize = 17)
figSt.suptitle('Best fit:'+str(bestfit))
figSt.savefig('Figures/SterilePWContour.png')
