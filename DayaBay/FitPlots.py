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
dir = 'PlotData/'

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

data = txt_to_array(dir+'SMChi2.dat')
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
axSM.scatter(2.5,0.0841, marker = '+', label = 'DB best fit')
axSM.grid(linestyle = '--')
axSM.legend(loc = 'upper right', fontsize = 16)
axSM.tick_params(axis='x', labelsize=13)
axSM.tick_params(axis='y', labelsize=13)
axSM.set_xlim([2.2,2.9])
axSM.set_ylim([0.02,0.13])
axSM.set_xlabel(r"$\Delta m^2_{13} (eV^2)$", fontsize = 16)
axSM.set_ylabel(r"$\sin^2 2 \theta_{13}$", fontsize = 16)
figSM.suptitle(r'PW best fit: $\Delta m^2_{13} = 2.5·10^{-3} eV^2$, $\sin^2 2\theta_{13} = 0.07821$', fontsize = 17)
figSM.savefig('Figures/SMContour.png')


# -------------------------------------------
# Same thing, but with decoherence
# -------------------------------------------

data = txt_to_array(dir+'SMWPChi2.dat')
# data[:,2] = data[:,2]-np.min(data[:,2])
data[:,0] *= 1e3
min_index = np.where(data[:,2] == np.min(data[:,2]))[0][0]
bestfit = data[min_index]
print(bestfit)
nang = 40
nmas = 40

figSM,axSM = plt.subplots(figsize = (7,7))
axSM.contour(data[:,0].reshape(nmas,nang),data[:,1].reshape(nmas,nang),(data[:,2]-np.min(data[:,2])).reshape(nmas,nang),levels = [2.30,6.18,11.83])
axSM.scatter(2.5,0.07821,marker = '+', label = 'PW best fit')
axSM.scatter(bestfit[0],bestfit[1],marker = '+', label = 'WP best fit')
axSM.scatter(2.5,0.0841, marker = '+', label = 'DB best fit')
axSM.grid(linestyle = '--')
axSM.legend(loc = 'upper right', fontsize = 16)
axSM.tick_params(axis='x', labelsize=13)
axSM.tick_params(axis='y', labelsize=13)
axSM.set_xlim([2.2,2.9])
axSM.set_ylim([0.02,0.13])
axSM.set_xlabel(r"$\Delta m^2_{13} (eV^2)$", fontsize = 16)
axSM.set_ylabel(r"$\sin^2 2 \theta_{13}$", fontsize = 16)
figSM.suptitle(r'WP best fit: $\Delta m^2_{13} = 2.49·10^{-3} eV^2$, $\sin^2 2\theta_{13} = 0.09051$', fontsize = 17)
figSM.savefig('Figures/WPContour.png')

# -------------------------------------------
# Sterile stuff - plane wave
# -------------------------------------------

data = txt_to_array(dir+'SterilePWChi2.dat')
# data[:,2] = data[:,2]-np.min(data[:,2])
min_index = np.where(data[:,2] == np.min(data[:,2]))[0][0]
bestfit = data[min_index]
print(bestfit)
nang = 40
nmas = 40

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
figSt.suptitle(r'Sterile PW best fit: $\Delta m^2_{14} = 1.8·10^{-2} eV^2$, $\sin^2 2\theta_{14} = 0.00346$', fontsize = 17)
figSt.savefig('Figures/SterilePWContour.png')

# -------------------------------------------
# Sterile stuff - wave packer
# -------------------------------------------

data = txt_to_array(dir+'SterileWPChi2.dat')
# data[:,2] = data[:,2]-np.min(data[:,2])
min_index = np.where(data[:,2] == np.min(data[:,2]))[0][0]
bestfit = data[min_index]
print(bestfit)
nang = 40
nmas = 40

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
figSt.suptitle(r'Sterile WP best fit: $\Delta m^2_{14} = 0.13695 eV^2$, $\sin^2 2\theta_{14} = 0.05878$', fontsize = 17)
figSt.savefig('Figures/SterileWPContour.png')

# ---------
# Comparar steriles
dataPW = txt_to_array(dir+'SterilePWChi2.dat')
dataWP = txt_to_array(dir+'SterileWPChi2.dat')

figSt,axSt = plt.subplots(figsize = (7,7))
axSt.contour(dataPW[:,1].reshape(nmas,nang),dataPW[:,0].reshape(nmas,nang),(dataPW[:,2]-np.min(data[:,2])).reshape(nmas,nang),levels = [6.18],colors='red')
axSt.contour(dataWP[:,1].reshape(nmas,nang),dataWP[:,0].reshape(nmas,nang),(dataWP[:,2]-np.min(data[:,2])).reshape(nmas,nang),levels = [6.18],colors='blue')
axSt.grid(linestyle = '--')
# axSt.legend(loc = 'upper right', fontsize = 16)
axSt.tick_params(axis='x', labelsize=13)
axSt.tick_params(axis='y', labelsize=13)
axSt.set_xscale('log')
axSt.set_yscale('log')
axSt.set_ylabel(r"$\Delta m^2_{14} (eV^2)$", fontsize = 16)
axSt.set_xlabel(r"$\sin^2 2 \theta_{14}$", fontsize = 16)
figSt.suptitle(r'Sterile WP best fit: $\Delta m^2_{14} = 0.13695 eV^2$, $\sin^2 2\theta_{14} = 0.05878$', fontsize = 17)
figSt.savefig('Figures/SterileComparison.png')
