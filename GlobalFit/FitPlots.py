import sys
import os
sys.path.append(os.getcwd()[:-10]+"/Common")
sys.path.append(os.getcwd()[:-10]+"/NEOS")
sys.path.append(os.getcwd()[:-10]+"/DayaBay")


import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib import rc

import Models
import GlobalFit as GF
import DayaBay as DB

DB_test = DB.DayaBay()

cwd = os.getcwd()
path_to_style=cwd+'/Figures'
# plt.style.use(path_to_style+r"/paper.mplstyle")

# matplotlib.rcParams.update({'text.usetex': True})

# matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

fitter = GF.GlobalFit()

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

def getChi2(mass = 2.5e-3,angl = 0.0841):
    model = Models.PlaneWaveSterile(Sin22Th14 = angl, DM2_41 = mass)
    chi2 = fitter.get_poisson_chi2(model)
    # chi2 = DB_test.get_poisson_chi2(model)
    print(mass,angl,chi2)
    return chi2

# -------------------------------------------
# Sterile stuff - plane wave
# -------------------------------------------

data = txt_to_array('PWSterileChi2_new.dat')
# data[:,2] = data[:,2]-np.min(data[:,2])

data = np.unique(data,axis=0)

nang = 145
nmas = int(len(data)/145)

xxyy = data[:,:2]

min_index = np.where(data[:,2] == np.min(data[:,2]))[0][0]
null_hyp = getChi2(0,0)
bestfit = data[min_index]
print(bestfit)
print(null_hyp)


figSt,axSt = plt.subplots(figsize = (7,7))
# axSt.contour(data[:,1].reshape(nmas,nang),data[:,0].reshape(nmas,nang),(data[:,2]-bestfit[2]).reshape(nmas,nang),levels = [2.30,6.18,11.83])
axSt.tricontour(data[:,1],data[:,0],(data[:,2]-bestfit[2]),levels = [2.30,6.18,11.83])
# axSM.scatter(,0.07821,marker = '+', label = 'Coherent best fit')
axSt.scatter(data[:,1],data[:,0],marker = '+', s = 1.)
axSt.scatter(bestfit[1],bestfit[0],marker = '+', label = 'Our best fit')
# axSM.scatter(2.471,0.0841, marker = '+', label = 'DB best fit')
axSt.grid(linestyle = '--')
axSt.legend(loc = 'lower right', fontsize = 16)
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
figSt.savefig('Figures/PWContour_bestfit_new.png')

# --------
# Null hypothesis
# --------

figSt,axSt = plt.subplots(figsize = (7,7))
# axSt.contour(data[:,1].reshape(nmas,nang),data[:,0].reshape(nmas,nang),(data[:,2]-null_hyp).reshape(nmas,nang),levels = [2.30,6.18,11.83])
axSt.tricontour(data[:,1],data[:,0],(data[:,2]-null_hyp),levels = [2.30,6.18,11.83])
# axSM.scatter(,0.07821,marker = '+', label = 'Coherent best fit')
axSt.scatter(data[:,1],data[:,0],marker = '+', s = 1.)
axSt.scatter(bestfit[1],bestfit[0],marker = '+', label = 'Our best fit')
# axSM.scatter(2.471,0.0841, marker = '+', label = 'DB best fit')
axSt.grid(linestyle = '--')
axSt.legend(loc = 'lower right', fontsize = 16)
axSt.tick_params(axis='x', labelsize=13)
axSt.tick_params(axis='y', labelsize=13)
axSt.set_xscale('log')
axSt.set_yscale('log')
axSt.set_ylabel(r"$\Delta m^2_{14} (eV^2)$", fontsize = 16)
axSt.set_xlabel(r"$\sin^2 2 \theta_{14}$", fontsize = 16)
axSt.set_xlim([1e-3,1])
axSt.set_ylim([1e-2,10])
# figSt.suptitle(r'Sterile PW best fit: $\Delta m^2_{14} = 1.8·10^{-2} eV^2$, $\sin^2 2\theta_{14} = 0.00346$', fontsize = 17)
figSt.suptitle('Null hypothesis:'+str(null_hyp))
figSt.savefig('Figures/PWContour_nullhyp_new.png')
