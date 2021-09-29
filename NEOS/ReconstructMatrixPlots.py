import numpy as np
import matplotlib.pyplot as plt
import NEOS

# RECONSTRUCTMATRIXPLOTS.PY
# -------------------------
# This program allows to read the reconstruction/response matrix R of the NEOS
# detector and to plot it in a 2D contour plot, and to make some slices for
# fixed neutrino energy.
# For more information on how the response matrix was built, see NEOSData.py
# -------------------------

exp = NEOS.Neos()

# Important!
# This axis must correspond to the energies in the response matrix.
x = np.arange(0,11.95,0.05)
y = np.arange(0,11.95,0.05)
mat = exp.FromEtrueToErec


# PLOT 1
# This routine plots a 2D contour subplot of R and some slices of it in another subplot.
fig,ax = plt.subplots(1,2,figsize=(12,6),gridspec_kw=dict(left=0.07, right=0.98,bottom=0.1, top=0.93))

contplot = ax[1].contourf(x,y,np.log10(mat))
cbar = fig.colorbar(contplot)
ax[1].set_xlabel("True energy (MeV)", fontsize = 14)
ax[1].set_ylabel("Prompt energy (MeV)", fontsize = 14)

# In this case we plot the slices of 4,5,6,7 MeV,
# which correspond to the columns 80,100,120,140 of R.
ax[0].plot(y,mat[:,80],label="4 MeV")
ax[0].plot(y,mat[:,100],label="5 MeV")
ax[0].plot(y,mat[:,120],label="6 MeV")
ax[0].plot(y,mat[:,140],label="7 MeV")
ax[0].set_xlabel("Prompt energy (MeV)", fontsize = 14)
ax[0].set_ylabel("Arbitrary units", fontsize = 14)
ax[0].set_xlim([1,7])
ax[0].legend(loc = 'upper left', fontsize = 14)

fig.savefig('Figures/ReconstructMatrix.png')

# ------------------------------------------


# PLOT 2
# This routine plots some slices of R at 4,5,6,7 MeV of neutrino energy.
fig2,ax2 = plt.subplots(1,1)
ax2.plot(y,mat[:,80],label="4 MeV")
ax2.plot(y,mat[:,100],label="5 MeV")
ax2.plot(y,mat[:,120],label="6 MeV")
ax2.plot(y,mat[:,140],label="7 MeV")
ax2.set_yscale('log')
ax2.set_xlabel("Prompt energy (MeV)", fontsize = 14)
ax2.set_ylabel("Arbitrary units", fontsize = 14)
ax2.set_xlim([1,7])
ax2.set_ylim([0.005,10])
ax2.legend(loc = 'upper left', fontsize = 14)
fig2.savefig('Figures/ReconstructMatrixSlices.png')
