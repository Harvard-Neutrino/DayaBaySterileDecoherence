import numpy as np
import matplotlib.pyplot as plt
import DayaBay as DB

exp = DB.DayaBay()

x = np.arange(0,12.,0.05)
y = np.arange(0,12.,0.05)
mat = exp.FromEtrueToErec

fig,ax = plt.subplots(1,2,figsize=(12,6),gridspec_kw=dict(left=0.07, right=0.98,bottom=0.1, top=0.93))
contplot = ax[1].contourf(x,y,np.log10(mat))
cbar = fig.colorbar(contplot)
ax[1].set_xlabel("True energy (MeV)", fontsize = 14)
ax[1].set_ylabel("Prompt energy (MeV)", fontsize = 14)
ax[0].plot(y,mat[:,80],label="4 MeV")
ax[0].plot(y,mat[:,100],label="5 MeV")
ax[0].plot(y,mat[:,120],label="6 MeV")
ax[0].plot(y,mat[:,140],label="7 MeV")
ax[0].set_xlabel("Prompt energy (MeV)", fontsize = 14)
ax[0].set_ylabel("Arbitrary units", fontsize = 14)
ax[0].set_xlim([1,7])
ax[0].legend(loc = 'upper left', fontsize = 14)
fig.savefig('Figures/ReconstructMatrix.png')

# ----

fig2,ax2 = plt.subplots(1,1)
ax2.plot(y,mat[:,80],label="4 MeV")
ax2.plot(y,mat[:,100],label="5 MeV")
ax2.plot(y,mat[:,120],label="6 MeV")
ax2.plot(y,mat[:,140],label="7 MeV")
ax2.set_xlabel("Prompt energy (MeV)", fontsize = 14)
ax2.set_ylabel("Arbitrary units", fontsize = 14)
ax2.set_xlim([1,7])
ax2.legend(loc = 'upper left', fontsize = 14)
fig2.savefig('Figures/ReconstructMatrixSlices.png')
