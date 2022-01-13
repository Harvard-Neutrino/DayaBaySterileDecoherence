import sys
import os
common_dir = '/Common_cython'
main_dir = os.getcwd()[:-9]
sys.path.append(main_dir+common_dir)

import matplotlib.pyplot as plt
import numpy as np
import matplotlib

import Models
import PROSPECT as PS

path_to_style= main_dir+common_dir
dir = 'PlotData/'
plt.style.use(path_to_style+r"/paper.mplstyle")
matplotlib.rcParams.update({'text.usetex': True})
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]


# -------------------------------------------------------
# PRELIMINAR FUNCTIONS
# -------------------------------------------------------

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

fitter = PS.Prospect()

def getChi2(mass,angl,wave_packet = False):
    if wave_packet == False:
        model = Models.PlaneWaveSterile(Sin22Th14 = angl, DM2_41 = mass)
    else:
        model = Models.WavePacketSterile(Sin22Th14 = angl, DM2_41 = mass)
    # chi2 = fitter.get_poisson_chi2(model)
    chi2 = fitter.get_chi2(model)
    return chi2

def stylize(axxis,contours,t_ax = [1e-3,1], m_ax = [1e-2,10]):
    axxis.grid(linestyle = '--')
    axxis.tick_params(axis='x')
    axxis.tick_params(axis='y')
    axxis.set_xscale('log')
    axxis.set_yscale('log')
    axxis.set_ylabel(r"$\Delta m^2_{41} (\text{eV}^2)$", fontsize = 24)
    axxis.set_xlabel(r"$\sin^2 2 \theta_{14}$", fontsize = 24)
    axxis.set_xlim([0.004,1])
    axxis.set_ylim([0.08,25.])
    labels = [r'$1\sigma$ (68\% C.L.)',r'$2\sigma$ (95\% C.L.)',r'$3\sigma$ (99\% C.L.)']
    for i in range(3):
        contours.collections[i].set_label(labels[i])
    axxis.legend(loc = 'lower left', fontsize = 20)

color1 = '#FFB14E'
color2 = '#EA5F94'
color3 = '#0000FF'

titlesize = 13.
size = (7,7)
margins = dict(left=0.16, right=0.97,bottom=0.1, top=0.93)

# -------------------------------------------------
# STERILE PLANE WAVE CONTOUR - PLANE WAVE FORMALISM
# -------------------------------------------------

print('Plane wave')
data_PW = txt_to_array(dir+'PWSterileChi2.dat')
data_PW = np.unique(data_PW,axis=0) # We remove duplicates from the list

# We find which is the point with minimum chi2, i.e. our best fit.
min_index = np.where(data_PW[:,2] == np.min(data_PW[:,2]))[0][0]
bestfit = data_PW[min_index]
print('Best fit values and chi2: ',bestfit)

# We find which is the chi2 of the null hypothesis
null_hyp_PW = getChi2(0.,0.)
print('Null hyp chi2: ',null_hyp_PW)

# PLOT WITH RESPECT TO THE BEST FIT
# ----------------------------------
figBF,axBF = plt.subplots(figsize = size,gridspec_kw=margins)


conts = axBF.tricontour(data_PW[:,1],data_PW[:,0],(data_PW[:,2]-bestfit[2]),levels = [2.30,6.18,11.83],  colors = [color1,color2,color3])
axBF.scatter(bestfit[1],bestfit[0],marker = '+', label = r'Best fit')
# axBF.scatter(data_PW[:,1],data_PW[:,0],marker = '+', s = 1.) # This tells us the resolution of our table

stylize(axBF,conts)

figBF.suptitle(r'Best fit:  $\Delta m^2_{41} = %.2f \text{ eV}^2$, $\sin^2 2\theta_{14} = %.3f$. Total $\chi^2 = %.2f$'%(bestfit[0],bestfit[1], bestfit[2]), fontsize = titlesize)
figBF.savefig('Figures/PWContour_bestfit.png')


# PLOT WITH RESPECT TO THE NULL HYPOTHESIS
# -----------------------------------------

figNH,axNH = plt.subplots(figsize = size, gridspec_kw = margins)

conts = axNH.tricontour(data_PW[:,1],data_PW[:,0],(data_PW[:,2]-null_hyp_PW),levels = [2.30,6.18,11.83],  colors = [color1,color2,color3])
axNH.scatter(bestfit[1],bestfit[0],marker = '+', label = 'Our best fit')
# axNH.scatter(data_PW[:,1],data_PW[:,0],marker = '+', s = 1.) # This tells us the resolution of our table

stylize(axNH,conts)

figNH.suptitle('Null hypothesis: total $\chi^2 = %.2f$'%(null_hyp_PW), fontsize = titlesize)
figNH.savefig('Figures/PWContour_nullhyp.png')



# -------------------------------------------------
# STERILE WAVE PACKET CONTOUR - WAVE PACKET FORMALISM
# -------------------------------------------------

print('\nWave Packet')
data_WP = txt_to_array(dir+'WPSterileChi2.dat')
data_WP = np.unique(data_WP,axis=0) # We remove duplicates from the list

# We find which is the point with minimum chi2, i.e. our best fit.
min_index = np.where(data_WP[:,2] == np.min(data_WP[:,2]))[0][0]
bestfit = data_WP[min_index]
print('Best fit values and chi2: ',bestfit)

# We find which is the chi2 of the null hypothesis
null_hyp_WP = getChi2(0,0, wave_packet = True)
print('Null hyp chi2: ',null_hyp_WP)



# PLOT WITH RESPECT TO THE BEST FIT
# ----------------------------------
figBF,axBF = plt.subplots(figsize = size, gridspec_kw = margins)

conts = axBF.tricontour(data_WP[:,1],data_WP[:,0],(data_WP[:,2]-bestfit[2]),levels = [2.30,6.18,11.83],  colors = [color1,color2,color3])
axBF.scatter(bestfit[1],bestfit[0],marker = '+', label = r'Best fit')
# axBF.scatter(data_WP[:,1],data_WP[:,0],marker = '+', s = 1.) # This tells us the resolution of our table

stylize(axBF,conts)

figBF.suptitle(r'Best fit:  $\Delta m^2_{41} = %.2f eV^2$, $\sin^2 2\theta_{14} = %.3f$. Total $\chi^2 = %.2f$'%(bestfit[0],bestfit[1], bestfit[2]), fontsize = titlesize)
figBF.savefig('Figures/WPContour_bestfit.png')


# PLOT WITH RESPECT TO THE NULL HYPOTHESIS
# -----------------------------------------

figNH,axNH = plt.subplots(figsize = size, gridspec_kw = margins)

conts = axNH.tricontour(data_WP[:,1],data_WP[:,0],(data_WP[:,2]-null_hyp_WP),levels = [2.30,6.18,11.83], colors = [color1,color2,color3])
axNH.scatter(bestfit[1],bestfit[0],marker = '+', label = 'Our best fit')
# axNH.scatter(data_WP[:,1],data_WP[:,0],marker = '+', s = 1.) # This tells us the resolution of our table

stylize(axNH,conts)

figNH.suptitle('Null hypothesis: total $\chi^2 = %.2f$'%(null_hyp_WP), fontsize = titlesize)
figNH.savefig('Figures/WPContour_nullhyp.png')




# ----------------------------------------------
# 2SIGMA PLOT COMPARISON
# ----------------------------------------------

margins = dict(left=0.16, right=0.97,bottom=0.1, top=0.97)
fig_comp,ax_comp = plt.subplots(figsize = size, gridspec_kw = margins)
cont_PW = ax_comp.tricontour(data_PW[:,1],data_PW[:,0],(data_PW[:,2]-null_hyp_PW),levels = [6.18], colors = color2, linestyles = ['solid'])
cont_PW.collections[0].set_label(r'$2\sigma$ Plane wave')
cont_WP = ax_comp.tricontour(data_WP[:,1],data_WP[:,0],(data_WP[:,2]-null_hyp_WP),levels = [6.18], colors = color3,linestyles = ['solid'])
cont_WP.collections[0].set_label(r'$2\sigma$ Wave package')

ax_comp.annotate('PROSPECT', xy = (5e-3,14), size = 42)
ax_comp.grid(linestyle = '--')
ax_comp.tick_params(axis='x')
ax_comp.tick_params(axis='y')
ax_comp.set_xscale('log')
ax_comp.set_yscale('log')
ax_comp.set_ylabel(r"$\Delta m^2_{41} (\text{eV}^2)$", fontsize = 24)
ax_comp.set_xlabel(r"$\sin^2 2 \theta_{14}$", fontsize = 24)
ax_comp.set_xlim([0.004,1])
ax_comp.set_ylim([0.08,25.])
ax_comp.legend(loc = 'lower left', fontsize = 20)

fig_comp.savefig('Figures/ContourComparison.png')
fig_comp.savefig('Figures/ContourComparison.pdf')
