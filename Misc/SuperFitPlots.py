import sys
import os
import time
homedir = os.path.realpath(__file__)[:-len('Misc/SuperFitPlots.py')]
common_dir = 'Common_cython/'
sys.path.append(homedir+common_dir)
sys.path.append(homedir+"NEOS/")
sys.path.append(homedir+"DayaBay/")
sys.path.append(homedir+"GlobalFit/")
sys.path.append(homedir+"PROSPECT/")
sys.path.append(homedir+"BEST/")

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.lines import Line2D

import Models
import GlobalFit as GF
import PROSPECT as PS
import BEST

import scipy.interpolate


path_to_style = homedir + common_dir
dirGF = homedir+'GlobalFit/PlotData/'
dirPS = homedir+'PROSPECT/PlotData/'
dirBEST = homedir+'BEST/PlotData/'
plotdir = homedir+'Misc/AllFitFigures/'
plt.style.use(path_to_style+r"/paper.mplstyle")
matplotlib.rcParams.update({'text.usetex': True})
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]


# -------------------------------------------------------
# PRELIMINAR FUNCTIONS
# -------------------------------------------------------

def txt_to_array(filename, sep = ",", section = 0):
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
    i = 0
    for line in file_lines:

        if len(line.strip().split(sep)) == 1:
            # print(line.strip().split(sep))
            if i == section:
                break
            elif i < section:
                mat = []
                i += 1
        else:
            mat.append(line.strip().split(sep))
    # print(mat)
    mat = np.array(mat).astype(np.float)
    return mat

# dataBEST1 = txt_to_array('BEST.dat',section = 0)
# dataBEST2 = txt_to_array('BEST.dat',section = 1)

# print(dataBEST2[:18])

sigma = 5.0e-4 # in nm

fitterGF = GF.GlobalFit()
fitterPS = PS.Prospect()
fitterBEST = BEST.Best()

def getChi2(mass,angl,wave_packet = False):
    if wave_packet == False:
        model = Models.PlaneWaveSterile(Sin22Th14 = angl, DM2_41 = mass)
    else:
        model = Models.WavePacketSterile(Sin22Th14 = angl, DM2_41 = mass, Sigma = sigma)
    chi2GF = fitterGF.get_chi2(model)
    chi2PS = fitterPS.get_chi2(model)
    return chi2GF + chi2PS

def getBESTChi2(mass,angl,wave_packet = False):
    if wave_packet == False:
        model = Models.PlaneWaveSterile(Sin22Th14 = angl, DM2_41 = mass)
    else:
        model = Models.WavePacketSterile(Sin22Th14 = angl, DM2_41 = mass, Sigma = sigma)
    chi2 = fitterBEST.get_chi2(model)
    return chi2

def stylize(axxis,contours,t_ax = [1e-3,1], m_ax = [1e-2,10]):
    axxis.grid(linestyle = '--')
    axxis.tick_params(axis='x')
    axxis.tick_params(axis='y')
    axxis.set_xscale('log')
    axxis.set_yscale('log')
    axxis.set_ylabel(r"$\Delta m^2_{41} (\textrm{eV}^2)$", fontsize = 24)
    axxis.set_xlabel(r"$\sin^2 2 \theta_{14}$", fontsize = 24)
    axxis.set_xlim([4e-3,1])
    axxis.set_ylim([8e-2,10])
    legend_elements = [Line2D([0], [0], color=color1, ls = '-', lw=2, label=r'$1\sigma$ (68\% C.L.)'),
                       Line2D([0], [0], color=color2, ls = '-', lw=2, label=r'$2\sigma$ (95\% C.L.)'),
                       Line2D([0], [0], color=color3, ls = '-', lw=2, label=r'$3\sigma$ (99\% C.L.)'),
                       Line2D([0], [0], marker='+', color='c', lw = 0, label='Best Fit', markerfacecolor='b', markersize=8)]
    axxis.legend(handles = legend_elements, loc = 'upper left', fontsize = 16)


titlesize = 13.
size = (7,7)
margins = dict(left=0.16, right=0.97,bottom=0.1, top=0.93)

color1 = '#FFB14E'
color2 = '#EA5F94'
color3 = '#0000FF'

# s90 = 0.0212
# s99 = 0.0446
# solar90 = np.sin(2*np.arcsin(np.sqrt(s90)))**2
# solar99 = np.sin(2*np.arcsin(np.sqrt(s99)))**2

# Solar bounds obtained from figure 1 of 2111.12530
solar90 = 0.158 
solar99 = 0.248
# print(solar90,solar99)


# -------------------------------------------------
# STERILE PLANE WAVE CONTOUR - PLANE WAVE FORMALISM
# -------------------------------------------------

dataGF_PW = txt_to_array(dirGF+'PWSterileChi2.dat')
dataGF_PW = np.unique(dataGF_PW,axis=0) # We remove duplicates from the list
dataPS_PW = txt_to_array(dirPS+'PWSterileChi2.dat')
dataPS_PW = np.unique(dataPS_PW,axis=0) # We remove duplicates from the list
dataBEST_PW = txt_to_array(dirBEST+'PWSterileChi2.dat')
dataBEST_PW = np.unique(dataBEST_PW,axis=0) # We remove duplicates from the list
begin = time.time()
# print(dataGF_PW[:,0],dataGF_PW[:,1],dataGF_PW[:,2])
# chi2GF = scipy.interpolate.interp2d(dataGF_PW[:,0],dataGF_PW[:,1],dataGF_PW[:,2])
# chi2PS = scipy.interpolate.interp2d(dataPS_PW[:,0],dataPS_PW[:,1],dataPS_PW[:,2])
# print(time.time()-begin)

y = np.logspace(np.log10(0.08),1,200)
x = np.logspace(-3,0,200)
yy,xx = np.meshgrid(y,x)

pointsGF_PW = dataGF_PW[:,:2]
valuesGF_PW = dataGF_PW[:,2]
chi2GF_PW = scipy.interpolate.griddata(pointsGF_PW, valuesGF_PW, (yy, xx), method='cubic',fill_value = 0)

pointsPS_PW = dataPS_PW[:,:2]
valuesPS_PW = dataPS_PW[:,2]
chi2PS_PW = scipy.interpolate.griddata(pointsPS_PW, valuesPS_PW, (yy, xx), method='cubic',fill_value = 0)


data_PW = np.vstack((yy.flatten(),xx.flatten(),(chi2GF_PW+chi2PS_PW).flatten())).transpose()

print(data_PW)
# print(x.shape)

file = open(homedir+'Misc/AllFitPointsPW.dat','w')
for i in range(x.shape[0]):
    for j in range(y.shape[0]):
        file.write('{0:1.8f},{1:1.6f},{2:7.4f}\n'.format(data_PW[i+200*j,0],data_PW[i+200*j,1],data_PW[i+200*j,2]))
file.close()


# We find which is the point with minimum chi2, i.e. our best fit.
min_index = np.where((data_PW[:,2] == np.min(data_PW[:,2])))[0][0]
bestfit = data_PW[min_index]
print('Best fit values and chi2: ',bestfit)

# We find which is the chi2 of the null hypothesis
null_hyp_PW = getChi2(0,0)
print('Null hyp chi2: ',null_hyp_PW)

min_index_BEST = np.where((dataBEST_PW[:,2] == np.min(dataBEST_PW[:,2])))[0][0]
bestfitBEST_PW = dataBEST_PW[min_index_BEST]


# -------------------------------------------------
# STERILE WAVE PACKET CONTOUR - WAVE PACKET FORMALISM
# -------------------------------------------------

if sigma == 2.1e-4:
    suffix = ''
elif sigma == 2.1e-3:
    suffix = '_2.1e-3'
elif sigma == 5.0e-4:
    suffix = '_5.0e-4'

dataGF_WP = txt_to_array(dirGF+'WPSterileChi2'+suffix+'.dat')
dataGF_WP = np.unique(dataGF_WP,axis=0) # We remove duplicates from the list
dataPS_WP = txt_to_array(dirPS+'WPSterileChi2'+suffix+'.dat')
dataPS_WP = np.unique(dataPS_WP,axis=0) # We remove duplicates from the list
dataBEST_WP = txt_to_array(dirBEST+'WPSterileChi2'+suffix+'.dat')
dataBEST_WP = np.unique(dataBEST_WP,axis=0) # We remove duplicates from the list
begin = time.time()


y = np.logspace(np.log10(0.08),1,200)
x = np.logspace(-3,0,200)
yy,xx = np.meshgrid(y,x)

pointsGF_WP = dataGF_WP[:,:2]
valuesGF_WP = dataGF_WP[:,2]
chi2GF_WP = scipy.interpolate.griddata(pointsGF_WP, valuesGF_WP, (yy, xx), method='cubic',fill_value= 0)

pointsPS_WP = dataPS_WP[:,:2]
valuesPS_WP = dataPS_WP[:,2]
chi2PS_WP = scipy.interpolate.griddata(pointsPS_WP, valuesPS_WP, (yy, xx), method='cubic',fill_value = 0)


data_WP = np.vstack((yy.flatten(),xx.flatten(),(chi2GF_WP+chi2PS_WP).flatten())).transpose()#


file = open(homedir+'Misc/AllFitPointsWP'+suffix+'.dat','w')
for i in range(x.shape[0]):
    for j in range(y.shape[0]):
        file.write('{0:1.8f},{1:1.6f},{2:7.4f}\n'.format(data_WP[i+200*j,0],data_WP[i+200*j,1],data_WP[i+200*j,2]))
file.close()

# We find which is the point with minimum chi2, i.e. our best fit.
min_index = np.where((data_WP[:,2] == np.min(data_WP[:,2])))[0][0]
print(min_index)
bestfit = data_WP[min_index]
print('Best fit values and chi2: ',bestfit)

# We find which is the chi2 of the null hypothesis
null_hyp_WP = getChi2(0,0, wave_packet = True)
print('Null hyp chi2: ',null_hyp_WP)

min_index_BEST = np.where((dataBEST_WP[:,2] == np.min(dataBEST_WP[:,2])))[0][0]
bestfitBEST_WP = dataBEST_WP[min_index_BEST]


# DATA FROM MINIMUM WAVE PACKET WIDTH
# ------------------------------------------------------------------------

dataGF_WP0 = txt_to_array(dirGF+'WPSterileChi2.dat')
dataGF_WP0 = np.unique(dataGF_WP0,axis=0) # We remove duplicates from the list
dataPS_WP0 = txt_to_array(dirPS+'WPSterileChi2.dat')
dataPS_WP0 = np.unique(dataPS_WP0,axis=0) # We remove duplicates from the list
dataBEST_WP0 = txt_to_array(dirBEST+'WPSterileChi2.dat')
dataBEST_WP0 = np.unique(dataBEST_WP0,axis=0) # We remove duplicates from the list


y = np.logspace(np.log10(0.08),1,200)
x = np.logspace(-3,0,200)
yy,xx = np.meshgrid(y,x)

pointsGF_WP0 = dataGF_WP0[:,:2]
valuesGF_WP0 = dataGF_WP0[:,2]
chi2GF_WP0 = scipy.interpolate.griddata(pointsGF_WP0, valuesGF_WP0, (yy, xx), method='cubic',fill_value= 0)

pointsPS_WP0 = dataPS_WP0[:,:2]
valuesPS_WP0 = dataPS_WP0[:,2]
chi2PS_WP0 = scipy.interpolate.griddata(pointsPS_WP0, valuesPS_WP0, (yy, xx), method='cubic',fill_value = 0)


data_WP0 = np.vstack((yy.flatten(),xx.flatten(),(chi2GF_WP0+chi2PS_WP0).flatten())).transpose()#

# We find which is the point with minimum chi2, i.e. our best fit.
min_index0 = np.where((data_WP0[:,2] == np.min(data_WP0[:,2])))[0][0]
print(min_index0)
bestfit0 = data_WP0[min_index0]
print('Best fit values and chi2: ',bestfit0)

# We find which is the chi2 of the null hypothesis
null_hyp_WP0 = getChi2(0,0, wave_packet = True)
print('Null hyp chi2: ',null_hyp_WP0)

min_index_BEST0 = np.where((dataBEST_WP0[:,2] == np.min(dataBEST_WP0[:,2])))[0][0]
bestfitBEST_WP0 = dataBEST_WP0[min_index_BEST0]

# ----------------------------------------------
# 2SIGMA PLOT COMPARISON
# ----------------------------------------------

margins = dict(left=0.16, right=0.97,bottom=0.11, top=0.97)
fig_comp,ax_comp = plt.subplots(figsize = size, gridspec_kw = margins)
# contBEST = ax_comp.plot(dataBEST1[:,0],dataBEST1[:,1], label = r'BEST $2\sigma$', color = color1, zorder = 2)
# ax_comp.fill_between(dataBEST1[:,0],dataBEST1[:,1],10, alpha = 0.4, color = color1, zorder = 0, label = 'BEST+SAGE\n+GALLEX'+r' $2\sigma$')
# contBEST = ax_comp.plot(dataBEST2[:,0],dataBEST2[:,1], color = color1, zorder = 3)
# ax_comp.fill_between(dataBEST2[:,0],dataBEST2[:,1], np.interp(dataBEST2[:,0],dataBEST2[:18,0],dataBEST2[:18,1]), alpha = 0.4, color = color1, zorder = 1)

cont_PW = ax_comp.tricontour(data_PW[:,1],data_PW[:,0],(data_PW[:,2]-null_hyp_PW),levels = [6.18], colors = color2, zorder = 10)
cont_WP = ax_comp.tricontour(data_WP[:,1],data_WP[:,0],(data_WP[:,2]-null_hyp_WP),levels = [6.18], colors = color3, zorder = 20)
# ax_comp.scatter(dataBEST_PW[:,1],dataBEST_PW[:,0],marker = '+', s = 1.) # This tells us the resolution of our table


cont_BEST_PW = ax_comp.tricontourf(dataBEST_PW[:,1],dataBEST_PW[:,0],(dataBEST_PW[:,2]-bestfitBEST_PW[2]), levels = [0.0,6.18], colors = color2, alpha = 0.3, zorder = 6)
cont_BEST_WP = ax_comp.tricontourf(dataBEST_WP[:,1],dataBEST_WP[:,0],(dataBEST_WP[:,2]-bestfitBEST_WP[2]), levels = [0.0,6.18], colors = color3, alpha = 0.3, zorder = 5)
cont_BEST_PW = ax_comp.tricontour(dataBEST_PW[:,1],dataBEST_PW[:,0],(dataBEST_PW[:,2]-bestfitBEST_PW[2]), levels = [6.18], colors = color2, linewidths = 1,zorder = 6.5)
cont_BEST_WP = ax_comp.tricontour(dataBEST_WP[:,1],dataBEST_WP[:,0],(dataBEST_WP[:,2]-bestfitBEST_WP[2]), levels = [6.18], colors = color3, linewidths = 1,zorder = 5.5)

cont_BEST_PW.collections[0].set_label(r'B+S+G $2\sigma$ PW')
cont_BEST_WP.collections[0].set_label(r'B+S+G $2\sigma$ WP')

proxy = [plt.Line2D([0], [0], color=color2, lw=2), plt.Line2D([0], [0], color=color3, lw=2),
         plt.Rectangle((0.1,0.1),0.8,0.8,fc = color2, alpha = 0.3),plt.Rectangle((0.1,0.1),0.8,0.8,fc = color3, alpha = 0.3),
         plt.Line2D([0],[0], color = 'gray', ls = 'dashed', lw = 2)]

# ax_comp.annotate('DB+NEOS', xy = (1.25e-3,5), size = 42)
ax_comp.grid(linestyle = '--')
ax_comp.tick_params(axis='x')
ax_comp.tick_params(axis='y')
ax_comp.set_xscale('log')
ax_comp.set_yscale('log')
ax_comp.set_ylabel(r"$\Delta m^2_{41} (\textrm{eV}^2)$", fontsize = 24)
ax_comp.set_xlabel(r"$\sin^2 2 \theta_{14}$", fontsize = 24)
ax_comp.set_xlim([4e-3,1])
ax_comp.set_ylim([8e-2,10])

ax_comp.vlines(solar90,0.08,10,colors='gray',linestyles = 'dashed')
# ax_comp.vlines(solar99,0.08,10,colors='gray',linestyles = 'dotted')
ax_comp.annotate("", xy=(solar90*1.5,1.2e-1), xytext=(solar90,1.2e-1),arrowprops=dict(arrowstyle="->", color = 'gray', lw = 1), zorder = 101)

ax_comp.legend(loc = 'upper left', fontsize = 18)
ax_comp.legend(proxy,[r'$2\sigma$ Plane wave',r'$2\sigma$ Wave packet',r'BEST $2\sigma$ PW',r'BEST $2\sigma$ WP', 'Solar'],loc = 'upper left', fontsize = 17)


fig_comp.savefig(plotdir+'ContourComparison'+suffix+'.pdf')
fig_comp.savefig(plotdir+'ContourComparison'+suffix+'.png')


# DIFFERENT WP WIDTH COMPARISON
# ---------------------------------------


margins = dict(left=0.16, right=0.97,bottom=0.11, top=0.97)
fig_comp,ax_comp = plt.subplots(figsize = size, gridspec_kw = margins)
# contBEST = ax_comp.plot(dataBEST1[:,0],dataBEST1[:,1], label = r'BEST $2\sigma$', color = color1, zorder = 2)
# ax_comp.fill_between(dataBEST1[:,0],dataBEST1[:,1],10, alpha = 0.4, color = color1, zorder = 0, label = 'BEST+SAGE\n+GALLEX'+r' $2\sigma$')
# contBEST = ax_comp.plot(dataBEST2[:,0],dataBEST2[:,1], color = color1, zorder = 3)
# ax_comp.fill_between(dataBEST2[:,0],dataBEST2[:,1], np.interp(dataBEST2[:,0],dataBEST2[:18,0],dataBEST2[:18,1]), alpha = 0.4, color = color1, zorder = 1)

cont_PW = ax_comp.tricontour(data_PW[:,1],data_PW[:,0],(data_PW[:,2]-null_hyp_PW),levels = [6.18], colors = color2, zorder = 10)
cont_WP = ax_comp.tricontour(data_WP[:,1],data_WP[:,0],(data_WP[:,2]-null_hyp_WP),levels = [6.18], colors = color1, zorder = 20)
cont_WP0 = ax_comp.tricontour(data_WP0[:,1],data_WP0[:,0],(data_WP0[:,2]-null_hyp_WP0),levels = [6.18], colors = color3, zorder = 20)
# ax_comp.scatter(dataBEST_PW[:,1],dataBEST_PW[:,0],marker = '+', s = 1.) # This tells us the resolution of our table


cont_BEST_PW = ax_comp.tricontourf(dataBEST_PW[:,1],dataBEST_PW[:,0],(dataBEST_PW[:,2]-bestfitBEST_PW[2]), levels = [0.0,6.18], colors = color2, alpha = 0.3, zorder = 6)
cont_BEST_WP = ax_comp.tricontourf(dataBEST_WP[:,1],dataBEST_WP[:,0],(dataBEST_WP[:,2]-bestfitBEST_WP[2]), levels = [0.0,6.18], colors = color1, alpha = 0.3, zorder = 5)
cont_BEST_PW = ax_comp.tricontour(dataBEST_PW[:,1],dataBEST_PW[:,0],(dataBEST_PW[:,2]-bestfitBEST_PW[2]), levels = [6.18], colors = color2, linewidths = 1,zorder = 6.5)
cont_BEST_WP = ax_comp.tricontour(dataBEST_WP[:,1],dataBEST_WP[:,0],(dataBEST_WP[:,2]-bestfitBEST_WP[2]), levels = [6.18], colors = color1, linewidths = 1,zorder = 5.5)
cont_BEST_WP0 = ax_comp.tricontourf(dataBEST_WP0[:,1],dataBEST_WP0[:,0],(dataBEST_WP0[:,2]-bestfitBEST_WP0[2]), levels = [0.0,6.18], colors = color3, alpha = 0.3, zorder = 5)
cont_BEST_WP0 = ax_comp.tricontour(dataBEST_WP0[:,1],dataBEST_WP0[:,0],(dataBEST_WP0[:,2]-bestfitBEST_WP0[2]), levels = [6.18], colors = color3, linewidths = 1,zorder = 5.5)

proxy = [plt.Line2D([0], [0], color=color2, lw=2), plt.Line2D([0], [0], color=color1, lw=2), plt.Line2D([0],[0],color = color3, lw = 2),
        #  plt.Rectangle((0.1,0.1),0.8,0.8,fc = color2, alpha = 0.3),plt.Rectangle((0.1,0.1),0.8,0.8,fc = color1, alpha = 0.3), plt.Rectangle((0.1,0.1),0.8,0.8,fc = color3, alpha = 0.3),
         plt.Line2D([0],[0], color = 'gray', ls = 'dashed', lw = 2)]

# ax_comp.annotate('DB+NEOS', xy = (1.25e-3,5), size = 42)
ax_comp.grid(linestyle = '--')
ax_comp.tick_params(axis='x')
ax_comp.tick_params(axis='y')
ax_comp.set_xscale('log')
ax_comp.set_yscale('log')
ax_comp.set_ylabel(r"$\Delta m^2_{41} (\textrm{eV}^2)$", fontsize = 24)
ax_comp.set_xlabel(r"$\sin^2 2 \theta_{14}$", fontsize = 24)
ax_comp.set_xlim([4e-3,1])
ax_comp.set_ylim([8e-2,10])

ax_comp.vlines(solar90,0.08,10,colors='gray',linestyles = 'dashed')
# ax_comp.vlines(solar99,0.08,10,colors='gray',linestyles = 'dotted')
ax_comp.annotate("", xy=(solar90*1.5,1.2e-1), xytext=(solar90,1.2e-1),arrowprops=dict(arrowstyle="->", color = 'gray', lw = 1), zorder = 101)

ax_comp.legend(proxy,[r'Plane wave',
                      r'$\sigma_x = 5.0\times 10^{-4}\ \textrm{nm}$',
                      r'$\sigma_x = 2.1\times 10^{-4}\ \textrm{nm}$', 
                    #   'Gallium PW', 
                    #   r'Gal $\sigma_x = 5.0\times 10^{-4}\ \textrm{nm}$',
                    #   r'Gal $ \sigma_x = 2.1\times 10^{-4}\ \textrm{nm}$',
                      'Solar'],loc = 'upper left', fontsize = 15)


fig_comp.savefig(plotdir+'ContourComparison_3.pdf')
fig_comp.savefig(plotdir+'ContourComparison_3.png')
