import sys
import os
import time
common_dir = '/Common_cython'
main_dir = os.getcwd()[:-5]
print(main_dir)
sys.path.append(main_dir+common_dir)
sys.path.append(main_dir+"/GlobalFit")
sys.path.append(main_dir+"/PROSPECT")
sys.path.append(main_dir+"/DayaBay")
sys.path.append(main_dir+"/NEOS")
sys.path.append(main_dir+"/BEST")

import matplotlib.pyplot as plt
import numpy as np
import matplotlib

import Models
import GlobalFit as GF
import PROSPECT as PS
import BEST

import scipy.interpolate


path_to_style = main_dir + common_dir
dirGF = main_dir+'/GlobalFit/PlotData/'
dirPS = main_dir+'/PROSPECT/PlotData/'
dirBEST = main_dir+'/BEST/PlotData/'
plotdir = 'AllFitFigures/'
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

fitterGF = GF.GlobalFit()
fitterPS = PS.Prospect()
fitterBEST = BEST.Best()

def getChi2(mass,angl,wave_packet = False):
    if wave_packet == False:
        model = Models.PlaneWaveSterile(Sin22Th14 = angl, DM2_41 = mass)
    else:
        model = Models.WavePacketSterile(Sin22Th14 = angl, DM2_41 = mass)
    chi2GF = fitterGF.get_chi2(model)
    chi2PS = fitterPS.get_chi2(model)
    return chi2GF + chi2PS

def getBESTChi2(mass,angl,wave_packet = False):
    if wave_packet == False:
        model = Models.PlaneWaveSterile(Sin22Th14 = angl, DM2_41 = mass)
    else:
        model = Models.WavePacketSterile(Sin22Th14 = angl, DM2_41 = mass)
    chi2 = fitterBEST.get_chi2(model)
    return chi2

def stylize(axxis,contours,t_ax = [1e-3,1], m_ax = [1e-2,10]):
    axxis.grid(linestyle = '--')
    axxis.tick_params(axis='x')
    axxis.tick_params(axis='y')
    axxis.set_xscale('log')
    axxis.set_yscale('log')
    axxis.set_ylabel(r"$\Delta m^2_{41} (\text{eV}^2)$", fontsize = 24)
    axxis.set_xlabel(r"$\sin^2 2 \theta_{14}$", fontsize = 24)
    axxis.set_xlim([4e-3,1])
    axxis.set_ylim([8e-2,10])
    labels = [r'$1\sigma$ (68\% C.L.)',r'$2\sigma$ (95\% C.L.)',r'$3\sigma$ (99\% C.L.)']
    for i in range(3):
        contours.collections[i].set_label(labels[i])
    axxis.legend(loc = 'lower right', fontsize = 18)


titlesize = 13.
size = (7,7)
margins = dict(left=0.16, right=0.97,bottom=0.1, top=0.93)

color1 = '#FFB14E'
color2 = '#EA5F94'
color3 = '#0000FF'


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

# print(data_chi2)


# We find which is the point with minimum chi2, i.e. our best fit.
min_index = np.where((data_PW[:,2] == np.min(data_PW[:,2])))[0][0]
bestfit = data_PW[min_index]
print('Best fit values and chi2: ',bestfit)

# We find which is the chi2 of the null hypothesis
null_hyp_PW = getChi2(0,0)
print('Null hyp chi2: ',null_hyp_PW)

min_index_BEST = np.where((dataBEST_PW[:,2] == np.min(dataBEST_PW[:,2])))[0][0]
bestfitBEST_PW = dataBEST_PW[min_index_BEST]


# PLOT WITH RESPECT TO THE BEST FIT
# ----------------------------------
figBF,axBF = plt.subplots(figsize = size,gridspec_kw=margins)

conts = axBF.tricontour(data_PW[:,1],data_PW[:,0],(data_PW[:,2]-bestfit[2]),levels = [2.30,6.18,11.83], colors = [color1,color2,color3])
axBF.scatter(bestfit[1],bestfit[0],marker = '+', label = r'Best fit')
# axBF.scatter(data_PW[:,1],data_PW[:,0],marker = '+', s = 1.) # This tells us the resolution of our table

stylize(axBF,conts)

figBF.suptitle(r'Best fit:  $\Delta m^2_{41} = %.2f \text{ eV}^2$, $\sin^2 2\theta_{14} = %.3f$. Total $\chi^2 = %.2f$'%(bestfit[0],bestfit[1], bestfit[2]), fontsize = titlesize)
figBF.savefig(plotdir+'PWContour_bestfit.png')


# PLOT WITH RESPECT TO THE NULL HYPOTHESIS
# -----------------------------------------

figNH,axNH = plt.subplots(figsize = size, gridspec_kw = margins)

conts = axNH.tricontour(data_PW[:,1],data_PW[:,0],(data_PW[:,2]-null_hyp_PW),levels = [2.30,6.18,11.83], colors = [color1,color2,color3])
axNH.scatter(bestfit[1],bestfit[0],marker = '+', label = 'Our best fit')
# axNH.scatter(data_PW[:,1],data_PW[:,0],marker = '+', s = 1.) # This tells us the resolution of our table

stylize(axNH,conts)

figNH.suptitle('Null hypothesis: total $\chi^2 = %.2f$'%(null_hyp_PW), fontsize = titlesize)
figNH.savefig(plotdir+'PWContour_nullhyp.png')



# -------------------------------------------------
# STERILE WAVE PACKET CONTOUR - WAVE PACKET FORMALISM
# -------------------------------------------------

dataGF_WP = txt_to_array(dirGF+'WPSterileChi2.dat')
dataGF_WP = np.unique(dataGF_WP,axis=0) # We remove duplicates from the list
dataPS_WP = txt_to_array(dirPS+'WPSterileChi2.dat')
dataPS_WP = np.unique(dataPS_WP,axis=0) # We remove duplicates from the list
dataBEST_WP = txt_to_array(dirBEST+'WPSterileChi2.dat')
dataBEST_WP = np.unique(dataBEST_WP,axis=0) # We remove duplicates from the list
begin = time.time()
# print(dataGF_WP[:,0],dataGF_WP[:,1],dataGF_WP[:,2])
# chi2GF = scipy.interpolate.interp2d(dataGF_WP[:,0],dataGF_WP[:,1],dataGF_WP[:,2])
# chi2PS = scipy.interpolate.interp2d(dataPS_WP[:,0],dataPS_WP[:,1],dataPS_WP[:,2])
# print(time.time()-begin)

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


# PLOT WITH RESPECT TO THE BEST FIT
# ----------------------------------
figBF,axBF = plt.subplots(figsize = size, gridspec_kw = margins)

conts = axBF.tricontour(data_WP[:,1],data_WP[:,0],(data_WP[:,2]-bestfit[2]),levels = [2.30,6.18,11.83], colors = [color1,color2,color3])
axBF.scatter(bestfit[1],bestfit[0],marker = '+', label = r'Best fit')
# axBF.scatter(data_WP[:,1],data_WP[:,0],marker = '+', s = 1.) # This tells us the resolution of our table

stylize(axBF,conts)

figBF.suptitle(r'Best fit:  $\Delta m^2_{41} = %.2f \text{ eV}^2$, $\sin^2 2\theta_{14} = %.3f$. Total $\chi^2 = %.2f$'%(bestfit[0],bestfit[1], bestfit[2]), fontsize = titlesize)
figBF.savefig(plotdir+'WPContour_bestfit.png')


# PLOT WITH RESPECT TO THE NULL HYPOTHESIS
# -----------------------------------------

figNH,axNH = plt.subplots(figsize = size, gridspec_kw = margins)

conts = axNH.tricontour(data_WP[:,1],data_WP[:,0],(data_WP[:,2]-null_hyp_WP),levels = [2.30,6.18,11.83], colors = [color1,color2,color3])
axNH.scatter(bestfit[1],bestfit[0],marker = '+', label = 'Our best fit')
# axNH.scatter(data_WP[:,1],data_WP[:,0],marker = '+', s = 1.) # This tells us the resolution of our table

stylize(axNH,conts)

figNH.suptitle('Null hypothesis: total $\chi^2 = %.2f$'%(null_hyp_WP), fontsize = titlesize)
figNH.savefig(plotdir+'WPContour_nullhyp.png')




# ----------------------------------------------
# 2SIGMA PLOT COMPARISON
# ----------------------------------------------

margins = dict(left=0.16, right=0.97,bottom=0.1, top=0.97)
fig_comp,ax_comp = plt.subplots(figsize = size, gridspec_kw = margins)
# contBEST = ax_comp.plot(dataBEST1[:,0],dataBEST1[:,1], label = r'BEST $2\sigma$', color = color1, zorder = 2)
# ax_comp.fill_between(dataBEST1[:,0],dataBEST1[:,1],10, alpha = 0.4, color = color1, zorder = 0, label = 'BEST+SAGE\n+GALLEX'+r' $2\sigma$')
# contBEST = ax_comp.plot(dataBEST2[:,0],dataBEST2[:,1], color = color1, zorder = 3)
# ax_comp.fill_between(dataBEST2[:,0],dataBEST2[:,1], np.interp(dataBEST2[:,0],dataBEST2[:18,0],dataBEST2[:18,1]), alpha = 0.4, color = color1, zorder = 1)

cont_PW = ax_comp.tricontour(data_PW[:,1],data_PW[:,0],(data_PW[:,2]-null_hyp_PW),levels = [6.18], colors = color2, zorder = 10)
cont_PW.collections[0].set_label(r'$2\sigma$ Plane wave')
cont_WP = ax_comp.tricontour(data_WP[:,1],data_WP[:,0],(data_WP[:,2]-null_hyp_WP),levels = [6.18], colors = color3, zorder = 20)
cont_WP.collections[0].set_label(r'$2\sigma$ Wave packet')
# ax_comp.scatter(dataBEST_PW[:,1],dataBEST_PW[:,0],marker = '+', s = 1.) # This tells us the resolution of our table


cont_BEST_PW = ax_comp.tricontourf(dataBEST_PW[:,1],dataBEST_PW[:,0],(dataBEST_PW[:,2]-bestfitBEST_PW[2]), levels = [0.0,6.18], colors = color2, alpha = 0.3, zorder = 6)
cont_BEST_WP = ax_comp.tricontourf(dataBEST_WP[:,1],dataBEST_WP[:,0],(dataBEST_WP[:,2]-bestfitBEST_WP[2]), levels = [0.0,6.18], colors = color3, alpha = 0.3, zorder = 5)
cont_BEST_PW = ax_comp.tricontour(dataBEST_PW[:,1],dataBEST_PW[:,0],(dataBEST_PW[:,2]-bestfitBEST_PW[2]), levels = [6.18], colors = color2, linewidths = 1,zorder = 6.5)
cont_BEST_WP = ax_comp.tricontour(dataBEST_WP[:,1],dataBEST_WP[:,0],(dataBEST_WP[:,2]-bestfitBEST_WP[2]), levels = [6.18], colors = color3, linewidths = 1,zorder = 5.5)

# cont_BEST_PW.collections[0].set_label(r'B+S+G $2\sigma$ PW')
# cont_BEST_WP.collections[0].set_label(r'B+S+G $2\sigma$ WP')

proxy = [plt.Line2D([0], [0], color=color2, lw=2), plt.Line2D([0], [0], color=color3, lw=2),
         plt.Rectangle((0.1,0.1),0.8,0.8,fc = color2, alpha = 0.3),plt.Rectangle((0.1,0.1),0.8,0.8,fc = color3, alpha = 0.3)]

# ax_comp.annotate('DB+NEOS', xy = (1.25e-3,5), size = 42)
ax_comp.grid(linestyle = '--')
ax_comp.tick_params(axis='x')
ax_comp.tick_params(axis='y')
ax_comp.set_xscale('log')
ax_comp.set_yscale('log')
ax_comp.set_ylabel(r"$\Delta m^2_{41} (\text{eV}^2)$", fontsize = 24)
ax_comp.set_xlabel(r"$\sin^2 2 \theta_{14}$", fontsize = 24)
ax_comp.set_xlim([4e-3,1])
ax_comp.set_ylim([8e-2,10])
ax_comp.legend(loc = 'upper left', fontsize = 18)
ax_comp.legend(proxy,[r'$2\sigma$ Plane wave',r'$2\sigma$ Wave packet',r'BEST $2\sigma$ PW',r'BEST $2\sigma$ WP'],loc = 'upper left', fontsize = 17)

fig_comp.savefig(plotdir+'ContourComparison.pdf')
fig_comp.savefig(plotdir+'ContourComparison.png')
