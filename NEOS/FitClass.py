import sys
import os
homedir = os.path.realpath(__file__)[:-len('NEOS/FitClass.py')]
common_dir = 'Common_cython'
sys.path.append(homedir+common_dir)

import NEOS
import Models
import time

import numpy as np

class SterileFit:

    def __init__(self, wave_packet = False, use_HM = True):
        # The wave packet argument will tell us whether to fit the data
        # according to the wave packet formalism or not (plane wave).
        self.WavePacket = wave_packet
        self.fitter = NEOS.Neos()
        self.UseHM = use_HM

    def what_do_we_do(self,mass):
        """
        When considering sterile neutrino oscillations, the mass of the neutrinos can give
        rise to very fast oscillations, for which it is necessary to integrate or average the probability.
        This function heuristically decides whether it is necessary or not to integrate or average
        for each detector, depending on the value of the mass of the sterile neutrino.

        Input:
        mass: the value of Delta m^2_{41} of the sterile neutrino.

        Output:
        A boolean dictionary with key 'integrate'/'average'
        """
        if mass <= 1.:
            return {'integrate':False,'average':False}
        elif (mass > 1.) and (mass <= 10.):
            return {'integrate':True,'average':False}
        elif (mass > 10.):
            return {'integrate':False,'average':True}

    def getChi2(self,mass,angl, sigma = 2.1e-4):
        """
        Computes the "chi2" value from formula (A12), taking into account
        every bin from every detector in the global fit, for a given square mass and mixing.

        Input:
        mass: the value of Delta m^2_{41} of the sterile neutrino.
        angl: the value of sin^2(2theta_{41})
        sigma: the value of the wave packet width (in nm)

        Output: (float) the chi2 value.
        """
        if self.WavePacket == False:
            model = Models.PlaneWaveSterile(Sin22Th14 = angl, DM2_41 = mass)
        elif self.WavePacket == True:
            model = Models.WavePacketSterile(Sin22Th14 = angl, DM2_41 = mass, Sigma = sigma)

        wdwd = self.what_do_we_do(mass)
        chi2 = self.fitter.get_chi2(model,integrate = wdwd['integrate'], average = wdwd['average'], use_HM = self.UseHM)
        print(mass,angl,chi2)
        return chi2

    def write_data_table(self,mass_ax,angl_ax,filename, sigma = 2.1e-4):
        """
        Writes a square table with the chi2 values of different masses and angles in a file.

        Input:
        mass_ax (array/list): the values of Delta m^2_{41}.
        angl_ax (array/list): the values of sin^2(2theta_{41})
        filename (str): the name of the file in which to write the table.
        sigma: the value of the wave packet width (in nm)
        """
        file = open(filename,'w')
        for m in mass_ax:
            for a in angl_ax:
                chi2 = self.getChi2(m,a, sigma = sigma)
                file.write('{0:1.5f},{1:1.5f},{2:7.4f}\n'.format(m,a,chi2))
        file.close()
