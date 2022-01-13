import sys
import os
common_dir = '/Common_cython'
sys.path.append(os.getcwd()[:-9]+common_dir)

import PROSPECT as PS
import Models
import time

import numpy as np



class SterileFit:

    def __init__(self, wave_packet = False):
        # The wave packet argument will tell us whether to fit the data
        # according to the wave packet formalism or not (plane wave).
        self.WavePacket = wave_packet
        self.fitter = PS.Prospect()

    def what_do_we_do(self,mass):
        """
        When considering sterile neutrino oscillations, the mass of the neutrinos can give
        rise to very fast oscillations, for which it is necessary to integrate or average the probability.
        This function heuristically decides whether it is necessary or not to integrate or average
        for each detector, depending on the value of the mass of the sterile neutrino.

        Input:
        mass: the value of Delta m^2_{41} of the sterile neutrino.

        Output:
        A boolean dictionary with first key 'DB'/'NEOS' and second key 'integrate'/'average'
        """
        if mass <= 2:
            return {'integrate':False,'average':False}
        elif (mass > 2) and (mass <= 20.):
            return {'integrate':True,'average':False}
        elif (mass > 20.):
            return {'integrate':False,'average':True}

    def getChi2(self,mass,angl):
        """
        Computes the "chi2" value from formula (A15),
        for a given square mass and mixing.

        Input:
        mass: the value of Delta m^2_{41} of the sterile neutrino.
        angl: the value of sin^2(2theta_{41})

        Output: (float) the chi2 value.
        """
        if self.WavePacket == False:
            model = Models.PlaneWaveSterile(Sin22Th14 = angl, DM2_41 = mass)
        elif self.WavePacket == True:
            model = Models.WavePacketSterile(Sin22Th14 = angl, DM2_41 = mass)

        wdwd = self.what_do_we_do(mass)
        chi2 = self.fitter.get_chi2(model,do_we_integrate = wdwd['integrate'], do_we_average = wdwd['average'])
        print(mass,angl,chi2)
        return chi2

    def write_data_table(self,mass_ax,angl_ax,filename):
        """
        Writes a square table with the chi2 values of different masses and angles in a file.

        Input:
        mass_ax (array/list): the values of Delta m^2_{41}.
        angl_ax (array/list): the values of sin^2(2theta_{41})
        filename (str): the name of the file in which to write the table.
        """
        file = open(filename,'w')
        for m in mass_ax:
            for a in angl_ax:
                chi2 = self.getChi2(m,a)
                file.write('{0:1.5f},{1:1.5f},{2:7.4f}\n'.format(m,a,chi2))
        file.close()
