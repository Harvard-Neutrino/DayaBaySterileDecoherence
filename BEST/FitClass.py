import sys
import os
common_dir = '/Common_cython'
sys.path.append(os.getcwd()[:-5]+common_dir)


import BEST
import Models
import time

import numpy as np

class SterileFit:

    def __init__(self, wave_packet = False):
        # The wave packet argument will tell us whether to fit the data
        # according to the wave packet formalism or not (plane wave).
        self.WavePacket = wave_packet
        self.fitter = BEST.Best()


    def getChi2(self,mass,angl):
        """
        Computes the "chi2" value from the Poisson probability, taking into account
        every bin from every detector in the global fit, for a given square mass and mixing.

        Input:
        mass: the value of Delta m^2_{41} of the sterile neutrino.
        angl: the value of sin^2(2theta_{41})

        Output: (float) the chi2 value.
        """
        if self.WavePacket == False:
            model = Models.PlaneWaveSterile(Sin22Th14 = angl, DM2_41 = mass)
        elif self.WavePacket == True:
            model = Models.WavePacketSterile(Sin22Th14 = angl, DM2_41 = mass)

        chi2 = self.fitter.get_chi2(model)
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
