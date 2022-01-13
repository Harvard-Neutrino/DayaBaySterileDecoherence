import sys
import os
common_dir = '/Common_cython'
sys.path.append(os.getcwd()[:-10]+common_dir)

import GlobalFit as GF
import Models
import time

import numpy as np


class SterileGlobalFit:

    def __init__(self, wave_packet = False):
        # The wave packet argument will tell us whether to fit the data
        # according to the wave packet formalism or not (plane wave).
        self.WavePacket = wave_packet
        self.fitter = GF.GlobalFit()
        if wave_packet == False:
            self.SMexpectation = self.fitter.PredictedNEOSDataSM
        if wave_packet == True:
            self.SMexpectation = self.fitter.PredictedNEOSDataSMWP

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
        if mass <= 0.15:
            return {'DB':{'integrate':False,'average':False},'NEOS':{'integrate':False,'average':False}}
        elif (mass > 0.15) and (mass <= 1.):
            return {'DB':{'integrate':True,'average':False},'NEOS':{'integrate':False,'average':False}}
        elif (mass > 1.) and (mass <= 2.):
            return {'DB':{'integrate':True,'average':False},'NEOS':{'integrate':True,'average':False}}
        elif (mass > 2.) and (mass <= 10.):
            return {'DB':{'integrate':False,'average':True},'NEOS':{'integrate':True,'average':False}}
        elif (mass > 10.):
            return {'DB':{'integrate':False,'average':True},'NEOS':{'integrate':False,'average':True}}

    def getChi2(self,mass,angl):
        """
        Computes the "chi2" value from (A14), taking into account
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

        wdwd = self.what_do_we_do(mass)
        chi2 = self.fitter.get_chi2(model,integrate_DB = wdwd['DB']['integrate'], integrate_NEOS = wdwd['NEOS']['integrate'],
                                            average_DB = wdwd['DB']['average'],     average_NEOS = wdwd['NEOS']['average'],
                                            exp_events_SM_NEOS = self.SMexpectation)
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
                file.write('{0:1.5f},{1:1.5f},{2:7.4f}\n'.format(m,a,self.getChi2(m,a)))
        file.close()
