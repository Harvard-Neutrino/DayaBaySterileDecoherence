import GlobalFit as GF
import Models
import time

import numpy as np

class StandardModelGlobalFit:
# Probably needs an init.
    def __init__(self, wave_packet = False):
        self.WavePacket = wave_packet
        self.fitter = GF.GlobalFit()


# From here on this needs implementation
    def getChi2(self,mass,angl):
        if self.WavePacket == False:
            model = Models.PlaneWaveSM(Sin22Th13 = angl,DM2_ee = mass)
        elif self.WavePacket == True:
            model = Models.WavePacketSM(Sin22Th13 = angl,DM2_ee = mass)
        chi2 = self.fitter.get_poisson_chi2(model)
        print(mass,angl,chi2)
        return chi2

    def write_data_table(self,mass_ax,angl_ax,filename):
        file = open(filename,'w')
        for m in mass_ax:
            for a in angl_ax:
                file.write('{0:1.5f},{1:1.5f},{2:7.4f}\n'.format(m,a,self.getChi2(m,a)))
        file.close()


class SterileGlobalFit:
# Probably needs an init.
    def __init__(self, wave_packet = False):
        self.WavePacket = wave_packet
        self.fitter = GF.GlobalFit()

    def what_do_we_do(self,mass):
        """Mass must be in eV^2. """
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

# From here on this needs implementation
    def getChi2(self,mass,angl):
        if self.WavePacket == False:
            model = Models.PlaneWaveSterile(Sin22Th14 = angl, DM2_41 = mass)
        elif self.WavePacket == True:
            model = Models.WavePacketSterile(Sin22Th14 = angl, DM2_41 = mass)

        wdwd = self.what_do_we_do(mass)
        chi2 = self.fitter.get_poisson_chi2(model,integrate_DB = wdwd['DB']['integrate'], integrate_NEOS = wdwd['NEOS']['integrate'],
                                             average_DB = wdwd['DB']['average'],     average_NEOS = wdwd['NEOS']['average'])
        print(mass,angl,chi2)
        return chi2

    def write_data_table(self,mass_ax,angl_ax,filename):
        file = open(filename,'w')
        for m in mass_ax:
            for a in angl_ax:
                file.write('{0:1.5f},{1:1.5f},{2:7.4f}\n'.format(m,a,self.getChi2(m,a)))
        file.close()


# -----------------------------------------------------------------------
# Best-fit from DayaBay
# mass = 2.5e-3,angl = 0.0841

datmass = np.logspace(-2,1,3)
datangl = np.logspace(-3,0,3)
print(datmass,datangl)

begin = time.time()
fit = SterileGlobalFit()
fit.write_data_table(datmass,datangl,'SMPWSterileChi2_new.dat')
end = time.time()
print('Time = '+str(end-begin)[:6]+' s.')
