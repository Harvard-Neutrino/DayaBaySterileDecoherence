import sys
import os
common_dir = '/Common_cython'
sys.path.append(os.getcwd()[:-5]+common_dir)

import InverseBetaDecayCrossSection as IBD
import HuberMullerFlux as HMF
import Models
import numpy as np
from scipy import integrate as integrate
from scipy import interpolate as interpolate

class Best:
    m = 1
    kg = 1
    day = 1
    nm = 1e-9*m
    cm = 0.01*m
    s = 1/3600/24*day
    Ci = 3.7e10/s
    eV = 50677.308/cm
    MeV = 1e6*eV

    def __init__(self):
        # We define some units. All results will be in units of m, kg and days.

        self.NeutrinoEnergies = [427e3*self.eV,432e3*self.eV,747e3*self.eV,752e3*self.eV]
        self.FissionFractions = {427e3*self.eV: 0.0895, 432e3*self.eV: 0.0093,
                                 747e3*self.eV: 0.8163, 752e3*self.eV: 0.0849}
        self.CrossSection = 5.81e-45*self.cm**2
        self.InitialActivity = 3.414e6*self.Ci
        self.GaNumberDensity = 2.1e22/self.cm**3

        self.r_in = 133.5*self.cm/2. # Radius of the spherical inner target
        self.r_Cr = (4.3+3)*self.cm  # Radius of the Cr source and its shield
        self.h_Cr = (10.8+3)*self.cm # Height of the Cr source and its shield
        self.r_out = 218.0*self.cm/2. # Radius of the cylindrical outer target
        self.h_out = 234.5*self.cm    # Height of the cylindrial outer target

        self.GeometricCorrectionInner = 1./1.0985196
        self.GeometricCorrectionOuter = 1./1.0891257

        self.ProductionRateData = np.array([49.4,44.9,62.9,73.3,49.8,69.5,64.6,53.8,49.9,69.1,
                                            41.1,63.6,51.4,66.6,46.9,87.3,50.4,59.7,43.0,78.8])/self.day
        self.StatisticalError = np.array([4.2,5.9,7.4,8.6,8.2,12.0,12.6,12.2,16.5,19.4,
                                         5.3,5.7,7.3,9.8,7.9,13.2,10.6,11.7,15.3,20.0])/self.day
        self.SystematicError = self.ProductionRateData*0.025
        self.CrossSectionError = self.ProductionRateData*0.035

    def is_in_inner_container(self,rho,z):
        r = np.sqrt(rho**2+z**2)
        if r > self.r_in or (rho < self.r_Cr and z > -1*self.h_Cr/2):
            return False
        else:
            return True

    def is_in_outer_container(self,rho,z):
        r = np.sqrt(rho**2+z**2)
        if (r < self.r_in or (rho < self.r_Cr and z > 0) or
           rho > self.r_out or z > self.h_out/2 or z < -1*self.h_out/2):
            return False
        else:
            return True

    def get_inner_production_rate(self,model):

        def integrand(z,rho):
            if self.is_in_inner_container(rho,z) == False:
                return 0
            else:
                r = np.sqrt(rho**2+z**2)
                sump = np.sum([self.FissionFractions[e]*model.oscProbability(e/self.MeV,r/self.m)
                                                for e in self.NeutrinoEnergies])
                return sump*rho/r**2

        def ubound(rho):
            return np.sqrt(self.r_in**2-rho**2)
        def lbound(rho):
            return -np.sqrt(self.r_in**2-rho**2)

        int = integrate.dblquad(integrand,0,self.r_in,lbound,ubound)[0]/2.
        # print('Inner: ',int,int*self.GeometricCorrection)
        # print(self.InitialActivity*self.GaNumberDensity*self.CrossSection*0.5203)
        return self.InitialActivity*self.GaNumberDensity*self.CrossSection*int*self.GeometricCorrectionInner



    def get_outer_production_rate(self,model):

        def integrand(z,rho):
            if self.is_in_outer_container(rho,z) == False:
                return 0
            else:
                r = np.sqrt(rho**2+z**2)
                sump = np.sum([self.FissionFractions[e]*model.oscProbability(e/self.MeV,r/self.m)
                                                for e in self.NeutrinoEnergies])
                return sump*rho/r**2

        def ubound1(rho):
            if rho > self.r_in:
                return 0
            else:
                return -np.sqrt(self.r_in**2-rho**2)

        def lbound2(rho):
            if rho > self.r_in:
                return 0
            else:
                return np.sqrt(self.r_in**2-rho**2)

        int  = integrate.dblquad(integrand,0,self.r_out,-self.h_out/2,ubound1)[0]/2.
        int += integrate.dblquad(integrand,self.r_Cr,self.r_out,lbound2,self.h_out/2)[0]/2.
        # print('Outer: ',int,int*self.GeometricCorrection)
        # print(self.InitialActivity*self.GaNumberDensity*self.CrossSection*0.5441)
        return self.InitialActivity*self.GaNumberDensity*self.CrossSection*int*self.GeometricCorrectionOuter


    def get_total_covariance_matrix(self):
        V = np.identity(20)*(self.SystematicError**2+self.StatisticalError**2)
        CS_err = self.CrossSectionError
        CS_err = np.tile(CS_err,(len(CS_err),1))*(np.tile(CS_err,(len(CS_err),1)).transpose())
        return V+CS_err

    def get_inverse_covariance_matrix(self):
        return np.linalg.inv(np.array(self.get_total_covariance_matrix()))

    def get_chi2_all(self,model):
        rate_in  = self.get_inner_production_rate(model)
        rate_out = self.get_outer_production_rate(model)
        pred = np.concatenate((np.repeat(rate_in,10),np.repeat(rate_out,10)))
        # print(pred)

        # print(rate_in -self.ProductionRateData[0],rate_out-self.ProductionRateData[1])
        Vinv = self.get_inverse_covariance_matrix()

        chi2 = (self.ProductionRateData-pred).dot(Vinv.dot(self.ProductionRateData-pred))
        return chi2

    def get_chi2(self,model):
        rate_in  = self.get_inner_production_rate(model)
        rate_out = self.get_outer_production_rate(model)

        # print(rate_in -self.ProductionRateData[0],rate_out-self.ProductionRateData[1])
        Vinv = self.get_inverse_covariance_matrix()

        chi2 = (54.9-rate_in)**2/(2.5**2+(0.02*rate_in)**2+(0.02*rate_in)**2)
        chi2 += (55.6-rate_out)**2/(2.7**2+(0.02*rate_out)**2+(0.02*rate_out)**2)
        return chi2













test = Best()

noosc = Models.NoOscillations()
# osc = Models.PlaneWaveSterile(DM2_41 = 1, Sin22Th14 = 0.4)
# chi = 1
# test.get_inner_production_rate(noosc)
# test.get_outer_production_rate(noosc)
# # print(test.get_inner_production_rate(noosc)*chi)
# # print(test.get_outer_production_rate(noosc)*chi)
test.get_chi2(noosc)
# print(test.get_chi2(noosc))
# print(test.get_chi2(osc))
