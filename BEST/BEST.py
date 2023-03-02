import sys
import os
homedir = os.path.realpath(__file__)[:-len('BEST/BEST.py')]
common_dir = 'Common_cython'
sys.path.append(homedir+common_dir)

import Models
import MyUnits as muns
import numpy as np
from scipy import integrate as integrate

class Best:


    def __init__(self):

        # The BEST experiment studies the decay of 50Cr, which decays mainly in 4 neutrinos
        # The energy of these neutrinos is
        self.NeutrinoEnergies = [427e3*muns.eV,432e3*muns.eV,747e3*muns.eV,752e3*muns.eV]
        # And the fission fractions of such neutrinos are
        self.FissionFractions = {427e3*muns.eV: 0.0895, 432e3*muns.eV: 0.0093,
                                 747e3*muns.eV: 0.8163, 752e3*muns.eV: 0.0849}

        # This is the neutrino capture cross-section
        self.CrossSection = 5.81e-45*muns.cm**2
        # This is the initial activity of the 50Cr
        self.InitialActivity = 3.414e6*muns.Ci
        # This is the number density of the gallium in the targets
        self.GaNumberDensity = 2.1e22/muns.cm**3

        self.r_in = 133.5*muns.cm/2. # Radius of the spherical inner target
        self.r_Cr = (4.3+3)*muns.cm  # Radius of the Cr source and its shield
        self.h_Cr = (10.8+3)*muns.cm # Height of the Cr source and its shield
        self.r_out = 218.0*muns.cm/2. # Radius of the cylindrical outer target
        self.h_out = 234.5*muns.cm    # Height of the cylindrial outer target

        # We introduce the xi correction factors from (A20)
        self.GeometricCorrectionInner = 1./1.0985196
        self.GeometricCorrectionOuter = 1./1.0891257

        # We copy the data from table I in 2109.11482.
        self.ProductionRateData = np.array([49.4,44.9,62.9,73.3,49.8,69.5,64.6,53.8,49.9,69.1,
                                            41.1,63.6,51.4,66.6,46.9,87.3,50.4,59.7,43.0,78.8])/muns.day
        self.StatisticalError = np.array([4.2,5.9,7.4,8.6,8.2,12.0,12.6,12.2,16.5,19.4,
                                         5.3,5.7,7.3,9.8,7.9,13.2,10.6,11.7,15.3,20.0])/muns.day

        # We introduce systematic errors and the theoretical uncertainties of the cross-sections
        self.SystematicError = self.ProductionRateData*0.025
        self.CrossSectionError = self.ProductionRateData*0.035

    def is_in_inner_container(self,rho,z):
        """
        This function implements the geometry of the inner target,
        in cylindrical coordinates.

        Input:
        rho, z (float): the cylindrical coordinates.

        Output:
        a boolean whether this point (rho,z) is inside the inner target.
        """
        r = np.sqrt(rho**2+z**2)
        return not(r > self.r_in or (rho < self.r_Cr and z > -1*self.h_Cr/2))

    def is_in_outer_container(self,rho,z):
        """
        This function implements the geometry of the outer target,
        in cylindrical coordinates.

        Input:
        rho, z (float): the cylindrical coordinates.

        Output:
        a boolean whether this point (rho,z) is inside the outer target.
        """
        r = np.sqrt(rho**2+z**2)
        return not((r < self.r_in or (rho < self.r_Cr and z > 0) or
                    rho > self.r_out or z > self.h_out/2 or z < -1*self.h_out/2))

    def get_inner_production_rate(self,model):
        """
        This function implements formula (A20) at the inner detector.

        Input:
        model: an object from Models.py with an oscProbability method.

        Output:
        (float) the production rate at the inner detector.
        """

        def integrand(z,rho):
            # This function returns the integrand of the integral in (A20).
            if not self.is_in_inner_container(rho,z):
                return 0
            else:
                r = np.sqrt(rho**2+z**2)
                sump = np.sum([self.FissionFractions[e]*model.oscProbability(e/muns.MeV,r/muns.m)
                                                for e in self.NeutrinoEnergies])
                return sump*rho/r**2

        def ubound(rho):
            # This function implements the upper bound of the integral.
            return np.sqrt(self.r_in**2-rho**2)
        def lbound(rho):
            # This function implements the lower bound of the integral.
            return -np.sqrt(self.r_in**2-rho**2)

        int = integrate.dblquad(integrand,0,self.r_in,lbound,ubound)[0]/2.
        return self.InitialActivity*self.GaNumberDensity*self.CrossSection*int*self.GeometricCorrectionInner



    def get_outer_production_rate(self,model):
        """
        This function implements formula (A20) at the outer detector.

        Input:
        model: an object from Models.py with an oscProbability method.

        Output:
        (float) the production rate at the outer detector.
        """
        def integrand(z,rho):
            # This function returns the integrand of the integral in (A20).
            if not self.is_in_outer_container(rho,z):
                return 0
            else:
                r = np.sqrt(rho**2+z**2)
                sump = np.sum([self.FissionFractions[e]*model.oscProbability(e/muns.MeV,r/muns.m)
                                                for e in self.NeutrinoEnergies])
                return sump*rho/r**2

        def ubound1(rho):
            # This function implements the upper bound of the lower integral.
            if rho > self.r_in:
                return 0
            else:
                return -np.sqrt(self.r_in**2-rho**2)

        def lbound2(rho):
            # This function implements the lower bound of the upper integral.
            if rho > self.r_in:
                return 0
            else:
                return np.sqrt(self.r_in**2-rho**2)

        # Our integral is split in two subsegments, to achieve faster computation times.
        int  = integrate.dblquad(integrand,0,self.r_out,-self.h_out/2,ubound1)[0]/2.
        int += integrate.dblquad(integrand,self.r_Cr,self.r_out,lbound2,self.h_out/2)[0]/2.

        return self.InitialActivity*self.GaNumberDensity*self.CrossSection*int*self.GeometricCorrectionOuter



    def get_chi2(self,model):
        """
        This function implements formula (A19), with the

        Input:
        model: an object from Models.py with an oscProbability method.

        Output:
        (float) the chi2 of the model.
        """
        rate_in  = self.get_inner_production_rate(model)
        rate_out = self.get_outer_production_rate(model)

        chi2 = (54.9-rate_in)**2/(2.5**2+(0.02*rate_in)**2+(0.02*rate_in)**2)
        chi2 += (55.6-rate_out)**2/(2.7**2+(0.02*rate_out)**2+(0.02*rate_out)**2)
        return chi2
