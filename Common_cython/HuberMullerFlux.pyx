#import numpy as np
from libc.math cimport sin, cos, sqrt, exp
from cpython cimport array
import numpy as np

huber_muller =  None#{'U235': [4.367, -4.577, 2.100, -5.294e-1, 6.185e-2, -2.777e-3],
#                  'U238': [4.833e-1, 1.927e-1, -1.283e-1, -6.762e-3, 2.233e-3, -1.536e-4],
#                  'PU239': [4.757, -5.392, 2.563, -6.596e-1, 7.820e-2, -3.536e-3],
#                  'PU241': [2.990, -2.882, 1.278, -3.343e-1, 3.905e-2, -1.754e-3]}

# # Obtained from Muller et al 1101.2663 table VI
# muller =  {'U235': [3.217, -3.111, 1.395, -3.690e-1, 4.445e-2, -2.053e-3],
#            'U238': [4.833e-1,  1.927e-1, -1.283e-1, -6.762e-3, 2.233e-3, -1.536e-4],
#            'PU239': [6.413, -7.432, 3.535, -8.820e-1, 1.025e-1, -4.550e-3],
#            'PU241': [3.251, -3.204, 1.428, -3.675e-1, 4.254e-2, -1.896e-3]}



#  Prescriptiions for all isotopes but U238 from Huber 1106.0687
#  The missing isotope (U238) is obtained from Muller et al
cdef double[:,:] HM = np.array([[4.367, -4.577, 2.100, -5.294e-1, 6.185e-2, -2.777e-3],
                       [4.833e-1, 1.927e-1, -1.283e-1, -6.762e-3, 2.233e-3, -1.536e-4],
                       [4.757, -5.392, 2.563, -6.596e-1, 7.820e-2, -3.536e-3],
                       [2.990, -2.882, 1.278, -3.343e-1, 3.905e-2, -1.754e-3]])

# Obtained from Muller et al 1101.2663 table VI

# M = np.array([[3.217, -3.111, 1.395, -3.690e-1, 4.445e-2, -2.053e-3],
#                                    [4.833e-1,  1.927e-1, -1.283e-1, -6.762e-3, 2.233e-3, -1.536e-4],
#                                    [6.413, -7.432, 3.535, -8.820e-1, 1.025e-1, -4.550e-3],
#                                    [3.251, -3.204, 1.428, -3.675e-1, 4.254e-2, -1.896e-3]])


cdef int L = 6

class reactor_isotope_flux:
    def __init__(self, isotope_name,flux_parameters):
        """
        Initialisation input:
        isotope_name (str): a key to choose which isotope
        we are studying. For instance, 'U235' or 'PU239'.
        flux_parameters (dict): a dictionary, with the names of
        the isotope as the keys, and a list of the flux_parameters
        associated to each key.

        Note to self: when the class is called, call it with
        the right flux_parameters as an argument, either
        hubber_muller or muller.
        """

        names = {'U235':0,'U238':1,'PU239':2,'PU241':3}
        self.isotope = names[isotope_name]
        #self.flux_parameters = flux_parameters

    def GetFlux(self, double Enu):
        """
        Input:
        The energy Enu of the outgoing neutrino (in MeV),
        of which we want to know its flux.

        Output:
        The flux of the outgoing neutrino with energy Enu
        after the beta decay of the isotope in isotope_name.
        Units: #neutrinos/(MeV · fission)

        More info: P. Vogel and J. Engel, Phys. Rev.D39, 3378 (1989).
        """

        # # According to 1704.01082, U235 flux must be corrected
        # # by reducing its yield by 7.8%.
        # norm = 1.
        # if self.isotope_name == 'U235':
        #     norm = 1-0.078

        # exponent = 0.0
        # for i in range(0,len(self.flux_parameters[self.isotope_name])):
        #     exponent += self.flux_parameters[self.isotope_name][i]*Enu**i

        cdef double exponent = 0.0
        cdef int iso=self.isotope
        cdef int i
        for i in range(L):
            #print(iso,i)
            exponent += HM[iso][i]*Enu**i

        # The physical meaning of this formula is found on (22) from 1106.0687,
        # it's some kind of perturbative development as a function of Enu.
        return exp(exponent)#*norm
