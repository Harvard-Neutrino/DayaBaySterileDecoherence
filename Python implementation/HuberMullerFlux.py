import math

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
        self.isotope_name = isotope_name
        self.flux_parameters = flux_parameters

    def GetFlux(self, Enu):
        """
        Input:
        The energy Enu of the outgoing neutrino (in MeV),
        of which we want to know its flux.

        Output:
        The flux of the outgoing neutrino with energy Enu
        after the beta decay of the isotope in isotope_name,
        according to the formula and parameters in 1106.0687.
        """
        exponent = 0.0
        for i in range(0,len(self.flux_parameters)):
            exponent += self.flux_parameters[self.isotope_name][i]*Enu**i
        # The physical meaning of this formula is found on (22) from 1106.0687,
        # it's some kind of perturbative development as a function of Enu.
        return math.exp(exponent)
