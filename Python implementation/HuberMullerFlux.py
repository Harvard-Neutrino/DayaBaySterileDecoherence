import math

class reactor_isotope_flux:
    def __init__(self, isotope_name,flux_parameters):
        self.isotope_name = isotope_name
        self.flux_parameters = flux_parameters
        # When the class is called, call it with the
        # right flux_parameters as an argument
        # (either hubber_muller or muller).

    def GetFlux(self, Enu):
        """ Returns the flux of the isotope, using the flux_parameters
        from 1106.0687."""
        exponent = 0.0
        for i in range(0,len(self.flux_parameters)):
            exponent += self.flux_parameters[self.isotope_name][i]*Enu**i
        # The physical meaning of this formula is found on (22) from 1106.0687,
        # it's some kind of perturbative development as a function of Enu.
        return math.exp(exponent)
