import numpy as np
# This module must return an oscillation probability dependent on L an enu.
# The way we compute this probability will entirely depend on the model that
# we pick (number of neutrinos, NSIs, steriles, decoherence...).
# The idea is that the rest of the program is model-independent.

# It might be interesting to define this as a class, since then we will be
# able to recover the values of the parameters from the class itself.

class PlaneWaveSM:
    def __init__(self):
        self.sin22th13 = 0.092
        self.dm2_31 = 2.494e-3

    def oscProbability(self,enu,L):
        """
        Input:
        enu (float): the energy of the electron antineutrino.
        L (float): the length travelled by the antineutrino.

        Output:
        The probability of the antineutrino remaining an antineutrino.
        This is computed according to (find paper!).
        """
        x = 1.267*self.dm2_31*L/enu
        return 1.- self.sin22th13*np.sin(x)**2
