import numpy as np
# This module must return an oscillation probability dependent on L an enu.
# The way we compute this probability will entirely depend on the model that
# we pick (number of neutrinos, NSIs, steriles, decoherence...).
# The idea is that the rest of the program is model-independent.

# It might be interesting to define this as a class, since then we will be
# able to recover the values of the parameters from the class itself.


# -----------------------------------------------------------
# No oscillations
# -----------------------------------------------------------
class NoOscillations:

    def oscProbability(self,enu,L):
        return 1.


# -----------------------------------------------------------
# Plane wave Standard Model: 3 neutrinos, no decoherence
# Uses the approximate formula and BF parameters from Daya Bay. No matter effects.
# -----------------------------------------------------------

class PlaneWaveDB:
    def __init__(self,Sin22Th13 = 0.0841,DM2_ee = 2.5e-3):
        self.th13 = np.arcsin(np.sqrt(Sin22Th13))/2.
        self.th12 = 0.583763
        self.dm2_ee = DM2_ee
        self.dm2_21 = 7.42e-5

    def oscProbability(self,enu,L):
        """
        Input:
        enu (float): the energy of the electron antineutrino, in MeV.
        L (float): the length travelled by the antineutrino, in meters.

        Output:
        The probability of the antineutrino remaining an antineutrino.
        This is computed according to (find paper!).
        """
        x21 = 1.267*self.dm2_21*L/enu
        xee = 1.267*self.dm2_ee*L/enu
        return (1- np.cos(self.th13)**4*np.sin(2*self.th12)**2*np.sin(x21)**2
                - np.sin(2*self.th13)**2*np.sin(xee)**2)


# -----------------------------------------------------------
# Plane wave Standard Model: 3 neutrinos, no decoherence
# Uses the full formula. Parameters taken from nu-fit.org
# -----------------------------------------------------------

class PlaneWaveSM:
    def __init__(self,Sin22Th13 = 0.0868525,DM2_31 = 2.515e-3):
        self.th13 = np.arcsin(np.sqrt(Sin22Th13))/2.
        self.th12 = 0.583763
        self.dm2_31 = DM2_31
        self.dm2_21 = 7.42e-5
        self.dm2_32 = self.dm2_31 - self.dm2_21

    def fosc(self,enu,L,deltam):
        return np.cos(L*deltam/(2*enu))

    def oscProbability(self,E,l):
        """
        Input:
        enu (float): the energy of the electron antineutrino, in MeV.
        L (float): the length travelled by the antineutrino, in meters.

        Output:
        The probability of the antineutrino remaining an antineutrino.
        This is computed according to (find paper!).
        """
        enu = 1e6*E # conversion from MeV to eV
        L = 5.06773e6*l # conversion from meters to 1/eV
        prob = 1.
        prob -= np.sin(2*self.th12)**2*np.cos(self.th13)**4*(1-self.fosc(enu,L,self.dm2_21))/2.
        prob -= np.sin(2*self.th13)**2*(np.cos(self.th12)**2*(1-self.fosc(enu,L,self.dm2_31))+
                                        np.sin(self.th12)**2*(1-self.fosc(enu,L,self.dm2_32)))/2.
        return prob


# -----------------------------------------------------------
# Wave packet Standard Model: 3 neutrinos, decoherence due to wave packet separation
# Uses the full formula. Parameters taken from nu-fit.org
# -----------------------------------------------------------
class WavePacketSM:
    def __init__(self,Sin22Th13 = 0.0868525,DM2_31 = 2.515e-3):
        self.th13 = np.arcsin(np.sqrt(Sin22Th13))/2.
        self.th12 = 0.583763
        self.dm2_31 = DM2_31
        self.dm2_21 = 7.42e-5
        self.dm2_32 = self.dm2_31 - self.dm2_21
        nm = 50677.308*1e-7 # 1/eV
        self.sigmax = 2.1e-4*nm # after multiplying by nm, sigmax in 1/eV

    def fosc(self,enu,L,deltam):
        return np.cos(L*deltam/(2*enu))*np.exp(-L**2*deltam**2/(32.*enu**4*self.sigmax**2))

    def oscProbability(self,E,l):
        """
        Input:
        enu (float): the energy of the electron antineutrino, in MeV.
        L (float): the length travelled by the antineutrino, in meters.

        Output:
        The probability of the antineutrino remaining an antineutrino.
        This is computed according to (find paper!).
        """
        enu = 1e6*E # conversion from MeV to eV
        L = 5.06773e6*l # conversion from meters to 1/eV
        prob = 1.
        prob -= np.sin(2*self.th12)**2*np.cos(self.th13)**4*(1-self.fosc(enu,L,self.dm2_21))/2.
        prob -= np.sin(2*self.th13)**2*(np.cos(self.th12)**2*(1-self.fosc(enu,L,self.dm2_31))+
                                        np.sin(self.th12)**2*(1-self.fosc(enu,L,self.dm2_32)))/2.
        return prob

# -----------------------------------------------------------
# Plane wave with sterile neutrino: 4 neutrinos, no decoherence
# Uses the final formula for computation efficiency. No matter effects.
# -----------------------------------------------------------
class PlaneWaveSterile:
    def __init__(self,Sin22Th14 = 0.01,DM2_41 = 0.1):
        self.th14 = np.arcsin(np.sqrt(Sin22Th14))/2.
        self.th13 = np.arcsin(np.sqrt(0.0868525))/2.
        self.th12 = 0.583763
        self.dm2_41 = DM2_41
        self.dm2_31 = 2.515e-3
        self.dm2_21 = 7.42e-5
        self.dm2_32 = self.dm2_31 - self.dm2_21
        self.dm2_42 = self.dm2_41 - self.dm2_21
        self.dm2_43 = self.dm2_41 - self.dm2_31

    def fosc(self,enu,L,deltam):
        return np.cos(L*deltam/(2*enu))

    def oscProbability(self,E,l):
        """
        Input:
        enu (float): the energy of the electron antineutrino, in MeV.
        L (float): the length travelled by the antineutrino, in meters.

        Output:
        The probability of the antineutrino remaining an antineutrino.
        This is computed according to (find paper!).
        """
        enu = 1e6*E # conversion from MeV to eV
        L = 5.06773e6*l # conversion from meters to 1/eV
        prob = 1.
        prob -= np.sin(2*self.th12)**2* np.cos(self.th13)**4* np.cos(self.th14)**4*(1-self.fosc(enu,L,self.dm2_21))/2.
        prob -= np.sin(2*self.th13)**2* np.cos(self.th14)**4*(np.cos(self.th12)**2*(1-self.fosc(enu,L,self.dm2_31))+
                                                              np.sin(self.th12)**2*(1-self.fosc(enu,L,self.dm2_32)))/2.
        prob -= np.sin(2*self.th14)**2*(np.cos(self.th13)**2*(np.cos(self.th12)**2*(1-self.fosc(enu,L,self.dm2_41))+
                                                              np.sin(self.th12)**2*(1-self.fosc(enu,L,self.dm2_42)))+
                                        np.sin(self.th13)**2*(1-self.fosc(enu,L,self.dm2_43)))/2.
        return prob

    def oscProbability_av(self,E,l):
        """
        Input:
        enu (float): the energy of the electron antineutrino, in MeV.
        L (float): the length travelled by the antineutrino, in meters.

        Output:
        The probability of the antineutrino remaining an antineutrino.
        This is computed according to (find paper!).
        """
        enu = 1e6*E # conversion from MeV to eV
        L = 5.06773e6*l # conversion from meters to 1/eV
        prob = 1.
        prob -= np.sin(2*self.th12)**2* np.cos(self.th13)**4* np.cos(self.th14)**4*(1-self.fosc(enu,L,self.dm2_21))/2.
        prob -= np.sin(2*self.th13)**2* np.cos(self.th14)**4*(np.cos(self.th12)**2*(1-self.fosc(enu,L,self.dm2_31))+
                                                              np.sin(self.th12)**2*(1-self.fosc(enu,L,self.dm2_32)))/2.
        prob -= np.sin(2*self.th14)**2/2.
        return prob


# -----------------------------------------------------------
# Wave packet with sterile neutrino: 4 neutrinos, decoherence due to wave packet separation
# Uses the final formula for computation efficiency. No matter effects.
# -----------------------------------------------------------
class WavePacketSterile:
    def __init__(self,Sin22Th14 = 0.01,DM2_41 = 0.1):
        self.th14 = np.arcsin(np.sqrt(Sin22Th14))/2.
        self.th13 = np.arcsin(np.sqrt(0.0868525))/2.
        self.th12 = 0.583763
        self.dm2_41 = DM2_41
        self.dm2_31 = 2.515e-3
        self.dm2_21 = 7.42e-5
        self.dm2_32 = self.dm2_31 - self.dm2_21
        self.dm2_42 = self.dm2_41 - self.dm2_21
        self.dm2_43 = self.dm2_41 - self.dm2_31
        nm = 50677.308*1e-7 # 1/eV
        self.sigmax = 2.1e-4*nm # after multiplying by nm, sigmax in 1/eV

    def fosc(self,enu,L,deltam):
        return np.cos(L*deltam/(2*enu))*np.exp(-L**2*deltam**2/(32.*enu**4*self.sigmax**2))

    def oscProbability(self,E,l):
        """
        Input:
        enu (float): the energy of the electron antineutrino, in MeV.
        L (float): the length travelled by the antineutrino, in meters.

        Output:
        The probability of the antineutrino remaining an antineutrino.
        This is computed according to (find paper!).
        """
        enu = 1e6*E # conversion from MeV to eV
        L = 5.06773e6*l # conversion from meters to 1/eV
        prob = 1.
        prob -= np.sin(2*self.th12)**2* np.cos(self.th13)**4* np.cos(self.th14)**4*(1-self.fosc(enu,L,self.dm2_21))/2.
        prob -= np.sin(2*self.th13)**2* np.cos(self.th14)**4*(np.cos(self.th12)**2*(1-self.fosc(enu,L,self.dm2_31))+
                                                              np.sin(self.th12)**2*(1-self.fosc(enu,L,self.dm2_32)))/2.
        prob -= np.sin(2*self.th14)**2*(np.cos(self.th13)**2*(np.cos(self.th12)**2*(1-self.fosc(enu,L,self.dm2_41))+
                                                              np.sin(self.th12)**2*(1-self.fosc(enu,L,self.dm2_42)))+
                                        np.sin(self.th13)**2*(1-self.fosc(enu,L,self.dm2_43)))/2.
        return prob

    def oscProbability_av(self,E,l):
        """
        Input:
        enu (float): the energy of the electron antineutrino, in MeV.
        L (float): the length travelled by the antineutrino, in meters.

        Output:
        The probability of the antineutrino remaining an antineutrino.
        This is computed according to (find paper!).
        """
        enu = 1e6*E # conversion from MeV to eV
        L = 5.06773e6*l # conversion from meters to 1/eV
        prob = 1.
        prob -= np.sin(2*self.th12)**2* np.cos(self.th13)**4* np.cos(self.th14)**4*(1-self.fosc(enu,L,self.dm2_21))/2.
        prob -= np.sin(2*self.th13)**2* np.cos(self.th14)**4*(np.cos(self.th12)**2*(1-self.fosc(enu,L,self.dm2_31))+
                                                              np.sin(self.th12)**2*(1-self.fosc(enu,L,self.dm2_32)))/2.
        prob -= np.sin(2*self.th14)**2/2.
        return prob


# -----------------------------------------------------------
# Plane wave Standard Model: 3 neutrinos, no decoherence
# Implements the most general formula and allows for matter effects.
# -----------------------------------------------------------
class PlaneWaveSM_full:
    def __init__(self):
        self.deltaCP = 0
        self.theta12 = np.arcsin(np.sqrt(0.846))/2.
        self.theta13 = np.arcsin(np.sqrt(0.0868525))/2.
        self.theta23 = np.arcsin(np.sqrt(0.999))/2.
        self.dm2_31 = 2.44e-3 # eV^2
        self.dm2_21 = 7.42e-5 # eV^2
        self.V = 0
        self.VCC = 0

    def get_mixing_matrix(self):
        """ Returns the PMNS matrix"""
        c12 = np.cos(self.theta12)
        c13 = np.cos(self.theta13)
        c23 = np.cos(self.theta23)
        s12 = np.sin(self.theta12)
        s13 = np.sin(self.theta13)
        s23 = np.sin(self.theta23)
        dCP = self.deltaCP
        U3 = np.matrix([[c12*c13, s12*c13, s13*np.exp(-1j*dCP)],
                       [-s12*c23 - c12*s13*s23*np.exp(-1j*dCP),
                        c12*c23 - s12*s13*s23*np.exp(-1j*dCP), c13*s23],
                       [s12*s23 - c12*s13*c23*np.exp(1j*dCP),
                        -c12*s23 - s12*s13*c23*np.exp(-1j*dCP), c13*c23]])
        return U3

    def ham(self,enu):
        """This function returns the three-neutrino hamiltonian in
        the flavor basis (e, mu, s) and under the ultrarelativistic
        approximation, for a given energy E."""
        massmat = np.matrix([[0,0,0],[0,self.dm2_21,0],[0,0,self.dm2_31]])
        U3 = self.get_mixing_matrix()
        return (1/(2*enu)*np.dot(U3,np.dot(massmat,U3.getH())) +
                np.matrix([[self.VCC+self.V,0,0],[0,self.V,0],[0,0,0]]))

    def DeltaEij(self,enu):
        """This function returns the difference in energies of
        the matter eigenstates for a given neutrino energy E."""
        vaps = np.linalg.eig(self.ham(enu))[0]
        return np.array([vaps[1]-vaps[0],vaps[2]-vaps[0],vaps[2]-vaps[1]])

    def get_matter_mixing_matrix(self,enu):
        """This function returns the mixing matrix in matter
        for a given energy E."""
        return np.linalg.eig(self.ham(enu))[1]

    def deltavels(self,E,h = 1e-5):
        """This function computes the difference in velocities between
        the wave packets of the eigenstates. The velocities are computed
        numerically by performing a numerical derivative.
        Here the step of the derivative has been picked as 0.00001E. Then,
        the estimated error is proporcional to (0.00001E)^2 and the third
        derivative of the energy eigenvalues."""
        eps = E*h
        vapspost = np.linalg.eig(self.ham(E+eps))[0]
        vapspre = np.linalg.eig(self.ham(E-eps))[0]
        vels = (vapspost-vapspre)/(2*eps)
        return np.array([vels[1]-vels[0],vels[2]-vels[0],vels[2]-vels[1]])

    def oscProbability(self,E,l):
        """This function returns the probability of an electron neutrino
        with energy E remaining an electron neutrino after travelling a
        distance L, in the wave packet formalism.
        L must be in meters, and E must be in MeV."""
        enu = 1e6*E # conversion from MeV to eV
        L = 5.06773e6*l # conversion from meters to 1/eV
        arg = -1j*self.DeltaEij(enu)*L
        U2 = np.square(np.abs(self.get_matter_mixing_matrix(enu)))
        return U2[0,0]**2+U2[0,1]**2+U2[0,2]**2 + 2*np.real(
               U2[0,0]*U2[0,1]*np.exp(arg[0])+
               U2[0,0]*U2[0,2]*np.exp(arg[1])+
               U2[0,1]*U2[0,2]*np.exp(arg[2]))

# -----------------------------------------------------------
# Wave packet Standard Model: 3 neutrinos, decoherence effects allowed.
# Implements the most general formula and allows for matter effects.
# -----------------------------------------------------------
class WavePacketSM_full:
    def __init__(self):
        nm = 50677.308*1e-7 # 1/eV
        self.deltaCP = 0
        self.theta12 = np.arcsin(np.sqrt(0.846))/2.
        self.theta13 = np.arcsin(np.sqrt(0.0868525))/2.
        self.theta23 = np.arcsin(np.sqrt(0.999))/2.
        self.dm2_31 = 2.44e-3 # eV^2
        self.dm2_21 = 7.42e-5 # eV^2
        self.V = 0
        self.VCC = 0
        self.sigmax = 2.1e-4*nm

    def get_mixing_matrix(self):
        """ Returns the PMNS matrix"""
        c12 = np.cos(self.theta12)
        c13 = np.cos(self.theta13)
        c23 = np.cos(self.theta23)
        s12 = np.sin(self.theta12)
        s13 = np.sin(self.theta13)
        s23 = np.sin(self.theta23)
        dCP = self.deltaCP
        U3 = np.matrix([[c12*c13, s12*c13, s13*np.exp(-1j*dCP)],
                       [-s12*c23 - c12*s13*s23*np.exp(-1j*dCP),
                        c12*c23 - s12*s13*s23*np.exp(-1j*dCP), c13*s23],
                       [s12*s23 - c12*s13*c23*np.exp(1j*dCP),
                        -c12*s23 - s12*s13*c23*np.exp(-1j*dCP), c13*c23]])
        return U3

    def ham(self,enu):
        """This function returns the three-neutrino hamiltonian in
        the flavor basis (e, mu, s) and under the ultrarelativistic
        approximation, for a given energy E."""
        massmat = np.matrix([[0,0,0],[0,self.dm2_21,0],[0,0,self.dm2_31]])
        U3 = self.get_mixing_matrix()
        return (1/(2*enu)*np.dot(U3,np.dot(massmat,U3.getH())) +
                np.matrix([[self.VCC+self.V,0,0],[0,self.V,0],[0,0,0]]))

    def DeltaEij(self,enu):
        """This function returns the difference in energies of
        the matter eigenstates for a given neutrino energy E."""
        vaps = np.linalg.eig(self.ham(enu))[0]
        return np.array([vaps[1]-vaps[0],vaps[2]-vaps[0],vaps[2]-vaps[1]])

    def get_matter_mixing_matrix(self,enu):
        """This function returns the mixing matrix in matter
        for a given energy E."""
        return np.linalg.eig(self.ham(enu))[1]

    def deltavels(self,E,h = 1e-5):
        """This function computes the difference in velocities between
        the wave packets of the eigenstates. The velocities are computed
        numerically by performing a numerical derivative.
        Here the step of the derivative has been picked as 0.001E. Then,
        the estimated error is proporcional to (0.001E)^2 and the third
        derivative of the energy eigenvalues."""
        eps = E*h
        vapspost = np.linalg.eig(self.ham(E+eps))[0]
        vapspre = np.linalg.eig(self.ham(E-eps))[0]
        vels = (vapspost-vapspre)/(2*eps)
        return np.array([vels[1]-vels[0],vels[2]-vels[0],vels[2]-vels[1]])

    def oscProbability(self,E,l):
        """This function returns the probability of an electron neutrino
        with energy E remaining an electron neutrino after travelling a
        distance L, in the wave packet formalism."""
        enu = 1e6*E # conversion from MeV to eV
        L = 5.06773e6*l # conversion from meters to 1/eV
        argosc = -1j*self.DeltaEij(enu)*L-self.sigmax**2*self.DeltaEij(enu)**2
        argcoh = -L**2/(8*self.sigmax**2)*self.deltavels(enu)**2
        arg = argcoh + argosc
        U2 = np.square(np.abs(self.get_matter_mixing_matrix(enu)))
        return U2[0,0]**2+U2[0,1]**2+U2[0,2]**2 + 2*np.real(
                U2[0,0]*U2[0,1]*np.exp(arg[0])+
                U2[0,0]*U2[0,2]*np.exp(arg[1])+
                U2[0,1]*U2[0,2]*np.exp(arg[2]))

# -----------------------------------------------------------
# Extended Standard Model: 3 neutrinos + 1 sterile, no decoherence
# Implements the most general formula and allows for matter effects.
# -----------------------------------------------------------
class PlaneWaveSterile_full:
    def __init__(self):
        r14 = 0.5
        r24 = 0.1
        r34 = 0.0
        self.deltaCP = 0.
        self.delta14 = 0.
        self.delta24 = 0.
        self.delta34 = 0.
         #DB best fit
        self.theta12 = np.arcsin(np.sqrt(0.304))
        self.theta13 = np.arcsin(np.sqrt(0.0868525))/2.  # nu-fit is sin^2theta = 0.02221
        self.theta23 = np.arcsin(np.sqrt(0.570))
        self.theta14 = np.arcsin(np.sqrt(r14))/2.
        self.theta24 = np.arcsin(np.sqrt(r24))/2.
        self.theta34 = np.arcsin(np.sqrt(r34))/2.
        self.dm2_31 = 2.44e-3 # eV^2
        self.dm2_21 = 7.42e-5 # eV^2
        self.dm2_41 = 0.1     # eV^2
        self.V = 0
        self.VCC = 0

    def get_mixing_matrix(self):
        """ Returns the PMNS matrix"""
        c12 = np.cos(self.theta12)
        c13 = np.cos(self.theta13)
        c23 = np.cos(self.theta23)
        s12 = np.sin(self.theta12)
        s13 = np.sin(self.theta13)
        s23 = np.sin(self.theta23)
        c14 = np.cos(self.theta14)
        c24 = np.cos(self.theta24)
        c34 = np.cos(self.theta34)
        s14 = np.sin(self.theta14)
        s24 = np.sin(self.theta24)
        s34 = np.sin(self.theta34)
        dCP = self.deltaCP
        R14 = np.matrix([[c14, 0, 0, s14*np.exp(-1j*self.delta14)], [0, 1, 0, 0], [0, 0, 1, 0],
                         [-s14*np.exp(1j*self.delta14), 0, 0, c14]])
        R24 = np.matrix([[1, 0, 0, 0], [0, c24, 0, s24*np.exp(-1j*self.delta24)], [0, 0, 1, 0],
                         [0, -s24*np.exp(1j*self.delta24), 0, c24]])
        R34 = np.matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, c34, s34*np.exp(-1j*self.delta34)],
                         [0, 0, -s34*np.exp(1j*self.delta34), c34]])
        R12 = np.matrix([[c12, s12, 0, 0], [-s12, c12, 0, 0], [0, 0, 1, 0], [0, 0, 0,1]])
        R13 = np.matrix([[c13, 0, s13*np.exp(-1j*self.deltaCP), 0], [0, 1, 0, 0],
                         [-s13*np.exp(1j*self.deltaCP), 0, c13, 0], [0, 0, 0, 1]])
        R23 = np.matrix([[1, 0, 0, 0], [0, c23, s23, 0], [0, -s23, c23, 0], [0, 0, 0, 1]])
        U4 = R34*R24*R14*R23*R13*R12
        return U4

    def ham(self,enu):
        """This function returns the three-neutrino hamiltonian in
        the flavor basis (e, mu, s) and under the ultrarelativistic
        approximation, for a given energy E."""
        massmat = np.matrix([[0,0,0,0],[0,self.dm2_21,0,0],[0,0,self.dm2_31,0],[0,0,0,self.dm2_41]])
        U4 = self.get_mixing_matrix()
        return (1/(2*enu)*np.dot(U4,np.dot(massmat,U4.getH())) +
                np.matrix([[self.VCC+self.V,0,0,0],[0,self.V,0,0],[0,0,self.V,0],[0,0,0,0]]))

    def DeltaEij(self,enu):
        """This function returns the difference in energies of
        the matter eigenstates for a given neutrino energy E."""
        vaps = np.linalg.eig(self.ham(enu))[0]
        return np.array([vaps[1]-vaps[0],vaps[2]-vaps[0],vaps[3]-vaps[0],
                         vaps[2]-vaps[1],vaps[3]-vaps[1],vaps[3]-vaps[2]])

    def get_matter_mixing_matrix(self,enu):
        """This function returns the mixing matrix in matter
        for a given energy E."""
        return np.linalg.eig(self.ham(enu))[1]

    def deltavels(self,E,h = 1e-5):
        """This function computes the difference in velocities between
        the wave packets of the eigenstates. The velocities are computed
        numerically by performing a numerical derivative.
        Here the step of the derivative has been picked as 0.001E. Then,
        the estimated error is proporcional to (0.001E)^2 and the third
        derivative of the energy eigenvalues."""
        eps = E*h
        vapspost = np.linalg.eig(self.ham(E+eps))[0]
        vapspre = np.linalg.eig(self.ham(E-eps))[0]
        vels = (vapspost-vapspre)/(2*eps)
        return np.array([vels[1]-vels[0],vels[2]-vels[0],vels[3]-vels[0],
                         vels[2]-vels[1],vels[3]-vels[1],vels[3]-vels[2]])

    def oscProbability(self,enu,L):
        """This function returns the probability of an electron neutrino
        with energy E remaining an electron neutrino after travelling a
        distance L, in the wave packet formalism."""
        arg = -1j*self.DeltaEij(enu)*L
        U2 = np.square(np.abs(self.get_matter_mixing_matrix(enu)))
        return U2[0,0]**2+U2[0,1]**2+U2[0,2]**2+U2[0,3]**2 + 2*np.real(
                U2[0,0]*U2[0,1]*np.exp(arg[0])+
                U2[0,0]*U2[0,2]*np.exp(arg[1])+
                U2[0,0]*U2[0,3]*np.exp(arg[2])+
                U2[0,1]*U2[0,2]*np.exp(arg[3])+
                U2[0,1]*U2[0,3]*np.exp(arg[4])+
                U2[0,2]*U2[0,3]*np.exp(arg[5]))


# -----------------------------------------------------------
# Extended Standard Model: 3 neutrinos + 1 sterile, decoherence allowed
# Implements the most general formula and allows for matter effects.
# -----------------------------------------------------------
class WavePacketSterile_full:
    def __init__(self):
        nm = 50677.308*1e-7 # 1/eV
        r14 = 0.5
        r24 = 0.1
        r34 = 0.0
        self.deltaCP = 0.
        self.delta14 = 0.
        self.delta24 = 0.
        self.delta34 = 0.
        self.theta12 = np.arcsin(np.sqrt(0.304))
        self.theta13 = np.arcsin(np.sqrt(0.02221))
        self.theta23 = np.arcsin(np.sqrt(0.570))
        self.theta14 = np.arcsin(np.sqrt(r14))/2.
        self.theta24 = np.arcsin(np.sqrt(r24))/2.
        self.theta34 = np.arcsin(np.sqrt(r34))/2.
        self.dm2_31 = 2.44e-3 # eV^2
        self.dm2_21 = 7.42e-5 # eV^2
        self.dm2_41 = 0.1     # eV^2
        self.V = 0
        self.VCC = 0
        self.sigmax = 2.1e-3*nm

    def get_mixing_matrix(self):
        """ Returns the 4 neutrino mixing matrix. """
        c12 = np.cos(self.theta12)
        c13 = np.cos(self.theta13)
        c23 = np.cos(self.theta23)
        s12 = np.sin(self.theta12)
        s13 = np.sin(self.theta13)
        s23 = np.sin(self.theta23)
        c14 = np.cos(self.theta14)
        c24 = np.cos(self.theta24)
        c34 = np.cos(self.theta34)
        s14 = np.sin(self.theta14)
        s24 = np.sin(self.theta24)
        s34 = np.sin(self.theta34)
        dCP = self.deltaCP
        R14 = np.matrix([[c14, 0, 0, s14*np.exp(-1j*self.delta14)], [0, 1, 0, 0], [0, 0, 1, 0],
                         [-s14*np.exp(1j*self.delta14), 0, 0, c14]])
        R24 = np.matrix([[1, 0, 0, 0], [0, c24, 0, s24*np.exp(-1j*self.delta24)], [0, 0, 1, 0],
                         [0, -s24*np.exp(1j*self.delta24), 0, c24]])
        R34 = np.matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, c34, s34*np.exp(-1j*self.delta34)],
                         [0, 0, -s34*np.exp(1j*self.delta34), c34]])
        R12 = np.matrix([[c12, s12, 0, 0], [-s12, c12, 0, 0], [0, 0, 1, 0], [0, 0, 0,1]])
        R13 = np.matrix([[c13, 0, s13*np.exp(-1j*self.deltaCP), 0], [0, 1, 0, 0],
                         [-s13*np.exp(1j*self.deltaCP), 0, c13, 0], [0, 0, 0, 1]])
        R23 = np.matrix([[1, 0, 0, 0], [0, c23, s23, 0], [0, -s23, c23, 0], [0, 0, 0, 1]])
        U4 = R34*R24*R14*R23*R13*R12
        return U4

    def ham(self,enu):
        """This function returns the three-neutrino hamiltonian in
        the flavor basis (e, mu, s) and under the ultrarelativistic
        approximation, for a given energy E."""
        massmat = np.matrix([[0,0,0,0],[0,self.dm2_21,0,0],[0,0,self.dm2_31,0],[0,0,0,self.dm2_41]])
        U4 = self.get_mixing_matrix()
        return (1/(2*enu)*np.dot(U4,np.dot(massmat,U4.getH())) +
                np.matrix([[self.VCC+self.V,0,0,0],[0,self.V,0,0],[0,0,self.V,0],[0,0,0,0]]))

    def DeltaEij(self,enu):
        """This function returns the difference in energies of
        the matter eigenstates for a given neutrino energy E."""
        vaps = np.linalg.eig(self.ham(enu))[0]
        return np.array([vaps[1]-vaps[0],vaps[2]-vaps[0],vaps[3]-vaps[0],
                         vaps[2]-vaps[1],vaps[3]-vaps[1],vaps[3]-vaps[2]])

    def get_matter_mixing_matrix(self,enu):
        """This function returns the mixing matrix in matter
        for a given energy E."""
        return np.linalg.eig(self.ham(enu))[1]

    def deltavels(self,E,h = 1e-5):
        """This function computes the difference in velocities between
        the wave packets of the eigenstates. The velocities are computed
        numerically by performing a numerical derivative.
        Here the step of the derivative has been picked as 0.001E. Then,
        the estimated error is proporcional to (0.001E)^2 and the third
        derivative of the energy eigenvalues."""
        eps = E*h
        vapspost = np.linalg.eig(self.ham(E+eps))[0]
        vapspre = np.linalg.eig(self.ham(E-eps))[0]
        vels = (vapspost-vapspre)/(2*eps)
        return np.array([vels[1]-vels[0],vels[2]-vels[0],vels[3]-vels[0],
                         vels[2]-vels[1],vels[3]-vels[1],vels[3]-vels[2]])

    def oscProbability(self,enu,L):
        """This function returns the probability of an electron neutrino
        with energy E remaining an electron neutrino after travelling a
        distance L, in the wave packet formalism."""
        argosc = -1j*self.DeltaEij(enu)*L-self.sigmax**2*self.DeltaEij(enu)**2
        argcoh = -L**2/(8*self.sigmax**2)*self.deltavels(enu)**2
        arg = argcoh + argosc
        U2 = np.square(np.abs(self.get_matter_mixing_matrix(enu)))
        return U2[0,0]**2+U2[0,1]**2+U2[0,2]**2+U2[0,3]**2 + 2*np.real(
                U2[0,0]*U2[0,1]*np.exp(arg[0])+
                U2[0,0]*U2[0,2]*np.exp(arg[1])+
                U2[0,0]*U2[0,3]*np.exp(arg[2])+
                U2[0,1]*U2[0,2]*np.exp(arg[3])+
                U2[0,1]*U2[0,3]*np.exp(arg[4])+
                U2[0,2]*U2[0,3]*np.exp(arg[5]))
