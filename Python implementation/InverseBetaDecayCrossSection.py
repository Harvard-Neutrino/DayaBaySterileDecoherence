import math

NeutrinoEnergyThreshold = 1.806 # MeV
DeltaNeutronToProtonMass = 1.29322 # MeV from PDG2018 mass differences
ElectronMass = 0.511 # MeV
sigma_0 = 0.0952e-42 # cm^2

def ElectronEnergyO0(Enu):
    """
    Input:
    Enu (double): real energy of the incoming neutrino.

    Output:
    Real energy of the electron after IBD (double).

    Note:
    Might return a negative energy if Enu is less than the
    difference of energy between neutron an proton.
    This problem is acknowledged in function CrossSection().
    """
    return Enu - DeltaNeutronToProtonMass


def ElectronMomentum(Eele):
    """
    Input:
    Eele (double): real energy of an electron.

    Output:
    Real momentum of the electron (double).

    Note:
    Might return an imaginary number if the energy is
    smaller than the mass. This is acknowledged in CrossSection().
    """
    return math.sqrt(Eele**2 - ElectronMass**2)



def CrossSection(Enu):
    """
    Input:
    Enu (double): Real energy of the incoming neutrino.

    Output:
    Cross-section of the IBD process.
    """
    Eele = ElectronEnergyO0(Enu)
    if (Eele < ElectronMass):
        return 0.0
    else:
        return sigma_0*Eele*ElectronMomentum(Eele)
