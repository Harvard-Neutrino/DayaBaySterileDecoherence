import math

NeutrinoEnergyThreshold = 1.806 # MeV
DeltaNeutronToProtonMass = 1.29322 # MeV from PDG2018 mass differences
ElectronMass = 0.511 # MeV
sigma_0 = 0.0952e-42 # cm^2
# More information on the IBD CrossSection can be found in
# A Oralbaev et al 2016 J. Phys.: Conf. Ser. 675 012003
# Using that data, sigma_0 should be 9.3516e-44 #cm^-2 at tree-level.
# One should check this better, but it should arise no big error.

def ElectronEnergyO0(Enu):
    """
    Input:
    Enu (double): real energy of the incoming neutrino.

    Output:
    Real energy of the electron after IBD (float).

    Note:
    Returns a negative energy if Enu is less than the
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
    Returns an error if the energy is smaller than the mass.
    This is acknowledged in CrossSection().
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
