#include "InverseBetaDecayCrossSection.h"

namespace IBD {

  double ElectronEnergyO0(double Enu){
    return Enu-DeltaNeutronToProtonMass;
  }

  double ElectronMomentum(double Eele){
    return sqrt(Eele*Eele - ElectronMass*ElectronMass);
  }

  double CrossSection(double Enu){
    double Eele = ElectronEnergyO0(Enu);
    if(Eele < ElectronMass) return 0.0;
    return sigma_0*Eele*ElectronMomentum(Eele);
  }

} // close IBD namespace
