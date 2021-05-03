#ifndef INVERSEBETADECAYCROSSSECTION_H
#define INVERSEBETADECAYCROSSSECTION_H

#include <cmath>

namespace IBD {
  // following hep-ph/9903554
  static const double NeutrinoEnergyThreshold = 1.806; // MeV
  static const double DeltaNeutronToProtonMass = 1.29322; // MeV from PDG2018 mass differences
  static const double ElectronMass = 0.511; // MeV
  static const double sigma_0 = 0.0952e-42; // cm^2

  double ElectronEnergyO0(double Enu);
  double ElectronMomentum(double Eele);
  double CrossSection(double Enu);
} // close IBD namespace

#endif
