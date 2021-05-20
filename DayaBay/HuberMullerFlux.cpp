#include "HuberMullerFlux.h"
#include <iostream>

namespace reactor_isotope_flux {

namespace huber_muller {
  // prescriptiions for all isotopes but U238 from Huber 1106.0687
  // the missing isotope is obtained from Muller et al 1101.2663

  std::map< std::string, std::vector<double> > flux_parameters =
  {
    {"U235" , {4.367, -4.577, 2.100, -5.294e-1, 6.185e-2, -2.777e-3}},
    {"U238" , {4.833e-1,  1.927e-1, -1.283e-1, -6.762e-3, 2.233e-3, -1.536e-4}},
    {"PU239", {4.757, -5.392, 2.563, -6.596e-1, 7.820e-2, -3.536e-3}},
    {"PU241", {2.990, -2.882, 1.278, -3.343e-1, 3.905e-2, -1.754e-3}}
  };
} // close huber_muller namespace

namespace muller {
  // obtained from Muller et al 1101.2663 table VI
  std::map< std::string, std::vector<double> > flux_parameters =
  {
    {"U235" , {3.217, -3.111, 1.395, -3.690e-1, 4.445e-2, -2.053e-3}},
    {"U238" , {4.833e-1,  1.927e-1, -1.283e-1, -6.762e-3, 2.233e-3, -1.536e-4}},
    {"PU239", {6.413, -7.432, 3.535, -8.820e-1, 1.025e-1, -4.550e-3}},
    {"PU241", {3.251, -3.204, 1.428, -3.675e-1, 4.254e-2, -1.896e-3}}
  };
} // close muller namespace

double GetFlux(double Enu, const std::vector<double> & flux_parameters){
  double exponent = 0.0;
  for(unsigned int i=0; i< flux_parameters.size(); i++)
    exponent += flux_parameters[i]*pow(Enu,i);
  return exp(exponent);
}

} // close reactorflux namespace


// int main(){
//
//   std::cout << reactor_isotope_flux::GetFlux(1,reactor_isotope_flux::huber_muller::flux_parameters["U235"]);
//   std::cout << "\n";
//   std::cout<<"Hola Món\n";
//
//   return 0;
// }
