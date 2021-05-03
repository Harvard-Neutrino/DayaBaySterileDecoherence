#ifndef HUBERMULLERFLUX_H
#define HUBERMULLERFLUX_H

#include <vector>
#include <string>
#include <map>
#include <math.h>

namespace reactor_isotope_flux {
  // comentari random escrit aquí
  namespace huber_muller {
    extern std::map< std::string, std::vector<double> > flux_parameters;
  } // close huber_muller namespace

  namespace muller {
    extern std::map< std::string, std::vector<double> > flux_parameters;
  } // close muller namespace

  double GetFlux(double Enu, const std::vector<double> & flux_parameters);

  class IsotopeFlux {
    private:
      const std::string isotope_name;
      const std::vector<double> flux_parameters;
    public:
      IsotopeFlux(std::string isotope_name, std::vector<double> flux_parameters):isotope_name(isotope_name), flux_parameters(flux_parameters){}
      double GetFlux(double Enu) const {
        double exponent = 0.0;
        for(unsigned int i=0; i< flux_parameters.size(); i++)
          exponent += flux_parameters[i]*pow(Enu,i);
        return exp(exponent);
      }
  };

} // close reactor_isotope_flux namespace

#endif
