#ifndef DAYABAY_H
#define DAYABAY_H

#include <map>
#include <assert.h>
#include "HuberMullerFlux.h"
#include "InverseBetaDecayCrossSection.h"
//#include <nuSQuIDS/tools.h>
//#include "Experiment.hpp"
//#include "DataSet.hpp"
//#include "Model.hpp"

namespace dayabay {

class DayaBay {
  private:
    static constexpr int n_bins = 35;
    static const std::vector<std::string> isotopes_to_consider;
    static const std::map<std::string,double> mean_fission_fractions;
    static const std::map<std::string,double> EfficiencyOfHall;
    static const std::map<std::pair<std::string,std::string>,double> DistanceFromReactorToHallSquare;
    static const std::map<std::string,double> FudgeFactorPerHall;
    // neutrino information
    static const std::vector<std::vector<double>> NeutrinoCovarianceMatrix;
    static const std::vector<double> NeutrinoUpperBinEdges;
    static const std::vector<double> NeutrinoLowerBinEdges;
    // map between true electron energy and reconstructed energy
    static const std::vector<double> DataUpperBinEdges;
    static const std::vector<double> DataLowerBinEdges;
    static const std::vector<std::vector<double>> FromEtrueToErec;
    const double deltaEFine= 0.05; // MeV
    const double DeltaNeutronToProtonMass = 1.29322; // MeV from PDG2018 mass differences
    const double ElectronMass = 0.511; // MeV
    const double Na =  6.022140857e23; // avogadros number
    const double FH = 12.02; // hydrogen fraction in GdLS
    const double IH1 = 0.9998; // H1 isotope abundance
    const double HidrogenMass = 1.673723e-27; // hidrogen mass in kg
    const double TotalMass = 20.e3; // in kg
    const double TotalNumberOfProtons = TotalMass*FH*Na*IH1/HidrogenMass;
    // observations
    static const std::vector<std::string> sets_names;
    static const std::vector<std::string> reactor_names;
    static const std::map<std::string,std::vector<double>> ObservedData;
    static const std::map<std::string,std::vector<double>> PredictedBackground;
    // flags
    bool ignore_oscillations = false;
  public:
    DayaBay();
    ~DayaBay(){}

    // Cost calc_costs(const Model& m) const override;

    // Still not implemented
    //std::vector<std::vector<Experiment<DayaBay>::Point>> get_data() const override;
    //std::vector<std::vector<Experiment<DayaBay>::Point>> get_expectation(const Model &m) const override;

    static std::string name() { return std::string("DayaBay");}
    // size_t num_bins() const override { return n_bins; }
    static void initialise_options() { }
    void set_ignore_oscillations(bool i_o){ ignore_oscillations = i_o; }
    bool are_we_ignoring_oscillations() const { return ignore_oscillations; }

    // nusquids not implemented
    // nusquids::marray<double,2> get_inverse_flux_covariance() const;
    // nusquids::marray<double,2> get_pseudo_inverse_flux_covariance() const;
    // nusquids::marray<double,2> get_flux_covariance() const;

    static std::vector<double> get_lower_neutrino_bin_edges() {return NeutrinoLowerBinEdges;}
    static std::vector<double> get_upper_neutrino_bin_edges() {return NeutrinoUpperBinEdges;}
  public:
    double get_flux(double enu, std::string isotope_name) const {
      return reactor_isotope_flux::GetFlux(enu,reactor_isotope_flux::huber_muller::flux_parameters[isotope_name]);
    }
    double get_cross_section(double enu) const {
      return IBD::CrossSection(enu);
    }
    std::vector<std::string> get_isotope_names() const {
      return isotopes_to_consider;
    }
    std::vector<std::vector<double>> get_resolution_matrix() const {
      return FromEtrueToErec;
    }
    std::vector<double> get_data_upper_bin_edges() const {
      return DataUpperBinEdges;
    }
    std::vector<double> get_data_lower_bin_edges() const {
      return DataLowerBinEdges;
    }
    std::vector<double> get_true_neutrino_energy_bin_centers() const {
			std::vector<double> enu_true(FromEtrueToErec.front().size());
			for(unsigned int j = 0; j < FromEtrueToErec.front().size(); j++){
					enu_true[j] = (j+0.5)*deltaEFine;// bin center
			}
			return enu_true;
    }
    size_t FindFineBinIndex(double energy) const {
      auto dindex = floor(energy/deltaEFine - 0.5);
      if(dindex < 0)
        return 0;
      else return (size_t)(dindex);
    }
    double oscProbability(double enu, double L) const {
      double sin22th13 = 0.092;
      double dm2_31 = 2.494e-3;
      double x = 1.267*dm2_31 * L / enu;
      //std::cout << (1. - sin22th13*sin(x)*sin(x)) << std::endl; // This prints the proba
      return 1. - sin22th13*sin(x)*sin(x);
    }
    void use_simple_convolution(bool simple_convolution){ simple_convolution_ = simple_convolution; }
  private:
    bool simple_convolution_ = false;
  public:
    //double from_
    double calculate_naked_event_expectation(std::string set_name, unsigned int i) const {
      using namespace reactor_isotope_flux;
      //if(std::none_of(sets_names.begin(),sets_names.end(),[&set_name](std::string set){return set_name == set;}))
      //  throw std::runtime_error("Dayabay:: Cannot calculate naked rate. Invalid set.");
      if(i>=n_bins)
        throw std::runtime_error("Dayabay:: Cannot calculate naked rate. Bin number is invalid.");

      double expectation = 0.0;
      size_t min_reco_energy_fine_index = FindFineBinIndex(DataLowerBinEdges[i]);
      size_t max_reco_energy_fine_index = FindFineBinIndex(DataUpperBinEdges[i]);

      for(std::string reactor_name : reactor_names){
        for(size_t erf = min_reco_energy_fine_index; erf < max_reco_energy_fine_index; erf++){
          for(size_t etf = 0; etf < FromEtrueToErec.front().size(); etf++){
            double enu = (etf+0.5)*deltaEFine + (DeltaNeutronToProtonMass - ElectronMass);// bin center
            double L = sqrt(DistanceFromReactorToHallSquare.at({set_name,reactor_name}));
            //double oscprob = model.amp<Model::E_TO_E, Model::ANTIPARTICLE, Model::FREE>(L/enu);
            double oscprob = 1.;
            if(ignore_oscillations) oscprob = 1.;
            assert(oscprob > 0.0);
            double flux = 0.0;
            for(std::string isotope_name : isotopes_to_consider){
              flux += mean_fission_fractions.at(isotope_name)*GetFlux(enu,huber_muller::flux_parameters[isotope_name]);
            }
            expectation += flux*IBD::CrossSection(enu)*FromEtrueToErec[erf][etf]*deltaEFine*oscprob;
          }
        }
        expectation *= deltaEFine;
        expectation *= EfficiencyOfHall.at(set_name);
        expectation /= DistanceFromReactorToHallSquare.at({set_name,reactor_name});
      }
      return expectation*TotalNumberOfProtons;
    }
};

} // close dayabay namespace

#endif
