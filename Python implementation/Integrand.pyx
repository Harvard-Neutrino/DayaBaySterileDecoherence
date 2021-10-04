cpdef float integrand(float enu,self,float L,model,int erf,int etf):
    flux = 0.0
    for isotope in self.isotopes_to_consider:
            flux += self.get_flux(enu,isotope)*self.mean_fission_fractions[isotope]

    return (flux*
            self.get_cross_section(enu) *
            self.FromEtrueToErec[erf][etf] *
            model.oscProbability(enu,L))

# De moment, cythonize només oscProbability? Dunno man.
