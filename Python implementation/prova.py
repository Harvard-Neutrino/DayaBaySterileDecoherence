import InverseBetaDecayCrossSection as IBD
import HuberMullerFlux as HMF

huber_muller =  {'U235': [4.367, -4.577, 2.100, -5.294e-1, 6.185e-2, -2.777e-3],
                 'U238': [4.833e-1, 1.927e-1, -1.283e-1, -6.762e-3, 2.233e-3, -1.536e-4],
                 'PU239': [4.757, -5.392, 2.563, -6.596e-1, 7.820e-2, -3.536e-3],
                 'PU241': [2.990, -2.882, 1.278, -3.343e-1, 3.905e-2, -1.754e-3]}

muller =  {'U235': [3.217, -3.111, 1.395, -3.690e-1, 4.445e-2, -2.053e-3],
           'U238': [4.833e-1,  1.927e-1, -1.283e-1, -6.762e-3, 2.233e-3, -1.536e-4],
           'PU239': [6.413, -7.432, 3.535, -8.820e-1, 1.025e-1, -4.550e-3],
           'PU241': [3.251, -3.204, 1.428, -3.675e-1, 4.254e-2, -1.896e-3]}

flux_test = HMF.reactor_isotope_flux('U235',huber_muller)
print(flux_test.isotope_name)
print(flux_test.flux_parameters[flux_test.isotope_name])

#print(HMF.reactor_isotope_flux('U235',huber_muller).isotope_name)
#print(HMF.reactor_isotope_flux('U238',muller).flux_parameters)

print(flux_test.GetFlux(0.1))
