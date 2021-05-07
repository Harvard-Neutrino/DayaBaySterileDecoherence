import numpy as np

# To understand the information on the bins, we must understand the physical process.
# An electron antineutrino arrives to the detector with an energy which can go between
# 0 and infty. However, only if the antineutrino has an energy > 1.8 MeV, it can
# produce IBD. In such case, a positron will be produced (with kinetic and rest energy).

# This positron will have total energy between 511 keV and infty. It will annihilate with
# an electron at rest. This will produce a flash of light, whose energy can go between
# 1.022 MeV and infty (due to the rest mass energy of the positron+electron).

# The energy of this light is called prompt energy, and is the one from the data bins.
# We can relate the prompt energy with the incoming antineutrino energy through
# Eprompt = Erealnu - 0.78 (MeV).
# For more information, the process is described in 1610.04802.

# -------------------------------------------------------------
# PS: The program begins to take prompt energies from ~0.78 MeV.
#     This is a bit stupid, since the first possible antineutrino energy should be
#     ~1.806 MeV and the first non-null prompt energy will be 1.022 MeV.
#     However, it will bring up no error, since this is taken into account
#     in the IBD cross-section, which is set to be 0 if the antineutrino does not
#     have enough energy.

# -------------------------------------------------------------
#   HISTOGRAM BINS
# -------------------------------------------------------------

# Histogram bins of the measured data.
number_of_bins = 35
datlowerbin = np.array([0.7,1.3,1.5,1.7,1.9,2.1,2.3,2.5,2.7,2.9,3.1,3.3,3.5,3.7,3.9,4.1,4.3,4.5,
               4.7,4.9,5.1,5.3,5.5,5.7,5.9,6.1,6.3,6.5,6.7,6.9,7.1,7.3,7.5,7.7,7.9])
datupperbin = np.array([1.3,1.5,1.7,1.9,2.1,2.3,2.5,2.7,2.9,3.1,3.3,3.5,3.7,3.9,4.1,4.3,4.5,4.7,
               4.9,5.1,5.3,5.5,5.7,5.9,6.1,6.3,6.5,6.7,6.9,7.1,7.3,7.5,7.7,7.9,12.0])

# -------------------------------------------------------------
#   DAYABAY DATA
# -------------------------------------------------------------

observed_data = {'EH1':  np.array([43781,31619,40024,47052,53432,58628,63823,66905,67597,67761,66409,
                         63852,60164,55403,51728,47916,44952,41353,37622,33677,29657,25660,
                         21537,18159,15236,12603,9792,7447,5698,4250,3031,2154,1308,862,2873]),
                 'EH2':  np.array([36051,26727,33786,40038,45745,50634,54820,57405,58109,58185,57631,
                         55398,51881,47954,44177,41219,38597,35364,32690,28834,25503,22057,
                         18786,15573,13162,10820,8558,6343,4836,3685,2710,1856,1178,686,2211]),
                 'EH3':  np.array([11952,8506,10396,12243,13696,15079,16168,16776,17119,17064,16887,
                         16134,15186,14047,13058,12261,11388,10542,9596,8881,7638,6667,5678,
                         4759,3847,3139,2516,1906,1480,1099,764,544,331,223,580])}

predicted_bkg = {'EH1':  np.array([8337.52,1890.66,1667.57,1065.15,690.76,503.43,517.92,649.8,774.96,480.69,239.63,
                         181.37,166.38,160.2,160.06,156.93,156.99,159.3,158.73,158.07,157.58,157.47,155.88,
                         157.69,153.5,152.4,149.8,147.74,143.61,136,131.72,126.95,121.55,114.85,1190.93]),
                 'EH2':  np.array([5749.96,1359.51,1204.36,753.72,485.96,350.64,361.83,459.7,559.01,338.95,162.74,
                         122.39,111.74,107.31,107.11,104.97,106.5,104.95,106.11,105.66,105.33,105.31,104.36,
                         105.67,103.01,102.5,100.97,99.61,96.6,91.23,88.16,84.89,81.24,76.78,800.55]),
                 'EH3':  np.array([2637.85,627.96,551.27,344.36,218.3,150.51,150.69,189.16,226.35,130.97,52.16,
                         32.43,27.08,24.73,24.12,23.38,23.5,23.6,24.38,24.07,23.73,24.15,25.01,26.42,
                         27.07,29.38,31.19,30.95,27.49,22.96,20.1,18.33,16.85,15.83,162.4])}
