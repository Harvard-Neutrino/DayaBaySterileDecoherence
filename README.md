# DayaBaySterileDecoherence

This is a very simple Python program designed to analyse different low-energy sterile neutrino experiments, and which is connected to our paper [here](https://arxiv.org/abs/2201.05108). Namely, the program is capable of analysing the [DayaBay](https://arxiv.org/abs/1610.04802v1), [NEOS](https://arxiv.org/abs/1610.05134v4), [PROSPECT](https://arxiv.org/abs/2006.11210v2) and [BEST](https://arxiv.org/abs/2109.11482v1) data. The analyses focus on studying sterile neutrino oscillations, addressing possible differences between the plane wave approximation and the more complete wave package formalism. However, this program can be extended to arbitrary oscillation formulas. Program written by Toni Bertólez-Martínez. For any doubt, please open an issue or send an email to antoni.bertolez@fqa.ub.edu.

In this ReadMe we briefly introduce the program and the different analyses and plots one can do with it.

## Before starting
In order for most scripts to work, one needs to compile the following file:

```
user@user: ~/DayaBaySterileDecoherence$ cd Common_cython
user@user: ~/DayaBaySterileDecoherence/Common_cython$ sh compile.sh
```

This will prepare the necessary libraries, which are written in Cython. You require `python3-dev` in order to run these files. This same command must be done each time a modification is done to the file `Models.py` or `HuberMullerFlux.py`.

## General structure
The main folder of the program contains 5+2 directories. Namely:
 - BEST, DayaBay, GlobalFit, NEOS, PROSPECT: the analysis of the respective experiments.
 - Common_cython: contains the libraries which are used for every experiment, written in Cython for a better performance.
 - Misc: some miscellaneous files to plot some interesting graphics, namely those of our paper.
All the Python files `.py` can be run with the `python3` command on your terminal.

__Note:__ All the programs here have been tested running the command in the same directory where the `.py` file is. That is, if one wants to run `NEOS/FitPlots.py`, the only procedure that has been tested is

```
user@user: ~/DayaBaySterileDecoherence$ cd NEOS
user@user: ~/DayaBaySterileDecoherence/NEOS$ python3 FitPlots.py
```

And not
```
user@user: ~/DayaBaySterileDecoherence$ python3 NEOS/FitPlots.py
```
This last command may (or may not) rise some errors. Sorry for the inconvenience.


## Experiment analysis directories
This section describes the content inside BEST, DayaBay, GlobalFit, NEOS and PROSPECT directories. For clarity, we will use NEOS as the example.

`NEOS.py` is the main file. It defines a class with all the methods necessary to compute event expectations and compare them with the data using a test statistic. Usually this program is never compiled, but accessed through the rest of the programs.

`NEOSData.py` and `NEOSParameters.py` are auxiliary files to `NEOS.py` which read all the data of the experiment (found in the subdirectory `/Data`) and  holds the parameters of the analysis. This allows for a better tuning and easier variation of the numbers.

`EventExpectationPlots.py` is, as the same name says, a program to compute event expectation and plot them different ways: measured spectrum, ratio to the standard oscillations, the value of the test statistic per bin... The figures are saved in the subdirectory `/Figures`

`FitClass.py` defines a class to ease the task of computing the test statistic for different values of the mass and the mixing. This class is then called by `PWSterileFitTable.py` and `WPSterileFitTable.py`, which write in a file the value of this statistic in the subdirectory `/PlotData`.

`FitPlots.py` draws the exclusion contours using the data written by these files, and found in `/PlotData/PWSterileChi2.dat` or `/PlotData/WPSterileChi2.dat`.

## Common
The `Common_cython` directory includes diferent programs which are required by all other programs and analysis.

`HuberMullerFlux.pyx` defines a class which returns the Huber-Muller flux for some given nuclear isotopes.

`InverseBetaDecayCrossSection.py` defines different functions to be compute the IBD cross-sections, as its own name states.

`Models.pyx` defines different classes of models which define different oscillation probabilities. For example, there is a class for the standard oscillations (with parameters from nu-fit.org), or some classe for oscillations with a sterile neutrino. The only requirement for this classes is that they have a method `oscProbability` and `oscProbability_av` which return the full and averaged oscillation probabilities at distance L and energy E, respectively. Feel free to write your own class!

## Miscellaneous
Finally, the `Misc` folder contains different and diverse files.

`Plot Neutrino Experiments.ipynb` reproduces figure 1 in our paper, e.g. plots the position in (L,E) of relevant neutrino oscillation experiments, and the relevant scales.

`SuperFitPlots.py` reproduces figure 4 of our paper, e.g. computes the total chi2 of all nuclear reactor experiments and of BEST, and plots the exclusion contours and preferred regions. These are saved in the subdirectory `/AllFitFigures`.

`Probability.py` is simply used to draw toy plots of the oscillation probabilities defined in `Models.py`.
