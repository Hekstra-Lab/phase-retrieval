# phase-retrieval
## A small library implementing phase retrieval algorithms for 2D images.

<div style="text-align:center"><img src="https://github.com/HekstraLab/phase-retrieval/blob/master/Images/disp_duck.png" width="350" height="350" /> <img src="https://github.com/HekstraLab/phase-retrieval/blob/master/Images/fourier_duck.png" width="350" height="350"/></div>

The code in the repository was created for a project in AM216 at Harvard University.

### PhaseRetrieval

The main piece of this repository is the `PhaseRetrieval` object. Once initialized with a set of Fourier Magnitudes. It can carry out several [Fineup algorithms](http://www2.optics.rochester.edu/workgroups/fienup/PUBLICATIONS/JRF_PR-Tour_AO2013.pdf) (Error reduction, HIO, CHIO) as well as [prRED](https://arxiv.org/pdf/1803.00212.pdf). The object also contains a few methods to asses the success or failure of a given phase retrieval algorithm including tracking of the Fourier space error, real space error, and successive real space guesses. 


### phase_mixing_utils

This file implements a few handy methods for experimenting with phase mixing, i.e. using the magnitudes from one image and the phases from another, as in the illustration at the top. Methods should be self explanatory.
