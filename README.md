# phase-retrieval
## A small library implementing phase retrieval algorithms for 2D images.

<div style="text-align:center"><img src="https://github.com/HekstraLab/phase-retrieval/blob/master/Images/disp_duck.png" width="350" height="350" /> <img src="https://github.com/HekstraLab/phase-retrieval/blob/master/Images/fourier_duck.png" width="350" height="350"/></div>

The code in the repository was created for a project in AM216 at Harvard University.

### PhaseRetrieval

The main piece of this repository is the `PhaseRetrieval` object. Once initialized with a set of Fourier Magnitudes. It can carry out several [Fienup algorithms](http://www2.optics.rochester.edu/workgroups/fienup/PUBLICATIONS/JRF_PR-Tour_AO2013.pdf) (Error reduction, HIO, CHIO) as well as [prRED](https://arxiv.org/pdf/1803.00212.pdf). The object also contains a few methods to asses the success or failure of a given phase retrieval algorithm including tracking of the Fourier space error, real space error, and successive real space guesses. 

#### Dependencies 

The notebooks in `Notebooks` are a hodgepodge of python 2.7 and 3.6. The few in the main directory use python 3.6.

```
numpy
scipy
matplotlib
skimage
keras (if using prDeep)
tensorflow 
fasta-python #https://github.com/phasepack/fasta-python 
```


#### Basic Usage

```python
im = np.load("some_file.npy") #Load an image
mags = np.abs(np.fft.fftn(im)) #Compute FFT and take only the magnitudes

pr = PhaseRetrieval(mags)
pr.CHIO(n_iter=500) #Run Fienup's CHIO algorithm for 500 iterations with default settings

plt.imshow(pr.real_space_guess)
plt.show() #View final result of CHIO

plt.plot(pr.err_track)
plt.show() #See progression of Fourier space error 
```

### DnCNN

Our implementation of [DnCNN](https://arxiv.org/pdf/1608.03981.pdf) is based on [this repository](https://github.com/husqin/DnCNN-keras). You can load our fully trained DnCNN into keras with

```python
from keras.models import load_model

denoiser = load_model("dncnn_50k.h5")
```


You may find it convenient to define a helper function to handle image reshaping:

```python

def denoise(x):
    shape = x.shape
    tensor_x = x.reshape(shape+(1,))
    return denoiser.predict(tensor_x).reshape(shape)
```

### phase_mixing_utils

This file implements a few handy methods for experimenting with phase mixing, i.e. using the magnitudes from one image and the phases from another. The most interesting method here might be `phase_intensity_plot` which generates images with pixel brightness determined by the Fourier Magnitued and color determined by the complex phase. An example of this is show at the top of this readme. Other functions mostly thinly wrap numpy and should be self explanatory.


### TO DO

- Post clear example notebooks
- Post code for creating and training DnCNN (and relevant datasets)
- Make `replicate_cat_duck.ipynb` actually replicate Cowtan figures
