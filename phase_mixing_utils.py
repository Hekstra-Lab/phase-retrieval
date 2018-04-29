"""
phase_mixing_utils.py

Utilities for messing around with phase retrieval algorithms and
visualizing the phase problem
"""

from scipy.stats import multivariate_normal
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from cmocean.cm import phase as phase_cm
from skimage.feature import register_translation

def get_mag(FT_arr):
    """
    Returns the magnitudes of the intensities from a Fourier transform

    Parameters
    ----------
    FT_arr : np.ndarray
        An array of complex numbers

    Returns
    -------
    np.ndarray
        Array of magnitudes from Fourier transform
    """
    return np.abs(FT_arr)

def get_phase(FT_arr):
    """
    Returns the angles of the complex phases from a Fourier transform

    Parameters
    ----------
    FT_arr : np.ndarray
        An array of complex numbers

    Returns
    -------
    np.ndarray
        Angles of the complex phases from Fourier transform
    """
    return np.angle(FT_arr)

def mix_FT(mag_arr, phase_arr):
    """
    Performs the Fourier transforms and then mixes the provided phases
    and magnitudes.

    Parameters
    ----------
    mag_arr : np.ndarray
        Array to be fourier transformed for its magnitudes
    phase_arr : np.ndarray
        Array to be fourier transformed for its complex phases

    Returns
    -------
    np.ndarray
        New complex array
    """
    return mix_FT_arr(np.fft.fft2(mag_arr),np.fft.fft2(phase_arr))

def mix_FT_arr(mag_arr, phase_arr):
    """
    Mixes two Fourier transform arrays by taking the magnitude from one
    and the phase from the other. Arrays must be the same shape.

    Parameters
    ----------
    mag_arr : np.ndarray
        Array of magnitudes
    phase_arr : np.ndarray
        Array of complex phases

    Returns
    -------
    np.ndarray
        New complex array
    """
    mag = get_mag(mag_arr)
    phase = get_phase(phase_arr)

    return mag*np.exp(1j*phase)

def phase_intensity_plot(arr, ax=None, cb=True, min_alpha=0.0, cmap='ocean'):
    """
    Plot the provided Fourier transform representing intensity using
    alpha value and phase using colormap.

    Parameters
    ----------
    arr : np.ndarray
        An array of complex numbers. This should be fftshifted already.
    ax : matplotlib axis object
        If provided, plot is drawn to axis object
    cb : boolean
        Show colorbar corresponding to phases (Default is True).
    min_alpha : float
        Map the magnitude of intensities to alphas: [min_alpha,1]
        (Default is 0.0).
    cmap : matplotlib colormap
        Colormap for phases -- string will be used if provided.
        (Default is cmocean phase cm -- hsv is a good alternative).

    Returns
    -------
        Phase plot following Kevin Cowtan's Book of Fourier convention.
    """

    r = get_mag(arr)
    theta = get_phase(arr)
    theta[theta<0]+=2*np.pi #correct for neg values of angle so no weird jumps going around 180 deg
    theta /= 2*np.pi
    disp_arr = phase_cm(theta)

    # Scale alpha-values between 0 and 1
    disp_arr[:,:,-1] = r/np.max(r)

    # Move minimal alpha value to min_alpha
    disp_arr[:,:,-1] += (1-disp_arr[:,:,-1])*min_alpha

    if ax is None:
        fig, ax = plt.subplots(figsize=(10,10))
    elif ax is not None and cb:
        raise ValueError('if you pass an axis you can\'t have cb=True')
    if cmap == 'ocean':
        cmap = phase_cm
    # Relabel the colorbar without rescaling theta to be in [0,2pi]
    if cb:
        cax = ax.imshow(np.flipud(disp_arr), cmap=cmap, vmin=0, vmax=1)
        cbar = fig.colorbar(cax, ticks=np.linspace(0, 1, 5))
        labels = [ r"0", r"$\frac{\pi}{2}$", r"$\pi$",
                   r"$\frac{3\pi}{2}$", r"$2\pi$"]
        cbar.ax.set_yticklabels(labels, fontsize=20)
    else:
        cax = ax.imshow(np.flipud(disp_arr), cmap=cmap)

    if ax is None:
        plt.show()

def gaussian_lattice(n_atoms, uc_size, n_tiles, means):
    """
    Create a square lattice composed of unit cells each containg n_atoms
    gaussians of unit variance.

    Lattice dimensions are (uc_size*n_tiles)x(uc_size*n_tiles)

    Parameters
    ----------
    n_atoms : int
        Number of atom-centered Gaussians
    uc_size : float
        Dimensions of unit cell
    n_tiles : float or int
        Number of unit cells to tile to generate lattice
    means : np.ndarray
        Array of locations for the gaussians. Each value must be less
        than uc_size. Default variance is uc_size/10

    Returns
    -------
        Array containg the lattice
    """
    uc = np.zeros((uc_size,uc_size))
    X,Y = np.meshgrid(np.arange(uc_size),np.arange(uc_size))
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X; pos[:, :, 1] = Y
    for i in range(n_atoms):
        mvn = multivariate_normal(mean=means[i],cov = uc_size/10.)
        uc = uc + mvn.pdf(pos)
    return np.tile(uc,(n_tiles,n_tiles))


def rings():
    """
    Example Gaussian lattice with a unit cell containing a ring of 10
    atoms

    Returns
    -------
    np.ndarray
        array containing the density at each point in the lattice
    """
    th = np.linspace(0,2*np.pi,11)
    circ_mu = np.array([[50+20*np.cos(t),50+20*np.sin(t)] for t in th[:-1]])
    return gaussian_lattice(10,100,20, circ_mu)
def align_and_plot(pr, true_im):
    """
    Determines the proper rotation and translation for matching the reconstructed real space image to the true one

    inputs
    --------
    pr : a PhaseRetrieval object that has run some reconstruction
    true_im : ndarry of the original real space image
    """
    errs = np.zeros(4)
    for i in range(4):
        shift, errs[i], blarg = register_translation(true_im, np.rot90(pr.real_space_guess,k=i))
    n_rot = np.argmin(errs)
    shift, error, blargh= register_translation(true_im,np.rot90(pr.real_space_guess,k=n_rot))


    fixed = np.roll(np.rot90(pr.real_space_guess,k=n_rot),shift.astype(np.int),axis=(0,1))
    plt.imshow(fixed)
    plt.show()
    plt.title('orig')
    plt.imshow(true_im)
    plt.show()

    error = np.zeros(pr.rs_track.shape[0])
    for i,im in enumerate(pr.rs_track):
        shift, error[i], diffphase = register_translation(true_im, np.rot90(im,k=n_rot))

    plt.plot(error)
    plt.ylabel('error')
    plt.show()
