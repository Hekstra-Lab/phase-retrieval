"""
phase_retrieval.py

Useful functions and algorithms related to phase-retrieval problems.
"""

def fourier_MSE(guess, reference): 
    """
    MSE in Fourier domain

    Parameters
    ----------
    guess : np.ndarray
        Fourier transform of current real-space estimate
    reference : np.ndarray
        Reference magnitudes in Fourier space

    Returns
    -------
    float
        Mean-squared error in the Fourier domain
    """
    abs_err = np.sum((np.abs(guess) - np.abs(reference))**2)
    norm = np.sum(np.abs(reference)**2)
    return abs_err/norm

def inputoutput(mags, guess=None, n_iters=1000, beta=0.3, freq=0.5):
    """
    Implementation of the input-output phase retrieval algorithm from 
    Fienup JR, Optics Letters (1978). 

    Parameters
    ----------
    mags : np.ndarray
        Array of magnitudes in Fourier space (modulus of Fourier transform)
    n_iters : int
        Number of iterations to run algorithm
    beta : float
        Scaling factor for updates to negative real-space components in iteration 
        of input-output algorithm
    freq : float
        Switching frequency between input-output updates to real-space density and
        error-reduction updates. If 1.0, input-output updates are always used. If
        0.0, negative real-space values are zeroed out at every iteration. (Default 
        value is 0.5)
        
    Returns
    -------
    (rs_track, error_track)
        The first array contains the estimate of the real-space density at each iteration.
        The second array contains the Fourier domain mean-squared error at each iteration.
    """

    # Initialize with guess if provided. Otherwise, generate random starting guess.
    shape = list(mags.shape)
    if guess is None:
        rs_const = np.random.random_sample(shape)
    else:
        assert guess.shape == mags.shape
        rs_const = guess
        
    rs_track  = np.zeros([n_iters+1] + shape)
    rs_track[0]= rs_const
    err_track = np.zeros(n_iters)
    
    for i in range(n_iters):
        
        ft = np.fft.fft2(rs_const)
        err_track[i] = fourier_MSE(ft, mags)
        
        # Mix known magnitudes and guessed phases
        ks_est = mags*np.exp(1j*phase_mixing_utils.get_phase(ft))
        
        # Inverse fourier transfrom your phase guess with the given magnitudes
        rs_est = np.fft.ifft2(ks_est)
        
        # Impose desired real-space density constraint
        gamma  = np.real(rs_est) > 0 # Mask of negative density
        if np.random.rand() < 0.5:
            # Input-Output update
            rs_const = np.abs(rs_est*gamma - (rs_est*(~gamma)*beta)) 
        else:
            # Error-Reduction update
            rs_const = np.abs(rs_est*gamma)
        rs_track[i+1] = rs_const
    
    return rs_track, err_track
