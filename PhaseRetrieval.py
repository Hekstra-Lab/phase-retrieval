import numpy as np
import phase_mixing_utils

class PhaseRetrieval():
    def __init__(self, fourier_mags, real_space_guess = None):
        self.measured_mags = fourier_mags
        self.shape = self.measured_mags.shape
        if real_space_guess is not None:
            self.real_space_guess = real_space_guess
        else:
            self.real_space_guess = np.random.random_sample(self.shape)

    # Methods
    def fourier_MSE(self, guess):
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
            Mean-squared error in the Fourier domain of guess against self.measured_mags
        """
        abs_err = np.sum((np.abs(guess) - np.abs(self.measured_mags))**2)
        norm = np.sum(np.abs(self.measured_mags)**2)
        return abs_err/norm

    def _step(self, density_mod_func,curr_iter):
        """
        One iteration of the hybrid input output (HIO) algorithm with given beta value

        Parameters
        ----------

        denisty_mod_func : callable
            Function to update pixel values.

        Returns
        -------
        fourier_err : float
            Mean squared error in fourier domain - see fourier_MSE above

        rs_non_density_modified : ndarray
            Updated real space guess without any density modificaiton applied

        new_real_space : nd_array

        """

        ft = np.fft.fftn(self.real_space_guess)
        fourier_err = self.fourier_MSE(ft)

        # Mix known magnitudes and guessed phases
        ks_est = self.measured_mags*np.exp(1j*phase_mixing_utils.get_phase(ft))

        # Inverse fourier transfrom your phase guess with the given magnitudes
        rs_non_density_modified = np.real(np.fft.ifftn(ks_est))

        # Impose desired real-space density constraint
        #gamma  = np.real(rs_non_density_modified) > 0 # Mask of positive density
        #new_real_space = rs_non_density_modified*gamma - (rs_non_density_modified*(~gamma)*beta)
        new_real_space = density_mod_func(rs_non_density_modified, self.real_space_guess, curr_iter)
        self.real_space_guess = new_real_space
        return fourier_err, rs_non_density_modified, new_real_space


    def _initialize_tracking(self, n_iter):
        """
        Set up tracking arrays for an iterative algorithm.

        Parameters
        ----------

        n_iter : int
            Number of density modification steps to take.
        """
        self.ndm_track = np.zeros((n_iter,)+self.shape)
        self.rs_track = np.zeros((n_iter+1,)+self.shape)
        self.err_track = np.zeros(n_iter)
        self.rs_track[0] = self.real_space_guess


    def _iterate(self, density_mod_func, n_iter):
        self._initialize_tracking(n_iter)
        for i in range(n_iter):
            self.err_track[i], self.ndm_track[i], self.rs_track[i+1] = self._step(density_mod_func, i)


    def _input_output_update(self, density,old_density, beta, curr_iter):
            gamma  = density > 0 # Mask of positive density
            return gamma*density - (~gamma)*(old_density-density*beta)


    def hybrid_input_output(self, beta, n_iter = None, freq = 0.5):
        """
        Implementation of the hybrid input-output phase retrieval algorithm from
        Fienup JR, Optics Letters (1978).

        Parameters
        ----------
        beta : np.ndarray of floats or single float
            Scaling factor for updates to negative real-space components in iteration
                of hybrid input-output algorithm. If an ndarray is passed then the beta values
                will be iterated over. A single float value will create beta = np.ones(n_iter)*beta
                with with individual elements set to one at a rate corresponding to the freq value.
        n_iters : int
            Number of iterations to run algorithm
        freq : float
            Switching frequency between input-output updates to real-space density and
            error-reduction updates. If 1.0, input-output updates are always used. If
            0.0, negative real-space values are zeroed out at every iteration. (Default
            value is 0.5)
        """
        if np.isscalar(beta):
            if n_iter is None:
                raise ValueError("With scalar beta n_iter must be an integer")
            elif np.int(n_iter) == n_iter:
                    beta = np.ones(n_iter)*beta
                    beta[np.random.rand(n_iter) > freq] = 1
            else:
                raise ValueError("With scalar beta n_iter must be an integer")
        n_iter = beta.shape[0]

        def f(density, old_density, beta,  curr_iter):
            beta_val = beta[curr_iter]
            return self._input_output_update(density, old_density, beta_val, curr_iter)

        density_mod_func = lambda density, old_density, curr_iter: f(density, old_density, beta, curr_iter= curr_iter)

        self._iterate(density_mod_func, n_iter)


    def input_output(self, beta=0.7, n_iter = None):
        """
        Implementation of the input-output phase retrieval algorithm from
        Fienup JR, Optics Letters (1978).

        Parameters
        ----------
        beta : np.ndarray of floats or single float
            Scaling factor for updates to negative real-space components in iteration
                of hybrid input-output algorithm. If an ndarray is passed then the beta values
                will be iterated over. A single float value will create beta = np.ones(n_iter)*beta

        n_iters : int
            Number of iterations to run algorithm. Not used if beta is an array.
        """
        self.hybrid_input_output(beta,n_iter, freq=1)

    def error_reduction(self, n_iter):
        """
        Implementation of the input-output phase retrieval algorithm from
        Fienup JR, Optics Letters (1978).

        This corresponds to the input-output methods with all beta values equal to one.
        Parameters.
        ----------
        n_iters : int
            Number of iterations to run algorithm.
        """
        density_mod_func = lambda density, curr_iter : density*(density>0)
        self._iterate(density_mod_func,n_iter)
