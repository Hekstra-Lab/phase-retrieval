import numpy as np
import phase_mixing_utils
from skimage.feature import register_translation
import matplotlib.pyplot as plt
import sys
from fasta import fasta, plots, Convergence
from fasta.linalg import LinearMap
from skimage.restoration import denoise_tv_chambolle

class PhaseRetrieval():
    """
    Class for reconstructing real-space images from Fourier magnitudes
    """

    def __init__(self, fourier_mags, real_space_guess=None):
        self.measured_mags = fourier_mags
        self.shape = self.measured_mags.shape
        if real_space_guess is not None:
            self.real_space_guess = real_space_guess
        else:
            self.real_space_guess = np.random.random_sample(self.shape)
        self.tracking =False
        self.moves = []
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
    def align(self,unaligned, ref, return_meta=False):
        """
        Aligns the array "unaligned" to the array "ref"

        inputs
        -------
        unaligned : ndarray to be aligned to ref
        ref : ndarray to align to
        return_meta : boolean, return the number of rotations or not
        output
        -------
        aligned : ndarray with the best alignment
        n_rot(optional) : interger of optimal number of np.rot90 to apply
        """
        errs = np.zeros(4)
        for i in range(4):
            shift, errs[i], blarg = register_translation(ref, np.rot90(unaligned,k=i))
        n_rot = np.argmin(errs)
        shift, error, blargh= register_translation(ref,np.rot90(unaligned,k=n_rot))
        aligned = np.roll(np.rot90(unaligned,k=n_rot),shift.astype(np.int),axis=(0,1))
        if return_meta:
            return aligned, n_rot
        else:
            return aligned
    def calc_real_space_error(self, true_im, plot=True):
        """
        Determines the proper rotation and translation for matching the reconstructed real space image to the true one

        inputs
        --------
        true_im : ndarry of the original real space image
        plot : boolean
            whether to plot the shifted image an and the final error plot.
        """


        fixed, n_rot = self.align(self.real_space_guess,true_im,True)
        if plot:
            plt.imshow(fixed)
            plt.show()
            plt.title('orig')
            plt.imshow(true_im)
            plt.show()

        self.real_space_err_track = np.zeros(self.rs_track.shape[0])
        for i,im in enumerate(self.rs_track):
            shift, self.real_space_err_track[i], diffphase = register_translation(true_im, np.rot90(im,k=n_rot))
        if plot:
            plt.plot(self.real_space_err_track)
            plt.ylabel('error')
            plt.show()

    def _step(self, density_mod_func, curr_iter, **kwargs):
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
        # gamma  = np.real(rs_non_density_modified) > 0 # Mask of positive density
        # new_real_space = rs_non_density_modified*gamma - (rs_non_density_Modified*(~gamma)*beta)
        new_real_space = density_mod_func(rs_non_density_modified, self.real_space_guess, curr_iter, **kwargs)
        self.real_space_guess = new_real_space.copy()
        return fourier_err, rs_non_density_modified, new_real_space.copy()

    def _initialize_tracking(self, n_iter):
        """
        Set up tracking arrays for an iterative algorithm.

        Parameters
        ----------
        n_iter : int
            Number of density modification steps to take.
        """
        if not self.tracking:
            self.tracking = True
            self.ndm_track = np.zeros((n_iter,)+self.shape)
            self.rs_track = np.zeros((n_iter+1,)+self.shape)
            self.err_track = np.zeros(n_iter)
            self.rs_track[0] = self.real_space_guess
            self.step_name = []
            self.total_steps = 0
        else:
            extend_2d = np.zeros([n_iter,*self.rs_track.shape[1:]])
            extend_1d = np.zeros(n_iter)
            self.ndm_track = np.concatenate((self.ndm_track,extend_2d),axis=0)
            self.rs_track = np.concatenate((self.rs_track,extend_2d),axis=0)
            self.err_track = np.concatenate((self.err_track,extend_1d),axis=0)
        return

    def iterate(self, density_mod_func, n_iter,prog_bar =False, move_name=None, **kwargs):
        """
        Run iterations of phase retrieval algorithm specified by the
        density modification function
        """
        self._initialize_tracking(n_iter)

        for i in range(n_iter):
            if prog_bar:
                sys.stdout.write('\r')
                eq = int(np.ceil(np.true_divide(i*100,n_iter*5)))
                sys.stdout.write("[{:20s}] {}/{} steps  ".format('='*eq, i+1,n_iter))
                sys.stdout.flush()
            self.err_track[self.total_steps], self.ndm_track[self.total_steps], self.rs_track[self.total_steps+1] = self._step(density_mod_func, i, **kwargs)
            self.total_steps+=1

        return

    def _ERupdate(self, density, old_density, curr_iter):
        return density*(density > 0)

    def _IOupdate(self, density, old_density, curr_iter, beta):
        gamma = density > 0
        return density*gamma - (~gamma*(old_density - (beta*density)))

    def _HIOupdate(self, density, old_density, curr_iter, beta, freq):
        # Input-Output
        if np.random.rand() < freq:
            return self._IOupdate(density, old_density, curr_iter, beta)
        # Error Reduction
        else:
            return self._ERupdate(density, old_density, curr_iter)

    def _CHIOupdate(self, density, old_density, curr_iter, alpha, beta, freq):
        gamma = density>alpha*old_density
        delta = (0<density)*(~gamma)
        negatives = ~(gamma+delta)
        # CHIO
        if np.random.rand() < freq:
            return density*gamma + delta*(old_density-((1-alpha)/alpha)*density) + (old_density - beta*density)*negatives
        # Error Reduction
        else:
            return self._ERupdate(density, old_density, curr_iter)

    def _BoundedCHIOupdate(self, density, old_density, curr_iter, alpha, beta, freq):
        gamma = density>alpha*old_density
        delta = (0<density)*(~gamma)
        negatives = ~(gamma+delta)
        # Bounded CHIO
        if np.random.rand() < freq:
            chio = density*gamma + delta*(old_density-((1-alpha)/alpha)*density) + (old_density - beta*density)*negatives
            return  chio*(np.abs(chio)<1) + (np.abs(chio)>1)
        # Error Reduction
        else:
            return self._ERupdate(density, old_density, curr_iter)

    def ErrorReduction(self, n_iter=None,**kwargs):
        """
        Implementation of the error reduction phase retrieval algorithm
        from Fienup JR, Optics Letters (1978).

        Parameters
        ----------
        n_iters : int
            Number of iterations to run algorithm
        """
        if n_iter is None:
            raise ValueError("Number of iterations must be specified")

        # Run error reduction for n_iter iterations
        self.iterate(self._ERupdate, n_iter,**kwargs)
        return

    def InputOutput(self, beta=0.7, n_iter=None,**kwargs):
        """
        Implementation of the input-output phase retrieval algorithm
        from Fienup JR, Optics Letters (1978).

        Parameters
        ----------
        n_iters : int
            Number of iterations to run algorithm
        beta : float
            Scaling coefficient for modifying negative real-space
            density
        """
        if n_iter is None:
            raise ValueError("Number of iterations must be specified")

        # Run input-output for n_iter iterations
        self.iterate(self._IOupdate, n_iter, beta=beta,**kwargs)
        return

    def HIO(self, beta=0.7, freq=0.95, n_iter=None,**kwargs):
        """
        Implementation of the hybrid input-output phase retrieval
        algorithm from Fienup JR, Optics Letters (1978).

        Parameters
        ----------
        beta : float
            Scaling coefficient for modifying negative real-space
            density
        freq : float
            Frequency with which to use input-output updates
        n_iters : int
            Number of iterations to run algorithm
        """
        if n_iter is None:
            raise ValueError("Number of iterations must be specified")

        # Run HIO for n_iter iterations
        self.iterate(self._HIOupdate, n_iter, beta=beta, freq=freq,**kwargs)
        return

    def CHIO(self, alpha=0.4, beta=0.7, freq=0.95, n_iter=None,**kwargs):
        """
        Implementation of the continuous hybrid input-output phase
        retrieval algorithm

        Parameters
        ----------
        alpha : float
            Scaling coefficient for modifying small real-space density
        beta : float
            Scaling coefficient for modifying negative real-space
            density
        freq : float
            Frequency with which to use input-output updates
        n_iters : int
            Number of iterations to run algorithm
        """
        if n_iter is None:
            raise ValueError("Number of iterations must be specified")

        # Run CHIO for n_iter iterations
        self.iterate(self._CHIOupdate, n_iter, alpha=alpha, beta=beta,
                     freq=freq,**kwargs)
        return

    def BoundedCHIO(self, alpha=0.4, beta=0.7, freq=0.95, n_iter=None,**kwargs):
        """
        Implementation of the continuous hybrid input-output phase
        retrieval algorithm

        Parameters
        ----------
        alpha : float
            Scaling coefficient for modifying small real-space density
        beta : float
            Scaling coefficient for modifying negative real-space
            density
        freq : float
            Frequency with which to use input-output updates
        n_iters : int
            Number of iterations to run algorithm
        """
        if n_iter is None:
            raise ValueError("Number of iterations must be specified")

        # Run Bounded CHIO for n_iter iterations
        self.iterate(self._BoundedCHIOupdate, n_iter, alpha=alpha,
                     beta=beta, freq=freq,**kwargs)
        return
    #======== RED stuff ====

    def _f(self, Z):
        return .5 * np.linalg.norm((np.abs(Z) - self.measured_mags),ord='fro')**2
    def _sub_grad_f(self, Z):
        """
        Subgradient function as in prDeep paper.
        params
        -----
        Z :
        """
        out = Z-self.measured_mags*Z/abs(Z)
        return out
    def _g(self,Z):
        """
        This function is only used if evaluate_object = True, it is not necessary for the optimization
        """
        x = Z.ravel()
        return .5*self.proximal_lambda*x.T @ (x-self.density_modifier(x))


    def _proxg(self, Z, t):
        """
        compute the proximal of g for RED.

        """
        x = np.real(Z)
        for i in range(1,self.proximal_iters+1):
            x = (1/(1+t*self.proximal_lambda))*(Z+t*self.proximal_lambda*self.density_modifier(x))
        return x

    def prRED(self,density_modifier = None, max_iter = 100,accelerate=True, evaluate_objective=False,
                            verbose=True, proximal_iters=1,prox_lambda=.01,record_iterates=True):
        if not density_modifier:
            self.density_modifier = lambda x: denoise_tv_chambolle(np.real(x))
        else:
            self.density_modifier = density_modifier
        self.proximal_iters=proximal_iters
        self.proximal_lambda=prox_lambda
        At = lambda x: np.real(np.fft.ifftn(x))
        linear_map = LinearMap(np.fft.fftn,At,Vshape=self.shape,Wshape=self.shape)
        self.fasta_solver = fasta(A = linear_map, f= self._f, gradf=self._sub_grad_f,g=self._g,proxg=self._proxg, x0=self.real_space_guess,max_iters=max_iter,
            accelerate=accelerate, # Other options: plain, adapative
                            evaluate_objective=evaluate_objective, #evaluate objective function at every step, slows it down
                            verbose = verbose,record_iterates=record_iterates)
        self.real_space_guess=self.fasta_solver.solution
        if record_iterates:
            n_steps = self.fasta_solver.iterates.shape[0]
            self._initialize_tracking(n_iter = n_steps)
            self.rs_track[-n_steps:] = self.fasta_solver.iterates
