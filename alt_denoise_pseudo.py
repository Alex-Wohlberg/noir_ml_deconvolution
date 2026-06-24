# Pseudo Code for proposed gradient step optimization
#
# Idea: use diffusion model denoiser as a regularizer in optimization problem for inversion of Deconvolution
#
# Optimizer solves:  argmin_x  (1/2) || y - C x ||_2^2   +   lambda * R(x)
#
# y	   : measurement, blurry image with noise
# C    : convolutiuon with PSF. forward operator
# R(x) : regularizer, denoising diffusion model applied onto obtained solution. only
#        ever need proximal operator to perform ADMM optimization.
# lamda : reularization wieght (balance bewteen bias and variance of optimization)
# x     : solution (sharp image)
#
#-------------------------------------------------------------------------------------------------
#
# C        : forward blur operator (psf)
# y        : blurred, noisy observation
# denoiser : diffusion model used as a Gaussian denoiser at a chosen strength.
#            Trained offline on SHARP star fields only -> it encodes the prior
#            p(x) = "what clean star fields look like." aimed at penalizing solutions with heavy confusion noise
# lambda   : regularization weight
# rho      : step size of gradient step
# iterations: number of ADMM iterations
#
#
def global_admm_opitimize(y ,C, denoiser, lambda_, rho, iterations):
	x = C.adjoint(y)   # starting guess ( assume C unitary)
	for i in range(iterations):

        # 1:GRADIENT of the smooth L2 data-fidelity term
        # f(x) = (1/2)||y - C x||^2     ->     grad f = C^T (C x - y)
        grad = C.adjoint(C(x) - y)
 
        # 2:GRADIENT STEP: form the new (pre-prox) x ----
        x_pre = x - rho * grad      # move to better explain the data
 
        # 3: Denoising Step with Difusion Model
        # prox of lambda*R has the form "denoise at strength sqrt(rho*lambda)".
        x = denoiser(x_half, strength = sqrt(rho * lambda)
        
        # From my understanding of prevalence of shot noise vs confusion noise         
        #   lambda too large -> prior overrides data in ambiguous blends
        #   lambda too small -> weak prior, shot noise amplifies
        #   so concern is about the optimzation of lamda

    return x