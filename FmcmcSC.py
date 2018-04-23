from __future__ import division  # Use "real" division

# Add another filepath to the Python path
import sys
#sys.path.insert(1, '/data/repos-git/quad/Functions-from-Chris')
#sys.path.insert(2, '/data/repos-git/quad/Functions-from-Chris/g2conda/GSASII')
#sys.path.insert(3, '/data/repos-git/quad')

# Import specific functions
from timeit import default_timer as timer  # Timing function
from scipy.stats import norm               # Normal distribution
from scipy.stats import multivariate_normal as mvnorm  # Multivariate normal distribution
import statsmodels.api as sm               # Library for lowess smoother
lowess = sm.nonparametric.lowess           # Lowess smoothing function
from bspline import Bspline                # Bspline function
from splinelab import augknt               # Bspline helper function

# Import entire modules
import numpy as np
import matplotlib.pyplot as plt
import CalculateSinCos as Calc

# Source mean function process from GSAS_Calculator_Opt.py
#import GSAS_Calculator_Opt as gsas

## Helper functions
# Transform between bounded parameter space and continuous space
def z2par(z, lower, upper, grad=False):
    if (grad):
        d = (upper-lower)*norm.pdf(z)
        return d
    else:
        par = lower + (upper-lower)*norm.cdf(z)
        # Fudge the parameter value if we've hit either boundary
        par[np.array([par[j]==upper[j] for j in range(len(par))])] -= 1e-10
        par[np.array([par[j]==lower[j] for j in range(len(par))])] += 1e-10
        return par

# Log likelihood of the parameters based on the prior distributions
def prior_loglike(par, m0, sd0):
    #print -0.5*(sd0**(-2))*np.inner(par-m0, par-m0)
    return -0.5*(sd0**(-2))*np.inner(par-m0, par-m0)

# Log of the posterior distribution
def logf(y_x,y_n, x, BG_x,BG_n, y_calc_x, y_calc_n, paramList, z, lower, upper, scale_x, scale_n, tau_y_x, tau_y_n, m0, sd0,n_share=1,n_x=2,n_n=2):
    # Update the calculator to reflect the current parameter estimates
    params = z2par(z=z, lower=lower, upper=upper)
    share = params[0:n_share]
    params_x = params[n_share:(n_share+n_x)]
    params_n = params[(n_share+n_x):(n_share+n_x+n_n)]
    z_x = (z[0:(n_share+n_x)])
    z_n = np.concatenate((np.array(z[0:n_share]),np.array(z[(n_share+n_x):(n_share+n_x+n_n)])))
    # Calculate the potential energy
    R_x = y_x-BG_x-y_calc_x               # Calculate residuals
    S_x = np.inner(R_x/np.sqrt(scale_x), R_x/np.sqrt(scale_x))  # Calculate weighted SSE
    l_x = 0.5*tau_y_x*S_x #- prior_loglike(par=z_x, m0=m0, sd0=sd0)
    R_n = y_n-BG_n-y_calc_n               # Calculate residuals
    S_n = np.inner(R_n/np.sqrt(scale_n), R_n/np.sqrt(scale_n))  # Calculate weighted SSE
    l_n = 0.5*tau_y_n*S_n #- prior_loglike(par=z_n, m0=m0, sd0=sd0)
    #print l_x," ",l_n," ",prior_loglike(par=z,m0=m0,sd0=sd0)
    return (-1)*(l_x+l_n-prior_loglike(par=z,m0=m0,sd0=sd0))

## MCMC function
def nlDRAM(paramList, init_z, lower, upper, y_x=None, y_n=None, x=None, L=20, shrinkage=0.2, s_p=(2.4**2), epsilon=1e-4, m0=0, sd0=1, c_y=0.1, d_y=0.1, c_g=0.1, d_g=0.1, c_b=0.1, d_b=0.1, adapt=20, thin=1, iters=5000, burn=2000, update=500, plot=True, fix=False):
    # Args:
    #   GPXfile - string, filepath for the GPX file underlying the current data
    #   paramList - (q x 1) list of GSASII parameter names in the same order as
    #               the upper and lower limits being provided
    #   init_z - (q x 1) vector of initial values in the z-space
    #   lower - (q x 1) vector of lower bounds for the parameter values
    #   upper - (q x 1) vector of upper bounds for the parameter values
    #   y - (n x 1) vector of intensities, default value is None. If no values
    #       are specified, the function uses the values from the provided GPX
    #       file
    #   x - (n x 1) vector of angles (2*theta), default value is None. If no
    #       values are specified, the function uses the values from the provided
    #       GPX file
    #   L - scalar, number of B-spline basis functions to model the background
    #       intensity, default is 20
    #   shrinkage - scalar, governs covariance change between proposal stages,
    #               default is 0.2
    #   s_p - scalar, scaling parameter for the adaptive covariance, default is
    #         set to (2.4**2)/d as in Gelman (1995), where d is the dimension of
    #         the parameter space
    #   epsilon - scalar, ridge constant to prevent singularity of the adaptive
    #             covariance, default is 0.0001
    #   m0, sd0 - scalars, govern the prior distribution on the latent Zs,
    #             default is a standard normal distribution
    #   c_y, d_y - scalars, govern the prior Gamma distribution for the error
    #              variance, default value is 0.1 for both
    #   c_g, d_g - scalars, govern the prior Gamma distribution for the error
    #              in the prior distribution for the basis function loadings,
    #   c_b, d_b - scalars, govern the prior Gamma distribution for scale of
    #              the proportional contribution to the error variance, default
    #              value is 0.1 for both
    #   adapt - scalar, controls the adaptation period, default is 20
    #   thin - scalar, degree of thinning, default is 1
    #   iters - scalar, number of total iterations to run, default is 5000
    #   burn - scalar, number of samples to consider as burn-in, default is 2000
    #   update - scalar, period between updates printed to the console, default
    #            is 500
    #   plot - boolean, indicator for whether or not to create trace plots as
    #          the sampler progresses, default is True
    #
    # Returns: 5-tuple containing the posterior samples for the parameters and
    #          the model timing, tuple entries are
    #            1 - (nSamples x q) matrix of posterior samples for the mean
    #                process parameters of interest
    #            2 - (nSamples x 1) vector of posterior samples for the constant
    #                factor on the smoothed observations in the proportional
    #                variance
    #            3 - (nSamples x 1) vector of posterior samples for the overall
    #                variance / temperature
    #            4 - (nSamples x L) matrix of posterior samples for the basis
    #                function loadings modeling the background intensity
    #            5 - scalar, number of minutes the sampler took to complete

    # Initialize the calculator based on the provided GPX file
    s_p = ((2.4**2)/len(paramList))

    # Assign the intensity vector (y) from the GPX file, if necessary
    #if y is None:
    #    y = np.array(Calc._Histograms[Calc._Histograms.keys()[0]]['Data'][1][:-1], copy=True)

    # Assign the grid of angles (x) from the GPX file, if no values are provided. If values ARE provided, overwrite the _tthsample parameter
    #if x is None:
    #    x = np.array(Calc._tth, copy=True)
    #else:
    #    Calc._tthsample = np.array(x, copy=True)

    # Calculate a B-spline basis for the range of x
    #unique_knots = np.percentile(a=x, q=np.linspace(0, 100, num=(L-2)))
    #knots = augknt(unique_knots, 3)
    #objB = Bspline(knots, order=3)
    #B = objB.collmat(x)
    #del unique_knots, knots, objB

    # Save dimensions
    n_x = len(y_x)       # Number of observations

    n_n = len(y_n)
    q = len(init_z)  # Number of parameters of interest

    n_share = 1
    n_x = 2
    n_n = 2

    # Smooth the observed Ys on the Xs, patch for negative or 0 values
    y_x_sm = lowess(endog=y_x, exog=x, frac=6.0/len(x), return_sorted=False)
    #y_x_sm = np.array([max(0, sm) for sm in y_x_sm])
    y_n_sm = lowess(endog=y_n, exog=x, frac=6.0/len(x), return_sorted=False)
    #y_n_sm = np.array([max(0, sm) for sm in y_n_sm])

    # Get indices of parameters to refine, even if they are "fixed" by bounds
    #useInd = [np.asscalar(np.where(np.array(Calc._varyList)==par)[0]) for par in paramList]
    #if (any(np.array(Calc._varyList)[useInd] != paramList)):
    #    raise ValueError("Parameter list specification is not valid.")

    # Make sure initial z values are given for every parameter in paramList
    if len(paramList)!=len(init_z):
        raise ValueError("Initial value specification for Z is not valid.")

    # Initialize parameter values
    z = np.array(init_z, copy=True)                     # Latent process
    params = z2par(z=init_z, lower=lower, upper=upper)  # Parameters of interest
    tau_y_x = 10                                          # Error variance for Y
    tau_y_n = 10
    #b_x = 1                                               # Contribution of y_sm
    #b_n = 1
    #gamma_x = np.ones(L)                                  # Loadings
    #gamma_n = np.ones(L)
    #tau_b_x = 1                                           # Variance for loadings
    #tau_b_n = 1
    #BG_x = np.matmul(B, gamma_x)                            # Background intensity
    #BG_n = np.matmul(B, gamma_n)

    share = params[0:n_share]
    params_x = params[n_share:(n_share+n_x)]
    params_n = params[(n_share+n_x):(n_share+n_x+n_n)]
    y_calc_x = Calc.CalcSin(x,params_x,share)
    y_calc_n = Calc.CalcCos(x,params_n,share)


    #var_scale_x = 1 + b_x*y_x_sm                              # Scale for y_sm/tau_y
    #var_scale_n = 1 + b_n*y_n_sm
    var_scale_x = np.zeros_like(x)+1.#np.abs(y_x_sm)                              # Scale for y_sm/tau_y
    var_scale_n = np.zeros_like(x)+1.#np.abs(y_n_sm)
    #var_scale_x = np.abs(y_x_sm)                            # Scale for y_sm/tau_y
    #var_scale_n = np.abs(y_n_sm)

    # Initialize covariance for proposal distribution using the Hessian
    # Calc.Calculate()
    # fullHess = Calc.CalculateHessian()
    # keepHess = (fullHess[:, useInd])[useInd, :]
    # scaleHess = keepHess * np.outer(z2par(z=init_z, lower=lower, upper=upper, grad=True), z2par(z=init_z, lower=lower, upper=upper, grad=True))
    # varS1 = np.linalg.inv(scaleHess)

    # Initialize covariance for proposal distribution
    varS1 = np.diag(0.0000005*np.ones(q))
    varS1 = np.diag(0.05*np.ones(q))

    # Set up counters for the parameters of interest
    att = att_S2 = acc_S1 = acc_S2 = 0  # Attempts / acceptances counters

    # Set Metropolis parameters for b
    #met_sd_b = 0.1     # Metropolis standard deviation
    #att_b = acc_b = 0  # Attempts / acceptances counters

    # Calculate the number of thinned samples to keep
    n_keep = np.floor_divide(iters-burn-1, thin) + 1
    curr_keep = 0

    # Initialize output objects
    all_Z = np.zeros((iters, q))
    keep_params = np.zeros((n_keep, q))
    #keep_gamma_x = np.zeros((n_keep, L))
    #keep_gamma_n = np.zeros((n_keep, L))
    #keep_b_x = np.zeros(n_keep)
    #keep_b_n = np.zeros(n_keep)
    keep_tau_y_x = np.zeros(n_keep)
    keep_tau_y_n = np.zeros(n_keep)
    #keep_tau_b_x = np.zeros(n_keep)
    #keep_tau_b_n = np.zeros(n_keep)

    BG_x = np.zeros(len(x))
    BG_n = np.zeros(len(x))

    tick = timer()
    for i in range(iters):
        #print " "
        ## Update basis function loadings and then background values
        #BtB_x = np.matmul(np.transpose(B)/var_scale_x, B)
        #VV_x = np.linalg.inv(tau_y_x*BtB_x + tau_b_x*np.identity(L))
        #err_x = (y_x-y_calc_x)/var_scale_x
        #MM_x = np.matmul(VV_x, tau_y_x*np.sum(np.transpose(B)*err_x, axis=1))
        #gamma_x = np.random.multivariate_normal(mean=MM_x, cov=VV_x)
        #BG_x = np.matmul(B, gamma_x)
        #del VV_x, err_x, MM_x

        #BtB_n = np.matmul(np.transpose(B)/var_scale_n, B)
        #VV_n = np.linalg.inv(tau_y_n*BtB_n + tau_b_n*np.identity(L))
        #err_n = (y_n-y_calc_n)/var_scale_n
        #MM_n = np.matmul(VV_n, tau_y_n*np.sum(np.transpose(B)*err_n, axis=1))
        #gamma_n = np.random.multivariate_normal(mean=MM_n, cov=VV_n)
        #BG_n = np.matmul(B, gamma_n)
        #del VV_n, err_n, MM_n

        ## Update mean process parameters using 2-stage DRAM
        att += 1
        #Stage 1:
        can_z1 = np.random.multivariate_normal(mean=z, cov=varS1)
        params1 = z2par(z=can_z1, lower=lower, upper=upper)  # Update params
        share = params1[0:n_share]
        params_x = params1[n_share:(n_share+n_x)]
        params_n = params1[(n_share+n_x):(n_share+n_x+n_n)]
        y_calc_x1 = Calc.CalcSin(x,params_x,share)
        y_calc_n1 = Calc.CalcCos(x,params_n,share)
        can1_ll = logf(y_x=y_x, y_n=y_n, x=x, BG_x=BG_x, BG_n=BG_n, y_calc_x=y_calc_x1, y_calc_n=y_calc_n1, paramList=paramList, z=can_z1, lower=lower, upper=upper, scale_x=var_scale_x, scale_n=var_scale_n, tau_y_x=tau_y_x, tau_y_n=tau_y_n, m0=m0, sd0=sd0)
        cur_ll = logf(y_x=y_x, y_n=y_n, x=x, BG_x=BG_x, BG_n=BG_n, y_calc_x=y_calc_x, y_calc_n=y_calc_n, paramList=paramList, z=z, lower=lower, upper=upper, scale_x=var_scale_x, scale_n=var_scale_n, tau_y_x=tau_y_x, tau_y_n=tau_y_n, m0=m0, sd0=sd0)
        R1 = can1_ll - cur_ll
        #print "can_z1:",can_z1
        #print " canl_ll:", can1_ll," cur_ll:", cur_ll
        #print "R1:", R1
        if (np.log(np.random.uniform()) < R1):
            #print "ACCEPTED STAGE 1"
            acc_S1 += 1
            z = np.array(can_z1, copy=True)                # Update latent
            params = z2par(z=z, lower=lower, upper=upper)  # Update params
            share = params[0:n_share]
            params_x = params[n_share:(n_share+n_x)]
            params_n = params[(n_share+n_x):(n_share+n_x+n_n)]
            y_calc_x = Calc.CalcSin(x,params_x,share)
            y_calc_n = Calc.CalcCos(x,params_n,share)
        else:
            #Stage 2:
            att_S2 += 1
            # Propose the candidate
            can_z2 = np.random.multivariate_normal(mean=z, cov=shrinkage*varS1)
            params2 = z2par(z=can_z2, lower=lower, upper=upper)  # Update params
            share = params2[0:n_share]
            params_x = params2[n_share:(n_share+n_x)]
            params_n = params2[(n_share+n_x):(n_share+n_x+n_n)]
            y_calc_x2 = Calc.CalcSin(x,params_x,share)
            y_calc_n2 = Calc.CalcCos(x,params_n,share)
            #print "can_z2:", can_z2
            if np.sum(np.abs(can_z2) > 3)==0:
                can2_ll = logf(y_x=y_x, y_n=y_n, x=x, BG_x=BG_x, BG_n=BG_n, y_calc_x=y_calc_x2, y_calc_n=y_calc_n2, paramList=paramList, z=can_z2, lower=lower, upper=upper, scale_x=var_scale_x, scale_n=var_scale_n, tau_y_x=tau_y_x, tau_y_n=tau_y_n, m0=m0, sd0=sd0)
                #print "can2_ll:", can2_ll
                #print can1_ll, can2_ll, np.exp(can1_ll-can2_ll)

                # Calculate the acceptance probability
                inner_n = 1 - np.min([1, np.exp(can1_ll - can2_ll)])
                inner_d = 1 - np.min([1, np.exp(can1_ll - cur_ll)])
                # Fudge factors for inner_n and inner_d too close to 0, 1
                inner_n = inner_n + 1e-10 if inner_n==0 else inner_n
                inner_n = inner_n - 1e-10 if inner_n==1 else inner_n
                inner_d = inner_d + 1e-10 if inner_d==0 else inner_d
                inner_d = inner_d - 1e-10 if inner_d==1 else inner_d
                numer = can2_ll + mvnorm.logpdf(x=can_z1, mean=can_z2, cov=varS1) + np.log(inner_n)
                denom = cur_ll + mvnorm.logpdf(x=can_z1, mean=z, cov=varS1) + np.log(inner_d)
                R2 = numer - denom
                #print "R2", R2
                if np.log(np.random.uniform()) < R2:
                    #print "ACCEPTED STAGE 2"
                    acc_S2 += 1
                    z = np.array(can_z2, copy=True)                # Update latent
                    params = z2par(z=z, lower=lower, upper=upper)  # Update params
                    share = params[0:n_share]
                    params_x = params[n_share:(n_share+n_x)]
                    params_n = params[(n_share+n_x):(n_share+n_x+n_n)]
                    y_calc_x = Calc.CalcSin(x,params_x,share)
                    y_calc_n = Calc.CalcCos(x,params_n,share)
                del can_z2, can2_ll, inner_n, inner_d, numer, denom, R2
            else:
                del can_z2
        del can_z1, can1_ll, cur_ll, R1
        all_Z[i] = z

        ## Adapt the proposal distribution covariance matrix
        if (0 < i) & (i % adapt is 0):
            varS1 = s_p*np.cov(all_Z[range(i+1)].transpose()) + s_p*epsilon*np.diag(np.ones(q))

        ## Update tau_b
        #rate = d_g + 0.5*np.inner(gamma_x, gamma_x)
        #tau_b_x = np.random.gamma(shape=(c_g + 0.5*L), scale=1/rate)
        #del rate
        #rate = d_g + 0.5*np.inner(gamma_n, gamma_n)
        #tau_b_n = np.random.gamma(shape=(c_g + 0.5*L), scale=1/rate)
        #del rate

        ## Update tau_y
        #err_x = (y_x-BG_x-y_calc_x)/np.sqrt(var_scale_x)
        #rate = d_y + 0.5*np.inner(err_x, err_x)
        #tau_y_x = np.random.gamma(shape=(c_y + 0.5*n_x), scale=1/rate)
        #del rate
        #err_n = (y_n-BG_n-y_calc_n)/np.sqrt(var_scale_n)
        #rate = d_y + 0.5*np.inner(err_n, err_n)
        #tau_y_n = np.random.gamma(shape=(c_y + 0.5*n_n), scale=1/rate)
        #del rate

        # ## Update b
        # att_b += 1
        # can_b = np.exp(np.random.normal(loc=np.log(b), scale=met_sd_b))
        # can_err = (y-BG-(Calc.Calculate())[1].data)/np.sqrt(1+can_b*y_sm)
        # can_ll = -0.5*sum(np.log(1+can_b*y_sm)) + (c_b-1)*np.log(can_b) - d_b*can_b - 0.5*tau_y*np.inner(can_err, can_err)
        # cur_ll = -0.5*sum(np.log(var_scale)) + (c_b-1)*np.log(b) - d_b*b - 0.5*tau_y*np.inner(err, err)
        # R = can_ll - cur_ll
        #
        # # Evaluate the candidate
        # if np.log(np.random.uniform()) < R:
        #     acc_b += 1
        #     b = can_b
        #     var_scale = 1 + b*y_sm
        # del err, can_b, can_err, can_ll, cur_ll, R
        #
        # ## Update Metropolis std deviation for b during burn-in
        # if (0 < i <= burn) & (i % 25 is 0):
        #     if acc_b/att_b < 0.3:
        #         met_sd_b *= 0.8
        #     elif acc_b/att_b > 0.7:
        #         met_sd_b *= 1.2
        #     att_b = acc_b = 0  # Reset counters for b

        ## Keep track of everything
        if i >= burn:
            # Store posterior draws if appropriate
            if (i-burn) % thin is 0:
                keep_params[curr_keep] = params
                #keep_gamma_x[curr_keep] = gamma_x
                #keep_b_x[curr_keep] = b_x
                keep_tau_y_x[curr_keep] = tau_y_x
                #keep_tau_b_x[curr_keep] = tau_b_x
                #keep_gamma_n[curr_keep] = gamma_n
                #keep_b_n[curr_keep] = b_n
                keep_tau_y_n[curr_keep] = tau_y_n
                #keep_tau_b_n[curr_keep] = tau_b_n
                curr_keep += 1

            # Print an update if necessary
            if curr_keep % update is 0:
                print "Collected %d of %d samples" % (curr_keep, n_keep)
                print '  %03.2f acceptance rate for Stage 1 (%d attempts)' % (acc_S1/att, att)
                if att_S2 > 0:
                    rate_2 = acc_S2/att_S2
                else:
                    rate_2 = 0
                print '  %03.2f acceptance rate for Stage 2 (%d attempts)' % (rate_2, att_S2)
                del rate_2

                # Produce trace plots
                if plot is True:
                    plt.figure(1, figsize=(20, 10))
                    plt.subplot(261)
                    plt.plot(keep_params[range(curr_keep), 0], 'k')
                    plt.xlabel("Iteration")
                    plt.ylabel(paramList[0])
                    plt.subplot(262)
                    plt.plot(keep_params[range(curr_keep), 1], 'k')
                    plt.xlabel("Iteration")
                    plt.ylabel(paramList[1])
                    plt.subplot(263)
                    plt.plot(keep_params[range(curr_keep), 2], 'k')
                    plt.xlabel("Iteration")
                    plt.ylabel(paramList[2])
                    plt.subplot(264)
                    plt.plot(keep_params[range(curr_keep), 3], 'k')
                    plt.xlabel("Iteration")
                    plt.ylabel(paramList[3])
                    plt.subplot(265)
                    plt.plot(keep_params[range(curr_keep), 4], 'k')
                    plt.xlabel("Iteration")
                    plt.ylabel(paramList[4])
                    plt.subplot(266)
                    plt.plot(keep_params[range(curr_keep), 5], 'k')
                    plt.xlabel("Iteration")
                    plt.ylabel(paramList[5])

                    plt.subplot(266)
                    plt.plot(keep_params[range(curr_keep), 6], 'k')
                    plt.xlabel("Iteration")
                    plt.ylabel(paramList[6])
                    plt.subplot(267)
                    plt.plot(keep_params[range(curr_keep), 7], 'k')
                    plt.xlabel("Iteration")
                    plt.ylabel(paramList[7])
                    plt.subplot(268)
                    plt.plot(keep_params[range(curr_keep), 8], 'k')
                    plt.xlabel("Iteration")
                    plt.ylabel(paramList[8])
                    plt.subplot(265)
                    plt.plot(keep_params[range(curr_keep), 9], 'k')
                    plt.xlabel("Iteration")
                    plt.ylabel(paramList[9])
                    plt.subplot(2,6,10)
                    plt.plot(keep_params[range(curr_keep), 10], 'k')
                    plt.xlabel("Iteration")
                    plt.ylabel(paramList[10])
                    plt.pause(0.1)
                    # plt.show()

    tock = timer()
    print (tock-tick)/60
    # Gather output into a tuple
    #output = (keep_params, varS1, keep_b_x, keep_b_n, 1.0/keep_tau_y_x, 1.0/keep_tau_y_n, keep_gamma_x, keep_gamma_n, (tock-tick)/60)
    output = (keep_params, varS1)
    return output
