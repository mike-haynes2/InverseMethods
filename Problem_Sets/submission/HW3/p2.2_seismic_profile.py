import numpy as np
import math as m
import scipy as sci

import matplotlib.pyplot as plt

## CMH
## created 02-11-2026
## Inverse Methods (EAS 6134) Problem Set 3



## translated function to generate plot of ellipse (plot_ellipse.m)
def plot_ellipse(DELTA2, C, m, n = 400):
    """
    Plots the error ellipse centered at model parameters m.
    
    Parameters:
    DELTA2 : The Chi-squared value (e.g., 5.99 for 95% confidence with 2 degrees of freedom)
    C      : The 2x2 Covariance Matrix Cov(m)
    m      : The 2-element array of model parameters [m1, m2]
    """
    theta = np.linspace(0, 2*np.pi, n)
    
    # Create unit vectors for all angles (n x 2 matrix)
    xhat = np.column_stack((np.cos(theta), np.sin(theta)))
    
    # Calculate inverse covariance (Information Matrix)
    Cinv = np.linalg.inv(C)
    
    # Vectorized Mahalanobis calculation:
    # We calculate (xhat * Cinv * xhat.T) for all points simultaneously
    # This result is the 'denominator' for the radius in every direction
    quad_form = np.sum((xhat @ Cinv) * xhat, axis=1)
    
    # Calculate radii for all points
    r_lengths = np.sqrt(DELTA2 / quad_form)
    
    # Scaled points relative to center
    r = xhat * r_lengths[:, np.newaxis]
    # Force m to be (2,) to ensure correct broadcasting
    m_centered = np.atleast_1d(m).flatten()
    
    # Calculate final points
    final_points = m_centered + r

    
    # Plotting column 0 (X) vs column 1 (Y)
    plt.plot(final_points[:, 0], final_points[:, 1], 'b-o', markersize=2, alpha=0.5, label='95% confidence ellipse')
    # Plotting: Shift points by the model parameter vector m
    #plt.plot(m[0] + r[:, 0], m[1] + r[:, 1], label=f'Conf Region $\Delta^2$={DELTA2}')
    plt.scatter(m[0], m[1], color='red', marker='+', label='Best-fit Least-Squares $m$', zorder=10)
    
    #plt.axis('equal')
    plt.title('Error Ellipsoid for Least-Squares seismic profiling model', fontweight='bold')
    plt.xlabel('Parameter $m_1=t_0$ [s]')
    plt.ylabel('Parameter $m_2=s_2$ [s/m]')
    plt.legend()
    plt.grid(True)
    # calculate confidence interval
    variances = np.diag(C)
    
    # 3. Calculate the +/- half-width
    half_widths = np.sqrt(DELTA2 * variances)
    
    # 4. Create bounds
    lower_bounds = m - half_widths
    upper_bounds = m + half_widths
    
    return list(zip(lower_bounds, upper_bounds))

## unvectorized version (degraded performance but more transparent calculation)
# def plot_ellipse2(DELTA2, C, m):
#     n = 100
#     # 1. Generate n angles and unit vectors
#     theta = np.linspace(0, 2*np.pi, n)
#     xhat = np.column_stack((np.cos(theta), np.sin(theta)))
    
#     Cinv = np.linalg.inv(C)
    
#     # 2. Preallocate output array
#     r = np.zeros((n, 2))
    
#     # 3. Calculate the radius for each angle
#     for i in range(n):
#         # Calculate scaling factor: sqrt( delta^2 / (u^T * Cinv * u) )
#         # This solves for the point where the quadratic form equals DELTA2
#         unit_vec = xhat[i, :]
#         scale = np.sqrt(DELTA2 / (unit_vec @ Cinv @ unit_vec.T))
#         r[i, :] = scale * unit_vec
        
#     # 4. Plot the result (shifted by the mean m)
#     plt.plot(m[0] + r[:, 0], m[1] + r[:, 1])
#     plt.axis('equal')
#     plt.grid(True)


####################################################################################################
########################################## Part (a) ################################################
####################################################################################################



## define positions and data
xvec= np.array([6., 10.1333, 14.2667, 18.4, 22.5333, 26.6667])
t_elapsed_vec = np.array([3.4935, 4.2853, 5.1374, 5.8181, 6.8632, 8.1841])
d = np.copy(t_elapsed_vec)

## define shape of G
nrow = len(xvec)
ncol = 2
DF = nrow - ncol

## define G
c2 = xvec[:,None]
c1 = np.ones(nrow)[:,None]
G = np.concatenate((c1,c2),axis=1)
## check G
# print(G)
# print(np.dot(G.T,G))
# print(np.sum(xvec))


## perform normal equations least squares calculation
GTGinv = np.linalg.inv(G.T @ G)
GTd = np.dot(G.T,d)
mL2 = np.dot(GTGinv,GTd)

## store output
t0_L2 = mL2[0]
s2_L2 = mL2[1]



## plot output
fig, ax = plt.subplots((2), figsize=(7,9))

## plot data, model fit
ax[0].plot(xvec, (t0_L2+ (s2_L2*xvec)), color='navy', lw=2.2, label='Least Squares Model Fit' )
ax[0].errorbar(xvec,d,yerr=0.1, color='orange',fmt='o', linestyle='', capsize=4, lw=1, label='Observation times')
ax[0].set_xlabel('Distance from source [km]')
ax[0].set_ylabel('Time since origination [s]')
ax[0].set_title('Least-Squares Model Predictions',fontweight='bold')
ax[0].grid()
ax[0].legend()

## plot residuals
mod_vec = t0_L2+ (s2_L2*xvec)
res_vec = d - mod_vec

ax[1].axhspan(0, 1, color='green', alpha=0.3, label='underestimated')
ax[1].axhspan(0, -1, color='purple', alpha=0.3, label='overestimated')
ax[1].axhspan(-0.1, .1, color='cyan', alpha=0.2, label=r'$± \sigma$')

ax[1].scatter(xvec,res_vec, marker='+', color='navy', lw=2, s=90)
ax[1].set_xlabel('Distance from source [km]')
ax[1].set_ylabel('Residual [s]')
ax[1].set_title('Least-Squares Model-Data Residuals',fontweight='bold')
ax[1].grid()
ax[1].legend()
res_mag = np.max(np.abs(res_vec)) + (np.max(np.abs(res_vec))/6.)
ax[1].set_ylim(-res_mag, res_mag)



fig.tight_layout()
plt.show()
#plt.savefig('p2a_soln.png', dpi=275)
#plt.close()






####################################################################################################
########################################## Part (b) ################################################
####################################################################################################

sigma_measurement = 0.1     # seconds

## mean squared error of the residuals
MSE_residuals = np.dot(res_vec,res_vec)/(DF)

## calculate covariance matrix Cov(m)
sigma_squared = MSE_residuals
Covm = sigma_squared * GTGinv
print(Covm)

## calculate correlation matrix Corr(mi , mj)
std_devs = np.sqrt(np.diag(Covm))
Corrm = Covm / np.outer(std_devs,std_devs)

print(Corrm)




####################################################################################################
########################################## Part (c) ################################################
####################################################################################################



## eigendecomposition of Correlation matrix for error ellipsoid:
evals, evecs = np.linalg.eig(Corrm)
# so for eigenvalue i: evals[i] <--> evecs[:,i]
idx = evals.argsort()[::-1]
evals, evecs = evals[idx], evecs[:,idx]

## calculate Error ellipse properties
tilt_angle = np.rad2deg(np.arctan2(evecs[1,0], evecs[0,0]))

confidence_level = 0.95
D2 = sci.stats.chi2.ppf(confidence_level,df=DF)
print('Delta squared: ',D2)

conf_intervals = plot_ellipse(D2, Covm, mL2)

#plt.savefig('p2c_soln.png', dpi=260)
plt.show()

print(f"m1: {conf_intervals[0][0]:.4f} to {conf_intervals[0][1]:.4f}")
print(f"m2: {conf_intervals[1][0]:.4f} to {conf_intervals[1][1]:.4f}")

t0_lower = conf_intervals[0][0]
t0_upper = conf_intervals[0][1]
s2_lower = conf_intervals[1][0]
s2_upper = conf_intervals[1][1]

####################################################################################################
########################################## Part (d) ################################################
####################################################################################################

## calc the Chi-squared statistic
chi2_obs = np.sum(np.dot(res_vec,res_vec)) / sigma_measurement**2
print('Model Chi-squared value: ',chi2_obs)

## calc p-value
pval = sci.stats.chi2.sf(chi2_obs,DF)
print('Model p-value: ',pval)





####################################################################################################
########################################## Part (e) ################################################
####################################################################################################

## function to perform monte carlo simulations and evaluate the chi-squared distribution value from each
def monte_carlo_chi2_analysis(y_pred, X, sigma, n_sims=1000):
    """
    Performs Monte Carlo simulations to evaluate the distribution of Chi-squared.
    
    Parameters:
    y_pred : array, the data predicted by your best-fit model (X @ m), or mod_vec
    X      : G matrix, the operator
    sigma  : 0.1 s, the standard deviation of the noise
    n_sims : integer number of Monte Carlo iterations
    """
    n, p = X.shape
    df = n - p
    chi2_simulated = []

    # Pre-calculate (X^T X)^-1 X^T for speed (the OLS operator)
    # m_hat = (X.T @ X)^-1 @ X.T @ y
    ols_operator = np.linalg.inv(X.T @ X) @ X.T

    for _ in range(n_sims):
        # 1. Generate synthetic data: Prediced + Random Noise
        noise = np.random.normal(0, sigma, size=y_pred.shape)
        y_synthetic = y_pred + noise
        
        # 2. Re-estimate model parameters m_sim
        m_sim = ols_operator @ y_synthetic
        
        # 3. Calculate residuals of the simulated fit
        y_fit_sim = X @ m_sim
        residuals = y_synthetic - y_fit_sim
        
        # 4. Calculate Chi-squared statistic for this simulation
        # Using the known sigma
        c2 = np.sum(residuals**2) / sigma**2
        chi2_simulated.append(c2)

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    
    # Histogram of simulated Chi-squared values
    count, bins, ignored = plt.hist(chi2_simulated, bins=30, density=True, 
                                    alpha=0.7, color='skyblue', label='Simulated $\chi^2$')

    # Theoretical Chi-squared distribution
    x_axis = np.linspace(sci.stats.chi2.ppf(0.01, df), sci.stats.chi2.ppf(0.999, df), 100)
    plt.plot(x_axis, sci.stats.chi2.pdf(x_axis, df), 'r-', lw=2, label=f'Theoretical $\chi^2$ (df={df})')

    plt.title(f'Monte Carlo Seismic Profiling Simulation: {n_sims} Iterations', fontweight='bold')
    plt.xlabel('$\chi^2$ Value')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(alpha=0.6)


    return np.array(chi2_simulated)


## call the routine: run 1000 DSMC simulations, calculate 
monte_carlo_chi2_analysis(mod_vec, G, sigma_measurement, n_sims=1000)

#plt.savefig('p2e_soln.png', dpi=285)
plt.show()




####################################################################################################
########################################## Part (f) ################################################
####################################################################################################

## Our p-value of 8 x 10^-4 indicates that our model does not represent a very consistent solution. 
# Either (a) the data represent an unlikely arrangement for normally distributed measurement errors,
# or (b) the model is not capturing all of the seismic physics and is not a fully appropriate representation



####################################################################################################
########################################## Part (g) ################################################
####################################################################################################

# function to iterate and compute IRLS 

def irls_custom(X, y, mL2, tol=1e-7, max_iter=50, suppressed=False):
    # calculate shape of G matrix
    n, p = X.shape
    m = mL2 # initial guess is least-squares solution calculated above
    
    for i in range(max_iter):
        # store previous iter
        m_old = m.copy()
        
        # calculate residuals
        res = np.abs(y - X @ m)
        
        # define Weights based on residuals 
        # Eqn 2.90 in Aster: L1-norm approximation (W = 1/|res|)
        # compare against small delta to avoid division by zero
        weights = 1.0 / np.maximum(res, 1e-10)
        W = np.diag(weights)
        
        # solve Weighted Least Squares: m = (X.T * W * X)^-1 * X.T * W * y
        # Using np.linalg.solve is more stable than inv()
        lhs = X.T @ W @ X
        rhs = X.T @ W @ y
        m = np.linalg.solve(lhs, rhs)
        
        # 4. Check Convergence
        if np.linalg.norm(m - m_old) < tol:
            if not suppressed:
                print(f"Converged in {i} iterations.")
            break
    return m

IRLS_model = irls_custom(G, d, mL2)


## plot output against part (a)
fig, ax = plt.subplots((2), figsize=(7,9))

## plot data, model fit
ax[0].plot(xvec, mod_vec, color='navy', lw=2.2, label='Least Squares Model Fit' )
ax[0].plot(xvec, (G @ IRLS_model), color='teal', lw=2.2, label='1-Norm Model Estimate (IRLS)' )
ax[0].errorbar(xvec,d,yerr=0.1, color='orange',fmt='o', linestyle='', capsize=4, lw=1, label='Observed travel times')
ax[0].set_xlabel('Distance from source [km]')
ax[0].set_ylabel('Time since origination [s]')
ax[0].set_title('Least-Squares vs. IRLS Model Predictions',fontweight='bold')
ax[0].grid()
ax[0].legend()


ax[1].axhspan(0, 1, color='green', alpha=0.3, label='underestimated')
ax[1].axhspan(0, -1, color='purple', alpha=0.3, label='overestimated')
ax[1].axhspan(-0.1, .1, color='cyan', alpha=0.2, label=r'$± \sigma$')

ax[1].scatter(xvec,res_vec, marker='+', color='navy', lw=2, s=90, label='Least Squares residuals')
ax[1].scatter(xvec,(d-(G @ IRLS_model)), marker='x', color='teal', lw=2, s=90, label='IRLS residuals')
ax[1].set_xlabel('Distance from source [km]')
ax[1].set_ylabel('Residual [s]')
ax[1].set_title('Least-Squares Model-Data Residuals',fontweight='bold')
ax[1].grid()
ax[1].legend()
ax[1].set_ylim(-res_mag-0.17, res_mag+0.17)



fig.tight_layout()
plt.show()
#plt.savefig('p2g_soln.png', dpi=275)


####################################################################################################
########################################## Part (h) ################################################
####################################################################################################

mod_vec_IRLS = G @ IRLS_model

def monte_carlo_IRLS_error_prop(X, y, mIRLS, mL2, sigma, N=1000):
    # conf: confidence interval width
    # X : G matrix operator
    # y : d, or dataset vector of measurements
    # IRLS_model : model parameters from IRLS L1 method
    baseline_dat = X @ mIRLS
    # create average to update in place
    m_avg = mIRLS.copy()
    # create array to store ith row
    A = []
    for i in range(N):
        # calc Gaussian noise
        noise = np.random.normal(0., sigma, size=y.shape)
        # update baseline data to include regenerated noise dist
        dvec = baseline_dat + noise
        # calculate IRLS iteration procedure for new, noisy data (important!: keep model from LeastSquares as initial guess for "fairness")
        m_ith_IRLS_model = irls_custom(G, dvec, mL2, suppressed=True)
        # calculate average IRLS model from ensemble (update in place)
        k = i+1
        m_AVG = m_avg + (1./k)*(m_ith_IRLS_model - m_avg)
        m_avg = m_AVG.copy()
        # calculate the distance from average, populate column of A matrix
        Ai = m_ith_IRLS_model - m_avg
        A.append(Ai)

    A = np.mat(np.array(A))
    # estimate covariance matrix CovmL1
    CovmL1 = (A.T @ A)/N
    return CovmL1, m_avg

# execute model error propagation using MC simulation
CovmL1, mL1_avg = monte_carlo_IRLS_error_prop(G, d, IRLS_model, mL2, sigma_measurement)

# calculate confidence interval using covariance matrix:
# calculate confidence interval
IRLS_variances = np.diag(CovmL1)
print(CovmL1)

#  Calculate the +/- half-width for 95% confidence
half_widths = np.sqrt( D2 * IRLS_variances)

#  Create bounds for CI
lower_bounds = mL1_avg - half_widths
upper_bounds = mL1_avg + half_widths

conf_intervals_L1 = list(zip(lower_bounds, upper_bounds))

print(f"m1: {conf_intervals_L1[0][0]:.4f} to {conf_intervals_L1[0][1]:.4f}")
print(f"m2: {conf_intervals_L1[1][0]:.4f} to {conf_intervals_L1[1][1]:.4f}")

