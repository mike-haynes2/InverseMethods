import numpy as np
from scipy.special import jv




## a function to calculate the B field from a dipole moment centered at Europa 
# (i.e., at the origin) with a specified M vector and position
## func is VECTORIZED: accepts INPUTS:
# r_pos: (N, 3) array of positions (meters)
# M: (3,) array representing the dipole moment vector
# returns(N, 3) B-field vectors (Tesla) matching the shape of rvec in
def calculate_B_dipole_vectorized(r_pos, M):


    # Standard Dipole Equation: B(r) = (mu0/4pi) * [3r(M.r)/r^5 - M/r^3]
    # Note: mu0/4pi = 1e-7
    r_mag = np.linalg.norm(r_pos, axis=1, keepdims=True)
    r_hat = r_pos / r_mag
    
    dot_product = np.sum(r_hat * M, axis=1, keepdims=True)
    
    term1 = 3 * r_hat * dot_product
    term2 = M
    
    B = 1e-7 * (term1 - term2) / (r_mag**3)
    return B


# def calculate_B_dipole_unit_agnostic(r_pos, M_vol):
#     """
#     Calculates B-field from a Volume-normalized dipole.
#     r_pos: (N, 3) array of positions (meters)
#     M_vol: (3,) array [Q * B0 * R^3] 
#     Returns: (N, 3) B-field in same units as B0 (e.g., nT)
#     """
#     r_mag = np.linalg.norm(r_pos, axis=1, keepdims=True)
#     r_hat = r_pos / r_mag
    
#     dot_product = np.sum(r_hat * M_vol, axis=1, keepdims=True)
    
#     # The 0.5 is the required geometric factor for a spherical dipole
#     term1 = 3 * r_hat * dot_product
#     term2 = M_vol
    
#     B_ind = 0.5 * (term1 - term2) / (r_mag**3)
#     return B_ind


def calculate_B_dipole_final(r_pos_km, M_km_units):
    """
    r_pos_km: (N, 3) array of positions in KILOMETERS
    M_km_units: (3,) dipole moment calculated using km (e.g., Q * B0 * R_km^3)
    """
    # Everything stays in km to match your 10^11 moment
    r_mag = np.linalg.norm(r_pos_km, axis=1, keepdims=True)
    r_hat = r_pos_km / r_mag
    
    dot_product = np.sum(r_hat * M_km_units, axis=1, keepdims=True)
    
    # Unit-agnostic dipole formula
    # B = (1/r^3) * [3(M·r_hat)r_hat - M]
    term1 = 3 * r_hat * dot_product
    term2 = M_km_units
    
    # We do NOT multiply by mu0/4pi because it's already implicitly 
    # handled in the way we defined the induction moment Q
    B_ind = (term1 - term2) / (r_mag**3)
    return B_ind


## compact function that will convert the model parameters 
# (inner radius, outer radius, conductivity) into an actual dipole moment
def model_to_dipole_moment(m, freq, B0_vec):
    """
    Converts model m = [r1, r0, sigma] to a complex dipole moment vector M.
    r1, r0 in km, sigma in S/m.
    """
    ## ONLY INDUCING COMPNENTS COUNT HERE!
    B0_vec = np.array([B0_vec[0],B0_vec[1], 0.])
    r1_km, r0_km, sigma = m
    mu0 = 4 * np.pi * 1e-7
    omega = 2 * np.pi * freq
    k = (1 - 1j) * np.sqrt(mu0 * sigma * omega / 2.0)
    
    # Radii to meters
    z0, z1 = r0_km * 1000 * k, r1_km * 1000 * k
    
    # Zimmer Eq (6) & (5)
    R_num = z1 * jv(-2.5, z1)
    R_den = 3 * jv(1.5, z1) - z1 * jv(0.5, z1)
    R = R_num / (R_den + 1e-20)
    
    ae_num = R * jv(2.5, z0) - jv(-2.5, z0)
    ae_den = R * jv(0.5, z0) - jv(-0.5, z0)
    Q = ae_num / (ae_den + 1e-20)
    
    # Dipole Moment M = -Q * B0 * R_outer^3 / (scaling constant)
    # Using the standard induced moment definition
    M_complex = -Q * B0_vec * (r0_km * 1000)**3
    return M_complex



# def model_to_dipole_moment_stable(m, freq, B0_vec):
#     r1_km, r0_km, sigma = m
#     B0_inducing = np.array([B0_vec[0], B0_vec[1], 0.]) # Only X, Y induce
    
#     mu0 = 4 * np.pi * 1e-7
#     omega = 2 * np.pi * freq
#     # k = sqrt(i * mu0 * sigma * omega)
#     k = np.sqrt(1j * mu0 * sigma * omega)
    
#     r0, r1 = r0_km * 1000.0, r1_km * 1000.0
#     d = r0 - r1 # Ocean thickness
    
#     # Large-argument approximation for Q of a shell
#     # For large |kz|, the induction response Q approaches:
#     # Q = 1 - (3 / (k*r0)) * [coth(k*d) - 1/(k*r0)]
#     # This is numerically stable for any sigma or radius.
#     kd = k * d
#     # Use np.where or a safety check for coth(kd)
#     coth_kd = 1.0 / np.tanh(kd)
    
#     Q = 1.0 - (3.0 / (k * r0)) * (coth_kd - 1.0 / (k * r0))
    
#     # Moment = -1/2 * Q * B0 * r0^3 (The 1/2 is the spherical geometry factor)
#     M_complex = -0.5 * Q * B0_inducing * (r0**3)
#     return M_complex




import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
def plot_mag_trajectory(t, x, y, z, Bx, By, Bz, curveBx=None, curveBy=None, curveBz=None, color='navy', lw=2.2, title='Galileo MAG: E14 Flyby', figname = 'E14_dat.png', nplots=3, save=False):

    r = np.sqrt(x**2 + y**2 + z**2)
    CA_idx = np.argmin(r)

    if nplots != 3 and nplots != 4:
        raise ValueError('NOTE: nplots must be either 3 (components) or 4 (components + magnitude)')

    fig, ax = plt.subplots(nplots)

    ax[0].plot(t, Bx, lw=lw, color=color, label='Galileo MAG')
    ax[0].set_ylabel('$B_x$ [nT]')

    ax[1].plot(t, By, lw=lw, color=color)
    ax[1].set_ylabel('$B_y$ [nT]')

    ax[2].plot(t, Bz, lw=lw, color=color)
    ax[2].set_ylabel('$B_z$ [nT]')
    ax[2].set_xlabel('Time [UTC]')

    if curveBx is not None:
        ax[0].plot(t, curveBx, lw=(lw/2.), color='teal', label='Recovered dipole')
        ax[1].plot(t, curveBy, lw=(lw/2.), color='teal')
        ax[2].plot(t, curveBz, lw=(lw/2.), color='teal')

    for i in range(nplots):
        ax[i].axvline(t[CA_idx], color='red', lw=3.2, ls='--')
        ax[i].grid(alpha=0.7,ls='dashed')
        if i < nplots-1:
            ax[i].xaxis.set_tick_params(labelbottom=False)
        else:
            ax[i].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    if curveBx is not None:
        ax[0].legend()

    fig.suptitle(title,fontweight='bold')
    fig.tight_layout()

    if save:
        plt.savefig(figname, dpi=290)
        plt.close()
    else:
        plt.show()
    


# plot synth trajectory: no UTC, label with x coord
def plot_mag_synth(x, y, z, Bx, By, Bz, curveBx=None, curveBy=None, curveBz=None, curve2Bx=None, curve2By=None, curve2Bz=None, color='navy', lw=2.2, title='Synthetic MAG Data: Max Distance Below Mag. Eq.', figname = 'synth_dat.png', nplots=3, save=False):

    r = np.sqrt(x**2 + y**2 + z**2)
    CA_idx = np.argmin(r)
    RE = 1560.8e+03

    if nplots != 3 and nplots != 4:
        raise ValueError('NOTE: nplots must be either 3 (components) or 4 (components + magnitude)')

    fig, ax = plt.subplots(nplots)

    ax[0].plot(x/RE, Bx, lw=lw, color=color, label='Synthetic data')
    ax[0].set_ylabel('$B_x$ [nT]')

    ax[1].plot(x/RE, By, lw=lw, color=color)
    ax[1].set_ylabel('$B_y$ [nT]')

    ax[2].plot(x/RE, Bz, lw=lw, color=color)
    ax[2].set_ylabel('$B_z$ [nT]')
    ax[2].set_xlabel('$x$ position [$R_E$]')

    if curveBx is not None:
        ax[0].plot(x/RE, curveBx, lw=(lw/2.), color='teal', label='Recovered dipole: LM')
        ax[1].plot(x/RE, curveBy, lw=(lw/2.), color='teal')
        ax[2].plot(x/RE, curveBz, lw=(lw/2.), color='teal')

    if curve2Bx is not None:
        ax[0].plot(x/RE, curve2Bx, lw=(lw/2.), color='cyan', label='Recovered dipole: GN')
        ax[1].plot(x/RE, curve2By, lw=(lw/2.), color='cyan')
        ax[2].plot(x/RE, curve2Bz, lw=(lw/2.), color='cyan')
    

    for i in range(nplots):
        ax[i].axvline(x[CA_idx]/RE, color='red', lw=3.2, ls='--')
        ax[i].grid(alpha=0.7,ls='dashed')
        if i < nplots-1:
            ax[i].xaxis.set_tick_params(labelbottom=False)
        else:
            pass
            #ax[i].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    if curveBx is not None:
        ax[0].legend()
    
    fig.suptitle(title,fontweight='bold')
    fig.tight_layout()

    if save:
        plt.savefig(figname, dpi=290)
        plt.close()
    else:
        plt.show()





# implementation of generating synthetic data as a closed function
def generate_synthetic_data(m_true, freq, B0_vec, noise_std=2.e-10, CA=100.):
    m_true = np.array(m_true, dtype=float)
    # Flyby from -5000km to 5000km along X, Closest Approach 100km at (0, 0, 1660 km)
    x = np.linspace(-5000, 5000, 100) * 1000
    y = np.ones_like(x) * 100.e+03
    z = np.ones_like(x) * (1560 + CA) * 1000
    r_pos = np.stack([x, y, z], axis=1)
    
    M_complex = model_to_dipole_moment(m_true, freq, B0_vec)
    # Real part represents the instantaneous induction response
    B_ind = calculate_B_dipole_final(r_pos, M_complex.real)
    #print(B_ind)
    
    # Add Gaussian Noise
    B_obs = B0_vec + B_ind + np.random.normal(0, noise_std, B_ind.shape)
    return r_pos, B_obs












def perform_gauss_newton(r_pos, B_obs, m_init, freq, B0_vec, max_iter=10, pseudo_inv=False,tikhonov_reg=False, do_col_norms=False):
    m = np.array(m_init, dtype=float)
    d_obs = B_obs.flatten() # Length 3N
    
    # Finite difference steps for Jacobian
    eps = np.array([0.1, 0.1, 0.01]) # 100m for radii, 0.01 for sigma
    
    print(f"{'Iter':<5} | {'r1':<10} | {'r0':<10} | {'sigma':<10} | {'Residual'}")
    print("-" * 60)

    for i in range(max_iter):
        # Forward Model G(m)
        M_curr = model_to_dipole_moment(m, freq, B0_vec)
        G_m = calculate_B_dipole_final(r_pos, M_curr.real).ravel()

        # 2. Residual (3N,)
        residual = d_obs - G_m
        res_norm = np.linalg.norm(residual)

        # 3. Jacobian (3N, 3)
        J = np.zeros((len(d_obs), 3))
        for j in range(3):
            m_plus = m.copy()
            m_plus[j] += eps[j]
            M_plus = model_to_dipole_moment(m_plus, freq, B0_vec)
            
            # Perturbed Prediction (N, 3) -> (3N,)
            G_plus = calculate_B_dipole_final(r_pos, M_plus.real).ravel()
            
            # Derivative: (G_plus_vector - G_m_vector) / epsilon
            J[:, j] = (G_plus - G_m) / eps[j]


        print(f"{i:<5} | {m[0]:<10.2f} | {m[1]:<10.2f} | {m[2]:<10.4f} | {res_norm:.2e}")
        if res_norm < 1e-12: break # Convergence

        if do_col_norms:
            # Inside the loop, after calculating J:
            # 1. Calculate the norm of each column
            col_norms = np.linalg.norm(J, axis=0)
            col_norms[col_norms == 0] = 1.0 # Avoid div by zero

            # 2. Scale the Jacobian
            J_scaled = J / col_norms
        
        
            
        # Gauss-Newton Update: delta_m = (J^T J)^-1 J^T * residual
        try:
            if pseudo_inv == True:
                if do_col_norms:
                    # Using lstsq is safer than solve when you have these singular values
                    v_scaled, _, _, _ = np.linalg.lstsq(J_scaled, residual, rcond=1e-15)

                    #un-scale the step to get back to physical units
                    step = v_scaled / col_norms
                else:
                    step, residuals, rank, s = np.linalg.lstsq(J, residual, rcond=1e-12)
                    #print('singular values:',s)

            # Adding small regularization to prevent singular matrix
            elif tikhonov_reg == True:
                alpha = 1e-8 
                H = J.T @ J + alpha * np.diag(np.diag(J.T @ J)) # Scale alpha to the matrix magnitude
                step = np.linalg.solve(H, J.T @ residual)
            else:
                if do_col_norms:
                    v_scaled = np.linalg.solve(J_scaled.T @ J_scaled + 1e-15 * np.eye(3), J_scaled.T @ residual)
                    step = v_scaled / col_norms
                else:
                    step = np.linalg.solve(J.T @ J + 1e-15 * np.eye(3), J.T @ residual)

                # Limit radius change to 50km and sigma change to 2.0 per iteration
            step[0:2] = np.clip(step[0:2], -50, 50)
            step[2] = np.clip(step[2], -1, 1)
            
            m += step
            
            # 5. PHYSICAL BOUNDS (The 'Guardrails')
            m[0] = np.clip(m[0], 500, m[1]-1) # r1 must be less than r0
            m[1] = np.clip(m[1], 1500, 1565) # r0 must be near surface
            m[2] = max(m[2], 0.01)           # sigma must be positive

        except np.linalg.LinAlgError:
            print("Singular Matrix! Inversion failed.")
            break
            
    return m





from scipy.optimize import least_squares

def residuals_wrapper(m_inv, r_pos, B_obs, freq, B0_vec):
    """
    The bridge between LM algorithm and Zimmer math.
    m_inv: [r1, r0, log10_sigma]
    """
    # 1. Unpack and transform back to physical space
    # Inverting for log10(sigma) makes the Jacobian columns comparable!
    r1, r0, log_sigma = m_inv
    sigma = 10**log_sigma
    
    # 2. Chain of functions
    m_phys = [r1, r0, sigma]
    M_curr = model_to_dipole_moment(m_phys, freq, B0_vec)
    G_m = calculate_B_dipole_final(r_pos, M_curr.real)
    
    # 3. Return the 1D residual vector
    # Scipy handles the squaring and summing internally
    return (G_m - B_obs).ravel()

def run_scipy_inversion(r_pos, B_obs, m_guess, freq, B0_vec):
    # m_guess = [r1, r0, sigma] -> convert to [r1, r0, log10_sigma]
    x0 = [m_guess[0], m_guess[1], np.log10(m_guess[2])]
    
    # Define physical bounds (Crucial for Europa!)
    # r1 must be > 0, r0 must be <= 1560, log_sigma between -2 and 1
    lower_bounds = [0.0, 1000.0, -3.0]
    upper_bounds = [1560.0, 1560.0, 2.0]
    
    res = least_squares(
        residuals_wrapper, 
        x0, 
        args=(r_pos, B_obs, freq, B0_vec),
        method='trf',         # Trust Region Reflective (better than pure LM for bounds)
        loss='soft_l1',       # Robust to outliers/noise
        x_scale='jac',        # Automatically scales parameters based on Jacobian
        ftol=1e-8, 
        xtol=1e-8,
        bounds=(lower_bounds, upper_bounds)
    )
    
    # Convert back to physical units
    m_final = res.x
    return [m_final[0], m_final[1], 10**m_final[2]]













######################### MCMC FUNCTIONS #############################
import numpy as np

def log_prior(m):
    """
    Defines the physical 'box' the parameters must stay inside.
    m = [thickness_km, r0_km, log10_sigma]
    Returns 0.0 if inside bounds, -infinity if outside.
    """
    r1, r0, log_sigma = m
    thickness = r0-r1
    # Define physical boundaries for Europa
    if (1.0 < thickness < 300.0) and \
       (1430.0 < r0 < 1550.0) and \
       (-2.0 < log_sigma < 2.):
        return 0.0
    return -np.inf

def log_likelihood(m, r_pos_km, B_obs, freq, B0_vec, noise_std):
    # m = [r1, r0, log_sigma]
    r1, r0, log_sigma = m
    sigma = 10**log_sigma
    
    # Forward model
    M_curr = model_to_dipole_moment([r1, r0, sigma], freq, B0_vec)
    B_pred = calculate_B_dipole_final(r_pos_km, M_curr.real).ravel()
    
    # Residuals in nT
    residuals = B_obs.ravel() - B_pred
    
    # chi2 = sum( (res/noise)^2 )
    return -0.5 * np.sum((residuals **2)/ noise_std)

def run_mcmc(r_pos_km, B_obs, m_init, freq, B0_vec, noise_std, num_steps=10000):
    """
    The Metropolis-Hastings Random Walk.
    """
    m_curr = np.array(m_init, dtype=float)
    
    # 1. INITIAL DIAGNOSTIC
    lp_curr = log_prior(m_curr)
    if lp_curr == -np.inf:
        raise ValueError("Starting point m_init is OUTSIDE the prior bounds! Check r1, r0, or log_sigma.")
    
    ll_curr = log_likelihood(m_curr, r_pos_km, B_obs, freq, B0_vec, noise_std)
    
    # If LL is NaN, there's a math error in your Zimmer/Bessel function
    if np.isnan(ll_curr):
        raise ValueError("Log-likelihood is NaN. Check for division by zero in forward model.")

    # [thickness_jump, r0_jump, log_sigma_jump]
    # Reduce these if acceptance is too low
    step_sizes = np.array([0.08, 0.04, 0.002])
    
    chain = np.zeros((num_steps, 3))
    accept_count = 0

    for i in range(num_steps):
        m_prop = m_curr + np.random.randn(3) * step_sizes
        
        lp_prop = log_prior(m_prop)
        
        if lp_prop == -np.inf:
            # Automatic rejection: outside prior
            ll_prop = -np.inf
        else:
            ll_prop = log_likelihood(m_prop, r_pos_km, B_obs, freq, B0_vec, noise_std)

        # 2. THE PROTECTED DECISION
        # If both are -inf, we just reject the proposal
        if ll_prop == -np.inf and ll_curr == -np.inf:
            accept = False
        else:
            # log(P_prop / P_curr)
            log_accept_ratio = (ll_prop + lp_prop) - (ll_curr + lp_curr)
            
            # Acceptance probability alpha = min(1, exp(log_ratio))
            # We use log(rand) < log_ratio to avoid overflow of exp()
            if np.log(np.random.rand()) < log_accept_ratio:
                m_curr = m_prop
                ll_curr = ll_prop
                lp_curr = lp_prop
                accept_count += 1

        chain[i, :] = m_curr
        if i % 100 == 0:
            print(f"Step {i} | LL_curr: {ll_curr:.2f} | Last Delta_LL: {ll_prop - ll_curr:.2f}")
        
    print(f"MCMC Finished. Acceptance Rate: {accept_count / num_steps:.3f}")
    return chain



## Plotting MCMC out
def plot_mcmc_traces(chain, param_names=["r1", "r0", "log10_sigma"]):
    fig, axes = plt.subplots(len(param_names), 1, figsize=(10, 8), sharex=True)
    
    for i, name in enumerate(param_names):
        axes[i].plot(chain[:, i], color='black', alpha=0.7)
        axes[i].set_ylabel(name)
        axes[i].grid(True, alpha=0.3)
        
    axes[-1].set_xlabel("Step Number")
    plt.tight_layout()
    plt.show()



def plot_mcmc_corner(chain, burn_in=1000, param_names=["r1", "r0", "log10_sigma"]):
    # Discard the initial steps where the algorithm was still 'finding' the solution
    clean_chain = chain[burn_in:, :]
    
    num_params = len(param_names)
    fig, axes = plt.subplots(num_params, num_params, figsize=(10, 10))
    
    for i in range(num_params):
        for j in range(num_params):
            if i == j:
                # 1D Histogram on the diagonal
                axes[i, j].hist(clean_chain[:, i], bins=30, color='skyblue', edgecolor='black')
                axes[i, j].set_title(param_names[i])
            elif i > j:
                # 2D Scatter plot for correlations
                axes[i, j].scatter(clean_chain[:, j], clean_chain[:, i], s=1, alpha=0.2, color='navy')
                axes[i, j].set_xlabel(param_names[j])
                axes[i, j].set_ylabel(param_names[i])
            else:
                # Hide the upper triangle
                axes[i, j].axis('off')
                
    plt.tight_layout()
    plt.show()