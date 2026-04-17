import numpy as np
from scipy.optimize import least_squares

def invert_cond_LM(sigma_data, T_data, error_threshold=0.05, m_init=None):
    """
    Inverts temperature and conductivity data for a 4-parameter Arrhenius model
    using the Levenberg-Marquardt algorithm.
    
    Parameters:
    -----------
    sigma_data : numpy.ndarray
        1D array of measured electrical conductivities (S/m).
    T_data : numpy.ndarray
        1D array of corresponding temperatures (Kelvin).
    error_threshold : float
        Proportional error threshold (e.g., 0.05 for 5%).
    m_init : list or numpy.ndarray, optional
        Initial guess for the parameters [sigma_0, sigma_1, A_0, A_1].
        Activation energies must be in eV.
        
    Returns:
    --------
    m_opt : numpy.ndarray
        Optimized model parameters [sigma_0, sigma_1, A_0, A_1].
    optimization_result : OptimizeResult
        Full SciPy optimization object containing covariance, residuals, etc.
    """
    
    # Boltzmann constant in eV/K for numerical stability
    k_B = 8.617333262145e-5
    
    # 1. Define the Forward Model
    def forward_model(m, T):
        sigma_0, sigma_1, A_0, A_1 = m
        term_0 = sigma_0 * np.exp(-A_0 / (k_B * T))
        term_1 = sigma_1 * np.exp(-A_1 / (k_B * T))
        return term_0 + term_1

    # 2. Define the Weighted Residual Function
    # least_squares minimizes the sum of squares of the returned array.
    # Returning (data - model) / uncertainty exactly implements chi-squared.
    def compute_residuals(m, T, sigma_measured, threshold):
        sigma_predicted = forward_model(m, T)
        uncertainty = threshold * sigma_measured
        return (sigma_measured - sigma_predicted) / uncertainty

    # 3. Handle Initial Guess
    # If no initial guess is provided, formulate a generic starting point based 
    # on typical mantle olivine values (e.g., small polaron vs ionic conduction).
    if m_init is None:
        # [sigma_0 (low T mech), sigma_1 (high T mech), A_0 (eV), A_1 (eV)]
        m_init = np.array([1e2, 1e5, 1.0, 2.5])
    else:
        m_init = np.array(m_init)

    # 4. Execute Levenberg-Marquardt Inversion
    # method='lm' forces the Levenberg-Marquardt algorithm.
    # Note: 'lm' does not support parameter bounds. If negative conductivities 
    # become an issue, switch method to 'trf' (Trust Region Reflective).
    result = least_squares(
        fun=compute_residuals,
        x0=m_init,
        method='lm',
        args=(T_data, sigma_data, error_threshold)
    )
    
    # 5. Extract optimized parameters
    m_opt = result.x
    
    return m_opt, result