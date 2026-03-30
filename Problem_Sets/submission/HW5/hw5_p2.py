import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

def build_2d_operators(z_nodes, y_nodes, z_obs, y_obs):
    """
    Constructs the sparse 2D roughness matrix (LtL) and the bilinear 
    interpolation forward operator (G).
    """
    Nz = len(z_nodes)
    Ny = len(y_nodes)
    M = Nz * Ny
    N_obs = len(z_obs)
    
    dz = np.diff(z_nodes)[0] # Assuming uniform grid spacing
    dy = np.diff(y_nodes)[0]

    # --- 1. Construct 2D Roughness Operator (L^T L) ---
    # 1D difference operators
    D_z_1d = sp.diags([-1, 1], [0, 1], shape=(Nz-1, Nz)) / dz
    D_y_1d = sp.diags([-1, 1], [0, 1], shape=(Ny-1, Ny)) / dy
    
    # Expand to 2D using Kronecker products
    I_y = sp.eye(Ny)
    I_z = sp.eye(Nz)
    
    Lz = sp.kron(I_y, D_z_1d) # Vertical derivatives
    Ly = sp.kron(D_y_1d, I_z) # Horizontal derivatives
    
    LtL = (Lz.T @ Lz) + (Ly.T @ Ly)

    # --- 2. Construct Forward Operator G (Bilinear Interpolation) ---
    # Find grid indices directly below/left of each observation
    idx_z = np.searchsorted(z_nodes, z_obs, side='right') - 1
    idx_y = np.searchsorted(y_nodes, y_obs, side='right') - 1
    
    # Clip to avoid out-of-bounds if points lie exactly on the upper boundary
    idx_z = np.clip(idx_z, 0, Nz - 2)
    idx_y = np.clip(idx_y, 0, Ny - 2)

    # Fractional distances
    fz = (z_obs - z_nodes[idx_z]) / dz
    fy = (y_obs - y_nodes[idx_y]) / dy
    
    # Bilinear weights
    w00 = (1 - fz) * (1 - fy) # Top-Left
    w10 = fz * (1 - fy)       # Bottom-Left
    w01 = (1 - fz) * fy       # Top-Right
    w11 = fz * fy             # Bottom-Right
    
    # Flattened grid indices for the 4 corners
    node00 = idx_y * Nz + idx_z
    node10 = idx_y * Nz + (idx_z + 1)
    node01 = (idx_y + 1) * Nz + idx_z
    node11 = (idx_y + 1) * Nz + (idx_z + 1)
    
    # Assemble sparse G
    row_idx = np.repeat(np.arange(N_obs), 4)
    col_idx = np.column_stack((node00, node10, node01, node11)).flatten()
    data = np.column_stack((w00, w10, w01, w11)).flatten()
    
    G = sp.csr_matrix((data, (row_idx, col_idx)), shape=(N_obs, M))
    
    return G, LtL

def run_2d_inversion_sweep(zmap, Nz=100, Ny=100, noise_floor=10.0):
    """
    Executes the 2D inversion sweep over lambda.
    zmap format: [z(northing), y(easting), data, error]
    """
    z_obs, y_obs, d_obs, err_obs = zmap[:, 0], zmap[:, 1], zmap[:, 2], zmap[:, 3]
    N = len(d_obs)
    
    # Create the fine mesh
    z_nodes = np.linspace(z_obs.min(), z_obs.max(), Nz)
    y_nodes = np.linspace(y_obs.min(), y_obs.max(), Ny)
    
    # Build Operators
    G, LtL = build_2d_operators(z_nodes, y_nodes, z_obs, y_obs)
    
    # Weighting Matrix W
    sigma_total = np.maximum(err_obs, noise_floor) # Implements the error floor
    W = sp.diags(1.0 / sigma_total)
    
    # Pre-compute weighted matrices for the normal equations
    Gw = W @ G
    dw = W @ d_obs
    GtG = Gw.T @ Gw
    Gtd = Gw.T @ dw
    
    # Define Lambda Sweep
    lambdas = np.logspace(-2, 5, 40)
    results = []
    
    for lam in lambdas:
        # Solve (Gw^T Gw + lam * L^T L) m = Gw^T dw
        A = GtG + lam * LtL
        
        # spsolve is an exact sparse solver. For massive grids (>500x500), 
        # swap this for an iterative solver like spla.cg or spla.lsqr
        m_est = spla.spsolve(A, Gtd) 
        
        # Metrics
        residuals = G @ m_est - d_obs
        weighted_res = (1.0 / sigma_total) * residuals
        rms = np.sqrt(np.mean(weighted_res**2)) # Target is 1.0
        model_norm = np.sqrt(m_est.T @ (LtL @ m_est))
        
        results.append({
            'lambda': lam,
            'm_grid': m_est.reshape((Ny, Nz)).T, # Reshape back to 2D
            'rms': rms,
            'norm': model_norm
        })
        
    return results, z_nodes, y_nodes

# --- Visualization ---
# Assuming 'sweep_results' is the output from run_2d_inversion_sweep()

def plot_results(sweep_results, z_nodes, y_nodes):
    rms_vals = [r['rms'] for r in sweep_results]
    norm_vals = [r['norm'] for r in sweep_results]
    lambdas = [r['lambda'] for r in sweep_results]
    
    # Find the model closest to RMS = 1.0
    best_idx = np.argmin(np.abs(np.array(rms_vals) - noise))
    best_model = sweep_results[best_idx]
    
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. The L-Curve (Colored by Lambda)
    sc = axs[0].scatter(rms_vals, norm_vals, c=np.log10(lambdas), cmap='viridis', zorder=5)
    axs[0].plot(rms_vals, norm_vals, 'k-', alpha=0.3, zorder=4)
    axs[0].axvline(noise, color='r', linestyle='--', label=f'Target RMS=5.0')
    axs[0].scatter(best_model['rms'], best_model['norm'], color='red', s=100, label='Target RMS Model', zorder=6)
    
    axs[0].set_xscale('log')
    axs[0].set_yscale('log')
    axs[0].set_xlabel('RMS Misfit')
    axs[0].set_ylabel('Model Roughness (L1 Tikhonov Norm)')
    axs[0].set_title('2D L-Curve Trade-off')
    axs[0].legend()
    plt.colorbar(sc, ax=axs[0], label='$\lambda$')
    
    # 2. The 2D E-Field Map (Target Model)
    Y, Z = np.meshgrid(y_nodes, z_nodes)
    c_map = axs[1].pcolormesh(Y, Z, -best_model['m_grid'].T, shading='gouraud', cmap='jet', vmin=-1.e+02, vmax=2.e+02)
    axs[1].set_xlabel('Easting [m]')
    axs[1].set_ylabel('Northing [m]')
    axs[1].set_title(f"Reconstructed E-Field (RMS = {best_model['rms']:.2f})")
    plt.colorbar(c_map, ax=axs[1], label='Electric Field Amplitude [$\mu$V/m]')
    
    plt.tight_layout()
    plt.show()
    #plt.savefig('hw5_p3.png', dpi=325)



zmap = np.loadtxt('HW5_data/zmap.txt')

noise = 5.

sweep_results, zgrid, ygrid = run_2d_inversion_sweep(zmap, Nz=100, Ny=100, noise_floor=noise)

plot_results(sweep_results,zgrid,ygrid)