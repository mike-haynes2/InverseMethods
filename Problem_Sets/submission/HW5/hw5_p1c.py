import numpy as np
import scipy
import math as m
import scipy.linalg as la
import matplotlib.pyplot as plt


zline = np.loadtxt('HW5_data/zline.txt')



def roundup_ceil(x):
  # Divide by 10, round up, then multiply by 10
  return m.ceil(x / 10) * 10

def minimum_norm_inversion(G, d, method=0):
    """
    Computes the minimum L2 norm solution for an underdetermined system d = Gm.
    """
    if method==0:
    # Option 1: The explicit analytical formula (can be unstable for ill-conditioned G)
        G_transpose = G.T
        G_G_transpose = np.dot(G, G_transpose)
        m_est = np.dot(G_transpose, la.solve(G_G_transpose, d))
    if method==1:    
    # Option 2: The SVD/Pseudoinverse approach (Numerically stable and preferred)
        G_pinv = la.pinv(G)
        m_est = np.dot(G_pinv, d)
    else:
        raise(ValueError,"Error: method must be equal to 0 or 1")

    return m_est


def L1_inversion(G, d, dx=1.0):
    """
    Solves the UDLS problem d = Gm by minimizing the first derivative norm ||Lm||.
    """
    N, M = G.shape
    
    # 1. Construct the First-Difference Matrix L ((M-1) x M)
    # Using np.eye and offsets to create the -1, 1 pattern
    L = (np.eye(M-1, M, k=1) - np.eye(M-1, M)) / dx
    
    # 2. Compute the Roughness Kernal (L^T * L)
    # We add a tiny 'epsilon' to the diagonal to ensure invertibility (Tikhonov-lite)
    LtL = L.T @ L + 1e-10 * np.eye(M)
    
    # 3. Apply the Analytical Formula
    # Let inv_LtL_Gt = (L^T L)^-1 * G^T
    inv_LtL_Gt = la.solve(LtL, G.T)
    
    # Inner term: G * (L^T L)^-1 * G^T
    inner_term = G @ inv_LtL_Gt
    
    # Solve for Lagrange multipliers (intermediate step)
    m_est = inv_LtL_Gt @ la.solve(inner_term, d)
    
    return m_est,L



def plot_zline_dat(noise_floor,show_err=True):
    plt.plot(zline[:,0],zline[:,1], lw=2,color='teal',label='zline data')

    if show_err: plt.errorbar(zline[:,0],zline[:,1],xerr=0.,yerr=(noise_floor+np.abs(zline[:,2])), fmt='none',capsize=3., lw=2.8, color='purple',label='data error')
    
    plt.xlabel('Northing track [m]')
    plt.ylabel('Vertical $E$ field [$\mu$V/m]')

    plt.grid()
    plt.legend()

    plt.show()
    return None



def construct_G_interpolation(x_model, x_obs):
    """
    Constructs the forward operator G mapping a 1D model mesh to arbitrary observation points.
    
    Parameters:
    x_model (ndarray): 1D array of strictly increasing model grid coordinates.
    x_obs (ndarray): 1D array of observation coordinates.
    
    Returns:
    G (ndarray): The N x M interpolation matrix.
    """
    M = len(x_model)
    N = len(x_obs)
    G = np.zeros((N, M))
    
    # Find the indices of the model nodes immediately to the right of each observation
    # side='right' ensures that if x_obs exactly equals a node, it is placed correctly
    right_indices = np.searchsorted(x_model, x_obs, side='right')
    
    for i in range(N):
        idx_right = right_indices[i]
        
        # Handle edge cases: if observation is exactly on the last node
        if idx_right == M:
            if x_obs[i] == x_model[-1]:
                G[i, -1] = 1.0
                continue
            else:
                raise ValueError(f"Observation {x_obs[i]} is outside the model mesh bounds.")
                
        idx_left = idx_right - 1
        
        # Coordinates of the bracketing nodes
        x_L = x_model[idx_left]
        x_R = x_model[idx_right]
        x_i = x_obs[i]
        
        # Calculate fractional weights
        dx = x_R - x_L
        w_left = (x_R - x_i) / dx
        w_right = (x_i - x_L) / dx
        
        # Populate the G matrix
        G[i, idx_left] = w_left
        G[i, idx_right] = w_right
        
    return G


# print(np.average(zline[:,2]))

# noise floor specified in the paper
noise=10.

plot_zline_dat(noise_floor = noise)

delx_vals = [zline[i+1,0] - zline[i,0] for i in range(len(zline[:,0])-1) ]
dx_avg = sum(delx_vals)/len(delx_vals)

xmin  = zline[0,0]
xmax  = zline[-1,0]

xarr_coarse = np.arange(start=xmin,stop=xmax,step=40.)











xarr_fine = np.arange(start=xmin,stop=roundup_ceil(xmax),step=10.)



#print(xarr_fine,zline[:,0])

G_UDLS = construct_G_interpolation(xarr_fine,zline[:,0])
#print(G_UDLS)

G_UDLS_pinv = la.pinv(G_UDLS)
#G_UDLLS_inv = np.linalg.inv(G_UDLS)
# minimum norm solution:
m_UDLS_L0 = G_UDLS_pinv @ zline[:,1]



res_L0 = np.abs(G_UDLS @ m_UDLS_L0 - zline[:,1]) ** 2.


m_UDLS_L1, L1 = L1_inversion(G_UDLS, zline[:,1], dx=0.5)




res_L1 = np.abs(G_UDLS @ m_UDLS_L1 - zline[:,1]) ** 2.










def evaluate_regularization_sweep(G, d, L, lambdas):
    """
    Sweeps through various lambda values to produce a suite of models.
    """
    N = len(d)
    results = []
    
    # Pre-calculate G.T @ G and L.T @ L for efficiency
    GtG = G.T @ G
    LtL = L.T @ L
    Gtd = G.T @ d

    for lam in lambdas:
        # 1. Solve the regularized system: (GtG + lambda*LtL)m = Gtd
        # Using la.solve is generally more stable than explicit inversion
        A = GtG + lam * LtL
        m_curr = la.solve(A, Gtd) # 'pos' for positive definite
        
        # 2. Calculate Metrics
        misfit_vec = G @ m_curr - d
        misfit_norm = np.linalg.norm(misfit_vec)
        rms_misfit = misfit_norm / np.sqrt(N)
        model_norm = np.linalg.norm(L @ m_curr)
        m_norm = np.linalg.norm(m_curr)
        
        results.append({
            'lambda': lam,
            'm_vec': m_curr,
            'rms': rms_misfit,
            'model_norm': model_norm,
            'true_norm': m_norm
        })
        
    return results

# --- Example Workflow ---

# Assuming G, d, and L (first-difference) are already defined from previous steps
lambdas = np.logspace(-5, 5, 30) # Sweep from 10^-4 to 10^6
sweep_results = evaluate_regularization_sweep(G_UDLS, d=zline[:,1], L=L1, lambdas=lambdas)

# Extract data for plotting
rms_vals = [r['rms'] for r in sweep_results]
norm_vals = [r['model_norm'] for r in sweep_results]
true_norm_vals = [r['true_norm'] for r in sweep_results]
log_lambdas = np.log10([r['lambda'] for r in sweep_results])


target_rms = noise 
idx_target = np.argmin([abs(r['rms'] - target_rms) for r in sweep_results])

m_target = sweep_results[idx_target]['m_vec']
lam_target = sweep_results[idx_target]['lambda']
rms_target = sweep_results[idx_target]['rms']
m_norm_target = sweep_results[idx_target]['model_norm']



## plot the L-curve and the lambda parameter tradeoff (saturation)

plt.subplot(1, 2, 1)
plt.loglog(rms_vals, norm_vals, 'b-o', markersize=4, label='($||\mathbb{L}_1⋅ \mathbf{m}||$, Roughness)')
plt.loglog(rms_vals, true_norm_vals, 'y-o', markersize=4, label='($||\mathbf{m}||$, magnitude)')
plt.scatter(rms_target,m_norm_target,marker='*',s=75,color='teal',zorder=10,label='target RMS parameters')
plt.xlabel("RMS Misfit")
plt.ylabel("Model Norm")
plt.title("L-Curve")
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.5)


plt.subplot(1, 2, 2)
plt.loglog(lambdas, rms_vals, color='salmon')
plt.axhline(y=noise, color='k', linestyle='--', label='Target RMS')
plt.scatter(lam_target,rms_target,marker='*',s=75,color='teal',zorder=10,label='target RMS parameters')
plt.xlabel("Lagrange Multiplier ($\lambda$)")
plt.ylabel("RMS Misfit")
plt.title("Acheived RMS misfit (regularization effectiveness) versus $\lambda$")
plt.legend()
plt.tight_layout()
plt.show()
plt.close()



## plot "winner" soln: the nearest to the target misfit via RMS comparison

plt.plot(zline[:,0],zline[:,1], lw=4,color='black',label='zline data')

plt.errorbar(zline[:,0],zline[:,1],xerr=0.,yerr=(noise+np.abs(zline[:,2])), fmt='none',capsize=3., lw=2.8, color='purple',label='data error + noise floor (Constable et al. 2018)')

plt.plot(xarr_fine, m_target, lw=2., color='teal', label = f"model: L1 norm minimzing, target misfit")

plt.xlabel('Northing track [m]')
plt.ylabel('Vertical $E$ field [$\mu$V/m]')

plt.title("Problem 1(c): ($||\mathbb{L}_1 ⋅ \mathbf{m}||$) minimizing model, target misfit = 10 [$\mu$V/m],"+f"$\lambda=${lam_target:.2e}")

plt.grid()
plt.legend()

plt.show()
plt.close()


## plot ALL solutions

plt.plot(zline[:,0],zline[:,1], lw=4,color='black',label='zline data')

plt.errorbar(zline[:,0],zline[:,1],xerr=0.,yerr=(noise+np.abs(zline[:,2])), fmt='none',capsize=3., lw=2.8, color='purple',label='data error + noise floor (Constable et al. 2018)')

for i in range(len(rms_vals)):
    plt.plot(xarr_fine, sweep_results[i]['m_vec'], lw=2.)

plt.xlabel('Northing track [m]')
plt.ylabel('Vertical $E$ field [$\mu$V/m]')

plt.title("Problem 1(c): All model solutions overlayed: "+"$\lambda\in[10^{-5},10^{5}]$")

plt.grid()
plt.legend()

plt.show()
plt.close()






## tabulate output values
import csv
csv_file_path = 'hw5_p1c.csv'

#output_dict = {key: value for key,value in sweep_results.items() if key not in ['m_vec']}
output_dict = [{k: v for k, v in d.items() if k != 'm_vec'} for d in sweep_results]

fieldnames = ['lambda',
            'rms',
            'model_norm',
            'true_norm'] # Explicitly define the column order
title_fieldnames = ['$\lambda$',
            'rms misfit',
            'model norm ($||\mathbb{L}_1 ⋅ \mathbf{m}||$)',
            'model magnitude ($||\mathbf{m}||$) '] 

with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)

    # Write the header row
    writer.writeheader() #

    # Write the data rows
    writer.writerows(output_dict) #





