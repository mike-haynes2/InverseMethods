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








# G_ODLS = np.zeros((len(xarr_coarse),len(zline[:,0])))
# G_ODLS = G_ODLS.T

# print(G_ODLS)
# print(np.shape(G_ODLS))



# # normal equations: 
# mod_ODLS = np.linalg.inv(G_ODLS.T @ G_ODLS) @ G_ODLS.T @ zline[:,1]


# plot normal eqn solution
# plt.plot(zline[:,0],zline[:,1], lw=4,color='black',label='zline data')

# plt.errorbar(zline[:,0],zline[:,1],xerr=0.,yerr=(noise+zline[:,2]), fmt='none',capsize=3., lw=2.8, color='purple',label='data error')

# plt.plot(xarr_coarse,mod_ODLS, lw=2., color='teal', label = 'normal eqns')

# plt.xlabel('Northing track [m]')
# plt.ylabel('Vertical $E$ field [$\mu$V/m]')

# plt.grid()
# plt.legend()

# plt.show()
# plt.close()

# since we need to compare arrays of like shape, resend the model to the dimension of the data via the G operator for the normal eqns
# res = np.abs(G_ODLS @ mod_ODLS - zline[:,1]) ** 2.

# plt.scatter(zline[:,0], res, color='teal',zorder=1)
# plt.title('Squared residuals from Normal Equations Inversion')
# plt.xlabel('Northing track [m]')
# plt.ylabel('Squared difference in vertical $E$ field [$\mu$V/m]$^2$')
# plt.grid(alpha=0.5,zorder=0)

# plt.show()


# norm_model_ODLS = np.sqrt(np.dot(mod_ODLS,mod_ODLS))
# assert norm_model_ODLS == np.linalg.norm(mod_ODLS)
# norm_res_ODLS = np.sqrt(np.dot(res,res))



# def get_quadratic_weights(x_data, x_grid):
#     # 1. Find the nearest node for each data point
#     # We want the 'middle' node of our quadratic triplet
#     j = np.searchsorted(x_grid, x_data)
    
#     # Stay away from the very edges to keep a triplet [j-1, j, j+1]
#     j = np.clip(j, 1, len(x_grid) - 2)
    
#     # 2. Identify the three coordinates for each triplet
#     x0, x1, x2 = x_grid[j-1], x_grid[j], x_grid[j+1]
#     x = x_data
    
#     # 3. Calculate Lagrange Weights
#     w0 = ((x - x1) * (x - x2)) / ((x0 - x1) * (x0 - x2))
#     w1 = ((x - x0) * (x - x2)) / ((x1 - x0) * (x1 - x2))
#     w2 = ((x - x0) * (x - x1)) / ((x2 - x0) * (x2 - x1))
    
#     return j, (w0, w1, w2)



xarr_fine = np.arange(start=xmin,stop=roundup_ceil(xmax),step=10.)
print(len(xarr_fine))


#print(xarr_fine,zline[:,0])

G_UDLS = construct_G_interpolation(xarr_fine,zline[:,0])
#print(G_UDLS)

G_UDLS_pinv = la.pinv(G_UDLS)
#G_UDLLS_inv = np.linalg.inv(G_UDLS)
# minimum norm solution:
m_UDLS_L0 = G_UDLS_pinv @ zline[:,1]


plt.plot(zline[:,0],zline[:,1], lw=4,color='black',label='zline data')

plt.errorbar(zline[:,0],zline[:,1],xerr=0.,yerr=(noise+np.abs(zline[:,2])), fmt='none',capsize=3., lw=2.8, color='purple',label='data error')

plt.plot(xarr_fine, m_UDLS_L0, lw=2., color='orange', label = 'zeroth order (magnitude) norm')

plt.xlabel('Northing track [m]')
plt.ylabel('Vertical $E$ field [$\mu$V/m]')

plt.title('Problem 1(b): model norm ($||\mathbf{m}||$) minimizing solution')

plt.grid()
plt.legend()

plt.show()
plt.close()

res_L0 = np.abs(G_UDLS @ m_UDLS_L0 - zline[:,1]) ** 2.


m_UDLS_L1, L1 = L1_inversion(G_UDLS, zline[:,1], dx=0.5)



plt.plot(zline[:,0],zline[:,1], lw=4,color='black',label='zline data')

plt.errorbar(zline[:,0],zline[:,1],xerr=0.,yerr=(noise+np.abs(zline[:,2])), fmt='none',capsize=3., lw=2.8, color='purple',label='data error')

plt.plot(xarr_fine, m_UDLS_L1, lw=2., color='teal', label = 'first order (derivative) norm')

plt.xlabel('Northing track [m]')
plt.ylabel('Vertical $E$ field [$\mu$V/m]')

plt.title('Problem 1(b): model roughness ($||\mathbb{L}_1 ⋅ \mathbf{m}||$) minimizing solution')

plt.grid()
plt.legend()

plt.show()
plt.close()



res_L1 = np.abs(G_UDLS @ m_UDLS_L1 - zline[:,1]) ** 2.


plt.scatter(zline[:,0], res_L0, color='orange',zorder=1)
plt.scatter(zline[:,0], res_L1, color='teal',zorder=2)

plt.title('Squared residuals from norm minimizing Inversions')
plt.xlabel('Northing track [m]')
plt.ylabel('Squared difference in vertical $E$ field [$\mu$V/m]$^2$')
plt.grid(alpha=0.5,zorder=0)

plt.yscale('log')

plt.show()
plt.close()

print("L0-minimizing model norm: ",np.linalg.norm(m_UDLS_L0))
print("L1-minimizing model norm: ",np.linalg.norm(L1 @ m_UDLS_L1))






