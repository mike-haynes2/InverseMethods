import numpy as np
import scipy

# Column 1: y values
y = np.array([
    0.0250, 0.0750, 0.1250, 0.1750, 0.2250, 0.2750, 0.3250, 0.3750, 0.4250, 0.4750,
    0.5250, 0.5750, 0.6250, 0.6750, 0.7250, 0.7750, 0.8250, 0.8750, 0.9250, 0.9750
])

# Column 2: d(y) values
d_y = np.array([
    0.2388, 0.2319, 0.2252, 0.2188, 0.2126, 0.2066, 0.2008, 0.1952, 0.1898, 0.1846,
    0.1795, 0.1746, 0.1699, 0.1654, 0.1610, 0.1567, 0.1526, 0.1486, 0.1447, 0.1410
])


delt = 1.e-04 * (np.sqrt(len(d_y)) / np.linalg.norm(d_y))
print(delt)

# N = 20 based on Table 3.2
N = len(y)
dx = 1.0 / N
x_edges = np.linspace(0, 1, N + 1)

# Function to compute the integral G[i,j] analytically
def kernel_integral(y, x_start, x_end):
    def indef_int(val):
        return - (val * np.exp(-y * val) / y) - (np.exp(-y * val) / (y**2))
    
    # Handle the y=0 case if necessary (though Table 3.2 starts at 0.025)
    if y == 0:
        return 0.5 * (x_end**2 - x_start**2)
    else:
        pass
    return indef_int(x_end) - indef_int(x_start)

# Initialize and fill G matrix
G = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        G[i, j] = kernel_integral(y[i], x_edges[j], x_edges[j+1])

# Solve the system Gm = d
# Note: This system is ill-conditioned (Fredholm IFK), so simple inv is risky.
m_sol = np.linalg.solve(G, d_y)
m_dag = scipy.linalg.pinv(G) @ d_y

print(f"G Matrix Shape: {G.shape}")
print(f"Condition Number: {np.linalg.cond(G):.2e}")

print(m_sol,m_dag)


U, s, Vh = np.linalg.svd(G)
Prats = np.abs(U.T @ d_y) / s

import matplotlib.pyplot as plt

plt.scatter(np.arange(len(Prats)),Prats, marker='o', color='navy')
plt.title('Picard Ratios', fontweight='bold')
plt.yscale('log')
plt.grid()
plt.xlabel('index n')
plt.ylabel('ratio')
plt.show()
# plt.savefig('p3.5c_Picard.png', dpi=250)
# plt.close()



def tsvd(A, k=None, full_matrices=False):
    """
    Custom wrapper for Truncated SVD.
    Returns Gdagger, V, and U
    """
    # 1. Perform standard SVD
    U, s, Vh = scipy.linalg.svd(A, full_matrices=full_matrices)
    
    # 2. Determine truncation point
    # If k is None, keep all components
    if k is None:
        k = len(s)
    
    # 3. Truncate components
    Uk = U[:, :k]   # Keep first k columns
    sk = s[:k]      # Keep first k singular values
    Vhk = Vh[:k, :] # Keep first k rows (since Vh is V-Hermitian)
    
    return (Vhk.T @ (np.linalg.inv(np.diag(sk)) @ Uk.T)), Vhk.T, Uk


Gdagg, V, U = tsvd(G, k=4, full_matrices=True)
mdagg = Gdagg @ d_y
plt.plot(y, mdagg, color='teal')
plt.title('Recovered model $m(x)$')
plt.xlabel('$y$')
plt.ylabel('$m$')
plt.grid()
plt.show()
#plt.savefig('p5c_m.png')

#print(f"Condition Number: {np.linalg.cond(np.linalg.inv(Gdagg)):.2e}")





##############################################################################################################
###################################################### HW4 ###################################################
##############################################################################################################

## part (a): demonstrate simple collocation discretization by depicting G
im = plt.imshow(G, cmap='CMRmap')
plt.title(r'$\mathbb{G}$ matrix from simple collocation discretization', fontweight='bold')
plt.xlabel('model space')
plt.ylabel('data space')
plt.colorbar(im)
plt.show()
# plt.savefig('p4.2a.png', dpi=280)
# plt.close()



## part (b): determine reasonable parameter value for delta
delta = 1.e-04
delta = delt



from SVD_tikhonov_dampedLS import SVD_tikhonov_dampedLS
from gen_Lcurve import calc_L_coordinates


## part (c): 
# define array of regularization parameter values to sweep through 
alphas = np.logspace(-6,-1, 50)
Lmodel = []
Ldata = []
gcurve = []


for alph in alphas:
    # run the tikhonov (0th order regularized) solution method
    msharp0, Gsharp0, Rm0, Rd0, discrepancy = SVD_tikhonov_dampedLS(G, d_y, a=alph, order=0)
    # extract the values of the regularization hyperparameters
    L_coords0, g0 = calc_L_coordinates(G, msharp0, d_y, Gsharp0, order=0)
    Lmodel.append(L_coords0[0])
    Ldata.append(L_coords0[1])
    gcurve.append(g0)



alpha0_discrepancy = discrepancy
alpha0_L = alphas[np.abs(np.array(Ldata) - 1.e-04).argmin()]       #TBD by looking at plot
alpha0_g = alphas[np.argmin(gcurve)]

print('\nZeroth order:')
print(alpha0_discrepancy, alpha0_L, alpha0_g)

plt.loglog(Ldata,Lmodel, color='salmon', lw=2)
plt.scatter(Ldata[np.abs(np.array(Ldata) - 1.e-04).argmin()], Lmodel[np.abs(np.array(Ldata) - 1.e-04).argmin()], marker='*', color='red', label=f'$α$ = {alpha0_L:.2e}; L-curve', s=100, zorder=10)
plt.scatter(Ldata[np.argmin(gcurve)], Lmodel[np.argmin(gcurve)], marker='*', color='navy', label=f'$α$ = {alpha0_g:.2e}; GCV', s=100, zorder=10)
plt.grid()
plt.title('L-Curve for Zeroth-order Tikhonov Regularization')
plt.legend()
plt.show()
# plt.savefig('L_zero_p4.2c.png', dpi=250)
# plt.close()

plt.loglog(alphas,gcurve,color='teal')

plt.grid()
plt.title('L-Curve for Zeroth-order Tikhonov Regularization')
plt.legend()
plt.show()



## calculate "final" solution for this order regularization
msharp0L, Gsharp0L, Rm0L, Rd0L, discrepancy = SVD_tikhonov_dampedLS(G, d_y, a=alpha0_L, order=0)
msharp0g, Gsharp0g, Rm0g, Rd0g, discrepancy = SVD_tikhonov_dampedLS(G, d_y, a=alpha0_g, order=0)
msharp0d, Gsharp0d, Rm0d, Rd0d, discrepancy = SVD_tikhonov_dampedLS(G, d_y, a=delta, order=0)










## part (d): 
# define array of regularization parameter values to sweep through 
alphas = np.logspace(-6,-1, 50)
Lmodel1 = []
Ldata1 = []
gcurve1 = []


for alph in alphas:
    # run the tikhonov (0th order regularized) solution method
    msharp1, Gsharp1, Rm1, Rd1, L1, discrepancy1 = SVD_tikhonov_dampedLS(G, d_y, a=alph, order=1)
    # extract the values of the regularization hyperparameters
    L_coords1, g1 = calc_L_coordinates(G, msharp1, d_y, Gsharp1, L=L1, order=1)
    Lmodel1.append(L_coords1[0])
    Ldata1.append(L_coords1[1])
    gcurve1.append(g1)




epsVal = 9.9e-05

alpha1_discrepancy = discrepancy1
alpha1_L = alphas[np.abs(np.array(Ldata1) - epsVal).argmin()]       #TBD by looking at plot
alpha1_g = alphas[np.argmin(gcurve1)]

print('\nFirst order:')
print(alpha1_discrepancy, alpha1_L, alpha1_g)

## calculate "final" solution for this order regularization
msharp1g, Gsharp1g, Rm1g, Rd1g, L1g, discrepancy1g = SVD_tikhonov_dampedLS(G, d_y, a=alpha1_g, order=1)
msharp1L, Gsharp1L, Rm1L, Rd1L, L1L, discrepancy1L = SVD_tikhonov_dampedLS(G, d_y, a=alpha1_L, order=1)
msharp1d, Gsharp1d, Rm1d, Rd1d, L1d, discrepancy1d = SVD_tikhonov_dampedLS(G, d_y, a=delta, order=1)


plt.loglog(Ldata1,Lmodel1, color='salmon', lw=2)
plt.scatter(Ldata1[np.abs(np.array(Ldata1) - epsVal).argmin()], Lmodel1[np.abs(np.array(Ldata1) - epsVal).argmin()], marker='*', color='red', label=f'$α$ = {alpha1_L:.2e}; L-curve', s=100, zorder=10)
plt.scatter(Ldata1[np.argmin(gcurve1)], Lmodel1[np.argmin(gcurve1)], marker='*', color='navy', label=f'$α$ = {alpha1_L:.2e}; GCV', s=100, zorder=10)
plt.grid()
plt.title('L-Curve for First-order Tikhonov Regularization')
plt.legend()
plt.show()
# plt.savefig('L_1_p4.2d.png', dpi=250)
# plt.close()











## part (e): 
# define array of regularization parameter values to sweep through 
alphas = np.logspace(-6,-1, 50)
Lmodel2 = []
Ldata2 = []
gcurve2 = []


for alph in alphas:
    # run the tikhonov (0th order regularized) solution method
    msharp2, Gsharp2, Rm2, Rd2, L2, discrepancy2 = SVD_tikhonov_dampedLS(G, d_y, a=alph, order=2)
    # extract the values of the regularization hyperparameters
    L_coords2, g2 = calc_L_coordinates(G, msharp2, d_y, Gsharp2, L=L2, order=2)
    Lmodel2.append(L_coords2[0])
    Ldata2.append(L_coords2[1])
    gcurve2.append(g2)




epsVal = 1.e-04

alpha2_discrepancy = discrepancy2
alpha2_L = alphas[np.abs(np.array(Ldata2) - epsVal).argmin()]       #TBD by looking at plot
alpha2_g = alphas[np.argmin(gcurve2)]

print('\Second order:')
print(alpha2_discrepancy, alpha2_L, alpha2_g)

## calculate "final" solution for this order regularization
msharp2g, Gsharp2g, Rm2g, Rd2g, L2g, discrepancy2g = SVD_tikhonov_dampedLS(G, d_y, a=alpha2_g, order=2)
msharp2L, Gsharp2L, Rm2L, Rd2L, L2L, discrepancy2L = SVD_tikhonov_dampedLS(G, d_y, a=alpha2_L, order=2)
msharp2d, Gsharp2d, Rm2d, Rd2d, L2d, discrepancy2d = SVD_tikhonov_dampedLS(G, d_y, a=delta, order=2)


plt.loglog(Ldata2,Lmodel2, color='salmon', lw=2)
plt.scatter(Ldata2[np.abs(np.array(Ldata2) - epsVal).argmin()], Lmodel2[np.abs(np.array(Ldata2) - epsVal).argmin()], marker='*', color='red', label=f'$α$ = {alpha2_L:.2e}; L-curve', s=100, zorder=10)
plt.scatter(Ldata2[np.argmin(gcurve2)], Lmodel2[np.argmin(gcurve2)], marker='*', color='navy', label=f'$α$ = {alpha2_L:.2e}; GCV', s=100, zorder=10)
plt.grid()
plt.title('L-Curve for Second-order Tikhonov Regularization')
plt.legend()
plt.show()
# plt.savefig('L_2_p4.2e.png', dpi=250)
# plt.close()
















## part (f): analyze the resolution of solutions, identify solutions in the inverse model that are not real, describe size/loc

colors = ['teal', 'cyan', 'magenta']
styles = ['-','-.',':']
# Organizing models into groups for each row (order 0, 1, 2)
# format: [m_L, m_g, m_d]
model_groups = [
    [msharp0L, msharp0g, msharp0d],
    [msharp1L, msharp1g, msharp1d],
    [msharp2L, msharp2g, msharp2d]
]
resolutions = [Rm0L, Rm1g, Rm2g]
labels = ['L-curve', 'GCV', 'Discrepancy']

fig, axes = plt.subplots(3, 2, figsize=(7.5, 9))

for i, (m_group, Rm) in enumerate(zip(model_groups, resolutions)):
    # --- Left Column: Discovered Models ---
    ax_m = axes[i, 0]
    
    # Plot the three Tikhonov variations
    k=0
    for m_curve, label in zip(m_group, labels):
        ax_m.plot(y, m_curve, label=f'Tikhonov ({label})', lw=2, color=colors[k], ls=styles[k])
        k+=1
    # Plot reference TSVD
    ax_m.plot(y, mdagg, 'k--', linewidth=1.5, label='truncated SVD (mdagg)')
    
    ax_m.set_xlabel('$y$')
    ax_m.set_ylabel('$m(x)$')
    ax_m.legend(fontsize='small', loc='best')
    ax_m.grid(True, alpha=0.3)
    if i == 0: ax_m.set_title('Discovered model', fontweight='bold')

    # --- Right Column: Resolution Matrices ---
    ax_r = axes[i, 1]
    im = ax_r.imshow(Rm, cmap='turbo', interpolation='nearest')
    cbar = plt.colorbar(im, ax=ax_r)
    cbar.set_label('resolution')
    if i == 0: ax_r.set_title('model Resolution matrix', fontweight='bold')

plt.suptitle(r"$\bf{Tikhonov\ Regularized\ models:\ Comparison\ of\ order\ (0-2)}$", fontsize=16)
#plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.show()
#plt.savefig('p4.2f_soln.png', dpi=300)






## old didn't plot all
# models = [[msharp0L], msharp1, msharp2]
# resolutions = [Rm0, Rm1, Rm2]
# fig, axes = plt.subplots(3, 2, figsize=(7.5, 9))

# for i, (m_sharp, Rm) in enumerate(zip(models, resolutions)):
#     # Left Column: Discovered Models
#     ax_m = axes[i, 0]
#     ax_m.plot(y, m_sharp, label='Tikhonov '+str(i), color='teal')
#     ax_m.plot(y, mdagg, 'k--', label='trunc. SVD')
#     ax_m.set_xlabel('$y$')
#     ax_m.set_ylabel('$m(x)$')
#     ax_m.legend()
#     ax_m.grid(True)
#     if i == 0: ax_m.set_title('Discovered model')

#     # Right Column: Resolution Matrices
#     ax_r = axes[i, 1]
#     im = ax_r.imshow(Rm, cmap='turbo')
#     cbar = plt.colorbar(im, ax=ax_r)
#     cbar.set_label('resolution')
#     if i == 0: ax_r.set_title('model Resolution matrix')

# plt.suptitle(r"$\bf{Tikhonov\ Regularized\ models:\ Comparison\ of\ order\ (0-2)}$", fontsize=14)
# #plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# #fig.tight_layout()
# plt.show()