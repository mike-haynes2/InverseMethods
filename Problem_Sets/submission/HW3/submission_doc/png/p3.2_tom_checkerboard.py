import numpy as np
import math as m
import scipy

import matplotlib.pyplot as plt







# define G matrix
G = np.array([
    [1, 0, 0, 1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 1, 0, 0, 1],
    [1, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 1],
    [np.sqrt(2), 0, 0, 0, np.sqrt(2), 0, 0, 0, np.sqrt(2)],
    [0, 0, 0, 0, 0, 0, 0, 0, np.sqrt(2)]
])


# checkerboard m_true
m_true = np.ones(9)
m_true[::2] = -1.
m_true_mat = m_true.reshape((int(len(m_true)/3),int(len(m_true)/3)))
print(m_true_mat)

# calculate synthetic dataset to invert
d_synth = G @ m_true

# perform SVD on G matrix operator to motivate solution 
print('Shape of G operator: ', G.shape)

U, s, VT = scipy.linalg.svd(G)
V = VT.T

# calculate dimension sizes and check for consistency
m_row = np.linalg.matrix_rank(U)
assert m_row == len(G[:,0])
n_col = np.linalg.matrix_rank(V)
assert n_col == len(G[0])
p_nonzero = 7
# calculate S matrix
S = scipy.linalg.diagsvd(s,m_row, n_col)

# check S matrix
with np.printoptions(suppress=True, precision=4):
    print(S)

print('m,n,p: ',m_row,n_col,p_nonzero)

# solve the system using normal equations, minimum length least squares solution
# Equations 3.29 - 3.38 in Aster
# since p<n, N(G) is nontrivial, but N(GT) is trivial. UT_p = Uinv_p. VT_p V_p = I_p. m dagger - m lies in N(GT G) = N(G)
# general solution is a sum of m_dagger plus an arbitrary model null space vector m0
# m0 is written as a linear combination of the basis of N(G), obviously.
# equation 3.38 yields:
# GGTinv = np.linalg.inv(G @ G.T)
# GT_GGTinv = G.T @ GGTinv
# m_dagger = GT_GGTinv @ d_synth
# HOWEVER, p is actually 7! there was a very near-zero singular value hiding, so p=7 < m,n and the actual solution is 
# given by path 4 in Aster
# UP = U[:,:p_nonzero]
# m_dagger = (UP @ UP.T )@ d_synth

# compare against SVD method (equations 3.20-3.22 in Aster)
# generate SVD decomp
Up, sp, VpT = scipy.linalg.svd(G, full_matrices=False)
#print(np.shape(Up), np.shape(Vp), Vp[:,:p_nonzero].shape, np.diag(sp).shape)
Sp = np.diag(sp)
Vp = VpT.T

Up_trunc = Up[:,:p_nonzero]
Vp_trunc = Vp[:,:p_nonzero]
Sp_trunc = np.diag(sp[:p_nonzero])

# with np.printoptions(suppress=True, precision=4):
#     print(Vp)
#     print(VpT)
# calculate matrix multiplier, G_dagger (psuedo inverse) from equation (3.22 in Aster)
Gdagger = Vp @ (np.linalg.inv(Sp) @ Up.T) 
m_dagger_SVD = Gdagger @ d_synth

m_dagger_SVD_trunc = (Vp_trunc @ (np.linalg.inv(Sp_trunc) @ Up_trunc.T)) @ d_synth

# calculate Moore-Penrose pseudo-inverse directly
Gdagger_MP = scipy.linalg.pinv(G)
mdagger_MP = Gdagger_MP @ d_synth

print(m_dagger_SVD_trunc, m_dagger_SVD, mdagger_MP)






# Visualize the matrix
N = 3

# 1. Prepare the Model Vectors (m) - Reshaped to 3x3
# (Using your variables from previous steps)
m_svd_grid = m_dagger_SVD.reshape((N, N))
m_tsvd_grid = m_dagger_SVD_trunc.reshape((N, N))

dev_svd = (m_dagger_SVD - m_true).reshape((N, N))
dev_tsvd = (m_dagger_SVD_trunc - m_true).reshape((N, N))


# 2. Prepare Resolution Data
# For SVD (Full rank p=8 or 9)
Rm_full = Vp @ Vp.T 
# For TSVD (Truncated p=7)
Rm_tsvd = Vp_trunc @ Vp_trunc.T

# --- 2. Create the 2x3 Plotting Grid ---
fig, ax = plt.subplots(2, 3, figsize=(15, 10))

# ROW 1: Full SVD Results
im1 = ax[0, 0].imshow(m_svd_grid, cmap='RdBu', vmin=-1., vmax=1.)
ax[0, 0].set_title("Full SVD Model")
fig.colorbar(im1, ax=ax[0, 0])

im2 = ax[0, 1].imshow(dev_svd, cmap='RdBu', vmin=-1, vmax=1)
ax[0, 1].set_title("Deviation (SVD - True)")
fig.colorbar(im2, ax=ax[0, 1])

im3 = ax[0, 2].imshow(Rm_full, cmap='hot', vmin=0, vmax=1)
ax[0, 2].set_title("Resolution Matrix (Full)")
fig.colorbar(im3, ax=ax[0, 2])

# ROW 2: Truncated SVD Results
im4 = ax[1, 0].imshow(m_tsvd_grid, cmap='RdBu', vmin=-1., vmax=1.)
ax[1, 0].set_title(f"TSVD Model (p={p_nonzero})")
fig.colorbar(im4, ax=ax[1, 0])

im5 = ax[1, 1].imshow(dev_tsvd, cmap='RdBu', vmin=-1, vmax=1)
ax[1, 1].set_title(f"Deviation (TSVD - True)")
fig.colorbar(im5, ax=ax[1, 1])

im6 = ax[1, 2].imshow(Rm_tsvd, cmap='hot', vmin=0, vmax=1)
ax[1, 2].set_title(f"Resolution Matrix (p={p_nonzero})")
fig.colorbar(im6, ax=ax[1, 2])

plt.tight_layout()
plt.show()
#plt.savefig('p3_soln.png', dpi=290)