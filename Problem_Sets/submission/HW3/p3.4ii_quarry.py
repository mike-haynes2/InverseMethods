import numpy as np
import math as m
import scipy

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm




colscan_mat = scipy.io.loadmat('hw3_src_provided/colscan.mat')
colscan = colscan_mat['colscan']
diag1scan_mat = scipy.io.loadmat('hw3_src_provided/diag1scan.mat')
diag1scan = diag1scan_mat['diag1scan']
diag2scan_mat = scipy.io.loadmat('hw3_src_provided/diag2scan.mat')
diag2scan = diag2scan_mat['diag2scan']
rowscan_mat = scipy.io.loadmat('hw3_src_provided/rowscan.mat')
rowscan = rowscan_mat['rowscan']

#print(np.shape(colscan), np.shape(rowscan), np.shape(diag1scan), np.shape(diag2scan))

# data format:
# x1, y1, x2, y2, time = rowscan[:]

idx_offset = 0.5
N = M = 16


lend = len(colscan)+len(rowscan)+len(diag1scan)+len(diag2scan)
d = np.zeros(lend)
count = 0


G = np.zeros((lend,N*M))

dset = np.concatenate((np.array(colscan[:,-1]), np.array(rowscan[:,-1]), np.array(diag2scan[:,-1]), np.array(diag1scan[:,-1])))


# start with columns::
for i,col in enumerate(colscan):
    # grab index of column being scanned
    col_idx = int(col[1] - idx_offset)
    assert col_idx == int(col[3] - idx_offset)
    tval = col[-1]
    G[count,col_idx::N] = 1.
    d[count] = tval
    count += 1

for i,row in enumerate(rowscan):
    row_idx = int(row[0]-idx_offset)
    assert row_idx == int(row[2]-idx_offset)
    tval=row[-1]
    d[count]=tval
    G[count, row_idx*N:(row_idx+1)*N]=1.
    count+=1
    


# top left to bottom right
for i,diag2 in enumerate(diag2scan):
    col_idx = int(diag2[1])
    row_idx = int(diag2[0])
    col_end = int(diag2[3])
    row_end = int(diag2[2])
    start = (row_idx)*N + col_idx
    # key: for top left to bottom right, the gap between populated elements increases by 1 each row (-1 for diag1 analogously)
    step = N + 1
    stop = start + (N-col_idx) * step
    stop = (row_end-1)*N + (col_end-1)

    if start==stop:
        G[count,start] = np.sqrt(2.)
    else:
        G[count,start:stop:step] = np.sqrt(2.)
    d[count]=diag2[-1]
    count+=1


# top right to bottom left
for i,diag1 in enumerate(diag1scan):
    col_idx = int(diag1[1])
    row_idx = int(diag1[0])
    col_end = int(diag1[3])
    row_end = int(diag1[2])
    start = (row_idx-1)*N + col_idx
    # key: for top right to bottom left, the gap between populated elements decreases by (-)1 
    step = -(N - 1)
    #stop = start + (N-col_idx) * step
    stop = row_end*N + col_end
    if start==stop:
        G[count,start] = np.sqrt(2.)
    else:
        G[count,start:stop:step] = np.sqrt(2.)
    d[count]=diag1[-1]
    count+=1







plt.imshow(G)
plt.show()


## part (a): calculate rank

print(np.linalg.matrix_rank(G))



## part (b): state / discuss the general solution,
#  data fit significance of the elements, 
# and dimensions of the data and model null spaces
# display a nonzero model that fits the trivial data set Gm = d = 0 exactly

Up, sp, VpT = scipy.linalg.svd(G, full_matrices=False)
#Up, sp, VpT = scipy.linalg.svd(G, full_matrices=True)
Vp = VpT.T
print(np.shape(Up), np.shape(Vp), Vp[:,:np.linalg.matrix_rank(G)].shape, np.diag(sp).shape)
Sp = np.diag(sp)
p_nonzero = len([x for x in sp if x > 1.e-15])


Up_trunc = Up[:,:p_nonzero]
Vp_trunc = Vp[:,:p_nonzero]
Vp_null = Vp[:,(p_nonzero-1):]
Sp_trunc = np.diag(sp[:p_nonzero])

print(Vp_null[0].shape)

m_null = (np.random.random() * Vp_null[:,np.random.randint(len(Vp_null[0]))]) + (np.random.random() * Vp_null[:,1]) + (np.random.random() * Vp_null[:,2])
m_null /= np.linalg.norm(m_null)

d_zero = G @ m_null

print(np.mean(np.abs(d_zero)))

# solve using truncated SVD manually
Gdagger = Vp @ (np.linalg.inv(Sp) @ Up.T) 
m_dagger_SVD = Gdagger @ d
## differs for some reason so use inbuilt method below

# solve using inbuilt SVD
Gdagger_MP = scipy.linalg.pinv(G)
m_dagger_MP = Gdagger_MP @ d



fig, ax = plt.subplots(len(Vp_null[0]), figsize=(6, 11))

for i in np.arange(len(Vp_null[0])):
    vnullp = Vp_null[:, i]
    vnull = vnullp.reshape(N, M)
    im = ax[i].imshow(vnull, cmap='bwr', vmin=-0.5, vmax=0.5)
    ax[i].set_title('i='+str(i)+' Model Null Space vector \nSum = '+str(np.sum(vnull)))
    #print(np.sum(vnull))


fig.suptitle('Model Null Space: Reshaped Visualization (256 --> 16 x 16)', fontweight='bold')

# 1. Call tight_layout BEFORE adding the manual colorbar axis
fig.tight_layout(rect=[0, 0, 0.9, 1]) # Reserve right 10% of space

# 2. Use 'cax' to place the colorbar exactly in the reserved axis
cbar_ax = fig.add_axes([0.8, 0.05, 0.07, 0.85]) 
fig.colorbar(im, cax=cbar_ax) # Removed 'ax', 'fraction', and 'pad'


fig.tight_layout
# plt.savefig('p3.4ii_b_soln.png', dpi=275)
# plt.close()
plt.show()





plt.imshow(m_null.reshape(N,N), cmap='bwr', vmin=-0.1,vmax=0.1)
cb = plt.colorbar()
cb.set_label('Seismic Slowness Perturbation')
plt.title(r'Model Null Space: Trivial solution $\mathbb{G}⋅ \mathbf{m}_\mathrm{null} = \mathbf{0}$')
# plt.savefig('p3.4ii_b_null.png', dpi=280)
# plt.close()
plt.show()




## part (c): note whether there are any model params that have perfect resolution: res matrix

Rm = Vp_trunc @ Vp_trunc.T




plt.imshow(Rm, cmap='hot', norm=LogNorm(vmin=1e-3, vmax=1))
#plt.imshow(Rm, cmap='hot', vmin=0, vmax=1)

plt.colorbar()
plt.title('Resolution Matrix (logscale)', fontweight='bold')
# plt.savefig('p3.4ii_c_ResMatrixLog.png', dpi=290)
# plt.close()
plt.show()


Rmflat = Rm.flatten()

print(len([x for x in Rmflat if x >=1.]))
print(np.where([Rmflat >=1.]))
## three instances of perfect resolution, probably the corners






## part (d): solve the system, plot 16x16 reshaped m_dagger (slowness model solution)
# show the min/max
# interperet the structures, geometrically, in terms of velocity propagation (where are the bones!)

# i: plot the reshaped m_dagger

plt.imshow(m_dagger_MP.reshape(N,N), cmap='bwr', vmin=-(1./3000),vmax=(1./3000))
cb = plt.colorbar()
cb.set_label(r'Seismic Slowness Perturbation ($± s_0$)')
plt.title(r'Model Space: Truncated SVD solution $\mathbb{G}⋅ \mathbf{m}_\dagger = \mathbf{d}$')
# plt.savefig('p3.4ii_d_mdagger.png', dpi=280)
# plt.close()
plt.show()


# ii: min / max
print(m_dagger_MP.min(), m_dagger_MP.max())


# iii: interperet m_dagger



## part (e): model resolution

Rm_diag = np.diag(Rm)

plt.imshow(Rm_diag.reshape(N,N), cmap='hot', vmin=0,vmax=1)
cb = plt.colorbar()
cb.set_label(r'Model Resolution')
plt.title(r'Model Space: Truncated SVD Resolution (diagonal)')
# plt.savefig('p3.4ii_e_Rmdiag.png', dpi=280)
# plt.close()
plt.show()





## part (f): show solution + null solution
plt.imshow((m_dagger_MP+m_null).reshape(N,N), cmap='bwr', vmin=-(1./3000),vmax=(1./3000))
cb = plt.colorbar()
cb.set_label(r'Seismic Slowness Perturbation ($± s_0$)')
plt.title(r'Truncated SVD solution $\mathbb{G}⋅ (\mathbf{m}_† + \mathbf{m}_\mathrm{null}) = \mathbf{d}$')
# plt.savefig('p3.4ii_f_mnet.png', dpi=280)
# plt.close()
plt.show()

print(G.shape)