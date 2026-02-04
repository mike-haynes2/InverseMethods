import numpy as np
import math as m
import matplotlib.pyplot as plt



## depth array of 100 evenly spaced points
# start at 0.2 m depth, end at 20 m
z_vec = np.arange(start=0.2,stop=20.2,step=0.2)
# define midpoints
z_mid = z_vec - 0.1
# define G operator (matrix) dimensions to solve eqn (1.21)
nrow = len(z_vec)
ncol = len(z_mid)
# define sensor array depth spacing
dz = z_vec[1]-z_vec[0]

# Calculate the G matrix: lower triangluar populated by dz
G_ij = np.tri(nrow,ncol,k=0,dtype=float)*dz

# specify the true slowness parameters
v0 = 1.
k = 40.
# calculate the slowness vector m_true (based on known model)
m_true = 1./(v0 + (k*z_mid))
# analytic data vector via integration 
## ! check whether z_vec or z_mid here:: z_vec because this is SENSOR depth
d_an = (1./k) * np.log(1. + (k*z_vec/v0))

# use "backslash" equivalent operator in python to solve the nxn system
m_calc_an = np.linalg.solve(G_ij,d_an)

# add noise to the analytically calculated travel-time vector (0.05 ms)
d_noisy = d_an + np.random.normal(0., 5.e-05, d_an.shape)

m_calc_noise = np.linalg.solve(G_ij,d_noisy)


fig, ax = plt.subplots(figsize=(6,8))

plt.plot(m_true,z_vec,label=r'$\mathbf{m}_{true}$',color='navy',lw=2.5)
plt.plot(m_calc_an,z_vec,label=r'$\mathbf{m}_{calculated}$',color='orange',lw=2,ls='dashed')
plt.plot(m_calc_noise,z_vec,label=r'$\mathbf{m}_{noisy}$',color='cyan',lw=1.5)

plt.ylim(z_vec[-1],z_vec[0])
plt.xlabel('Slowness [s/m]')
plt.ylabel('depth [m]')
#plt.xscale('log')

plt.title('Slowness vs. Depth: \nAnalytical vs. Simple Collocation (noisy)')

plt.grid()
plt.legend()
#plt.show()
plt.savefig('haynes_hw2_p3d_linear.png',dpi=250)

