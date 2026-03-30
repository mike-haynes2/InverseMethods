import numpy as np
import scipy
import math as m

import matplotlib.pyplot as plt


zline = np.loadtxt('HW5_data/zline.txt')



def plot_zline_dat(noise_floor,show_err=True):
    plt.plot(zline[:,0],zline[:,1], lw=2,color='teal',label='zline data')

    if show_err: plt.errorbar(zline[:,0],zline[:,1],xerr=0.,yerr=(noise_floor+np.abs(zline[:,2])), fmt='none',capsize=3., lw=2.8, color='purple',label='data error')
    
    plt.xlabel('Northing track [m]')
    plt.ylabel('Vertical $E$ field [$\mu$V/m]')

    plt.grid()
    plt.legend()

    plt.show()
    return None


# print(np.average(zline[:,2]))

# noise floor specified in the paper
noise=10.

plot_zline_dat(noise_floor = noise)

delx_vals = [zline[i+1,0] - zline[i,0] for i in range(len(zline[:,0])-1) ]
dx_avg = sum(delx_vals)/len(delx_vals)

xmin  = zline[0,0]
xmax  = zline[-1,0]

xarr_coarse = np.arange(start=xmin,stop=xmax,step=40.)




## simplest model: nearest neighbor
d_nn = np.zeros_like(xarr_coarse)

for i,xval in enumerate(xarr_coarse):
    ind = np.abs(zline[:,0] - xval).argmin()
    d_nn[i]=zline[ind,1]



# plot simplest nn solution
plt.plot(zline[:,0],zline[:,1], lw=2,color='teal',label='zline data')

plt.errorbar(zline[:,0],zline[:,1],xerr=0.,yerr=(noise+np.abs(zline[:,2])), fmt='none',capsize=3., lw=2.8, color='purple',label='data error')

plt.plot(xarr_coarse, d_nn, lw=1.5, color='firebrick', label='nearest neighbor')

plt.xlabel('Northing track [m]')
plt.ylabel('Vertical $E$ field [$\mu$V/m]')

plt.grid()
plt.legend()

#plt.show()
plt.close()







## 2nd simplest model: linear interpolation
d_LI = np.zeros_like(xarr_coarse)

G_ODLS = np.zeros((len(xarr_coarse),len(zline[:,0])))

for i,xval in enumerate(xarr_coarse):
    #ind = np.abs(zline[:,0] - xval).argmin()
    indices = np.argpartition(np.abs(zline[:,0] - xval), 1)[:2]
    d_LI[i]=(zline[indices[0],1]+zline[indices[1],1])/2.
    G_ODLS[i,indices[0]] = 1./2.
    G_ODLS[i,indices[1]] = 1./2.



# plot LI solution
plt.plot(zline[:,0],zline[:,1], lw=4,color='teal',label='zline data')

plt.errorbar(zline[:,0],zline[:,1],xerr=0.,yerr=(noise+np.abs(zline[:,2])), fmt='none',capsize=3., lw=2.8, color='purple',label='data error')

plt.plot(xarr_coarse, d_nn, lw=1.5, color='firebrick', label='nearest neighbor')
plt.plot(xarr_coarse, d_LI, lw=1.5, color='magenta', label='average 2nn')


plt.xlabel('Northing track [m]')
plt.ylabel('Vertical $E$ field [$\mu$V/m]')

plt.grid()
plt.legend()

#plt.show()
plt.close()

G_ODLS = G_ODLS.T

# print(G_ODLS)
print(np.shape(G_ODLS))



# normal equations: 
mod_ODLS = np.linalg.inv(G_ODLS.T @ G_ODLS) @ G_ODLS.T @ zline[:,1]
mod_nn = d_nn.copy()
mod_LI = d_LI.copy()

# plot normal eqn solution
plt.plot(zline[:,0],zline[:,1], lw=4,color='black',label='zline data')

plt.errorbar(zline[:,0],zline[:,1],xerr=0.,yerr=(noise+np.abs(zline[:,2])), fmt='none',capsize=3., lw=2.8, color='purple',label='data error')

plt.plot(xarr_coarse, mod_nn, lw=1.5, color='firebrick', label='nearest neighbor')
plt.plot(xarr_coarse, mod_LI, lw=1.5, color='magenta', label='average neighbors')
plt.plot(xarr_coarse,mod_ODLS, lw=1.75, color='teal', label = 'normal eqns')

plt.xlabel('Northing track [m]')
plt.ylabel('Vertical $E$ field [$\mu$V/m]')

plt.title('Comparison of ODLS model solutions')

plt.grid()
plt.legend()

plt.show()
plt.close()

# since we need to compare arrays of like shape, resend the model to the dimension of the data via the G operator for the normal eqns
res = np.abs(G_ODLS @ mod_ODLS - zline[:,1]) ** 2.


plt.scatter(zline[:,0], res, color='teal',zorder=1)
plt.title('Squared residuals from Normal Equations Inversion')
plt.xlabel('Northing track [m]')
plt.ylabel('Squared difference in vertical $E$ field [$\mu$V/m]$^2$')
plt.grid(alpha=0.5,zorder=0)

plt.show()


norm_model_ODLS = np.sqrt(np.dot(mod_ODLS,mod_ODLS))
assert norm_model_ODLS == np.linalg.norm(mod_ODLS)
norm_res_ODLS = np.sqrt(np.dot(res,res))

print("model norm:",norm_model_ODLS,"\nres norm:",norm_res_ODLS)



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
