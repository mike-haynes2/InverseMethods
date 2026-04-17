import numpy as np
import math as m
import scipy





## HW6: nonlinear inversion for the Arrhenius relationship
## parameterizing the thermally-activated semiconduction \sigma
 

# define boltz const
kB = scipy.constants.k

data_arr = np.loadtxt('data/jcd.dat',skiprows=1)

temperature_C = data_arr[:,0]
cond_dat = data_arr[:,1]
# convert data to kelvin for appropriate usage in expressions
temp_dat = temperature_C + 273.15*np.ones_like(temperature_C)



## p1(a): plot the data
## import user functions
from plot_dat import plot_conductivity
#plot_conductivity(temp_dat, cond_dat)




## p1(b): perform least squares (explain)
## p1(c): perform least squares
# truncate first 14 rows of data
trunc_idx = 13

cond_dat_trunc = cond_dat[trunc_idx:]
temp_dat_trunc = temp_dat[trunc_idx:]

log_cond_dat_trunc = np.log(cond_dat_trunc)
d = log_cond_dat_trunc.copy()

x_unconditioned = 1./ (kB * temp_dat_trunc)
# print(x_unconditioned)
# since the elements of x are huge (O(10^19)), need to precondition: acheive via dimension
kB_eV = kB / scipy.constants.eV
#print(kB_eV)
## !NOTE: This implies that A we obtain will be in units of eV!!
x = 1./ (kB_eV*temp_dat_trunc)
#print(x)

G = np.vstack((np.ones_like(x),-x))
Gun = np.vstack((np.ones_like(x_unconditioned),-x_unconditioned))
G = G.T
#print(G)
# G[:,0] = 1s , corresponding to s = log(sigma_0)
# G[:,1] = -x , corresponding to A/kBT

# test of preconditioning effectiveness:
# print(np.linalg.cond(G))
# print(np.linalg.cond(Gun))


# solve with pseudo inverse
Gpinv = scipy.linalg.pinv(G)


[log_sigma0, A_eV] = Gpinv @ d

print(np.exp(log_sigma0),A_eV,A_eV * scipy.constants.eV)


from plot_dat_ac import plot_linearized

plot_linearized(d, x, log_sigma0, A_eV)

# solve with normal equations
GT = G.T
GTG = GT @ G
GTGinv = np.linalg.inv(GTG)


m_normal = GTGinv @ GT @ d
#print(m_normal)

plot_linearized(d, x, log_sigma0, A_eV, second_est_s=m_normal[0], second_est_A=m_normal[1])

print(d)
print(x)
print(G)