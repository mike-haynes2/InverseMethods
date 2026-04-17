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
plot_conductivity(temp_dat, cond_dat, save=False)




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


from plot_dat import plot_linearized

#plot_linearized(d, x, log_sigma0, A_eV, save=False)

# solve with normal equations
GT = G.T
GTG = GT @ G
GTGinv = np.linalg.inv(GTG)


m_normal = GTGinv @ GT @ d
#print(m_normal)

#plot_linearized(d, x, log_sigma0, A_eV, second_est_s=m_normal[0], second_est_A=m_normal[1], save=False)

from plot_dat import plot_fit_OG

sigma0 = np.exp(log_sigma0)
Ajoule = A_eV * scipy.constants.eV

curv = sigma0 * np.exp(-Ajoule/(kB*temp_dat_trunc))



plot_fit_OG(temp_dat_trunc, cond_dat_trunc, curve=curv, plot_milliunits=True, save=False)



## part d) compute covariance matrix
std_dev = 0.05

Cov = (std_dev ** 2.) * GTGinv
print(Cov)

res = (sigma0 * np.exp(-A_eV/(kB_eV*temp_dat_trunc))) - cond_dat_trunc

from plot_dat import plot_res

plot_res(temp_dat_trunc, res, save=False)

cc = Cov[0,1]/np.sqrt(Cov[0,0]*Cov[1,1])
print(cc)
# print(Cov[0,1],Cov[0,0])


## part e) redo analysis with entire dataset

from run_LM import invert_cond_LM

# set error threshold 
er = 0.01

mvec_LM, sp_result = invert_cond_LM(cond_dat, temp_dat, error_threshold=er, m_init=None)

cond_out = mvec_LM[0] * np.exp(-mvec_LM[2]/(kB_eV*temp_dat)) + (mvec_LM[1]/temp_dat) * np.exp(-mvec_LM[3]/(kB_eV*temp_dat))

plot_conductivity(temp_dat, cond_dat, curve=cond_out,curve2=None , size=(8,5),err=(er*cond_dat), save=True, plot_milliunits = True)

print(len(curv))
print(len(cond_out))

plot_res(temp_dat,sp_result.fun)