import numpy as np
import scipy
import math as m



# path to galileo data
path_to_gal = '../../galileo_Europa_flybys/'

# setup formatting for np.loadtxt to expect datetime in first column
names = ['time', 'Bx', 'By', 'Bz', 'Bmag', 'x', 'y', 'z']
formats = ['M8[ms]', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8']

# setup rows to start and end at 
# index starting at 1:10 UTC
idx_110 = 789
# index starting at 1:30 UTC
idx_130 = 4370

# define start / stop rows
idx_start = 0
idx_stop = 6000



# Load the data to handle datetimes
dat_E14 = np.loadtxt(
    path_to_gal+'E14/ORB14_EUR_EPHIO.TAB',
    skiprows=idx_start,
    max_rows=(idx_stop-idx_start), 
    dtype={'names': names, 'formats': formats},
    # Converter for index 0 (the date string)
    converters={0: lambda s: np.datetime64(s.decode('utf-8'))}
)


# define the variable arrays
t_arr = dat_E14['time']
Bx_arr_nT = dat_E14['Bx']
By_arr_nT = dat_E14['By']
Bz_arr_nT = dat_E14['Bz']
Bmag_arr_nT = dat_E14['Bmag']
X_arr_RE = dat_E14['x']
Y_arr_RE = dat_E14['y']
Z_arr_RE = dat_E14['z']

# define variables
mu0 = scipy.constants.mu_0
RE = 1560.8e+03 
B0_detrend_nT = np.array([10,-216,-409])
B0_detrend = B0_detrend_nT*1.e-09
signal_period = 11.1 # hr
signal_freq = 1. / (11.1 * 3600) #convert to seconds, then reciprocal to frequency
test_vals = True



from library import plot_mag_trajectory

plot_mag_trajectory(t_arr,X_arr_RE,Y_arr_RE,Z_arr_RE, 
                    Bx_arr_nT-B0_detrend_nT[0], By_arr_nT-B0_detrend_nT[1], Bz_arr_nT-B0_detrend_nT[2], 
                    save=False)


from library import generate_synthetic_data, calculate_B_dipole_vectorized, model_to_dipole_moment, perform_gauss_newton, plot_mag_synth, calculate_B_dipole_final

B0_vec_synth = np.array([4., -209., -385.])*1.e-09

m_true = [1400.0, 1540.0, 2.5] # r1, r0, sigma
m_guess = [1300.0, 1560.0, 1.]

rsynth, Bsynth = generate_synthetic_data(m_true, signal_freq, B0_vec_synth,noise_std=1.e-9)


Bx_synth_detrend = (Bsynth[:,0]-B0_vec_synth[0])*1.e+09
By_synth_detrend = (Bsynth[:,1]-B0_vec_synth[1])*1.e+09
Bz_synth_detrend = (Bsynth[:,2]-B0_vec_synth[2])*1.e+09

plot_mag_synth(rsynth[:,0],rsynth[:,1],rsynth[:,2], 
                    Bx_synth_detrend, By_synth_detrend, Bz_synth_detrend,figname='synthetic_dat_Noise2p5nT.png',  
                    save=False)


## Gauss newton: difficult to get wrangled, tried the following:
# - column normalization in the Jacobian as a form of regularization
# - running pseudo inversion as a way to handle large ratio of singular values
# - including tikhonov component of inversion to assign small buffer against singularity in inversion
# pass in r values in KM!!
m_final = perform_gauss_newton(rsynth/1.e+03, Bsynth, m_guess, signal_freq, B0_vec_synth, max_iter=100, pseudo_inv=False,do_col_norms=False)



from library import run_scipy_inversion

# pass in r values in KM!!
m_recovered = run_scipy_inversion(rsynth/1.e+03, Bsynth, m_guess, signal_freq, B0_vec_synth)
print(m_recovered)



## testing 

if test_vals:
    m = [1400.0, 1560.0, 2.5]
    mu0 = 4 * np.pi * 1e-7
    omega = 2 * np.pi * signal_freq
    k = np.sqrt(1j * mu0 * m[2] * omega)

    # Check 1: Is k too large? (Skin depth check)
    delta = np.sqrt(2 / (mu0 * m[2] * omega)) / 1000.0 
    print(f"Skin Depth: {delta:.2f} km")
    print(f"Ocean Thickness: {m[1] - m[0]:.2f} km")

    # Check 2: The Response Factor Q
    r0, r1 = m[1]*1000, m[0]*1000
    kd = k * (r0 - r1)
    coth_kd = 1.0 / np.tanh(kd)
    Q = 1.0 - (3.0 / (k * r0)) * (coth_kd - 1.0 / (k * r0))
    print(f"Response Factor Q: {Q}") 

    # Check 3: The Moment vs Distance
    M = -0.5 * Q * 210.0 * (r0**3) # Using 210 nT as a test B0
    r_test = (m[1] + 200) * 1000.0 # 200km altitude in meters
    B_test = np.linalg.norm(M) / (r_test**3)
    print(f"Predicted Induction Signal at 200km: {B_test:.2f} nT")

# testing influence of small change in ocean on dipole field:
# m_base = [1400.0, 1560.0, 2.5]
# m_plus = [1401.0, 1560.0, 2.5] # Shift bottom by 1km

# M_base = model_to_dipole_moment(m_base, signal_freq, B0_vec_synth)
# M_plus = model_to_dipole_moment(m_plus, signal_freq, B0_vec_synth)

# print(f"Base Moment: {np.linalg.norm(M_base):.4e}")
# print(f"Shifted Moment: {np.linalg.norm(M_plus):.4e}")
# print(f"Delta M: {np.linalg.norm(M_base - M_plus):.4e}")


# B_base = calculate_B_dipole_final(rsynth/1.e+03, M_base.real)
# B_plus = calculate_B_dipole_final(rsynth/1.e+03, M_plus.real)
# print(f"Max B change: {np.max(np.abs(B_base - B_plus))} nT")



r_rec, B_rec = generate_synthetic_data(m_recovered, signal_freq, B0_vec_synth,noise_std=1.e-11)



Bx_rec_detrend = (B_rec[:,0]-B0_vec_synth[0])*1.e+09
By_rec_detrend = (B_rec[:,1]-B0_vec_synth[1])*1.e+09
Bz_rec_detrend = (B_rec[:,2]-B0_vec_synth[2])*1.e+09

print(np.max(Bx_rec_detrend))


m_rec_GN = [1349.04, 1550.64, 2.279]
r_rec2, B_rec2 = generate_synthetic_data(m_rec_GN, signal_freq, B0_vec_synth,noise_std=1.e-11)



Bx_rec2_detrend = (B_rec2[:,0]-B0_vec_synth[0])*1.e+09
By_rec2_detrend = (B_rec2[:,1]-B0_vec_synth[1])*1.e+09
Bz_rec2_detrend = (B_rec2[:,2]-B0_vec_synth[2])*1.e+09


plot_mag_synth(rsynth[:,0],rsynth[:,1],rsynth[:,2], 
                    Bx_synth_detrend, By_synth_detrend, Bz_synth_detrend, 
                    curveBx=Bx_rec_detrend,curveBy=By_rec_detrend,curveBz=Bz_rec_detrend,
                    curve2Bx=Bx_rec2_detrend,curve2By=By_rec2_detrend,curve2Bz=Bz_rec2_detrend,
                    figname='synthetic_recovery_comparison_2.png',  
                    save=False)
print(m_recovered)







from library import run_mcmc

## NOW all conductivities passed in in log form!!

m_guess = [1330, 1525, np.log10(2.)]

print(np.max(Bsynth))

markov_chain = run_mcmc(rsynth/1.e+03, Bsynth, m_guess, signal_freq, B0_vec_synth, noise_std=5.e-09, num_steps=50000)



from library import plot_mcmc_traces, plot_mcmc_corner


plot_mcmc_traces(markov_chain)

plot_mcmc_corner(markov_chain)