import matplotlib.pyplot as plt
import math as m




def plot_conductivity(temp, cond, curve=None, size=(8,5),err=None, save=False, plot_milliunits = True):

    fig, ax = plt.subplots(figsize=size)

    if plot_milliunits:
        cond*= 1.e+03
        if curve != None: curve*= 1.e+03
        unitlab = '[mS/m]'
    else:
        unitlab = '[S/m]'

    plt.plot(temp,cond,color='teal',lw=2.5,label='conductivity $\sigma$')
    if err != None:
        plt.errorbar(temp,cond, xerr=0., yerr=err, fmt='none', capsize=3., lw=2.8, color='purple',label='data error')
    
    if curve != None:
        plt.plot(temp,curve,color='magenta',lw=2.4,label='fitted conductivity $\sigma(T)$')
        plt.title('Jackson County Dunite: Recovered $\sigma$ vs. Observations', fontweight='bold')
    else:
        plt.title('Jackson County Dunite: $\sigma(T)$', fontweight='bold')

    plt.xlabel('Temperature [k]')
    plt.ylabel('Conductivity $\sigma(T)$ '+unitlab)
    
    plt.grid(alpha=0.5,zorder=0)
    plt.legend()

    if save:
        plt.savefig('p1a_data_plot.png', dpi=275)
    else:
        plt.show()





def plot_linearized(logSigmaT, x, est_s, est_A, second_est_s=None, second_est_A=None, size=(8,5),save=False):
    fig, ax = plt.subplots(figsize=size)

    plt.scatter(x,logSigmaT, color='teal',s=20,marker='o', label='Observations')
    linearized_est = est_s - est_A*x
    plt.plot(x, linearized_est, color='magenta', lw=3.1, label='Transformed fit: psuedoinverse')
    if (second_est_s!= None) & (second_est_A!= None):
        linearized_est2 = second_est_s - second_est_A*x
        plt.plot(x, linearized_est2, color='navy', lw=1.1, label='Transformed fit: normal equations') 

    plt.title('Transformed-Linearized Parameter Inversion', fontweight='bold')

    plt.xlabel('Inverse Thermal Energy [eV]$^{-1}$')
    plt.ylabel(r'Logarithmic conductivity $\log{\left( \sigma(T) / [1\,\mathrm{S}/\mathrm{m}] \right) }$')

    lab_string = '$A$ [eV] = '+str(est_A)+';\n $\sigma_0\,[\mathrm{S}/\mathrm{m}]=$ '+str(m.exp(est_s))
    plt.text(8,-6.5,lab_string, fontweight='bold', fontsize=10)

    plt.grid(alpha=0.5,zorder=0)
    plt.legend()

    if save:
        plt.savefig('p1c_fit_plot_pinv.png', dpi=275)
    else:
        plt.show()