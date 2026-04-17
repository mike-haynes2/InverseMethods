import matplotlib.pyplot as plt
import math as m

import copy


def plot_conductivity(temp, cond, curve=None, curve2 = None, size=(8,5),err=None, save=False, plot_milliunits = True):

    fig, ax = plt.subplots(figsize=size)
    cond = copy.deepcopy(cond)
    curve = copy.deepcopy(curve)

    if plot_milliunits:
        cond*= 1.e+03
        if curve is not None: curve*= 1.e+03
        if curve2 is not None: curve2*= 1.e+03
        unitlab = '[mS/m]'
    else:
        unitlab = '[S/m]'

    plt.plot(temp,cond,color='teal',lw=2.5,label='conductivity $\sigma$')
    if err is not None:
        plt.errorbar(temp,cond, xerr=0., yerr=err, fmt='none', capsize=3., lw=2.8, color='purple',label='data error')
    
    if curve is not None:
        plt.plot(temp,curve,color='cyan',lw=2.4,label='fitted (LM) conductivity $\sigma(T)$')
        plt.title('Jackson County Dunite: Recovered $\sigma$ vs. Observations', fontweight='bold')
        if curve2 is not None:
            plt.plot(temp[13:],curve2,color='magenta',lw=2.4,label='fitted (transformed) conductivity $\sigma(T)$')
            plt.title('Jackson County Dunite: Recovered $\sigma$ vs. Observations', fontweight='bold')
    else:
        plt.title('Jackson County Dunite: $\sigma(T)$', fontweight='bold')

    plt.xlabel('Temperature [k]')
    plt.ylabel('Conductivity $\sigma(T)$ '+unitlab)
    
    plt.grid(alpha=0.5,zorder=0)
    plt.legend()

    if save:
        plt.savefig('p1e_model.png', dpi=275)
        plt.close()
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
    plt.text(8,-13.8,lab_string, fontweight='bold', fontsize=10)

    plt.grid(alpha=0.5,zorder=0)
    plt.legend()

    if save:
        plt.savefig('p1c_fit_plot_pinv.png', dpi=275)
        plt.close()
    else:
        plt.show()


def plot_fit_OG(temp, cond, curve=None, size=(8,5),err=None, save=False, plot_milliunits = True):

    fig, ax = plt.subplots(figsize=size)
    cond = copy.deepcopy(cond)
    curve = copy.deepcopy(curve)



    if plot_milliunits:
        cond*= 1.e+03
        if curve.all() != None: curve*= 1.e+03
        unitlab = '[mS/m]'
    else:
        unitlab = '[S/m]'

    plt.scatter(temp,cond,color='teal',s=20, marker='o', label='Observations')
    if err != None:
        plt.errorbar(temp,cond, xerr=0., yerr=err, fmt='none', capsize=3., lw=2.8, color='purple',label='data error')
    
    if curve.all() != None:
        plt.plot(temp,curve,color='magenta',lw=2.4,label='fitted conductivity $\sigma(T)$')
        plt.title('Jackson County Dunite: Recovered $\sigma$ vs. Observations', fontweight='bold')
    else:
        plt.title('Jackson County Dunite: $\sigma(T)$', fontweight='bold')

    plt.xlabel('Temperature [k]')
    plt.ylabel('Conductivity $\sigma(T)$ '+unitlab)
    
    plt.grid(alpha=0.5,zorder=0)
    plt.legend()

    if save:
        plt.savefig('p1c_datafit_plot.png', dpi=275)
        plt.close()
    else:
        plt.show()




def plot_res(temp, res, cond=None, size=(8,5), save=False):
    fig, ax = plt.subplots(figsize=size)

    temp = temp.copy()
    res = res.copy()

    plt.scatter(temp, res, color='navy', marker='+', s=40, label = 'signed residuals')
    if cond!=None:
        plt.plot(temp, -0.05*cond, color='red', ls='dashed', label='observation error $\sigma=0.05$')

    plt.ylabel('Model Misfit [S/m]')
    plt.xlabel('Temperature [k]')

    fig.suptitle('Difference between Observations and Model from Transformed LS Inversion')

    plt.grid(alpha=0.5,zorder=0)
    plt.legend()

    if save:
        plt.savefig('p1d_res_LS.png', dpi=275)
        plt.close()
    else:
        plt.show()