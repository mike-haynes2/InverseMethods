import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
import math as m
from scipy import constants
from scipy.interpolate import RBFInterpolator, InterpolatedUnivariateSpline
from datetime import datetime, timedelta

import os
import warnings
from sys import argv

rcParams.update({'font.size': 28})
plt.style.use('dark_background')
warnings.filterwarnings( "ignore")


from juno_model_field_vector import *


## TO CALL: juno_field_vector( (360 - sys3 west longitude), (theta), (L-shell) )



Bx, By, Bz = juno_field_vector((360-136)*np.pi/180, np.pi/2, 9.374)

mag1 = np.sqrt(Bx**2. + By**2. + Bz**2.)*1.e+09
print(Bx*1e9, By*1e9, Bz*1e9)

Bx, By, Bz = juno_field_vector((360-136)*np.pi/180, np.pi/2, 9.394)
mag2 = np.sqrt(Bx**2. + By**2. + Bz**2.)*1.e+09
print(Bx*1e9, By*1e9, Bz*1e9)

print(mag1-mag2)

# define array of entire sys-3 EAST longitudes

res = 0.02*m.pi
sys3 = np.arange(start=0.0,stop=2*m.pi,step=res)

# define where in jovian magnetosphere (L shell of callisto)
# moon name
moon_name = str('Europa')
L_shell = 9.384
# moon_name = str('Io')
# L_shell = 5.899
# moon_name = str('Ganymede')
# L_shell = 14.972

# Europa
#L_shell = 9.38



# initialize arrays to store
Bx_arr = []
By_arr = []
Bz_arr = []

# loop over all longitudes to generate arrays of an entire synodic rotation

for i in range(len(sys3)):
    sys3_long = sys3[i]
    # calculate fields
    Bx, By, Bz = juno_field_vector(sys3_long, np.pi/2, L_shell)

    Bx_arr.append(Bx)
    By_arr.append(By)
    Bz_arr.append(Bz)

sys3_deg = ((sys3*180.)/m.pi)

Bz_arr = np.array(Bz_arr)
By_arr = np.array(By_arr)
Bx_arr = np.array(Bx_arr)

## find sys3 where a component is maximized
# print(np.max(Bz_arr))
# print(np.argmax(Bz_arr))
# print(sys3_deg[np.argmax(Bz_arr)])

## plotting attributes

#linewidth
lw = 1.8
#font size
fsz = 20



Bm_arr = np.sqrt( Bx_arr ** 2 + By_arr ** 2 + Bz_arr ** 2)

print(np.min(Bm_arr))
print(np.max(Bm_arr))

rcParams.update({'font.size': fsz})

plt.plot(sys3_deg, Bx_arr/1.0e-9, label=r'$B_x\,(\lambda_{III})$', color='cyan')
plt.plot(sys3_deg, By_arr/1.0e-9, label=r'$B_y\,(\lambda_{III})$', color='orange')
plt.plot(sys3_deg, Bz_arr/1.0e-9, label=r'$B_z\,(\lambda_{III})$', color='magenta')

plt.plot(sys3_deg, Bm_arr/1.0e-9, label=r'$|\mathbf{B}\,(\lambda_{III})|$', color='greenyellow')

plt.title(str("Magnetospheric field at "+moon_name+"'s orbit"))
plt.ylabel(r"$B_i (\lambda_{III})$ [nT]")
plt.xlabel(r"$\lambda_{III} \,\, [^{\circ}]$")

plt.legend(loc=4)
plt.grid()

plt.tight_layout()

plt.show()
