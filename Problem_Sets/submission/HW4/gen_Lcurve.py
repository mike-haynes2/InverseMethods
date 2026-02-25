import numpy as np
import scipy


def calc_L_coordinates(G, m, d, Gsharp, L=np.eye(2), order=0):

    if order == 0:
        modelNorm = np.linalg.norm(m)
    else:
        modelNorm = np.linalg.norm(L@m)

    data_norm = np.linalg.norm(G@m - d)

    L_coordinates = modelNorm, data_norm

    GGS = G @ Gsharp

    g_factor = (G.shape[0] * (data_norm**2.))/(np.trace(np.eye(*GGS.shape) - GGS)**2. )

    return L_coordinates, g_factor
