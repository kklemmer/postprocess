from scipy import ndimage
import numpy as np


def finite_diff(f, dx):

    gf = ndimage.gaussian_filter1d(f, sigma=1, order=1) / dx

    return gf

def derivative_z_ghost_cells(f, dz):

    f_ghost = -f.copy()[:,0]

    print(np.shape(f_ghost))
    print(np.shape(f))
    f_new = np.vstack([f_ghost, f.T]).T



    dfdz = finite_diff(f_new, dz)

    return(dfdz[:,1:])