from scipy import ndimage
import numpy as np


def finite_diff(f, dx):

    gf = ndimage.gaussian_filter1d(f, sigma=1, order=1) / dx

    return gf

def wall_bc(f):
    """
    Creates a wall BC in the z direction (last dimesion)
    using ghost points
    """

    z1_points = np.expand_dims(-f[...,0], axis=-1)
    

    f_new = np.concatenate((f,z1_points), axis=-1)

    return f_new

def top_bc(f):
    """
    Creates a wall BC in the z direction (last dimesion)
    using ghost points
    """

    z1_points = np.expand_dims(f[...,-1], axis=-1)
    

    f_new = np.concatenate((z1_points,f), axis=-1)

    return f_new

def streamtube_avg(var, mask, var_true=1):
    """
    Calculate and return the model streamtube averaged quantity
    normalized by the true streamtube averaged quantity
    """

    return np.sum(var * mask, axis=(1,2))/np.sum(var_true * mask, axis=(1,2))

def compute_centroid(y, z, vel):
    # Calculate the weighted sum of y and z based on the velocity deficit
    Y, Z = np.meshgrid(y, z)
    if np.ndim(vel) > 2:
        Y_x = np.tile(Y.T, (np.shape(vel)[0],1,1))
        Z_x = np.tile(Z.T, (np.shape(vel)[0],1,1))
        np.shape(Y_x)
        y_centroid = np.sum(Y_x * vel, axis=(1,2)) / np.sum(vel, axis=(1,2))
        z_centroid = np.sum(Z_x * vel, axis=(1,2)) / np.sum(vel, axis=(1,2))
    else:
        y_centroid = np.sum(Y.T * vel) / np.sum(vel)
        z_centroid = np.sum(Z.T * vel) / np.sum(vel)
    return y_centroid, z_centroid

def compute_fwhm(arr):
    half_max = np.max(np.abs(arr))/2
    indices_above_half = np.where(np.abs(arr) >= half_max) [0]
    if len(indices_above_half) == 0:
        first_cross = 0
        last_cross = 0
    else:
        # The first and last index where the curve is above half-maximum
        first_cross = indices_above_half[0]
        last_cross = indices_above_half[-1]

    return [first_cross, last_cross]

def compute_ambient_width(arr, val=0.05):
    indices_above_half = np.where(arr <= val) [0]
    if len(indices_above_half) == 0:
        first_cross = 0
        last_cross = 0
    else:
        # The first and last index where the curve is above half-maximum
        first_cross = indices_above_half[0]
        last_cross = indices_above_half[-1]

    return [first_cross, last_cross]