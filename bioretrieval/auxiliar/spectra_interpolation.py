"""
    This file is for the band selection and
    interpolation of the spectra if needed
"""

import numpy as np
from scipy.interpolate import interp1d, make_interp_spline
from multiprocessing import Pool, cpu_count


# trying interp_spine

def spline_interpolation(current_wl, reflectances, expected_wl):
    interpolator = make_interp_spline(current_wl, reflectances, axis=2)
    interpolated_datacube = interpolator(expected_wl)

    return interpolated_datacube


# Function definitions for the band selection
'''
def spline(arg):
    current_wl, block, expected_wl = arg

    # Create a spline interpolation function
    spline_interp = interp1d(current_wl, block, kind='cubic')
    # Evaluate the spline at the new points
    block_new = spline_interp(expected_wl)

    return block_new


def block_spline(current_wl, reflectances, expected_wl):
    # interpolate the reflectance data for every pixel with spline method (row,col,dim)
    """
      Interpolate reflectance data for every pixel using spline method.

      Args:
            current_wl (array-like): Array of current wavelengths.
            reflectances (array-like): Array of reflectance data with shape (rows, cols, wavelengths).
            expected_wl (array-like): Array of expected wavelengths.

      Returns:
      array-like: Interpolated reflectance data with shape (rows, cols, len(expected_wl)).
    """
    row = reflectances.shape[0]
    col = reflectances.shape[1]
    dim = expected_wl.shape[0]

    args_list = [(current_wl, reflectances[i, j, :], expected_wl)
                 for i in range(row) for j in range(col)]
    with Pool(processes=cpu_count()) as pool:  # process on different cpus (as many as we have)
        results = pool.map(spline, args_list)
    reflect_intercept = np.reshape(results, [row, col, dim])
    return reflect_intercept
'''
