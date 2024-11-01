"""
    This file is for the band selection and
    interpolation of the spectra if needed
"""

import logging

from scipy.interpolate import make_interp_spline

# Retrieve the loggers by name
app_logger = logging.getLogger("app_logger")
image_logger = logging.getLogger("image_logger")


def spline_interpolation(current_wl, reflectances, expected_wl):
    try:
        app_logger.info("Starting spline interpolation.")
        image_logger.info("Starting spline interpolation.")
        interpolator = make_interp_spline(current_wl, reflectances, axis=2)
        interpolated_datacube = interpolator(expected_wl)
        app_logger.info("Spline interpolation completed successfully.")
        image_logger.info("Spline interpolation completed successfully.")
        return interpolated_datacube
    except Exception as e:
        app_logger.error(f"Error during spline interpolation: {e}")
        image_logger.error(f"Error during spline interpolation: {e}")
        raise


# Function definitions for the band selection
'''
def spline(arg):
    try:
        current_wl, block, expected_wl = arg
        logging.info("Starting spline function.")

        # Create a spline interpolation function
        spline_interp = interp1d(current_wl, block, kind='cubic')
        # Evaluate the spline at the new points
        block_new = spline_interp(expected_wl)

        logging.info("Spline function completed successfully.")
        return block_new
    except Exception as e:
        logging.error(f"Error in spline function: {e}")
        raise

def block_spline(current_wl, reflectances, expected_wl):
    try:
        logging.info("Starting block spline interpolation.")
        # interpolate the reflectance data for every pixel
        # with spline method (row,col,dim)
        """
          Interpolate reflectance data for every pixel using spline method.

          Args:
                current_wl (array-like):    Array of current wavelengths.
                reflectances (array-like):  Array of reflectance data (rows, cols, wavelengths).
                expected_wl (array-like):   Array of expected wavelengths.

          Returns:
          array-like: Interpolated reflectance data (rows, cols, len(expected_wl)).
        """
        row = reflectances.shape[0]
        col = reflectances.shape[1]
        dim = expected_wl.shape[0]

        args_list = [(current_wl, reflectances[i, j, :], expected_wl)
                     for i in range(row) for j in range(col)]
        with Pool(processes=cpu_count()) as pool: # Use all available cores
            results = pool.map(spline, args_list)
        reflect_intercept = np.reshape(results, [row, col, dim])
        logging.info("Block spline interpolation completed successfully.")
        return reflect_intercept
    except Exception as e:
        logging.error(f"Error during block spline interpolation: {e}")
        raise
'''
