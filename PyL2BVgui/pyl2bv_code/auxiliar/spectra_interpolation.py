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
