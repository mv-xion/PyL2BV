"""
    This file is for the band selection and
    interpolation of the spectra if needed
"""

import logging

from scipy.interpolate import make_interp_spline

# Retrieve the loggers by name
image_logger = logging.getLogger("image_logger")


def spline_interpolation(current_wl, reflectance_batch, expected_wl):
    """Interpolate a batch of pixels (each pixel has a spectral dimension)."""
    try:
        interpolator = make_interp_spline(current_wl, reflectance_batch, axis=2)
        return interpolator(expected_wl)
    except Exception as e:
        image_logger.error(f"Error in batch interpolation: {e}")
        raise
