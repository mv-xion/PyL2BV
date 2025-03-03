"""
    This file is for the band selection and
    interpolation of the spectra if needed
"""

import logging

from joblib import Parallel, delayed
import numpy as np
from scipy.interpolate import make_interp_spline
from multiprocessing import cpu_count


# Retrieve the loggers by name
image_logger = logging.getLogger("image_logger")


def process_spline_batch(current_wl, reflectance_batch, expected_wl):
    """Interpolate a batch of pixels (each pixel has a spectral dimension)."""
    try:
        interpolator = make_interp_spline(current_wl, reflectance_batch, axis=1)
        return interpolator(expected_wl)
    except Exception as e:
        image_logger.error(f"Error in batch interpolation: {e}")
        raise

def spline_interpolation(current_wl, image, expected_wl):
    try:
        image_logger.info("Starting spline interpolation.")

        rows, cols, dims = image.shape  # Extract dimensions
        num_pixels = rows * cols  # Total number of pixels
        num_cores = min(cpu_count(), num_pixels)  # Don't exceed available data
        batch_size = num_pixels // num_cores  # Split pixels into balanced chunks

        # Reshape image to (num_pixels, dims) for easy processing
        flattened_image = image.reshape(num_pixels, dims)

        # Split pixel spectra into batches
        pixel_batches = np.array_split(flattened_image, num_cores, axis=0)

        # Parallel interpolation
        results = Parallel(n_jobs=num_cores)(
            delayed(process_spline_batch)(current_wl, batch, expected_wl)
            for batch in pixel_batches
        )

        # Concatenate and reshape back to (rows, cols, expected_wl_dims)
        interpolated_datacube = np.concatenate(results, axis=0).reshape(rows, cols, -1)

        image_logger.info("Spline interpolation completed successfully.")
        return interpolated_datacube
    except Exception as e:
        image_logger.error(f"Error during spline interpolation: {e}")
        raise