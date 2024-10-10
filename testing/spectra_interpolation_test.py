import unittest

import numpy as np

from bioretrieval.auxiliar.spectra_interpolation import spline_interpolation


class TestSpectraInterpolation(unittest.TestCase):
    def test_spline_interpolation(self):
        current_wl = np.array([400, 500, 600])
        reflectances = np.array(
            [
                [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]],
            ]
        )
        expected_wl = np.array([450, 550])

        interpolated_datacube = spline_interpolation(
            current_wl, reflectances, expected_wl
        )

        # Check the shape of the interpolated data
        self.assertEqual(interpolated_datacube.shape, (2, 2, 2))

        # Check the values of the interpolated data
        expected_values = np.array(
            [[[0.15, 0.25], [0.45, 0.55]], [[0.75, 0.85], [1.05, 1.15]]]
        )
        np.testing.assert_array_almost_equal(
            interpolated_datacube, expected_values, decimal=2
        )


if __name__ == "__main__":
    unittest.main()
