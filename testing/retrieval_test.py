import os
import shutil
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from bioretrieval.processing.retrieval import Retrieval, norm_data


class TestRetrieval(unittest.TestCase):
    def setUp(self):
        self.logfile = MagicMock()
        self.show_message = MagicMock()
        self.input_file = "test_input.nc"
        self.input_type = "CHIME netCDF"
        self.output_file = "test_output"
        self.model_path = "test_models"
        self.conversion_factor = 0.0001

        # Create test directories and files
        os.makedirs(self.model_path, exist_ok=True)
        with open(self.input_file, "w") as f:
            f.write("test data")
        with open(os.path.join(self.model_path, "test_model.py"), "w") as f:
            f.write(
                "model = 'test_model'\nmodel_type = 'GPR'\nveg_index = 'CCC'\nunits = 'g/m^2'\nwave_length = [400, 500, 600]\nmx_GREEN = 0.5\nsx_GREEN = 0.1\npca_mat = []"
            )

        self.retrieval = Retrieval(
            self.logfile,
            self.show_message,
            self.input_file,
            self.input_type,
            self.output_file,
            self.model_path,
            self.conversion_factor,
        )

    def tearDown(self):
        # Remove test directories and files
        if os.path.exists(self.input_file):
            os.remove(self.input_file)
        if os.path.exists(self.model_path):
            shutil.rmtree(self.model_path)
        if os.path.exists(self.output_file):
            os.remove(self.output_file)

    @patch("retrieval.read_netcdf")
    @patch("retrieval.show_reflectance_img")
    def test_bio_retrieval(self, mock_show_reflectance_img, mock_read_netcdf):
        mock_read_netcdf.return_value = (
            np.random.rand(10, 10, 3),
            [400, 500, 600],
        )

        result = self.retrieval.bio_retrieval

        self.assertFalse(result)
        self.show_message.assert_any_call("Reading image...")
        self.show_message.assert_any_call("Image read. Elapsed time:")
        self.show_message.assert_any_call("Retrieval of CCC was successful.")

    def test_band_selection(self):
        self.retrieval.img_wavelength = np.array([400, 500, 600])
        self.retrieval.img_reflectance = np.random.rand(10, 10, 3)
        self.retrieval.bio_models = [MagicMock()]
        self.retrieval.bio_models[0].wave_length = [400, 500, 600]

        result = self.retrieval.band_selection(0)

        self.assertEqual(result.shape, (10, 10, 3))
        self.show_message.assert_any_call("Matching bands found.")

    def test_norm_data(self):
        data = np.array([1, 2, 3])
        mean = 2
        std = 1
        result = norm_data(data, mean, std)
        expected = np.array([-1, 0, 1])

        np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
