import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from bioretrieval.auxiliar.image_read import (get_lat_lon_envi, read_envi,
                                              read_netcdf,
                                              show_reflectance_img)


class TestImageRead(unittest.TestCase):
    @patch("image_read.nc.Dataset")
    def test_read_netcdf(self, mock_nc):
        # Mock the netCDF dataset
        mock_dataset = MagicMock()
        mock_dataset.__getitem__.side_effect = lambda key: {
            "l2a_BOA_rfl": np.array([[[0.1, 0.2], [0.3, 0.4]]]),
            "central_wavelength": np.array([500, 600]),
        }[key]
        mock_nc.return_value = mock_dataset

        path = "mock_path"
        conversion_factor = 0.0001
        data_refl, data_wavelength = read_netcdf(path, conversion_factor)

        np.testing.assert_array_almost_equal(
            data_refl, np.array([[[0.00001, 0.00002], [0.00003, 0.00004]]])
        )
        np.testing.assert_array_equal(data_wavelength, np.array([500, 600]))

    @patch("image_read.envi.open")
    def test_read_envi(self, mock_envi_open):
        # Mock the ENVI image
        mock_envi_image = MagicMock()
        mock_envi_image.asarray.return_value = np.array(
            [[[0.1, 0.2], [0.3, 0.4]]]
        )
        mock_envi_image.metadata = {
            "wavelength": ["500", "600"],
            "map info": [
                "map",
                "info",
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
                "8",
                "datum",
            ],
        }
        mock_envi_open.return_value = mock_envi_image

        path = "mock_path"
        conversion_factor = 0.0001
        data, data_wavelength, longitude, latitude = read_envi(
            path, conversion_factor
        )

        np.testing.assert_array_almost_equal(
            data, np.array([[[0.00001, 0.00002], [0.00003, 0.00004]]])
        )
        np.testing.assert_array_equal(data_wavelength, np.array([500, 600]))
        self.assertIsNotNone(longitude)
        self.assertIsNotNone(latitude)

    def test_get_lat_lon_envi(self):
        map_info = [
            "map",
            "info",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "datum",
        ]
        lon = 2
        lat = 2
        longitude, latitude = get_lat_lon_envi(map_info, lon, lat)

        self.assertEqual(len(longitude), lon)
        self.assertEqual(len(latitude), lat)

    @patch("image_read.plt.show")
    def test_show_reflectance_img(self, mock_show):
        data_refl = np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]])
        data_wavelength = np.array([463, 547, 639])
        show_reflectance_img(data_refl, data_wavelength)
        mock_show.assert_called_once()


if __name__ == "__main__":
    unittest.main()
