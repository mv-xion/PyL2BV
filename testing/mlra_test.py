import unittest

import numpy as np

from bioretrieval.processing.mlra import MLRA_Methods


class TestMLRA_Methods(unittest.TestCase):
    def setUp(self):
        self.image = np.random.rand(10, 10, 10)  # Example 3D image
        self.bio_model = {"param1": 1, "param2": 2}  # Example bio_model
        self.mlra_methods = MLRA_Methods(self.image, self.bio_model)

    def test_initialization(self):
        self.assertTrue(np.array_equal(self.mlra_methods.image, self.image))
        self.assertEqual(self.mlra_methods.bio_model, self.bio_model)

    def test_perform_mlra_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.mlra_methods.perform_mlra()


if __name__ == "__main__":
    unittest.main()
