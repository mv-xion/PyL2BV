import unittest

import numpy as np

from bioretrieval.processing.mlra_gpr import MLRA_GPR


class TestMLRAGPR(unittest.TestCase):

    def setUp(self):
        # Sample data for testing
        self.image = np.random.rand(10, 5, 5)  # Example 3D image
        self.bio_model = {
            "hyp_ell_GREEN": np.random.rand(10),
            "X_train_GREEN": np.random.rand(10, 10),
            "mean_model_GREEN": np.random.rand(),
            "hyp_sig_GREEN": np.random.rand(),
            "XDX_pre_calc_GREEN": np.random.rand(10),
            "alpha_coefficients_GREEN": np.random.rand(10),
            "Linv_pre_calc_GREEN": np.random.rand(10, 10),
            "hyp_sig_unc_GREEN": np.random.rand(),
        }
        self.gpr = MLRA_GPR(self.image, self.bio_model)

    def test_GPR_mapping_pixel(self):
        # Sample arguments for a single pixel
        args = (
            self.image[:, 0, 0],
            self.bio_model["hyp_ell_GREEN"],
            self.bio_model["X_train_GREEN"],
            self.bio_model["mean_model_GREEN"],
            self.bio_model["hyp_sig_GREEN"],
            self.bio_model["XDX_pre_calc_GREEN"],
            self.bio_model["alpha_coefficients_GREEN"],
            self.bio_model["Linv_pre_calc_GREEN"],
            self.bio_model["hyp_sig_unc_GREEN"],
        )

        mean_pred, variance = self.gpr.GPR_mapping_pixel(args)

        # Check if the output is a tuple
        self.assertIsInstance((mean_pred, variance), tuple)

        # Check if the mean_pred and variance are floats
        self.assertIsInstance(mean_pred, float)
        self.assertIsInstance(variance, float)

        # Check if mean_pred is non-negative
        self.assertGreaterEqual(mean_pred, 0)


if __name__ == "__main__":
    unittest.main()
