# GPR class

import logging
import math
from multiprocessing import Pool, cpu_count

import numpy as np

from bioretrieval.processing.mlra import MLRA_Methods

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# GPRMapping inherits from MLRA_Methods
class MLRA_GPR(MLRA_Methods):
    def __init__(self, image: np.ndarray, bio_model) -> None:
        super().__init__(image, bio_model)
        logging.info("Initialized MLRA_GPR with image and bio_model.")

        # Load large arrays once
        self.hyp_ell_GREEN = bio_model["hyp_ell_GREEN"]
        self.X_train_GREEN = bio_model["X_train_GREEN"]
        self.mean_model_GREEN = bio_model["mean_model_GREEN"]
        self.hyp_sig_GREEN = bio_model["hyp_sig_GREEN"]
        self.XDX_pre_calc_GREEN = bio_model["XDX_pre_calc_GREEN"]
        self.alpha_coefficients_GREEN = bio_model["alpha_coefficients_GREEN"]
        self.Linv_pre_calc_GREEN = bio_model["Linv_pre_calc_GREEN"]
        self.hyp_sig_unc_GREEN = bio_model["hyp_sig_unc_GREEN"]

    def GPR_mapping_pixel(self, pixel_spectra) -> tuple:
        try:
            im_norm_ell2D = pixel_spectra
            im_norm_ell2D_hypell = im_norm_ell2D * self.hyp_ell_GREEN

            im_norm_ell2D = im_norm_ell2D.reshape(-1, 1)
            im_norm_ell2D_hypell = im_norm_ell2D_hypell.reshape(-1, 1)

            PtTPt = np.matmul(
                np.transpose(im_norm_ell2D_hypell), im_norm_ell2D
            ).ravel() * (-0.5)
            PtTDX = (
                np.matmul(self.X_train_GREEN, im_norm_ell2D_hypell)
                .ravel()
                .flatten()
            )

            arg1 = np.exp(PtTPt) * self.hyp_sig_GREEN
            k_star = np.exp(PtTDX - (self.XDX_pre_calc_GREEN.ravel() * 0.5))

            mean_pred = (
                np.dot(k_star.ravel(), self.alpha_coefficients_GREEN.ravel())
                * arg1
            ) + self.mean_model_GREEN
            filterDown = np.greater(mean_pred, 0).astype(int)
            mean_pred = mean_pred * filterDown

            k_star_uncert = (
                np.exp(PtTDX - (self.XDX_pre_calc_GREEN.ravel() * 0.5)) * arg1
            )
            Vvector = np.matmul(
                self.Linv_pre_calc_GREEN, k_star_uncert.reshape(-1, 1)
            ).ravel()

            Variance = math.sqrt(
                abs(self.hyp_sig_unc_GREEN - np.dot(Vvector, Vvector))
            )

            return mean_pred.item(), Variance
        except Exception as e:
            logging.error(f"Error in GPR_mapping_pixel: {e}")
            raise

    @property
    def perform_mlra(self) -> tuple:
        try:
            logging.info("Starting perform_mlra.")
            ydim, xdim = self.image.shape[1:]

            variable_map = np.empty((ydim, xdim))
            uncertainty_map = np.empty((ydim, xdim))

            args_list = [
                self.image[:, f, v] for f in range(ydim) for v in range(xdim)
            ]

            with Pool(processes=cpu_count()) as pool:
                results = pool.map(self.GPR_mapping_pixel, args_list)

            for i, (mean_pred, Variance) in enumerate(results):
                f = i // xdim
                v = i % xdim
                variable_map[f, v] = mean_pred
                uncertainty_map[f, v] = Variance

            logging.info("Completed perform_mlra.")
            return variable_map, uncertainty_map
        except Exception as e:
            logging.error(f"Error in perform_mlra: {e}")
            raise
