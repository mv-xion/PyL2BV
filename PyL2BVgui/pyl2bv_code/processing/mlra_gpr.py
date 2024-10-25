import logging
import math
from multiprocessing import Pool, cpu_count

import numpy as np
from joblib import Parallel, delayed

from pyl2bv_code.processing.mlra import MLRA_Methods


class MLRA_GPR(MLRA_Methods):
    def __init__(self, image: np.ndarray, bio_model) -> None:
        super().__init__(image, bio_model)
        logging.info("Initialized MLRA_GPR with image and bio_model.")

        # Load large arrays once
        self.hyp_ell_GREEN = bio_model["hyp_ell_GREEN"]
        self.X_train_GREEN = bio_model["X_train_GREEN"]
        self.mean_model_GREEN = bio_model["mean_model_GREEN"]
        self.hyp_sig_GREEN = bio_model["hyp_sig_GREEN"]
        self.XDX_pre_calc_GREEN = bio_model["XDX_pre_calc_GREEN"].flatten()
        self.alpha_coefficients_GREEN = bio_model[
            "alpha_coefficients_GREEN"
        ].flatten()
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

    def process_pixel_batch(self, batch: np.ndarray) -> tuple:
        im_norm_ell2D = batch
        im_norm_ell2D_hypell = im_norm_ell2D * self.hyp_ell_GREEN

        PtTPt = -0.5 * np.sum(im_norm_ell2D_hypell * im_norm_ell2D, axis=1)
        PtTDX = im_norm_ell2D_hypell @ self.X_train_GREEN.T

        arg1 = np.exp(PtTPt) * self.hyp_sig_GREEN
        k_star = np.exp(PtTDX - (0.5 * self.XDX_pre_calc_GREEN))

        mean_pred = (
            k_star @ self.alpha_coefficients_GREEN
        ) * arg1 + self.mean_model_GREEN
        mean_pred = np.maximum(mean_pred, 0)

        k_star_uncert = k_star * arg1[:, np.newaxis]
        Vvector = self.Linv_pre_calc_GREEN @ k_star_uncert.T
        Variance = np.sqrt(
            np.abs(self.hyp_sig_unc_GREEN - np.sum(Vvector**2, axis=0))
        )

        return mean_pred, Variance

    def perform_mlra(self) -> tuple:
        try:
            logging.info("Starting perform_mlra.")
            ydim, xdim = self.image.shape[1:]
            num_pixels = ydim * xdim

            pixels = self.image.reshape(self.image.shape[0], num_pixels).T

            # Split into smaller batches for parallel processing
            num_cores = cpu_count()
            batch_size = num_pixels // num_cores
            pixel_batches = np.array_split(pixels, num_cores)

            # Parallelize pixel batch processing
            results = Parallel(n_jobs=num_cores)(
                delayed(self.process_pixel_batch)(batch)
                for batch in pixel_batches
            )

            mean_pred = np.concatenate([res[0] for res in results])
            Variance = np.concatenate([res[1] for res in results])

            variable_map = mean_pred.reshape(ydim, xdim)
            uncertainty_map = Variance.reshape(ydim, xdim)

            logging.info("Completed perform_mlra.")
            return variable_map, uncertainty_map
        except Exception as e:
            logging.error(f"Error in perform_mlra: {e}")
            raise
