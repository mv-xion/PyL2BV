import logging
import numpy as np

from multiprocessing import cpu_count
from joblib import parallel_backend, Parallel, delayed

from PyL2BV.pyl2bv_code.processing.mlra import MLRA_Methods

# Retrieve the loggers by name
image_logger = logging.getLogger("image_logger")


class MLRA_GPR(MLRA_Methods):
    def __init__(self, image: np.ndarray, bio_model) -> None:
        super().__init__(image, bio_model)
        image_logger.debug("Initialized MLRA_GPR with image and bio_model.")

        # Load large arrays once
        self.hyp_ell_GREEN = bio_model["hyp_ell_GREEN"]
        self.X_train_GREEN = bio_model["X_train_GREEN"]
        self.mean_model_GREEN = bio_model["mean_model_GREEN"]
        self.hyp_sig_GREEN = bio_model["hyp_sig_GREEN"]
        self.XDX_pre_calc_GREEN = bio_model["XDX_pre_calc_GREEN"].flatten()
        self.alpha_coefficients_GREEN = bio_model["alpha_coefficients_GREEN"].flatten()
        self.Linv_pre_calc_GREEN = bio_model["Linv_pre_calc_GREEN"]
        self.hyp_sig_unc_GREEN = bio_model["hyp_sig_unc_GREEN"]

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
        Variance = np.sqrt(np.abs(self.hyp_sig_unc_GREEN - np.sum(Vvector**2, axis=0)))

        return mean_pred, Variance

    def perform_mlra(self) -> tuple:
        try:

            ydim, xdim = self.image.shape[1:]
            num_pixels = ydim * xdim

            pixels = self.image.reshape(self.image.shape[0], num_pixels).T

            # Split into smaller batches for parallel processing
            num_cores = cpu_count()

            batch_size = max(1000, num_pixels // (num_cores * 2))
            pixel_batches = [pixels[i:i + batch_size] for i in range(0, num_pixels, batch_size)]

            # Parallelize pixel batch processing
            with parallel_backend("loky", n_jobs=num_cores):
                results = Parallel()(
                    delayed(self.process_pixel_batch)(batch) for batch in pixel_batches
                )

            mean_pred = np.concatenate([res[0] for res in results])
            Variance = np.concatenate([res[1] for res in results])

            variable_map = mean_pred.reshape(ydim, xdim)
            uncertainty_map = Variance.reshape(ydim, xdim)

            return variable_map, uncertainty_map
        except Exception as e:
            image_logger.error(f"Error in perform_mlra: {e}")
            raise
