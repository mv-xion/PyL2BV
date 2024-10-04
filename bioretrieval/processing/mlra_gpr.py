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
        """
        Initialize GPRMapping by inheriting from MLRA_Methods.
        :param image: The image (3D cube) from MLRA_Methods.
        :param bio_model: The single biological model or hyperparameters from MLRA_Methods.
        """
        super().__init__(
            image, bio_model
        )  # Initialize the parent class (MLRA_Methods)
        logging.info("Initialized MLRA_GPR with image and bio_model.")

    def GPR_mapping_pixel(self, args) -> tuple:
        """
        GPR retrieval function for one pixel:
        outputs the mean and variance value calculated
        """
        try:
            # Access args
            (
                pixel_spectra,
                hyp_ell_GREEN,
                X_train_GREEN,
                mean_model_GREEN,
                hyp_sig_GREEN,
                XDX_pre_calc_GREEN,
                alpha_coefficients_GREEN,
                Linv_pre_calc_GREEN,
                hyp_sig_unc_GREEN,
            ) = args

            # Use the bio_model attributes
            im_norm_ell2D = pixel_spectra
            im_norm_ell2D_hypell = im_norm_ell2D * hyp_ell_GREEN

            # Flatten to array
            im_norm_ell2D = im_norm_ell2D.reshape(-1, 1)
            im_norm_ell2D_hypell = im_norm_ell2D_hypell.reshape(-1, 1)

            PtTPt = np.matmul(
                np.transpose(im_norm_ell2D_hypell), im_norm_ell2D
            ).ravel() * (-0.5)
            PtTDX = (
                np.matmul(X_train_GREEN, im_norm_ell2D_hypell)
                .ravel()
                .flatten()
            )

            arg1 = np.exp(PtTPt) * hyp_sig_GREEN
            k_star = np.exp(PtTDX - (XDX_pre_calc_GREEN.ravel() * 0.5))

            mean_pred = (
                np.dot(k_star.ravel(), alpha_coefficients_GREEN.ravel()) * arg1
            ) + mean_model_GREEN
            filterDown = np.greater(mean_pred, 0).astype(int)
            mean_pred = mean_pred * filterDown

            k_star_uncert = (
                np.exp(PtTDX - (XDX_pre_calc_GREEN.ravel() * 0.5)) * arg1
            )
            Vvector = np.matmul(
                Linv_pre_calc_GREEN, k_star_uncert.reshape(-1, 1)
            ).ravel()

            Variance = math.sqrt(
                abs(hyp_sig_unc_GREEN - np.dot(Vvector, Vvector))
            )

            return mean_pred.item(), Variance
        except Exception as e:
            logging.error(f"Error in GPR_mapping_pixel: {e}")
            raise

    @property
    def perform_mlra(self) -> tuple:
        """
        GPR function parallel processing:
        Output: retrieved variable map, map of uncertainty
        """
        try:
            logging.info("Starting perform_mlra.")
            ydim, xdim = self.image.shape[1:]

            variable_map = np.empty((ydim, xdim))
            uncertainty_map = np.empty((ydim, xdim))

            hyp_ell_GREEN = self.bio_model["hyp_ell_GREEN"]
            X_train_GREEN = self.bio_model["X_train_GREEN"]
            mean_model_GREEN = self.bio_model["mean_model_GREEN"]
            hyp_sig_GREEN = self.bio_model["hyp_sig_GREEN"]
            XDX_pre_calc_GREEN = self.bio_model["XDX_pre_calc_GREEN"]
            alpha_coefficients_GREEN = self.bio_model[
                "alpha_coefficients_GREEN"
            ]
            Linv_pre_calc_GREEN = self.bio_model["Linv_pre_calc_GREEN"]
            hyp_sig_unc_GREEN = self.bio_model["hyp_sig_unc_GREEN"]

            # Create a list of arguments (only the pixel spectra) for each pixel in the image
            args_list = [
                (
                    self.image[:, f, v],
                    hyp_ell_GREEN,
                    X_train_GREEN,
                    mean_model_GREEN,
                    hyp_sig_GREEN,
                    XDX_pre_calc_GREEN,
                    alpha_coefficients_GREEN,
                    Linv_pre_calc_GREEN,
                    hyp_sig_unc_GREEN,
                )
                for f in range(ydim)
                for v in range(xdim)
            ]

            # Use multiprocessing to process the pixels in parallel
            with Pool(processes=cpu_count()) as pool:
                results = pool.map(self.GPR_mapping_pixel, args_list)

            # Store results in the variable and uncertainty maps
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


# GPR function
'''
def GPR_mapping_pixel(args):
    pixel_spectra, hyp_ell_GREEN, X_train_GREEN, mean_model_GREEN, hyp_sig_GREEN, \
        XDX_pre_calc_GREEN, alpha_coefficients_GREEN, Linv_pre_calc_GREEN, hyp_sig_unc_GREEN = args
    """
    GPR retrieval function for one pixel:
    input arguments are given in the main parallel function
    outputs the mean and variance value calculated
    """
    # Image is already normalised
    im_norm_ell2D = pixel_spectra

    # Multiply with hyperparams
    im_norm_ell2D_hypell = im_norm_ell2D * hyp_ell_GREEN

    # flatten toarray toarray1
    im_norm_ell2D = im_norm_ell2D.reshape(-1, 1)
    im_norm_ell2D_hypell = im_norm_ell2D_hypell.reshape(-1, 1)

    PtTPt = np.matmul(np.transpose(im_norm_ell2D_hypell), im_norm_ell2D).ravel() * (-0.5)
    PtTDX = np.matmul(X_train_GREEN, im_norm_ell2D_hypell).ravel().flatten()

    arg1 = np.exp(PtTPt) * hyp_sig_GREEN
    k_star = np.exp(PtTDX - (XDX_pre_calc_GREEN.ravel() * 0.5))

    mean_pred = (np.dot(k_star.ravel(), alpha_coefficients_GREEN.ravel()) * arg1) + mean_model_GREEN
    filterDown = np.greater(mean_pred, 0).astype(int)
    mean_pred = mean_pred * filterDown

    k_star_uncert = np.exp(PtTDX - (XDX_pre_calc_GREEN.ravel() * 0.5)) * arg1
    Vvector = np.matmul(Linv_pre_calc_GREEN, k_star_uncert.reshape(-1, 1)).ravel()

    Variance = math.sqrt(abs(hyp_sig_unc_GREEN - np.dot(Vvector, Vvector)))

    return mean_pred.item(), Variance


def GPR_mapping_parallel(image, hyp_ell_GREEN, X_train_GREEN, mean_model_GREEN, hyp_sig_GREEN,
                         XDX_pre_calc_GREEN, alpha_coefficients_GREEN, Linv_pre_calc_GREEN, hyp_sig_unc_GREEN):
    """
    GPR function parallel processing:

    Input parameters: image(dim,y,x), hyperparams mx, X train, mean model,
    hyperparam estimate, XDX, alpha coefficients, Linverse, hyperparams uncertainty mx

    Output: retrieved variable map, map of uncertainty
    """
    ydim, xdim = image.shape[1:]

    variable_map = np.empty((ydim, xdim))
    uncertainty_map = np.empty((ydim, xdim))

    args_list = [(image[:, f, v], hyp_ell_GREEN, X_train_GREEN, mean_model_GREEN,
                  hyp_sig_GREEN, XDX_pre_calc_GREEN, alpha_coefficients_GREEN, Linv_pre_calc_GREEN, hyp_sig_unc_GREEN)
                 for f in range(ydim) for v in range(xdim)]

    with Pool(processes=cpu_count()) as pool:  # process on different cpus (as many as we have)
        results = pool.map(GPR_mapping_pixel, args_list)

    for i, (mean_pred, Variance) in enumerate(results):
        f = i // xdim
        v = i % xdim
        variable_map[f, v] = mean_pred
        uncertainty_map[f, v] = Variance

    return variable_map, uncertainty_map
'''
