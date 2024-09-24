# GPR class

import numpy as np
import math
import time
from multiprocessing import Pool, cpu_count

from bioretrieval.processing.retrieval import Retrieval


class GPRMapping(Retrieval):
    def __init__(self, retrieval_object: Retrieval):
        """
        Initialize the GPRMapping class by copying all attributes from the provided Retrieval object.
        :param retrieval_object: an instance of the Retrieval class
        """
        # Initialize the parent class (Retrieval) by passing the same parameters as the provided Retrieval object
        super().__init__(
            retrieval_object.logger,
            retrieval_object.show_message,
            retrieval_object.input_file,
            retrieval_object.input_type,
            retrieval_object.output_file,
            retrieval_object.model_path,
            retrieval_object.conversion_factor
        )

        # Copy all attributes from the passed Retrieval object to the new instance
        self.img_array = None
        self.__dict__.update(retrieval_object.__dict__)

    def GPR_mapping_pixel(self, pixel_spectra):
        """
        GPR retrieval function for one pixel:
        outputs the mean and variance value calculated
        """
        # Access the bio_model attributes directly
        hyp_ell_GREEN = self.bio_model.hyp_ell_GREEN
        X_train_GREEN = self.bio_model.X_train_GREEN
        mean_model_GREEN = self.bio_model.mean_model_GREEN
        hyp_sig_GREEN = self.bio_model.hyp_sig_GREEN
        XDX_pre_calc_GREEN = self.bio_model.XDX_pre_calc_GREEN
        alpha_coefficients_GREEN = self.bio_model.alpha_coefficients_GREEN
        Linv_pre_calc_GREEN = self.bio_model.Linv_pre_calc_GREEN
        hyp_sig_unc_GREEN = self.bio_model.hyp_sig_unc_GREEN

        # Use the bio_model attributes
        im_norm_ell2D = pixel_spectra
        im_norm_ell2D_hypell = im_norm_ell2D * hyp_ell_GREEN

        # Flatten to array
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

    def GPR_mapping_parallel(self):
        """
        GPR function parallel processing:
        Output: retrieved variable map, map of uncertainty
        """

        ydim, xdim = self.img_array.shape[1:]

        variable_map = np.empty((ydim, xdim))
        uncertainty_map = np.empty((ydim, xdim))

        # Create a list of arguments (only the pixel spectra) for each pixel in the image
        args_list = [self.img_array[:, f, v] for f in range(ydim) for v in range(xdim)]

        # Use multiprocessing to process the pixels in parallel
        with Pool(processes=cpu_count()) as pool:
            results = pool.map(self.GPR_mapping_pixel, args_list)

        # Store results in the variable and uncertainty maps
        for i, (mean_pred, Variance) in enumerate(results):
            f = i // xdim
            v = i % xdim
            variable_map[f, v] = mean_pred
            uncertainty_map[f, v] = Variance

        return variable_map, uncertainty_map

    def perform_GPR(self):

        self.logger.open()
        print('Running GPR...')
        self.show_message('Running GPR...')

        # Changing axes to because GPR function takes dim,y,x
        self.data_norm = np.swapaxes(self.data_norm, 0, 1)  # swapping axes to have the right order after transpose
        self.img_array = np.transpose(self.data_norm)

        self.start = time.time()
        # Starting GPR
        variable_map, uncertainty_map = self.GPR_mapping_parallel()
        self.end = time.time()

        # Logging
        self.process_time = self.end - self.start
        print(f'Elapsed time of GPR: {self.process_time}')
        self.show_message(f'Elapsed time of GPR: {self.process_time}')
        self.logger.log_message(f'Elapsed time of GPR: {self.process_time}\n')
        self.variable_maps.append(variable_map)
        self.uncertainty_maps.append(uncertainty_map)

        print(f'Retrieval of {self.bio_model[i].veg_index} was successful.')
        self.show_message(f'Retrieval of {self.bio_model[i].veg_index} was successful.')
        self.logger.log_message(f'Retrieval of {self.bio_model[i].veg_index} was successful.\n')

        self.logger.close()

        return 0


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
