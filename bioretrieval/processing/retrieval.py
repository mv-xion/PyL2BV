"""
    Contains the retrieval class which performs the retrieval
    from reading the image to writing the retrieved result
"""

import concurrent.futures
import importlib
import logging
import os
import pickle
import sys
from time import time

import numpy as np
from matplotlib import pyplot as plt
from netCDF4 import Dataset
from spectral.io import envi

from bioretrieval.auxiliar.image_read import (
    read_envi,
    read_netcdf,
    show_reflectance_img,
)
from bioretrieval.auxiliar.logger_class import Logger
from bioretrieval.auxiliar.spectra_interpolation import spline_interpolation
from bioretrieval.processing.mlra_gpr import MLRA_GPR

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class Retrieval:
    def __init__(
        self,
        logfile: Logger,
        show_message: callable,
        input_file: str,
        input_type: str,
        output_file: str,
        model_path: str,
        conversion_factor: float,
        plotting: bool,
    ):
        """
        Initialise the retrieval class
        :param logfile: path to the log file
        :param show_message: function for printing the message to gui
        :param input_file: path to the input file
        :param input_type: type of input file
        :param output_file: path to the output file
        :param model_path: path to the models directory
        :param conversion_factor: image conversion factor
        :param plotting: bool to plot the results or not
        """
        self.plotting = plotting
        self.conversion_factor = conversion_factor
        self.number_of_models = None
        self.bio_models = []  # Storing the models
        self.variable_maps = []  # Storing variable maps
        self.uncertainty_maps = []  # Storing uncertainty maps
        self.start = None
        self.end = None
        self.process_time = None
        self.map_info = None
        self.longitude = None
        self.latitude = None
        self.img_wavelength = None
        self.img_reflectance = None
        self.data_norm = None
        self.logger = logfile
        self.show_message = show_message
        self.input_file = input_file
        self.input_type = input_type
        self.output_file = output_file
        self.model_path = model_path

    # TODO: Try catch blocks, exceptions
    @property
    def bio_retrieval(self) -> bool:
        self.logger.open()
        logging.info("Reading image...")
        self.show_message("Reading image...")
        self.start = time()
        # __________________________Split image read by file type______________

        if self.input_type == "CHIME netCDF":
            image_data = read_netcdf(self.input_file, self.conversion_factor)
            self.img_reflectance = image_data[0]  # save reflectance
            self.img_wavelength = image_data[1]  # save wavelength
            self.map_info = False
        elif self.input_type == "ENVI Standard":
            image_data = read_envi(self.input_file, self.conversion_factor)
            self.img_reflectance = image_data[0]  # save reflectance
            self.img_wavelength = image_data[1]  # save wavelength
            if len(image_data) == 4:
                self.show_message("Map info included")
                self.map_info = True
                self.latitude = image_data[2]
                self.longitude = image_data[3]
            else:
                self.show_message("No map info")
                self.map_info = False
        self.end = time()
        self.process_time = self.end - self.start
        self.rows, self.cols, self.dims = self.img_reflectance.shape

        logging.info(f"Image read. Elapsed time: {self.process_time}")
        self.show_message(f"Image read. Elapsed time: {self.process_time}")
        self.logger.log_message(
            f"Image read. Elapsed time: {self.process_time}\n"
        )

        # Showing image
        if self.plotting:
            show_reflectance_img(self.img_reflectance, self.img_wavelength)

        # ___________________________Reading models____________________________

        # Getting path of the model files
        try:
            list_of_files = os.listdir(self.model_path)
            if not list_of_files:
                raise FileNotFoundError(
                    f"No models found in path: {self.model_path}"
                )
        except Exception as e:
            logging.error(e)
            self.show_message(str(e))
            self.logger.log_message(f"{e}\n")
            return True
        list_of_models = list(
            filter(lambda file: file.endswith(".py"), list_of_files)
        )
        self.number_of_models = len(list_of_models)
        logging.info(
            f"Getting model {self.number_of_models} names was successful."
        )
        self.show_message(
            f"Getting model {self.number_of_models} names was successful."
        )
        self.logger.log_message(
            f"Getting model {self.number_of_models} names was successful.\n"
        )

        # Importing the models
        sys.path.append(self.model_path)

        # Reading the models
        def import_and_log_model(model_file, bio_models, show_message, logger):
            # Importing model
            module = importlib.import_module(
                os.path.splitext(model_file)[0], package=None
            )
            bio_models.append(module)
            message = f"{module.model} imported"
            logging.info(message)
            show_message(message)
            logger.log_message(message + "\n")

        # Assuming self.bio_models, self.show_message, and self.logger are defined
        list(
            map(
                lambda model_file: import_and_log_model(
                    model_file, self.bio_models, self.show_message, self.logger
                ),
                list_of_models,
            )
        )

        # _________________________________Retrieval___________________________________________

        def run_model(i):
            logging.info(f"Running {self.bio_models[i].model} model")
            self.show_message(f"Running {self.bio_models[i].model} model")
            self.logger.log_message(
                f"Running {self.bio_models[i].model} model\n"
            )

            logging.info("Band selection...")
            self.show_message("Band selection...")

            # Band selection of the image
            self.start = time()
            data_refl_new = self.band_selection(i)
            self.end = time()
            self.process_time = self.end - self.start

            logging.info(
                f"Bands selected. Shape: {data_refl_new.shape} \nElapsed time: {self.process_time}"
            )
            self.show_message(
                f"Bands selected. Shape: {data_refl_new.shape} \nElapsed time: {self.process_time}"
            )
            self.logger.log_message(
                f"Image read.Bands selected. Shape: {data_refl_new.shape}"
                f" \nElapsed time: {self.process_time}\n"
            )

            # Normalising the image
            self.data_norm = norm_data(
                data_refl_new,
                self.bio_models[i].mx_GREEN,
                self.bio_models[i].sx_GREEN,
            )

            # Perform PCA if there is data
            if (
                hasattr(self.bio_models[i], "pca_mat")
                and len(self.bio_models[i].pca_mat) > 0
            ):
                self.data_norm = self.data_norm.dot(self.bio_models[i].pca_mat)

            if self.bio_models[i].model_type == "GPR":
                # Changing axes to because GPR function takes dim,y,x
                self.data_norm = np.swapaxes(
                    self.data_norm, 0, 1
                )  # swapping axes to have the right order after transpose
                self.img_array = np.transpose(self.data_norm)

                # Transform model to dictionary
                model_dict = module_to_dict(
                    self.bio_models[i]
                )  # we dont use it now

                logging.info("Running GPR...")
                self.show_message("Running GPR...")

                gpr_object = MLRA_GPR(self.img_array, model_dict)
                self.start = time()

                # Starting GPR
                variable_map, uncertainty_map = gpr_object.perform_mlra()
                self.end = time()

                # Logging
                self.process_time = self.end - self.start
                logging.info(f"Elapsed time of GPR: {self.process_time}")
                self.show_message(f"Elapsed time of GPR: {self.process_time}")
                self.logger.log_message(
                    f"Elapsed time of GPR: {self.process_time}\n"
                )
                self.variable_maps.append(variable_map)
                self.uncertainty_maps.append(uncertainty_map)

                logging.info(
                    f"Retrieval of {self.bio_models[i].veg_index} was successful."
                )
                self.show_message(
                    f"Retrieval of {self.bio_models[i].veg_index} was successful."
                )
                self.logger.log_message(
                    f"Retrieval of {self.bio_models[i].veg_index} was successful.\n"
                )

        # Use ThreadPoolExecutor to run models in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(run_model, i)
                for i in range(self.number_of_models)
            ]
            for future in concurrent.futures.as_completed(futures):
                future.result()  # This will raise any exceptions caught during execution

        # _________________________________Finishing tasks_____________________
        self.logger.close()
        return False

    def band_selection(self, i: int) -> np.ndarray:
        current_wl = self.img_wavelength
        expected_wl = self.bio_models[i].wave_length
        # Find the intersection of the two lists of wavelength
        if len(np.intersect1d(current_wl, expected_wl)) == len(expected_wl):
            reflectances_new = self.img_reflectance[
                :, :, np.where(np.in1d(current_wl, expected_wl))[0]
            ]
            logging.info("Matching bands found.")
            self.show_message("Matching bands found.")
            self.logger.log_message("Matching bands found.\n")
        else:
            logging.info(
                "No matching bands found, spline interpolation is applied."
            )
            self.show_message(
                "No matching bands found, spline interpolation is applied."
            )
            self.logger.log_message(
                "No matching bands found, spline interpolation is applied.\n"
            )
            reflectances_new = spline_interpolation(
                current_wl, self.img_reflectance, expected_wl
            )

        return reflectances_new  # returning the selected bands

    def export_retrieval(self) -> bool:
        self.logger.open()
        logging.info("Exporting image...")
        self.show_message("Exporting image...")
        self.start = time()
        # __________________________Split image export by file type______________

        if self.input_type == "CHIME netCDF":
            self.export_netcdf()
        elif self.input_type == "ENVI Standard":
            self.export_envi()
        self.end = time()
        self.process_time = self.end - self.start

        logging.info(f"Image exported. Elapsed time:{self.process_time}")
        self.show_message(f"Image exported. Elapsed time:{self.process_time}")
        self.logger.log_message(
            f"Image exported. Elapsed time:{self.process_time}\n"
        )

        logging.info(f"Show images")
        self.show_message(f"Show images")
        self.logger.log_message(f"Show images")
        if self.plotting:
            self.show_results()

        self.logger.close()
        return False

    def export_netcdf(self):
        # Creating output image
        # Create a new netCDF file
        nc_file = Dataset(self.output_file, "w", format="NETCDF4")

        # Set global attributes
        nc_file.title = "CHIME-E2E Level-2B product data"
        nc_file.institution = "University of Valencia (UVEG)"
        nc_file.source = "L2GPP"
        nc_file.history = "File generated by L2B Module"
        nc_file.references = "L2B.MO.01"
        nc_file.comment = "n/a"

        # Create groups
        for i in range(self.number_of_models):
            group = nc_file.createGroup(self.bio_models[i].veg_index)
            if self.bio_models[i].veg_index == "LCC":
                group.long_name = "Leaf Chlorophyll Content (LCC)"
            elif self.bio_models[i].veg_index == "LWC":
                group.long_name = "Leaf Water Content (LWC)"
            elif self.bio_models[i].veg_index == "LNC":
                group.long_name = "Leaf Nitrogen Content (LNC)"
            elif self.bio_models[i].veg_index == "LMA":
                group.long_name = "Leaf Mass Area (LMA)"
            elif self.bio_models[i].veg_index == "LAI":
                group.long_name = "Leaf Area Index (LAI)"
            elif self.bio_models[i].veg_index == "CCC":
                group.long_name = "Canopy Chlorophyll Content (CCC)"
            elif self.bio_models[i].veg_index == "CWC":
                group.long_name = "Canopy Water Content (CWC)"
            elif self.bio_models[i].veg_index == "CDMC":
                group.long_name = "Canopy Dry Matter Content (CDMC)"
            elif self.bio_models[i].veg_index == "CNC":
                group.long_name = "Canopy Nitrogen Content (CNC)"
            elif self.bio_models[i].veg_index == "FVC":
                group.long_name = "Fractional Vegetation Cover (FVC)"
            elif self.bio_models[i].veg_index == "FAPAR":
                group.long_name = "Fraction of Absorbed Photosynthetically Active Radiation (FAPAR)"

            # Create dimensions for group
            nl_dim = group.createDimension("Nl", self.rows)
            nc_dim = group.createDimension("Nc", self.cols)

            # Create variables for group
            retrieval_var = group.createVariable(
                "Retrieval", "f4", dimensions=("Nc", "Nl")
            )
            retrieval_var.units = self.bio_models[
                i
            ].units  # Adding the 'Units' attribute
            sd_var = group.createVariable("SD", "f4", dimensions=("Nc", "Nl"))
            sd_var.units = self.bio_models[i].units
            cv_var = group.createVariable("CV", "f4", dimensions=("Nc", "Nl"))
            cv_var.units = "%"
            qf_var = group.createVariable("QF", "i1", dimensions=("Nc", "Nl"))
            qf_var.units = "adim"

            # Assign data to the variable
            # Transpose for matlab type output
            retrieval_var[:] = np.transpose(self.variable_maps[i])
            sd_var[:] = np.transpose(self.uncertainty_maps[i])

        logging.info(
            f"NetCDF file created successfully at: {self.output_file}"
        )
        self.show_message(
            f"NetCDF file created successfully at: {self.output_file}"
        )
        self.logger.log_message(
            f"NetCDF file created successfully at: {self.output_file}\n"
        )

    def export_envi(self):
        # Open the ENVI file
        envi_image = envi.open(
            self.input_file,
            os.path.join(
                os.path.dirname(self.input_file),
                os.path.splitext(os.path.basename(self.input_file))[0],
            ),
        )
        # Storing all the metadata
        info = envi_image.metadata

        # Construct band names
        band_names = []
        for i in range(self.number_of_models):
            band_names.append(self.bio_models[i].veg_index)
            band_names.append(f"{self.bio_models[i].veg_index}_sd")

        # Define metadata ENVI Standard
        metadata = {
            "description": "Exported from Python",
            "samples": info["samples"],
            "lines": info["lines"],
            "bands": 2,
            "header offset": 0,
            "file type": info["file type"],
            "data type": 5,  # Float32 data type
            "interleave": info["interleave"],
            "sensor type": "unknown",
            "byte order": info["byte order"],
            "map info": info["map info"],
            "coordinate system string": info["coordinate system string"],
            "band names": band_names,
        }

        # Specify file paths
        file_path = self.output_file + ".hdr"

        # Check that both lists have the same length
        # Both lists must have the same length
        assert len(self.variable_maps) == len(self.uncertainty_maps)

        # Create an interleaved list of matrices
        interleaved_matrices = [
            matrix
            for pair in zip(self.variable_maps, self.uncertainty_maps)
            for matrix in pair
        ]

        # Stack the interleaved matrices along a new axis
        stacked_data = np.stack(interleaved_matrices, axis=-1)

        # Save the data to an ENVI file
        envi.save_image(
            file_path,
            stacked_data,
            interleave=info["interleave"],
            metadata=metadata,
        )

        logging.info(f"ENVI file created successfully at: {self.output_file}")
        self.show_message(
            f"ENVI file created successfully at: {self.output_file}"
        )
        self.logger.log_message(
            f"ENVI file created successfully at: {self.output_file}\n"
        )

    # TODO: maybe in a new GUI window?
    def show_results(self):
        """
        Show results and export function of retrieval
        :return: plot images, save in files
        !! Only for CCC,CWC, LAI for now !!
        """
        # Create directories for images
        img_dir = os.path.join(os.path.dirname(self.output_file), "images")
        # Check if the directory exists, and create it if it does not
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

        vec_dir = os.path.join(os.path.dirname(self.output_file), "vectors")
        # Check if the directory exists, and create it if it does not
        if not os.path.exists(vec_dir):
            os.makedirs(vec_dir)

        for i in range(self.number_of_models):
            if self.bio_models[i].veg_index == "CCC":
                colormap = "Greens"
            elif self.bio_models[i].veg_index == "CWC":
                colormap = "Blues"
            elif self.bio_models[i].veg_index == "LAI":
                colormap = "YlGn"
            else:
                colormap = "viridis"

            # Showing the result image
            plt.imshow(self.variable_maps[i], cmap=colormap)
            if self.bio_models[i].veg_index == "LAI":
                plt.title(
                    f"Estimated {self.bio_models[i].veg_index} map (m$^2$/m$^2$)"
                )
            else:
                plt.title(
                    f"Estimated {self.bio_models[i].veg_index} map (g/m$^2$)"
                )
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    img_dir,
                    f"{os.path.basename(self.output_file)}{self.bio_models[i].veg_index}.png",
                ),
                bbox_inches="tight",
            )
            plt.savefig(
                os.path.join(
                    vec_dir,
                    f"{os.path.basename(self.output_file)}{self.bio_models[i].veg_index}.pdf",
                ),
                bbox_inches="tight",
            )
            plt.show()

            # Showing the uncertainty image
            plt.imshow(self.uncertainty_maps[i], cmap="jet")
            if self.bio_models[i].veg_index == "LAI":
                plt.title(
                    f"Uncertainty of {self.bio_models[i].veg_index} map (m$^2$/m$^2$)"
                )
            else:
                plt.title(
                    f"Uncertainty of {self.bio_models[i].veg_index} map (g/m$^2$)"
                )
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    img_dir,
                    f"{os.path.basename(self.output_file)}{self.bio_models[i].veg_index}_uncertainty.png",
                ),
                bbox_inches="tight",
            )
            plt.savefig(
                os.path.join(
                    vec_dir,
                    f"{os.path.basename(self.output_file)}{self.bio_models[i].veg_index}_uncertainty.pdf",
                ),
                bbox_inches="tight",
            )
            plt.show()


# Normalise data function
def norm_data(data: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (data - mean) / std


# Parallel multiprocess cant pick modules
def module_to_dict(bio_model) -> dict:
    """
    Converts the attributes of an already existing module into a dictionary.

    :param bio_model: The module or object containing hyperparameters
    :return: Dictionary with the module's attributes
    """
    # Convert the module's attributes to a dictionary, excluding special
    # methods/attributes
    module_dict = {
        key: value
        for key, value in vars(bio_model).items()
        if not key.startswith("__")
    }

    module_dict = {k: v for k, v in module_dict.items() if is_picklable(v)}

    return module_dict


# Function to check if an object is picklable
def is_picklable(obj):
    try:
        pickle.dumps(obj)
    except (pickle.PicklingError, TypeError):
        return False
    return True
