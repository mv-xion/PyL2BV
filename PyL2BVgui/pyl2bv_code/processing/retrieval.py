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

from PyL2BVgui.pyl2bv_code.auxiliar.image_read import (
    read_envi,
    read_netcdf,
    show_reflectance_img,
)
import PyL2BVgui.pyl2bv_code.auxiliar.logger_config
from PyL2BVgui.pyl2bv_code.auxiliar.spectra_interpolation import (
    spline_interpolation,
)
from PyL2BVgui.pyl2bv_code.processing.mlra_gpr import MLRA_GPR

# Retrieve the loggers by name
app_logger = logging.getLogger("app_logger")
image_logger = logging.getLogger("image_logger")


class Retrieval:
    def __init__(
        self,
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
        self.show_message = show_message
        self.input_file = input_file
        self.input_type = input_type
        self.output_file = output_file
        self.model_path = model_path

    @property
    def bio_retrieval(self) -> bool:
        message = "Reading image..."
        app_logger.info(message)
        image_logger.info(message)
        self.show_message(message)

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
                message = "Map info included"
                app_logger.info(message)
                image_logger.info(message)
                self.show_message(message)
                self.map_info = True
                self.latitude = image_data[2]
                self.longitude = image_data[3]
            else:
                message = "No map info"
                app_logger.info(message)
                image_logger.info(message)
                self.show_message(message)
                self.map_info = False
        self.end = time()
        self.process_time = self.end - self.start
        self.rows, self.cols, self.dims = self.img_reflectance.shape

        message = f"Image read. Elapsed time: {self.process_time}"
        app_logger.info(message)
        image_logger.info(message)
        self.show_message(message)

        # Showing image
        if self.plotting:
            show_reflectance_img(self.img_reflectance, self.img_wavelength)

        # ___________________________Reading models____________________________

        # Getting path of the model files
        try:
            list_of_files = os.listdir(self.model_path)
            if not list_of_files:
                raise FileNotFoundError(f"No models found in path: {self.model_path}")
        except Exception as e:
            message = f"Error: {e}"
            app_logger.error(message)
            image_logger.error(message)
            self.show_message(message)
            return True
        list_of_models = list(filter(lambda file: file.endswith(".py"), list_of_files))
        self.number_of_models = len(list_of_models)
        message = f"Getting {self.number_of_models} names was successful."
        app_logger.info(message)
        image_logger.info(message)
        self.show_message(message)

        # Importing the models
        sys.path.append(self.model_path)

        # Reading the models
        def import_and_log_model(model_file, bio_models):
            # Importing model
            module = importlib.import_module(
                os.path.splitext(model_file)[0], package=None
            )
            bio_models.append(module)
            message = f"{module.model} imported"
            app_logger.info(message)
            image_logger.info(message)
            self.show_message(message)

        # Assuming self.bio_models and self.show_message are defined
        list(
            map(
                lambda model_file: import_and_log_model(model_file, self.bio_models),
                list_of_models,
            )
        )

        # _________________________________Retrieval___________________________________________

        def run_model(i):
            message = f"Running {self.bio_models[i].model} model"
            app_logger.info(message)
            image_logger.info(message)
            self.show_message(message)

            message = "Band selection..."
            app_logger.info(message)
            image_logger.info(message)
            self.show_message(message)

            # Band selection of the image
            self.start = time()
            data_refl_new = self.band_selection(i)
            self.end = time()
            self.process_time = self.end - self.start

            message = f"Bands selected."
            app_logger.info(message)
            image_logger.info(message)
            self.show_message(message)

            message = f"Elapsed time: {self.process_time}"
            app_logger.info(message)
            image_logger.info(message)
            self.show_message(message)

            message = f"Shape: {data_refl_new.shape}"
            app_logger.debug(message)
            image_logger.debug(message)

            # Normalising the image
            data_norm = norm_data(
                data_refl_new,
                self.bio_models[i].mx_GREEN,
                self.bio_models[i].sx_GREEN,
            )
            message = f"Data normalised: {data_norm.shape}"
            app_logger.debug(message)
            image_logger.debug(message)

            # Perform PCA if there is data
            if (
                hasattr(self.bio_models[i], "pca_mat")
                and len(self.bio_models[i].pca_mat) > 0
            ):
                message = f"PCA found in model, performing PCA."
                app_logger.info(message)
                image_logger.info(message)
                self.show_message(message)

                data_norm = data_norm.dot(self.bio_models[i].pca_mat)
                message = f"Shape: {data_norm.shape}"
                app_logger.debug(message)
                image_logger.debug(message)

            if self.bio_models[i].model_type == "GPR":
                # Changing axes to because GPR function takes dim,y,x
                data_norm = np.swapaxes(
                    data_norm, 0, 1
                )  # swapping axes to have the right order after transpose
                self.img_array = np.transpose(data_norm)

                # Transform model to dictionary
                model_dict = module_to_dict(self.bio_models[i])

                message = "Running GPR..."
                app_logger.info(message)
                image_logger.info(message)
                self.show_message(message)

                gpr_object = MLRA_GPR(self.img_array, model_dict)
                self.start = time()

                # Starting GPR
                variable_map, uncertainty_map = gpr_object.perform_mlra()
                self.end = time()

                # Logging
                self.process_time = self.end - self.start
                message = f"Elapsed time of GPR: {self.process_time}"
                app_logger.info(message)
                image_logger.info(message)
                self.show_message(message)

                # Appending results
                self.variable_maps.append(variable_map)
                self.uncertainty_maps.append(uncertainty_map)

                message = f"Retrieval of {self.bio_models[i].veg_index} was successful."
                app_logger.info(message)
                image_logger.info(message)
                self.show_message(message)

        # Use ThreadPoolExecutor to run models in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(run_model, i) for i in range(self.number_of_models)
            ]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()  # This will raise any exceptions caught during execution
                except Exception as e:
                    app_logger.error(f"Error in model: {e}")
                    raise

        return False

    def band_selection(self, i: int) -> np.ndarray:
        current_wl = self.img_wavelength
        expected_wl = self.bio_models[i].wave_length
        # Find the intersection of the two lists of wavelength
        if len(np.intersect1d(current_wl, expected_wl)) == len(expected_wl):
            reflectances_new = self.img_reflectance[
                :, :, np.where(np.in1d(current_wl, expected_wl))[0]
            ]
            message = "Matching bands found."
            app_logger.info(message)
            image_logger.info(message)
            self.show_message(message)
        else:
            message = "No matching bands found, spline interpolation is applied."
            app_logger.info(message)
            image_logger.info(message)
            self.show_message(message)

            reflectances_new = spline_interpolation(
                current_wl, self.img_reflectance, expected_wl
            )

        return reflectances_new  # returning the selected bands

    def export_retrieval(self) -> bool:
        message = "Exporting image..."
        app_logger.info(message)
        image_logger.info(message)
        self.show_message(message)

        self.start = time()
        # __________________________Split image export by file type______________

        if self.input_type == "CHIME netCDF":
            self.export_netcdf()
        elif self.input_type == "ENVI Standard":
            self.export_envi()
        self.end = time()
        self.process_time = self.end - self.start

        message = f"Image exported. Elapsed time:{self.process_time}"
        app_logger.info(message)
        image_logger.info(message)
        self.show_message(message)

        if self.plotting:
            message = f"Plotting result images"
            app_logger.info(message)
            image_logger.info(message)
            self.show_message(message)
            self.show_results()
        return False

    def export_netcdf(self):
        # Creating output image
        # Create a new netCDF file
        try:
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
        finally:
            nc_file.close()  # Closing the file

        message = f"NetCDF file created successfully at: {self.output_file}"
        app_logger.info(message)
        image_logger.info(message)
        self.show_message(message)

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

        message = f"ENVI file created successfully at: {self.output_file}"
        app_logger.info(message)
        image_logger.info(message)
        self.show_message(message)

    # TODO: maybe in a new GUI window?
    # Only for CCC, CWC, LAI yet
    def show_results(self):
        """
        Show results and export function of retrieval
        :return: plot images, save in files
        !! Colors only for CCC,CWC, LAI for now !!
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

        # Define vegetation indices and their dimensions
        veg_index_to_dimension = {
            "LAI": "(m$^2$/m$^2$)",
            "FAPAR": "([-])",
            "FVC": "([-])",
            "CCC": "(g/m$^2$)",
            "CWC": "(g/m$^2$)",
            "CNC": "(g/m$^2$)",
        }
        # Define vegetation indices and their associated colormaps
        veg_index_to_colormap = {
            "LAI": "YlGn",
            "FAPAR": "Reds",
            "FVC": "Oranges",
            "CCC": "Greens",
            "CWC": "Blues",  # TODO: Need more colors
        }

        def plot_and_save(
            data,
            veg_index,
            colormap,
            dimension,
            output_file,
            img_dir,
            vec_dir,
            suffix="",
        ):
            """
            Generalized function to plot, save, and display images.

            Parameters:
                data: The data to be plotted (e.g., variable or uncertainty map).
                veg_index: The vegetation index (e.g., "LAI", "CCC").
                colormap: The colormap to use for plotting.
                dimension: The dimension to display in the title.
                output_file: The base file name for saving images.
                img_dir: Directory to save PNG images.
                vec_dir: Directory to save PDF images.
                suffix: Optional suffix for filenames (e.g., "_uncertainty").
            """
            # Plot the data
            plt.imshow(data, cmap=colormap)
            plt.title(
                f"{'Uncertainty of ' if suffix else 'Estimated '}{veg_index} map {dimension}"
            )
            plt.colorbar()
            plt.tight_layout()

            # Save the image in PNG and PDF formats
            base_name = os.path.basename(output_file)
            plt.savefig(
                os.path.join(img_dir, f"{base_name}{veg_index}{suffix}.png"),
                bbox_inches="tight",
            )
            plt.savefig(
                os.path.join(vec_dir, f"{base_name}{veg_index}{suffix}.pdf"),
                bbox_inches="tight",
            )
            plt.show()

        # Loop through models
        for i in range(self.number_of_models):
            veg_index = self.bio_models[i].veg_index
            colormap = veg_index_to_colormap.get(veg_index, "viridis")
            dimension = veg_index_to_dimension.get(veg_index, "(unknown dimension)")

            # Plot and save variable map
            plot_and_save(
                data=self.variable_maps[i],
                veg_index=veg_index,
                colormap=colormap,
                dimension=dimension,
                output_file=self.output_file,
                img_dir=img_dir,
                vec_dir=vec_dir,
            )

            # Plot and save uncertainty map
            plot_and_save(
                data=self.uncertainty_maps[i],
                veg_index=veg_index,
                colormap="jet",
                dimension=dimension,
                output_file=self.output_file,
                img_dir=img_dir,
                vec_dir=vec_dir,
                suffix="_uncertainty",
            )


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
        key: value for key, value in vars(bio_model).items() if not key.startswith("__")
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
