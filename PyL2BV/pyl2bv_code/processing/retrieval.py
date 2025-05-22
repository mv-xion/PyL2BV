"""
    Contains the retrieval class which performs the retrieval
    from reading the image to writing the retrieved result
"""

import importlib
import logging
import os
import pickle
import sys
import concurrent.futures
import numpy as np

from matplotlib import pyplot as plt
from netCDF4 import Dataset
from spectral.io import envi

from PyL2BV.pyl2bv_code.auxiliar.image_read import (
    read_envi,
    read_netcdf,
    show_reflectance_img,
)

from PyL2BV.pyl2bv_code.auxiliar.spectra_interpolation import (
    spline_interpolation,
)
from PyL2BV.pyl2bv_code.processing.mlra_gpr import MLRA_GPR
from time import time

import warnings

warnings.filterwarnings("ignore", message="Starting a Matplotlib GUI outside of the main thread")


# Retrieve the loggers by name
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
            chunk_size: int,
            plotting: bool,
            debug_log: bool,
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
        self.initial_plot = None
        self.chunk_size = chunk_size
        self.gpr_models = {}
        self.chunk_num = None  # Storing current chunk number
        self.number_of_models = None  # Storing number models
        self.bio_models = []  # Storing the models
        self.variable_maps = []  # Storing variable maps
        self.uncertainty_maps = []  # Storing uncertainty maps
        self.model_order = []  # Storing the order the models ran
        self.map_info = None
        self.longitude = None
        self.latitude = None
        self.img_wavelength = None
        self.img_reflectance = None
        self.plotting = plotting
        self.conversion_factor = conversion_factor
        self.show_message = show_message
        self.input_file = input_file
        self.input_type = input_type
        self.output_file = output_file
        self.model_path = model_path
        self.debug_log = debug_log

    # ___________________________________ Read image _____________________________
    def read_image(self):
        message = "Reading image..."
        image_logger.debug(message)
        if self.show_message and self.debug_log:
            self.show_message(message)

        start = time()
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
                image_logger.debug(message)
                if self.show_message and self.debug_log:
                    self.show_message(message)
                self.map_info = True
                self.latitude = image_data[2]
                self.longitude = image_data[3]
            else:
                message = "No map info"
                image_logger.debug(message)
                if self.show_message and self.debug_log:
                    self.show_message(message)
                self.map_info = False
        end = time()
        process_time = end - start
        self.rows, self.cols, self.dims = self.img_reflectance.shape

        message = f"Image read. Elapsed time: {process_time}"
        image_logger.debug(message)
        if self.show_message and self.debug_log:
            self.show_message(message)

        # Save for GUI or CLI later
        if self.plotting:
            rgb_image, title, cmap = show_reflectance_img(self.img_reflectance, self.img_wavelength)
            self.initial_plot = (rgb_image, title, cmap)

    # _____________________________ Reading models ____________________________
    def load_models(self):
        # Getting path of the model files
        list_of_models = [f for f in os.listdir(self.model_path) if f.endswith('.py')]
        if not list_of_models:
            raise FileNotFoundError(f"No models found in path: {self.model_path}")

        self.number_of_models = len(list_of_models)

        # Importing the models
        sys.path.append(self.model_path)

        # Reading the models
        for model_file in list_of_models:
            # Importing model
            module = importlib.import_module(
                os.path.splitext(model_file)[0], package=None
            )
            self.bio_models.append(module_to_dict(module))
            if self.debug_log:
                message = f"{module.model} imported"
                image_logger.debug(message)
                if self.show_message and self.debug_log:
                    self.show_message(message)
        self.model_order = [model["veg_index"] for model in self.bio_models]
        message = f"Loading {self.number_of_models} models was successful: {self.model_order}"
        image_logger.info(message)
        if self.show_message:
            self.show_message(message)

    # _________________________________ Retrieval ___________________________________________

    def yield_chunks(self, mode="row"):
        if mode == "column":
            for col in range(0, self.cols, self.chunk_size):
                yield 0, self.rows, col, min(col + self.chunk_size, self.cols)
        elif mode == "row":
            for row in range(0, self.rows, self.chunk_size):
                yield row, min(row + self.chunk_size, self.rows), 0, self.cols
        else:
            for row in range(0, self.rows, self.chunk_size):
                for col in range(0, self.cols, self.chunk_size):
                    yield row, min(row + self.chunk_size, self.rows), col, min(col + self.chunk_size, self.cols)

    def band_selection(self, i, row_start, row_end, col_start, col_end: int) -> np.array:
        current_wl = self.img_wavelength
        expected_wl = self.bio_models[i]["wave_length"]
        reflectance_chunk = self.img_reflectance[row_start:row_end, col_start:col_end, :]
        # Find the intersection of the two lists of wavelength
        if len(np.intersect1d(current_wl, expected_wl)) == len(expected_wl):
            reflectance_chunk_new = reflectance_chunk[
                                    :, :, np.where(np.in1d(current_wl, expected_wl))[0]
                                    ]
            if self.chunk_num == 1:
                message = f"Matching bands found for {self.bio_models[i]['veg_index']}"
                image_logger.debug(message)
                if self.show_message and self.debug_log:
                    self.show_message(message)
        else:
            if self.chunk_num == 1:
                message = f"No matching bands for {self.bio_models[i]['veg_index']} spline interpolation is applied."
                image_logger.debug(message)
                if self.show_message and self.debug_log:
                    self.show_message(message)

            reflectance_chunk_new = spline_interpolation(
                current_wl, reflectance_chunk, expected_wl
            )
        return reflectance_chunk_new

    # Normalise data function
    def norm_data(self, i, reflectance_chunk) -> np.array:
        return (reflectance_chunk - self.bio_models[i]["mx_GREEN"]) / self.bio_models[i]["sx_GREEN"]

    def process_chunk(self, chunk_coords):
        row_start, row_end, col_start, col_end = chunk_coords
        chunk_results = []

        # Inner method to process a single model
        def process_single_model(i):
            model = self.bio_models[i]
            model_key = model["veg_index"]

            # Initialisation message (only first chunk)
            if self.chunk_num == 1:
                message = f"Initializing model {model_key}"
                image_logger.debug(message)
                if self.show_message and self.debug_log:
                    self.show_message(message)

            # Band selection and normalization
            reflectance_chunk = self.band_selection(i, row_start, row_end, col_start, col_end)
            reflectance_chunk = self.norm_data(i, reflectance_chunk)

            if "pca_mat" in model and model["pca_mat"].size > 0:
                if self.chunk_num == 1:
                    message = f"PCA found in model {model_key}"
                    image_logger.debug(message)
                    if self.show_message and self.debug_log:
                        self.show_message(message)
                reflectance_chunk = reflectance_chunk.dot(model["pca_mat"])

            if model["model_type"] == "GPR":
                # Changing axes to because GPR function takes dim,y,x
                reflectance_chunk = np.transpose(reflectance_chunk, (2, 0, 1))

                # Initialise MLRA_GPR once per model
                if self.chunk_num == 1:
                    message = f"Initializing GPR model {model_key}"
                    image_logger.info(message)
                    if self.show_message:
                        self.show_message(message)
                    gpr_object = MLRA_GPR(image=reflectance_chunk, bio_model=model)
                    self.gpr_models[model_key] = gpr_object
                else:
                    gpr_object = self.gpr_models[model_key]
                    gpr_object.image = reflectance_chunk

                # Perform retrieval
                variable_map, uncertainty_map = gpr_object.perform_mlra()
                return model_key, variable_map, uncertainty_map

        # Parallelize processing of different models within the chunk
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(len(self.bio_models), os.cpu_count() // 2)) as executor:
            future_to_model = {
                executor.submit(process_single_model, i): self.bio_models[i]["veg_index"]
                for i in range(len(self.bio_models))
            }

            for future in concurrent.futures.as_completed(future_to_model):
                model_key = future_to_model[future]
                try:
                    veg_index, variable_map, uncertainty_map = future.result()
                    chunk_results.append((veg_index, variable_map, uncertainty_map))
                except Exception as e:
                    message = f"Model {model_key} generated an exception: {e}"
                    image_logger.error(message)
                    if self.show_message:
                        self.show_message(message)
                    raise

        return row_start, row_end, col_start, col_end, chunk_results

    def perform_chunked_retrieval(self):
        message = f"Running chunked retrieval for {self.number_of_models} models with {self.chunk_size} chunk size"
        image_logger.info(message)
        if self.show_message:
            self.show_message(message)
        start = time()
        self.variable_maps = {
            model["veg_index"]: np.zeros((self.rows, self.cols)) for model in self.bio_models
        }
        self.uncertainty_maps = {
            model["veg_index"]: np.zeros((self.rows, self.cols)) for model in self.bio_models
        }

        chunks = list(self.yield_chunks())
        total_chunks = len(chunks)

        for idx, chunk_coords in enumerate(chunks, start=1):
            chunk_start = time()
            message = f"Processing chunk {idx}/{total_chunks} - coords {chunk_coords}..."
            self.chunk_num = idx
            image_logger.info(message)
            if self.show_message:
                self.show_message(message)

            row_start, row_end, col_start, col_end, chunk_results = self.process_chunk(
                chunk_coords
            )

            for veg_index, variable_map, uncertainty_map in chunk_results:
                self.variable_maps[veg_index][row_start:row_end, col_start:col_end] = variable_map
                self.uncertainty_maps[veg_index][row_start:row_end, col_start:col_end] = uncertainty_map
            chunk_end = time()
            message = f"Chunk time: {chunk_end - chunk_start}s"
            image_logger.debug(message)
            if self.show_message and self.debug_log:
                self.show_message(message)

        end = time()
        process_time = end - start
        message = "Chunked retrieval completed successfully. Elapsed time: {:.2f}".format(process_time)
        image_logger.info(message)
        if self.show_message:
            self.show_message(message)

    def bio_retrieval(self) -> None:
        try:
            self.read_image()
            self.load_models()
            self.perform_chunked_retrieval()
        except Exception as e:
            image_logger.error(f"Error during bio_retrieval: {e}")
            if self.show_message:
                self.show_message(f"Error during bio_retrieval: {e}")
            raise RuntimeError(f"Error during bio_retrieval: {e}") from e

    def export_retrieval(self) -> None:
        try:
            message = "Exporting image..."
            image_logger.debug(message)
            if self.show_message and self.debug_log:
                self.show_message(message)

            start = time()
            # __________________________Split image export by file type______________

            if self.input_type == "CHIME netCDF":
                self.export_netcdf()
            elif self.input_type == "ENVI Standard":
                self.export_envi()
            end = time()
            process_time = end - start

            message = f"Image exported. Elapsed time:{process_time}"
            image_logger.debug(message)
            if self.show_message and self.debug_log:
                self.show_message(message)
        except Exception as e:
            image_logger.error(f"Error during export_retrieval: {e}")
            raise RuntimeError(f"Error during export_retrieval: {e}")

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
                group = nc_file.createGroup(self.model_order[i])
                if self.model_order[i] == "LCC":
                    group.long_name = "Leaf Chlorophyll Content (LCC)"
                elif self.model_order[i] == "LWC":
                    group.long_name = "Leaf Water Content (LWC)"
                elif self.model_order[i] == "LNC":
                    group.long_name = "Leaf Nitrogen Content (LNC)"
                elif self.model_order[i] == "LMA":
                    group.long_name = "Leaf Mass Area (LMA)"
                elif self.model_order[i] == "LAI":
                    group.long_name = "Leaf Area Index (LAI)"
                elif self.model_order[i] == "CCC":
                    group.long_name = "Canopy Chlorophyll Content (CCC)"
                elif self.model_order[i] == "CWC":
                    group.long_name = "Canopy Water Content (CWC)"
                elif self.model_order[i] == "CDMC":
                    group.long_name = "Canopy Dry Matter Content (CDMC)"
                elif self.model_order[i] == "CNC":
                    group.long_name = "Canopy Nitrogen Content (CNC)"
                elif self.model_order[i] == "FVC":
                    group.long_name = "Fractional Vegetation Cover (FVC)"
                elif self.model_order[i] == "FAPAR":
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
                ]["units"]  # Adding the 'Units' attribute
                sd_var = group.createVariable("SD", "f4", dimensions=("Nc", "Nl"))
                sd_var.units = self.bio_models[i]["units"]
                cv_var = group.createVariable("CV", "f4", dimensions=("Nc", "Nl"))
                cv_var.units = "%"
                qf_var = group.createVariable("QF", "i1", dimensions=("Nc", "Nl"))
                qf_var.units = "adim"

                # Assign data to the variable
                # Transpose for matlab type output
                retrieval_var[:] = np.transpose(self.variable_maps[self.model_order[i]])
                sd_var[:] = np.transpose(self.uncertainty_maps[self.model_order[i]])
        finally:
            nc_file.close()  # Closing the file

        message = f"NetCDF file created successfully at: {self.output_file}"
        image_logger.info(message)
        if self.show_message:
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
            band_names.append(self.model_order[i])
            band_names.append(f"{self.model_order[i]}_sd")

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

        #  Create an interleaved list of matrices using correct order
        interleaved_matrices = []
        for veg_index in self.model_order:
            interleaved_matrices.append(self.variable_maps[veg_index])
            interleaved_matrices.append(self.uncertainty_maps[veg_index])

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
        image_logger.info(message)
        if self.show_message:
            self.show_message(message)

    # Only for CCC, CWC, LAI, FAPAR, FVC yet
    def show_results(self):
        """
        Show results and export function of retrieval
        :return: plot images, save in files
        !! Colors only for CCC,CWC, LAI for now !!
        """
        message = f"Plotting result images"
        image_logger.info(message)
        if self.show_message:
            self.show_message(message)

        plots_to_display = []  # to store data to plot later in GUI
        if hasattr(self, "initial_plot"):
            plots_to_display.insert(0, self.initial_plot)

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
            "CWC": "Blues",
            "CNC": "Purples"
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
            title = f"{'Uncertainty of ' if suffix else 'Estimated '}{veg_index} map {dimension}"
            plt.title(title)
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
            plt.close()
            # Return data for GUI display
            return data, title, colormap

        # Loop through models
        for i in range(self.number_of_models):
            veg_index = self.model_order[i]
            colormap = veg_index_to_colormap.get(veg_index, "viridis")
            dimension = veg_index_to_dimension.get(veg_index, "(unknown dimension)")

            # Plot and save variable map
            var_data = self.variable_maps[veg_index]
            plots_to_display.append(plot_and_save(
                data=var_data,
                veg_index=veg_index,
                colormap=colormap,
                dimension=dimension,
                output_file=self.output_file,
                img_dir=img_dir,
                vec_dir=vec_dir,
            ))

            # Plot and save uncertainty map
            unc_data = self.uncertainty_maps[veg_index]
            plots_to_display.append(plot_and_save(
                data=unc_data,
                veg_index=veg_index,
                colormap="jet",
                dimension=dimension,
                output_file=self.output_file,
                img_dir=img_dir,
                vec_dir=vec_dir,
                suffix="_uncertainty",
            ))

        return plots_to_display


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
