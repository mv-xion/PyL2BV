"""
    This file contains the function to start the retrieval module of BioRetrieval.
    This includes making initial tests, creating output folder and running the retrieval function.
"""

import logging
import os
from time import time
from datetime import datetime
from shutil import copyfile, rmtree
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

from PyL2BV.pyl2bv_code.auxiliar.logger_config import setup_logger, close_logger
from PyL2BV.pyl2bv_code.processing.retrieval import Retrieval

app_logger = logging.getLogger("app_logger")  # Retrieve the logger by name


# Return structure after processing
@dataclass
class RetrievalResult:
    success: bool
    message: str
    plots: Optional[List[Tuple[np.ndarray, str, str]]] = None  # (image, title, colormap)


def pyl2bv_processing(
        input_folder_path: str,
        input_type: str,
        model_folder_path: str,
        conversion_factor: float,
        chunk_size: int,
        show_message: callable,
        plotting: bool,
        debug_log: bool,
):
    """
    PyL2BV retrieval module for ARTMO based models:
    LEO-IPL - University of Valencia. June 2024.
    by Mészáros Viktor Ixion

    This function reads all the files in the folder input_folder_path and
    reads the indicated input_type than starts the retrieval process with the models
    specified in model_folder_path.
    Parameters:


    :type conversion_factor: conversion factor for the retrieval
    :param input_folder_path: path to folder containing input files
    :param input_type: type of input files
    :param model_folder_path: path to folder containing model files
    :param chunk_size: chunk size in pixels being processed at once (rectangular area of image size x size)
    :param show_message: function for printing messages on GUI
    :param plotting: bool to plot the results or not
    :param debug_log: bool to set log level to debug mode
    :return: RetrievalResult: success True or False, message and plots
    """

    # __________________________Construct variables___________________________
    # Create output path
    input_path = os.path.abspath(input_folder_path)  # Create a path object
    path_components = input_path.split(
        os.sep
    )  # Split the path into its components
    path_components[-1] = "output"  # Replace last part with output
    output_path = os.sep.join(
        path_components
    )  # Join the path components back together

    # Create logfile path
    logfile_path = os.path.join(output_path, "logfile.log")

    # __________________________Split processing by file type_________________

    if input_type == "CHIME netCDF":
        if show_message:
            show_message("Type: " + input_type)
        app_logger.info(f"Processing input type: {input_type}")

        # Check input files, log if number of files are not correct
        list_of_files = os.listdir(input_path)
        # Filter files with .nc extension
        nc_files = [file for file in list_of_files if file.endswith(".nc")]
        if not nc_files or len(nc_files) % 4 != 0:
            # If output folder exists, delete it and make a new one
            make_output_folder(output_path)
            app_logger.error(
                f"FAIL: Wrong number of inputs or error loading input image."
            )
            if show_message:
                show_message("Missing input nc file.")
            with open(logfile_path, "w") as fileID:
                fileID.write(
                    "FAIL: Wrong number of .nc inputs or error loading input image."
                    " Consider checking Input Path/File \n"
                )
                fileID.write(f"Input Path: {input_path} \n")
            raise FileNotFoundError("Missing or incomplete .nc input files.")

        # Counting input files
        pos_img_files = [i for i, name in enumerate(nc_files) if "IMG" in name]
        num_images = len(pos_img_files)  # Number of image files
        input_files = [
            os.path.join(input_path, nc_files[i]) for i in pos_img_files
        ]
        input_names = [nc_files[i] for i in pos_img_files]

        # Concatenate the name of output files
        l2b_output = os.path.join(output_path, "CHI_E2ES_PRO_L2VIMG_")
        # If output folder exists, delete it and make a new one
        flag_out = make_output_folder(output_path)

        # Generate processing time
        proces_time = datetime.now().strftime("%Y%m%dT%H%M%SZ")
        l2b_output_files = []  # Collecting the output filenames here

        # Process input files, copy GEO, QUA
        for i in range(num_images):
            input_file_name = input_names[i]
            # Configuration works for CHIME image name convention now
            pos = [j for j, char in enumerate(input_file_name) if char == "_"]
            scene_time = input_file_name[pos[3] + 1: pos[4]]
            tile = input_file_name[pos[5] + 1:]
            l2b_output_files.append(
                f"{l2b_output}{scene_time}_{proces_time}_{tile}"
            )

            try:
                input_file_geo = input_files[i].replace("IMG", "GEO")
                l2b_output_file_geo = l2b_output_files[i].replace("IMG", "GEO")
                copyfile(input_file_geo, l2b_output_file_geo)
                input_file_qua = input_files[i].replace("IMG", "QUA")
                l2b_output_file_qua = l2b_output_files[i].replace("IMG", "QUA")
                copyfile(input_file_qua, l2b_output_file_qua)
            except FileNotFoundError as e:
                app_logger.error(f"Error: {e}")
                if show_message:
                    show_message("Missing complementary files for CHIME image.")
                with open(logfile_path, "w") as fileID:
                    fileID.write(
                        "FAIL: Missing complementary files for CHIME image."
                        " Consider checking Input Path/File \n"
                    )
                    fileID.write(f"Input Path: {input_path} \n")
                return RetrievalResult(success=False, message="Something went wrong", plots=None)
            except Exception as e:
                app_logger.error(f"Unexpected error: {e}")
                if show_message:
                    show_message(
                        "An unexpected error occurred while copying complementary file."
                    )
                with open(logfile_path, "w") as fileID:
                    fileID.write(
                        "FAIL: An unexpected error occurred."
                        " Consider checking Input Path/File \n"
                    )
                    fileID.write(f"Input Path: {input_path} \n")
                return RetrievalResult(success=False, message="Something went wrong", plots=None)

    elif input_type == "ENVI Standard":
        if show_message:
            show_message("Type: " + input_type)
        app_logger.info(f"Processing input type: {input_type}")

        # Check input files, log if number of files are not correct
        list_of_files = os.listdir(input_path)
        # Filter files with .hdr extension
        hdr_files = [file for file in list_of_files if file.endswith(".hdr")]
        if not hdr_files:
            # If output folder exists, delete it and make a new one
            if os.path.exists(output_path):
                rmtree(output_path)
                os.makedirs(output_path)
            else:
                os.makedirs(output_path)
            app_logger.error(
                f"FAIL: Wrong number of inputs or error loading input file."
            )
            if show_message:
                show_message("Missing input hdr file.")
            with open(logfile_path, "w") as fileID:
                fileID.write(
                    "FAIL: Wrong number of inputs or error loading input file."
                    " Consider checking Input Path/File \n"
                )
                fileID.write(f"Input Path: {input_path} \n")
            raise FileNotFoundError("Missing input hdr file.")

        # Counting input files
        num_images = len(hdr_files)  # Number of image files
        input_files = [
            os.path.join(input_path, hdr_files[i]) for i in range(num_images)
        ]
        input_names = [
            os.path.splitext(os.path.basename(hdr_files[i]))[0]
            for i in range(num_images)
        ]

        # Generate processing time
        proces_time = datetime.now().strftime("%Y%m%dT%H%M%SZ")
        # Collecting the output filenames here
        l2b_output_files = [
            os.path.join(output_path, f"{input_names[i]}_{proces_time}")
            for i in range(num_images)
        ]
        # Create output folder
        flag_out = make_output_folder(output_path)
    else:
        app_logger.error("Invalid input format")
        raise ValueError(f"Unsupported input type: {input_type}")

    # ____________________________________Retrieval________________________________
    all_plots = []  # Collect plots from all images if GUI+plotting
    # Biophysical parameters retrieval
    for i in range(num_images):
        start = time()
        img_name = os.path.basename(l2b_output_files[i])
        log_path = os.path.splitext(l2b_output_files[i])[0] + "_logfile.log"
        log_level = logging.DEBUG if debug_log else logging.INFO
        image_logger = setup_logger(
            logger_name="image_logger",
            logfile_name=log_path,
            log_level=log_level,
        )
        if i == 0:
            # Log information to logfile
            if flag_out:
                image_logger.debug(
                    "Output folder already exists. Folder was overwritten."
                )
            else:
                image_logger.debug("Output folder does not exist. Folder was created.")

        # Log image information
        if show_message:
            show_message(f"Tile: {img_name}")
        image_logger.info(f"Processing tile:: {img_name}")

        try:
            # Creating Retrieval object and call function
            retrieval_object = Retrieval(
                show_message,
                input_files[i],
                input_type,
                l2b_output_files[i],
                model_folder_path,
                conversion_factor,
                chunk_size,
                plotting,
                debug_log,
            )

            retrieval_object.bio_retrieval()
            retrieval_object.export_retrieval()
            if plotting:
                image_plots = retrieval_object.show_results()
                if image_plots:
                    all_plots.extend(image_plots)

            end = time()
            process_time = end - start

            image_logger.info(f"Total retrieval. Elapsed time: {process_time}")
            image_logger.info(f"Retrieval of {img_name} successful.\n")
            close_logger("image_logger")
        except Exception as e:
            close_logger("image_logger")
            app_logger.error(f"Error processing tile {img_name}: {e}")
            if show_message:
                show_message(f"Error processing tile {img_name}: {e}")
            return RetrievalResult(success=False, message="Something went wrong", plots=None)

    if plotting:
        return RetrievalResult(
            success=True,
            message="Model ran successfully with plots",
            plots=all_plots
        )
    else:
        return RetrievalResult(
            success=True,
            message="Model ran successfully",
            plots=None
        )


def make_output_folder(output_path: str) -> bool:
    """
    Create output folder
    :param output_path: path of output folder
    :return: flag: 1 if overwritten, 0 if not
    """
    if os.path.exists(output_path):
        rmtree(output_path)
        os.makedirs(output_path)
        app_logger.debug(
            f"Output folder {output_path} already existed and was overwritten."
        )
        return True
    else:
        os.makedirs(output_path)
        app_logger.debug(f"Output folder {output_path} was created.")
        return False
