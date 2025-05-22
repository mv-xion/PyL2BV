import logging
from PyL2BV.pyl2bv_code.processing.processing_module import pyl2bv_processing, RetrievalResult

app_logger = logging.getLogger("app_logger")  # Retrieve the logger by name


def run_retrieval(
        input_folder_path: str,
        input_type: str,
        model_folder_path: str,
        conversion_factor: float = 0.0001,
        chunk_size: int = 300,
        show_message_callback=None,  # Optional callback for GUI messages
        plotting: bool = False,
        debug_log: bool = False,
) -> RetrievalResult:
    """
    Runs the retrieval function, shared between CLI and GUI.
    :param input_folder_path: path to the input folder
    :param input_type: type of input file
    :param model_folder_path: path to the model folder
    :param conversion_factor: image conversion factor
    :param chunk_size: chunk size in pixels being processed at once (rectangular area of image size x size)
    :param show_message_callback: Optional callback function for GUI messages
    :param plotting: bool to plot the results or not
    :param debug_log: bool to enable debug logging
    :return: Completion message
    """
    app_logger.info("Starting retrieval.")
    try:
        result = pyl2bv_processing(
            input_folder_path,
            input_type,
            model_folder_path,
            conversion_factor,
            chunk_size,
            show_message_callback,
            plotting,
            debug_log,
        )

        if not result.success:
            app_logger.error(result.message)
        else:
            app_logger.info(result.message)
        return result

    except Exception as e:
        message = f"Error in preprocessing: {e}"
        app_logger.error(message)
        return RetrievalResult(success=False, message="Something went wrong", plots=None)
