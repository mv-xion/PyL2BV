import logging
import tracemalloc
import psutil
import threading
import os
import time
import threading
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

    # --- Start tracing memory ---
    tracemalloc.start()
    memory_usage = []

    def trace_memory():
        while tracing[0]:
            current, peak = tracemalloc.get_traced_memory()
            memory_usage.append((time.time(), current / 10**6, peak / 10**6))
            time.sleep(0.1)  # every 0.1 second

    tracing = [True]
    memory_thread = threading.Thread(target=trace_memory)
    memory_thread.start()

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
    finally:
        # --- Stop tracing ---
        tracing[0] = False
        memory_thread.join()

        # Stop tracing memory allocations
        tracemalloc.stop()

        # Log the memory usage
        for timestamp, current, peak in memory_usage:
            logging.info(f"Time: {timestamp}, Current memory usage: {current} MB, Peak memory usage: {peak} MB")

        # Plot the memory usage over time
        try:
            import matplotlib.pyplot as plt

            times, currents, peaks = zip(*memory_usage)
            plt.figure(figsize=(10, 5))
            plt.plot(times, currents, label='Current Memory Usage (MB)')
            plt.plot(times, peaks, label='Peak Memory Usage (MB)')
            plt.xlabel('Time (s)')
            plt.ylabel('Memory Usage (MB)')
            plt.title('Memory Usage Over Time')
            plt.legend()
            plt.show()

        except ImportError:
            logging.warning("matplotlib is not installed. Memory usage plot will not be displayed.")

