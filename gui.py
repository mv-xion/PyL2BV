"""
    GUI for BioRetrieval programme using the tkinter library
"""

import logging
import os
import threading
import tkinter as tk
from tkinter import filedialog, ttk

from bioretrieval.processing.processing_module import bio_retrieval_module

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class SimpleGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Retrieval of biophysical variables")
        self.create_widgets()
        logging.info("Initialized SimpleGUI.")
        self.model_thread = None

    def create_widgets(self):
        default_folder = os.getcwd()
        # Selecting the input folder
        self.label_input_folder = tk.Label(self, text="Select Input Folder:")
        self.label_input_folder.grid(row=0, column=0, padx=5, pady=5)

        self.entry_input_folder = tk.Entry(self, width=50)
        self.entry_input_folder.insert(
            0, os.path.join(default_folder, "input")
        )
        self.entry_input_folder.grid(row=0, column=1, padx=5, pady=5)

        self.button_browse_input_folder = tk.Button(
            self, text="Browse", command=self.browse_input_folder
        )
        self.button_browse_input_folder.grid(row=0, column=2, padx=5, pady=5)

        # Selecting the input folder type
        self.label_input_type = tk.Label(self, text="Input Type:")
        self.label_input_type.grid(row=0, column=3, padx=5, pady=5)

        self.input_type_var = tk.StringVar(self)
        self.input_type_var.set("CHIME netCDF")  # Default option
        self.input_type_dropdown = ttk.Combobox(
            self,
            textvariable=self.input_type_var,
            values=["CHIME netCDF", "ENVI Standard"],
            state="readonly",  # This makes the dropdown non-editable
        )
        self.input_type_dropdown.grid(row=0, column=4, padx=5, pady=5)

        # Selecting the model folder
        self.label_model_folder = tk.Label(self, text="Select Model Folder:")
        self.label_model_folder.grid(row=1, column=0, padx=5, pady=5)

        self.entry_model_folder = tk.Entry(self, width=50)
        self.entry_model_folder.insert(
            0, os.path.join(default_folder, "models")
        )
        self.entry_model_folder.grid(row=1, column=1, padx=5, pady=5)

        self.button_browse_model_folder = tk.Button(
            self, text="Browse", command=self.browse_model_folder
        )
        self.button_browse_model_folder.grid(row=1, column=2, padx=5, pady=5)

        # Adding the Image Conversion Factor entry box
        self.label_conversion_factor = tk.Label(
            self, text="Image Conversion Factor:"
        )
        self.label_conversion_factor.grid(row=1, column=3, padx=5, pady=5)

        self.entry_conversion_factor = NonNegativeNumberEntry(self, width=20)
        self.entry_conversion_factor.insert(0, "0.0001")  # Set default value
        self.entry_conversion_factor.grid(row=1, column=4, padx=5, pady=5)

        # Run button
        self.button_run = tk.Button(
            self, text="Run", command=self.start_model_thread
        )
        self.button_run.grid(row=2, column=1, padx=5, pady=5)

        logging.info("Created GUI widgets.")

    def browse_input_folder(self):
        input_folder = filedialog.askdirectory(
            initialdir=os.getcwd(), title="Select Input Folder"
        )
        if input_folder:
            self.entry_input_folder.delete(0, tk.END)
            self.entry_input_folder.insert(0, input_folder)
            logging.info(f"Selected input folder: {input_folder}")

    def browse_model_folder(self):
        model_folder = filedialog.askdirectory(
            initialdir=os.getcwd(), title="Select Model Folder"
        )
        if model_folder:
            self.entry_model_folder.delete(0, tk.END)
            self.entry_model_folder.insert(0, model_folder)
            logging.info(f"Selected model folder: {model_folder}")

    def start_model_thread(self):
        """
        After pushing Run button this function starts a new thread and runs
        the retrieval function, shows the process in a new window
        :return:
        """
        input_folder_path = self.entry_input_folder.get()
        model_folder_path = self.entry_model_folder.get()
        input_type = self.input_type_var.get()
        conversion_factor = float(self.entry_conversion_factor.get())
        try:
            if os.path.isdir(input_folder_path) and os.path.isdir(
                model_folder_path
            ):
                logging.info("Starting model thread.")
                logging.info(f"Input folder: {input_folder_path}")
                logging.info(f"Model folder: {model_folder_path}")
                logging.info(f"Input type: {input_type}")
                logging.info(f"Conversion factor: {conversion_factor}")

                # Disable the Run button
                self.button_run.config(state=tk.DISABLED)

                # Create a new window for showing progress
                self.progress_window = tk.Toplevel(self)
                self.progress_window.title("Running Model")
                self.progress_window.geometry("400x200")
                self.progress_label = tk.Label(
                    self.progress_window, text="Running model..."
                )
                self.progress_label.pack(pady=10)

                # Create progress bar
                self.progress_bar = ttk.Progressbar(
                    self.progress_window, mode="indeterminate"
                )
                self.progress_bar.pack(pady=5)
                self.progress_bar.start(20)

                # Create a text widget for displaying messages
                self.text_widget = tk.Text(
                    self.progress_window, height=5, width=45
                )
                self.text_widget.pack(pady=5)

                # Run the model in a separate thread
                self.model_thread = threading.Thread(
                    target=self.run_model,
                    args=(
                        input_folder_path,
                        input_type,
                        model_folder_path,
                        conversion_factor,
                    ),
                )
                self.model_thread.start()

                # Check the thread status periodically
                self.check_thread_status()
            else:
                raise FileNotFoundError(
                    "Invalid or no input or model folder selected"
                )
        except Exception as e:
            logging.error(
                "Error occurred while starting model thread", exc_info=True
            )
            self.show_message("Error occurred while starting model thread")

    def check_thread_status(self):
        if self.model_thread.is_alive():
            self.after(100, self.check_thread_status)
        else:
            self.on_model_thread_complete()

    def on_model_thread_complete(self):
        # Stop the progress bar
        self.progress_bar.stop()

        # Enable the Run button
        self.button_run.config(state=tk.NORMAL)

        # Close the progress window
        self.progress_window.destroy()

    def run_model(
        self,
        input_folder_path: str,
        input_type: str,
        model_folder_path: str,
        conversion_factor: float,
    ):
        """
        This function runs on the new thread and starts the retrieval function
        :param conversion_factor: image conversion factor
        :param input_folder_path: path to the input folder
        :param input_type: type of input file
        :param model_folder_path: path to the model folder
        :return: Shows completion message and is able to run again
        """
        logging.info("Running model.")

        message = bio_retrieval_module(
            input_folder_path,
            input_type,
            model_folder_path,
            conversion_factor,
            self.show_message,
        )

        if message == 1:
            completion_message = "Something went wrong"
            logging.error(completion_message)
        elif message == 0:
            completion_message = "Model ran successfully"
            logging.info(completion_message)
        else:
            completion_message = "Unknown Error"
            logging.warning(completion_message)

        # Schedule the display of the completion message in the main thread
        self.after(0, self.show_message, completion_message)

    def show_message(self, message):
        self.text_widget.insert(tk.END, message + "\n")
        self.text_widget.see(tk.END)


class NonNegativeNumberEntry(tk.Entry):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.validation_command = self.register(self.validate_input)
        self.config(
            validate="key", validatecommand=(self.validation_command, "%P")
        )

    def validate_input(self, new_value):
        if new_value == "":
            return True
        if new_value.isdigit():
            return True
        try:
            if float(new_value) >= 0:
                return True
        except ValueError:
            return False

        return False


def main():
    """
    Main starts the programme with the GUI
    :return:
    """
    logging.info("Starting the BioRetrieval GUI application.")
    gui = SimpleGUI()
    gui.mainloop()
    logging.info("BioRetrieval GUI application closed.")


if __name__ == "__main__":
    main()
