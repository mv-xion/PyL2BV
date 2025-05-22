"""
    GUI for PyL2BV programme using the tkinter library
"""

import logging
import os
import threading
import time
import webbrowser
import tkinter as tk
from tkinter import filedialog, ttk

from datetime import datetime
from pyl2bv_code.model_runner import run_retrieval

app_logger = logging.getLogger("app_logger")  # Retrieve the logger by name


class SimpleGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PyL2BV for the retrieval of biophysical variables")
        self.create_widgets()
        self.create_menu()  # Initialize the menu
        app_logger.info("Initialized SimpleGUI.")
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

        # Adding the Image Conversion Factor entry box
        self.label_chunk_size= tk.Label(
            self, text="Chunk Size in Pixels:"
        )
        self.label_chunk_size.grid(row=2, column=3, padx=5, pady=5)

        self.entry_chunk_size = NonNegativeNumberEntry(self, width=20)
        self.entry_chunk_size.insert(0, "300")  # Set default value
        self.entry_chunk_size.grid(row=2, column=4, padx=5, pady=5)

        # Move the Run button one row down
        self.button_run = tk.Button(
            self, text="Run", command=self.start_model_thread
        )
        self.button_run.grid(row=3, column=1, padx=5, pady=5)

        app_logger.info("Created GUI widgets.")

    def create_menu(self):
        # Creating menu bar
        self.menu_bar = tk.Menu(self)

        # File menu with Exit button
        self.file_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.file_menu.add_command(
            label="Exit", command=self.quit
        )  # Close the app
        self.menu_bar.add_cascade(label="File", menu=self.file_menu)

        # Advanced Settings menu with the Plotting results and Debug log checkboxes
        self.advanced_menu = tk.Menu(self.menu_bar, tearoff=0)

        # Plotting results checkbox
        self.plotting_results_var = tk.BooleanVar()
        self.advanced_menu.add_checkbutton(
            label="Plotting results",
            onvalue=True,
            offvalue=False,
            variable=self.plotting_results_var,
        )

        # Debug log checkbox
        self.debug_log_var = tk.BooleanVar()
        self.advanced_menu.add_checkbutton(
            label="Debug log",
            onvalue=True,
            offvalue=False,
            variable=self.debug_log_var,
        )

        # Add Advanced Settings menu to the menu bar
        self.menu_bar.add_cascade(
            label="Advanced Settings", menu=self.advanced_menu
        )

        # Help menu with About button
        self.help_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.help_menu.add_command(
            label="About",
            command=lambda: webbrowser.open(
                "https://github.com/mv-xion/PyL2BV"
            ),
        )  # Open the GitHub repository
        self.menu_bar.add_cascade(label="Help", menu=self.help_menu)

        # Configuring the menu bar
        self.config(menu=self.menu_bar)

    def browse_input_folder(self):
        input_folder = filedialog.askdirectory(
            initialdir=os.getcwd(), title="Select Input Folder"
        )
        if input_folder:
            self.entry_input_folder.delete(0, tk.END)
            self.entry_input_folder.insert(0, input_folder)
            app_logger.info(f"Selected input folder: {input_folder}")

    def browse_model_folder(self):
        model_folder = filedialog.askdirectory(
            initialdir=os.getcwd(), title="Select Model Folder"
        )
        if model_folder:
            self.entry_model_folder.delete(0, tk.END)
            self.entry_model_folder.insert(0, model_folder)
            app_logger.info(f"Selected model folder: {model_folder}")

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
        chunk_size = int(self.entry_chunk_size.get())
        plotting = self.plotting_results_var.get()
        debug_log = bool(self.debug_log_var.get())
        try:
            if os.path.isdir(input_folder_path) and os.path.isdir(
                model_folder_path
            ):
                app_logger.info("Starting model thread.")
                app_logger.info(f"Input folder: {input_folder_path}")
                app_logger.info(f"Model folder: {model_folder_path}")
                app_logger.info(f"Input type: {input_type}")
                app_logger.info(f"Conversion factor: {conversion_factor}")
                app_logger.info(f"Chunk size: {chunk_size}")
                app_logger.info(f"Plotting request: {plotting}")
                app_logger.info(f"Debug log request: {debug_log}")

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
                    name="RetrievalThread",
                    target=self.run_retrieval_wrapper,
                    args=(
                        input_folder_path,
                        input_type,
                        model_folder_path,
                        conversion_factor,
                        chunk_size,
                        plotting,
                        debug_log,
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
            app_logger.error(
                "Error occurred while starting model thread", exc_info=True
            )
            self.show_message("Error occurred while starting model thread")

    def check_thread_status(self):
        if self.model_thread.is_alive():
            self.after(100, self.check_thread_status)
        else:
            self.on_model_thread_complete()

    def display_plot_in_window(self, image, title="Plot", cmap="viridis"):
        window = tk.Toplevel(self)
        window.title(title)

        fig = Figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        cax = ax.imshow(image, cmap=cmap)
        fig.colorbar(cax)
        ax.set_title(title)

        canvas = FigureCanvasTkAgg(fig, master=window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def on_model_thread_complete(self):
        # Stop the progress bar
        self.progress_bar.stop()

        # Enable the Run button
        self.button_run.config(state=tk.NORMAL)

        # Wait 5 seconds before closing the progress window
        self.progress_window.after(5000, self.progress_window.destroy)

    def run_retrieval_wrapper(
        self,
        input_folder_path: str,
        input_type: str,
        model_folder_path: str,
        conversion_factor: float,
        chunk_size: int,
        plotting: bool = False,
        debug_log: bool = False,
    ):
        """
        This function runs on the new thread and starts the retrieval function
        :param debug_log: bool to set log level to debug mode
        :param plotting: bool to plot the results or not
        :param conversion_factor: image conversion factor
        :param input_folder_path: path to the input folder
        :param input_type: type of input file
        :param model_folder_path: path to the model folder
        :return: Shows completion message and is able to run again
        """

        result = run_retrieval(
            input_folder_path=input_folder_path,
            input_type=input_type,
            model_folder_path=model_folder_path,
            conversion_factor=conversion_factor,
            chunk_size=chunk_size,
            show_message_callback=self.show_message,
            plotting=plotting,
            debug_log=debug_log,
        )

        # Safely update the GUI
        self.after(0, self.on_model_thread_complete)
        self.after(0, self.show_message, result.message)
        self.after(0, self.progress_label.config, {"text": result.message})

        if result.success and result.plots:
            for img, title, cmap in result.plots:
                self.after(0, lambda i=img, t=title, c=cmap: self.display_plot_in_window(i, t, c))


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


    app_logger.info("Starting the PyL2BV GUI application.")
    gui = SimpleGUI()
    gui.mainloop()
    app_logger.info("PyL2BV GUI application closed.")


if __name__ == "__main__":
    main()
