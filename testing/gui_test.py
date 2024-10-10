import os
import tkinter as tk
import unittest
from unittest.mock import patch

from gui import NonNegativeNumberEntry, SimpleGUI


class TestSimpleGUI(unittest.TestCase):
    def setUp(self):
        self.gui = SimpleGUI()
        self.gui.update()

    def tearDown(self):
        self.gui.destroy()

    def test_initial_state(self):
        self.assertEqual(
            self.gui.title(), "Retrieval of biophysical variables"
        )
        self.assertEqual(
            self.gui.entry_input_folder.get(), f"{os.getcwd()}/input"
        )
        self.assertEqual(
            self.gui.entry_model_folder.get(), f"{os.getcwd()}/models"
        )
        self.assertEqual(self.gui.entry_conversion_factor.get(), "0.0001")
        self.assertEqual(self.gui.input_type_var.get(), "CHIME netCDF")

    def test_browse_input_folder(self):
        with patch(
            "tkinter.filedialog.askdirectory", return_value="/mock/input"
        ):
            self.gui.browse_input_folder()
            self.assertEqual(self.gui.entry_input_folder.get(), "/mock/input")

    def test_browse_model_folder(self):
        with patch(
            "tkinter.filedialog.askdirectory", return_value="/mock/models"
        ):
            self.gui.browse_model_folder()
            self.assertEqual(self.gui.entry_model_folder.get(), "/mock/models")

    def test_run_button_disabled(self):
        self.gui.entry_input_folder.delete(0, tk.END)
        self.gui.entry_input_folder.insert(0, os.getcwd())
        self.gui.entry_model_folder.delete(0, tk.END)
        self.gui.entry_model_folder.insert(0, os.getcwd())
        self.gui.start_model_thread()
        self.assertEqual(self.gui.button_run["state"], tk.DISABLED)

    def test_non_negative_number_entry(self):
        entry = NonNegativeNumberEntry(self.gui)
        self.assertTrue(entry.validate_input("0"))
        self.assertTrue(entry.validate_input("0.1"))
        self.assertTrue(entry.validate_input("123"))
        self.assertFalse(entry.validate_input("-1"))
        self.assertFalse(entry.validate_input("abc"))


if __name__ == "__main__":
    unittest.main()
