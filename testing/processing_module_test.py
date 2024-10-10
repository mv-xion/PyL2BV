import os
import shutil
import unittest
from unittest.mock import MagicMock, patch

from bioretrieval.processing.processing_module import (bio_retrieval_module,
                                                       make_output_folder)


class TestBioRetrievalModule(unittest.TestCase):
    def setUp(self):
        self.input_folder_path = "test_input"
        self.model_folder_path = "test_models"
        self.output_folder_path = "test_input/output"
        self.conversion_factor = 0.0001
        self.input_type = "CHIME netCDF"
        self.show_message = MagicMock()

        # Create test directories and files
        os.makedirs(self.input_folder_path, exist_ok=True)
        os.makedirs(self.model_folder_path, exist_ok=True)
        with open(
            os.path.join(self.input_folder_path, "test_IMG_001.nc"), "w"
        ) as f:
            f.write("test data")
        with open(
            os.path.join(self.input_folder_path, "test_GEO_001.nc"), "w"
        ) as f:
            f.write("test data")
        with open(
            os.path.join(self.input_folder_path, "test_QUA_001.nc"), "w"
        ) as f:
            f.write("test data")

    def tearDown(self):
        # Remove test directories and files
        shutil.rmtree(self.input_folder_path, ignore_errors=True)
        shutil.rmtree(self.model_folder_path, ignore_errors=True)

    @patch("processing_module.Retrieval")
    @patch("processing_module.Logger")
    def test_bio_retrieval_module_success(self, MockLogger, MockRetrieval):
        # Mock the Retrieval and Logger classes
        mock_retrieval_instance = MockRetrieval.return_value
        mock_retrieval_instance.bio_retrieval = 0
        mock_retrieval_instance.export_retrieval.return_value = 0

        result = bio_retrieval_module(
            self.input_folder_path,
            self.input_type,
            self.model_folder_path,
            self.conversion_factor,
            self.show_message,
        )

        self.assertEqual(result, 0)
        self.show_message.assert_any_call("Type: " + self.input_type)
        self.show_message.assert_any_call(
            "Retrieval of test_IMG_001.nc successful."
        )

    @patch("processing_module.Retrieval")
    @patch("processing_module.Logger")
    def test_bio_retrieval_module_missing_files(
        self, MockLogger, MockRetrieval
    ):
        # Remove one of the required files to simulate missing file scenario
        os.remove(os.path.join(self.input_folder_path, "test_GEO_001.nc"))

        result = bio_retrieval_module(
            self.input_folder_path,
            self.input_type,
            self.model_folder_path,
            self.conversion_factor,
            self.show_message,
        )

        self.assertEqual(result, 1)
        self.show_message.assert_any_call(
            "Missing complementary files for CHIME image."
        )

    def test_make_output_folder(self):
        # Test the make_output_folder function
        output_path = "test_output"
        if os.path.exists(output_path):
            shutil.rmtree(output_path)

        flag = make_output_folder(output_path)
        self.assertFalse(flag)
        self.assertTrue(os.path.exists(output_path))

        flag = make_output_folder(output_path)
        self.assertTrue(flag)
        self.assertTrue(os.path.exists(output_path))

        shutil.rmtree(output_path)


if __name__ == "__main__":
    unittest.main()
