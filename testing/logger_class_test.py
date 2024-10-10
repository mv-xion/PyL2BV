import os
import unittest

from bioretrieval.auxiliar.logger_class import Logger


class TestLogger(unittest.TestCase):
    def setUp(self):
        self.log_path = "test_log"
        self.logger = Logger(self.log_path)

    def tearDown(self):
        try:
            os.remove(f"{self.log_path}_logfile.log")
        except FileNotFoundError:
            pass

    def test_log_message(self):
        message = "Test log message"
        self.logger.log_message(message)
        self.logger.close()

        with open(f"{self.log_path}_logfile.log", "r") as log_file:
            content = log_file.read()
            self.assertIn(message, content)

    def test_open(self):
        self.logger.close()
        self.assertTrue(self.logger.log_file_id.closed)

        self.logger.open()
        self.assertFalse(self.logger.log_file_id.closed)

    def test_close(self):
        self.logger.close()
        self.assertTrue(self.logger.log_file_id.closed)


if __name__ == "__main__":
    unittest.main()
