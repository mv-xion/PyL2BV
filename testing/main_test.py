import unittest

from gui import main


class TestGuiMain(unittest.TestCase):
    def test_main(self):
        # Assuming main() has some side effects or return value to test
        result = main()
        self.assertIsNotNone(result)  # Example assertion


if __name__ == "__main__":
    unittest.main()
