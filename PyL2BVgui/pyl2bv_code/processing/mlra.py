# Base Prototype Class
import numpy as np


class MLRA_Methods:
    def __init__(self, image: np.ndarray, bio_model):
        """
        Initialize MLRA Methods with shared attributes such as image and bio_models.
        :param image: The 3D image or data cube to process.
        :param bio_model: The list of biological models for the analysis.
        """
        self.image = image  # This is the 3D image or data cube
        self.bio_model = bio_model  # Model hyperparameters for ML methods

    def perform_mlra(self) -> tuple:
        """
        Prototype method for MLRA. Should be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")
