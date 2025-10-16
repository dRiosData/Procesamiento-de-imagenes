# app/detection/base.py
from abc import ABC, abstractmethod

class BaseDetector(ABC):
    """Clase base abstracta para todos los detectores de objetos."""

    @abstractmethod
    def detect(self, image):
        """Detecta objetos en una imagen."""
        pass

    @abstractmethod
    def load_model(self, model_path: str):
        """Carga el modelo desde un archivo."""
        pass

