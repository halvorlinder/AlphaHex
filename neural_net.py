from abc import ABC, abstractmethod

import numpy as np


class NeuralNet(ABC):

    model : any

    @abstractmethod
    def train(self, examples : np.ndarray):
        pass

    @abstractmethod
    def predict(self, input : np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def save(self, filename: str):
        pass

    @abstractmethod
    def load(self, filename: str):
        pass