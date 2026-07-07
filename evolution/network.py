"""
network.py — Generic Neural Network Architecture
================================================

Provides the fundamental mathematical model for fully-connected neural networks
using NumPy, decoupled from creature-specific logic or genome encoding.
"""

from __future__ import annotations

import numpy as np


class NeuralNetwork:
    """A fully-connected feedforward neural network with two hidden layers.

    Parameters
    ----------
    input_size : int
        Number of input neurons.
    hidden1_size : int
        Number of neurons in the first hidden layer.
    hidden2_size : int
        Number of neurons in the second hidden layer.
    output_size : int
        Number of output neurons.
    rng : np.random.Generator
        Random number generator for weight initialisation.
    """

    def __init__(
        self,
        input_size: int,
        hidden1_size: int,
        hidden2_size: int,
        output_size: int,
        rng: np.random.Generator,
    ) -> None:
        self.input_size: int = input_size
        self.hidden1_size: int = hidden1_size
        self.hidden2_size: int = hidden2_size
        self.output_size: int = output_size

        # Initialise weights with standard normal distribution scaled by 0.5
        self.weights_input_hidden1: np.ndarray = (
            rng.standard_normal((input_size, hidden1_size)) * 0.5
        )
        self.bias_hidden1: np.ndarray = np.zeros(hidden1_size)

        self.weights_hidden1_hidden2: np.ndarray = (
            rng.standard_normal((hidden1_size, hidden2_size)) * 0.5
        )
        self.bias_hidden2: np.ndarray = np.zeros(hidden2_size)

        self.weights_hidden2_output: np.ndarray = (
            rng.standard_normal((hidden2_size, output_size)) * 0.5
        )
        self.bias_output: np.ndarray = np.zeros(output_size)

    def forward_raw(self, inputs: np.ndarray) -> np.ndarray:
        """Execute a forward pass through the network.

        Parameters
        ----------
        inputs : np.ndarray
            Input array of shape (input_size,).

        Returns
        -------
        np.ndarray
            Raw tanh activations of shape (output_size,) in range [-1, 1].
        """
        hidden1_raw = inputs @ self.weights_input_hidden1 + self.bias_hidden1
        hidden1 = np.tanh(hidden1_raw)
        hidden2_raw = hidden1 @ self.weights_hidden1_hidden2 + self.bias_hidden2
        hidden2 = np.tanh(hidden2_raw)
        output_raw = hidden2 @ self.weights_hidden2_output + self.bias_output
        return np.tanh(output_raw)
