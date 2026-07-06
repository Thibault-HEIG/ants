"""
network.py — Generic Neural Network Architecture
================================================

Provides the fundamental mathematical model for fully-connected neural networks
using NumPy, decoupled from creature-specific logic or genome encoding.
"""

from __future__ import annotations

import numpy as np


class NeuralNetwork:
    """A fully-connected feedforward neural network with one hidden layer.

    Parameters
    ----------
    input_size : int
        Number of input neurons.
    hidden_size : int
        Number of hidden neurons.
    output_size : int
        Number of output neurons.
    rng : np.random.Generator
        Random number generator for weight initialisation.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        rng: np.random.Generator,
    ) -> None:
        self.input_size: int = input_size
        self.hidden_size: int = hidden_size
        self.output_size: int = output_size

        # Initialise weights with standard normal distribution scaled by 0.5
        self.weights_input_hidden: np.ndarray = (
            rng.standard_normal((input_size, hidden_size)) * 0.5
        )
        self.bias_hidden: np.ndarray = np.zeros(hidden_size)

        self.weights_hidden_output: np.ndarray = (
            rng.standard_normal((hidden_size, output_size)) * 0.5
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
        hidden_raw = inputs @ self.weights_input_hidden + self.bias_hidden
        hidden = np.tanh(hidden_raw)
        output_raw = hidden @ self.weights_hidden_output + self.bias_output
        return np.tanh(output_raw)
