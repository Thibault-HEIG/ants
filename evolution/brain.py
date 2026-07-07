"""
brain.py — Neural Network "Brain" for Creatures
===============================================

Wraps the generic NeuralNetwork with creature-specific dimensions (22→8→4),
motor command interpretation, and genome vector encoding/decoding.
"""

from __future__ import annotations

import numpy as np

from core.constants import NN_INPUTS, NN_HIDDEN, NN_OUTPUTS, GENOME_SIZE
from evolution.network import NeuralNetwork


class Brain:
    """The decision-making AI unit for a creature.

    Parameters
    ----------
    rng : np.random.Generator
        Seeded random generator for reproducibility.
    """

    def __init__(self, rng: np.random.Generator) -> None:
        self.net: NeuralNetwork = NeuralNetwork(
            input_size=NN_INPUTS,
            hidden_size=NN_HIDDEN,
            output_size=NN_OUTPUTS,
            rng=rng,
        )

    # ------------------------------------------------------------------
    # Properties delegating to underlying network
    # ------------------------------------------------------------------

    @property
    def weights_input_hidden(self) -> np.ndarray:
        return self.net.weights_input_hidden

    @weights_input_hidden.setter
    def weights_input_hidden(self, val: np.ndarray) -> None:
        self.net.weights_input_hidden = val

    @property
    def bias_hidden(self) -> np.ndarray:
        return self.net.bias_hidden

    @bias_hidden.setter
    def bias_hidden(self, val: np.ndarray) -> None:
        self.net.bias_hidden = val

    @property
    def weights_hidden_output(self) -> np.ndarray:
        return self.net.weights_hidden_output

    @weights_hidden_output.setter
    def weights_hidden_output(self, val: np.ndarray) -> None:
        self.net.weights_hidden_output = val

    @property
    def bias_output(self) -> np.ndarray:
        return self.net.bias_output

    @bias_output.setter
    def bias_output(self, val: np.ndarray) -> None:
        self.net.bias_output = val

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Run the neural network on a single set of sensor inputs.

        Parameters
        ----------
        inputs : np.ndarray, shape (22,)
            Normalised sensor readings and internal state vector.

        Returns
        -------
        np.ndarray, shape (4,)
            [turn, speed_factor, attack_boolean, eat_boolean]
            - turn ∈ [-1, 1]: negative = left, positive = right
            - speed_factor ∈ [0, 1]: 0 = stop, 1 = full speed
            - attack_boolean ∈ {0.0, 1.0}: 1.0 = attack, 0.0 = hold fire
            - eat_boolean ∈ {0.0, 1.0}: 1.0 = attempt to eat, 0.0 = don't eat
        """
        output = self.net.forward_raw(inputs)

        result = np.array([
            output[0],                        # turn (keep sign)
            (output[1] + 1.0) / 2.0,          # speed: map [-1,1] → [0,1]
            1.0 if output[2] > 0.0 else 0.0,  # attack boolean
            1.0 if output[3] > 0.0 else 0.0,  # eat boolean
        ])
        return result

    # ------------------------------------------------------------------
    # Genome encoding / decoding
    # ------------------------------------------------------------------

    def get_genome(self) -> np.ndarray:
        """Flatten all weights and biases into a single 1-D vector (size 219)."""
        return np.concatenate([
            self.weights_input_hidden.flatten(),
            self.bias_hidden.flatten(),
            self.weights_hidden_output.flatten(),
            self.bias_output.flatten(),
        ])

    def set_genome(self, genome: np.ndarray) -> None:
        """Reconstruct all weight matrices from a flat genome vector."""
        assert genome.shape == (GENOME_SIZE,), (
            f"Expected genome of length {GENOME_SIZE}, got {genome.shape}"
        )

        idx = 0

        size_ih = NN_INPUTS * NN_HIDDEN
        self.weights_input_hidden = genome[idx:idx + size_ih].reshape(
            NN_INPUTS, NN_HIDDEN
        )
        idx += size_ih

        self.bias_hidden = genome[idx:idx + NN_HIDDEN].copy()
        idx += NN_HIDDEN

        size_ho = NN_HIDDEN * NN_OUTPUTS
        self.weights_hidden_output = genome[idx:idx + size_ho].reshape(
            NN_HIDDEN, NN_OUTPUTS
        )
        idx += size_ho

        self.bias_output = genome[idx:idx + NN_OUTPUTS].copy()

    @classmethod
    def genome_size(cls) -> int:
        """Total number of floats in the genome vector."""
        return GENOME_SIZE
