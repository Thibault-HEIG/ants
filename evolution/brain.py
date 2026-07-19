"""
brain.py — Neural Network "Brain" for Creatures
===============================================

Wraps the generic NeuralNetwork with creature-specific dimensions (71→16→8→4),
motor command interpretation, and genome vector encoding/decoding.
"""

from __future__ import annotations

import numpy as np

from core.constants import NN_INPUTS, NN_HIDDEN_1, NN_HIDDEN_2, NN_OUTPUTS, GENOME_SIZE
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
            hidden1_size=NN_HIDDEN_1,
            hidden2_size=NN_HIDDEN_2,
            output_size=NN_OUTPUTS,
            rng=rng,
        )
        self._cached_genome: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Properties delegating to underlying network
    # ------------------------------------------------------------------

    @property
    def weights_input_hidden1(self) -> np.ndarray:
        return self.net.weights_input_hidden1

    @weights_input_hidden1.setter
    def weights_input_hidden1(self, val: np.ndarray) -> None:
        self.net.weights_input_hidden1 = val
        self._cached_genome = None

    @property
    def bias_hidden1(self) -> np.ndarray:
        return self.net.bias_hidden1

    @bias_hidden1.setter
    def bias_hidden1(self, val: np.ndarray) -> None:
        self.net.bias_hidden1 = val
        self._cached_genome = None

    @property
    def weights_hidden1_hidden2(self) -> np.ndarray:
        return self.net.weights_hidden1_hidden2

    @weights_hidden1_hidden2.setter
    def weights_hidden1_hidden2(self, val: np.ndarray) -> None:
        self.net.weights_hidden1_hidden2 = val
        self._cached_genome = None

    @property
    def bias_hidden2(self) -> np.ndarray:
        return self.net.bias_hidden2

    @bias_hidden2.setter
    def bias_hidden2(self, val: np.ndarray) -> None:
        self.net.bias_hidden2 = val
        self._cached_genome = None

    @property
    def weights_hidden2_output(self) -> np.ndarray:
        return self.net.weights_hidden2_output

    @weights_hidden2_output.setter
    def weights_hidden2_output(self, val: np.ndarray) -> None:
        self.net.weights_hidden2_output = val
        self._cached_genome = None

    @property
    def bias_output(self) -> np.ndarray:
        return self.net.bias_output

    @bias_output.setter
    def bias_output(self, val: np.ndarray) -> None:
        self.net.bias_output = val
        self._cached_genome = None

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Run the neural network on a single set of sensor inputs.

        Parameters
        ----------
        inputs : np.ndarray, shape (87,)
            Normalised sensor readings and internal state vector.

        Returns
        -------
        np.ndarray, shape (6,)
            [turn, speed_factor, attack_boolean, eat_boolean, take_boolean, release_boolean]
            - turn ∈ [-1, 1]: negative = left, positive = right
            - speed_factor ∈ [0, 1]: 0 = stop, 1 = full speed
            - attack_boolean ∈ {0.0, 1.0}: 1.0 = attack, 0.0 = hold fire
            - eat_boolean ∈ {0.0, 1.0}: 1.0 = attempt to eat, 0.0 = don't eat
            - take_boolean ∈ {0.0, 1.0}: 1.0 = attempt to take object, 0.0 = don't take
            - release_boolean ∈ {0.0, 1.0}: 1.0 = attempt to release object, 0.0 = don't release
        """
        output = self.net.forward_raw(inputs)

        result = np.array([
            output[0],                        # turn (keep sign)
            (output[1] + 1.0) / 2.0,          # speed: map [-1,1] → [0,1]
            1.0 if output[2] > 0.0 else 0.0,  # attack boolean
            1.0 if output[3] > 0.0 else 0.0,  # eat boolean
            1.0 if output[4] > 0.0 else 0.0,  # take boolean
            1.0 if output[5] > 0.0 else 0.0,  # release boolean
        ])
        return result

    # ------------------------------------------------------------------
    # Genome encoding / decoding
    # ------------------------------------------------------------------

    def get_genome(self) -> np.ndarray:
        """Flatten all weights and biases into a single 1-D vector (size 1468)."""
        if self._cached_genome is None:
            self._cached_genome = np.concatenate([
                self.weights_input_hidden1.flatten(),
                self.bias_hidden1.flatten(),
                self.weights_hidden1_hidden2.flatten(),
                self.bias_hidden2.flatten(),
                self.weights_hidden2_output.flatten(),
                self.bias_output.flatten(),
            ])
        return self._cached_genome

    def set_genome(self, genome: np.ndarray) -> None:
        """Reconstruct all weight matrices from a flat genome vector."""
        assert genome.shape == (GENOME_SIZE,), (
            f"Expected genome of length {GENOME_SIZE}, got {genome.shape}"
        )

        idx = 0

        size_ih1 = NN_INPUTS * NN_HIDDEN_1
        self.weights_input_hidden1 = genome[idx:idx + size_ih1].reshape(
            NN_INPUTS, NN_HIDDEN_1
        )
        idx += size_ih1

        self.bias_hidden1 = genome[idx:idx + NN_HIDDEN_1].copy()
        idx += NN_HIDDEN_1

        size_h1h2 = NN_HIDDEN_1 * NN_HIDDEN_2
        self.weights_hidden1_hidden2 = genome[idx:idx + size_h1h2].reshape(
            NN_HIDDEN_1, NN_HIDDEN_2
        )
        idx += size_h1h2

        self.bias_hidden2 = genome[idx:idx + NN_HIDDEN_2].copy()
        idx += NN_HIDDEN_2

        size_h2o = NN_HIDDEN_2 * NN_OUTPUTS
        self.weights_hidden2_output = genome[idx:idx + size_h2o].reshape(
            NN_HIDDEN_2, NN_OUTPUTS
        )
        idx += size_h2o

        self.bias_output = genome[idx:idx + NN_OUTPUTS].copy()
        self._cached_genome = genome.copy()

    @classmethod
    def genome_size(cls) -> int:
        """Total number of floats in the genome vector."""
        return GENOME_SIZE
