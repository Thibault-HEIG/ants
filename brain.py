"""
brain.py — Neural Network "Brain" for Each Ant
================================================

This is the core AI component.  Each ant has a fully-connected neural
network that takes 22 sensor inputs and produces 2 outputs (turn + speed).

**Why a neural network?**
  A neural network is essentially a big mathematical function that turns
  inputs into outputs.  The beauty is that we don't have to *program*
  the correct behaviour — instead we let evolution discover the right
  weights through trial and error.

**Architecture**::

    22 inputs  →  8 hidden neurons (tanh)  →  3 outputs (tanh)

The 22 inputs comprise:
  - 12 ray-based vision (left/right/forward × food/enemy/ally/wall)
  -  4 omnidirectional sensing (nearest food dist/angle, enemy dist/angle)
  -  4 internal state (hp, zone, speed, age)
  -  2 teamwork (ally density, enemy density)

**No machine learning libraries** are used here.  The forward pass is
implemented manually with NumPy so every step is transparent.
"""

from __future__ import annotations

import numpy as np

from ant_simulator.constants import NN_INPUTS, NN_HIDDEN, NN_OUTPUTS, GENOME_SIZE


class Brain:
    """A fully-connected neural network (22 → 8 → 2).

    The network maps 22 sensor/state readings to 2 motor commands.  All the
    "intelligence" of an ant lives in its weight matrices — these are the
    values that evolution tunes across generations.

    Parameters
    ----------
    rng : np.random.Generator
        Seeded random generator for reproducibility.
    """

    def __init__(self, rng: np.random.Generator) -> None:
        # ------------------------------------------------------------------
        # Weight initialisation
        # ------------------------------------------------------------------
        # We use small random values drawn from a standard normal distribution.
        # We scale them by 0.5 so they don't immediately saturate the tanh
        # activation function (which would just make the creature spin in circles).
        # Layer 1: Input → Hidden
        self.weights_input_hidden: np.ndarray = rng.standard_normal(
            (NN_INPUTS, NN_HIDDEN)
        ) * 0.5

        # Bias for each hidden neuron
        self.bias_hidden: np.ndarray = np.zeros(NN_HIDDEN)

        # Layer 2: Hidden → Output
        self.weights_hidden_output: np.ndarray = rng.standard_normal(
            (NN_HIDDEN, NN_OUTPUTS)
        ) * 0.5

        # Bias for each output neuron.
        self.bias_output: np.ndarray = np.zeros(NN_OUTPUTS)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Run the neural network on a single set of inputs.

        This is called every frame for every living ant.

        Parameters
        ----------
        inputs : np.ndarray, shape (22,)
            Normalised sensor readings and internal state.  See sensors.py
            for the full list of 22 inputs spanning vision rays,
            omnidirectional sensing, internal state, and teamwork.

        Returns
        -------
        np.ndarray, shape (3,)
            [turn, speed_factor, attack_boolean]
            - turn ∈ [-1, 1]:  negative = turn left, positive = turn right
            - speed_factor ∈ [0, 1]:  0 = stop, 1 = full speed
            - attack_boolean ∈ {0.0, 1.0}: 1.0 = attack, 0.0 = hold fire

        How it works, step by step
        --------------------------
        1. **Matrix multiply** inputs by weights — this computes a weighted
           sum of every input for each hidden neuron.
        2. **Add bias** — shifts the sum, letting the neuron activate even
           when inputs are all zero.
        3. **Activation function (tanh)** — squashes the result into [-1, 1].
           Without this non-linearity the entire network would collapse into
           a single linear transformation, no matter how many layers we add.
        4. Repeat for the output layer.
        """
        # --- Layer 1: Input → Hidden ---
        #
        # Matrix multiplication: each hidden neuron computes a weighted sum
        # of ALL 8 inputs.  The @ operator is shorthand for np.matmul.
        #
        #   hidden_raw[j] = Σᵢ inputs[i] * weights_input_hidden[i, j]
        #
        hidden_raw: np.ndarray = inputs @ self.weights_input_hidden + self.bias_hidden

        # Activation function: tanh squashes values into (-1, 1).
        # Why tanh and not sigmoid or ReLU?
        #   - tanh is centred at zero, which helps when outputs represent
        #     signed quantities like turning direction.
        #   - It's simple to understand and implement.
        hidden: np.ndarray = np.tanh(hidden_raw)

        # --- Layer 2: Hidden → Output ---
        #
        # Same process: weighted sum of hidden activations, then tanh.
        output_raw: np.ndarray = hidden @ self.weights_hidden_output + self.bias_output
        output: np.ndarray = np.tanh(output_raw)

        # Post-processing:
        #   output[0] = turn       ∈ [-1, 1]  (used directly)
        #   output[1] = speed_raw  ∈ [-1, 1]  → rescale to [0, 1]
        #   output[2] = attack_raw ∈ [-1, 1]  → threshold (> 0.0 → 1.0 else 0.0)
        #
        # We rescale speed because negative speed doesn't make sense for
        # forward-only movement.
        result = np.array([
            output[0],                    # turn (keep sign)
            (output[1] + 1.0) / 2.0,      # speed: map [-1,1] → [0,1]
            1.0 if output[2] > 0.0 else 0.0,  # attack boolean (0 or 1)
        ])
        return result

    # ------------------------------------------------------------------
    # Genome encoding / decoding
    # ------------------------------------------------------------------
    # The "genome" is just every weight and bias concatenated into a flat
    # 1-D array.  This makes it trivial for the genetic algorithm to
    # manipulate — it just operates on a vector of 46 floats.
    # ------------------------------------------------------------------

    def get_genome(self) -> np.ndarray:
        """Flatten all weights and biases into a single 1-D vector.

        Layout (total = 211 floats):
            [weights_ih (176) | bias_h (8) | weights_ho (24) | bias_o (3)]
        """
        return np.concatenate([
            self.weights_input_hidden.flatten(),    # 22×8 = 176
            self.bias_hidden.flatten(),             #         8
            self.weights_hidden_output.flatten(),   #  8×2 = 16
            self.bias_output.flatten(),             #         2
        ])

    def set_genome(self, genome: np.ndarray) -> None:
        """Reconstruct all weight matrices from a flat genome vector.

        This is the inverse of ``get_genome``.  The genetic algorithm calls
        this after breeding a child genome to install the new weights into
        the brain.

        Parameters
        ----------
        genome : np.ndarray, shape (211,)
            The flat vector of weights produced by evolution.
        """
        assert genome.shape == (GENOME_SIZE,), (
            f"Expected genome of length {GENOME_SIZE}, got {genome.shape}"
        )

        idx = 0

        size_ih = NN_INPUTS * NN_HIDDEN  # 176
        self.weights_input_hidden = genome[idx:idx + size_ih].reshape(
            NN_INPUTS, NN_HIDDEN
        )
        idx += size_ih

        self.bias_hidden = genome[idx:idx + NN_HIDDEN].copy()
        idx += NN_HIDDEN

        size_ho = NN_HIDDEN * NN_OUTPUTS  # 24
        self.weights_hidden_output = genome[idx:idx + size_ho].reshape(
            NN_HIDDEN, NN_OUTPUTS
        )
        idx += size_ho

        self.bias_output = genome[idx:idx + NN_OUTPUTS].copy()

    @classmethod
    def genome_size(cls) -> int:
        """Total number of floats in the genome vector.

        Useful when the genetics module needs to know how many values
        to allocate for a new genome.
        """
        return GENOME_SIZE
