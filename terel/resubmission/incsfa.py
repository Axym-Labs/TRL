"""Python 3 port of Kompella et al.'s official linear IncSFA implementation.

The update equations follow official commit a4972d2 (CCIPCA whitening followed
by sequential MCA). The port adds numerical guards, deterministic RNG, explicit
episode boundaries, and a scikit-learn-like interface.
"""

import numpy as np


class _RunningMean:
    def __init__(self):
        self.value = 0.0
        self.count = 0

    def update_and_center(self, sample):
        sample = np.asarray(sample, dtype=np.float64)
        self.value = (self.count * self.value + sample) / (self.count + 1)
        self.count += 1
        return sample if self.count == 1 else sample - self.value

    def center(self, samples):
        return np.asarray(samples, dtype=np.float64) - self.value


class _CCIPCAWhitening:
    def __init__(self, input_dim, output_dim, rng):
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.count = 1
        self._vectors = 0.1 * rng.standard_normal((output_dim, input_dim))
        self._values = np.linalg.norm(self._vectors, axis=1).clip(min=1e-12)
        self._normalized = self._vectors / self._values[:, None]
        self.components = self._normalized / np.sqrt(self._values[:, None])

    @staticmethod
    def _amnesic_weights(index):
        n1, n2, m, coefficient = 20.0, 200.0, 2000.0, 3.0
        if index < n1:
            amnesia = 0.0
        elif index < n2:
            amnesia = coefficient * (index - n1) / (n2 - n1)
        else:
            amnesia = coefficient + (index - n2) / m
        return (index - 1 - amnesia) / index, (1 + amnesia) / index

    def update(self, sample):
        self.count += 1
        old_weight, new_weight = self._amnesic_weights(self.count)
        residual = np.asarray(sample, dtype=np.float64).copy()
        for index in range(self.output_dim):
            vector = self._vectors[index].copy()
            vector = (
                old_weight * vector
                + new_weight * (residual @ vector) / max(self._values[index], 1e-12) * residual
            )
            value = max(float(np.linalg.norm(vector)), 1e-12)
            normalized = vector / value
            residual -= (residual @ normalized) * normalized
            self._vectors[index] = vector
            self._values[index] = value
            self._normalized[index] = normalized
        self.components = self._normalized / np.sqrt(self._values[:, None].clip(min=1e-12))

    def transform_one(self, sample):
        return np.asarray(sample, dtype=np.float64) @ self.components.T


class _SequentialMCA:
    def __init__(self, input_dim, output_dim, learning_rate, rng):
        self.output_dim = int(output_dim)
        self.learning_rate = float(learning_rate)
        vectors = 0.1 * rng.standard_normal((output_dim, input_dim))
        self.components = vectors / np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-12)
        self.gamma = 1.2 * (0.2 / self.learning_rate)

    def update(self, sample):
        covariance = np.outer(sample, sample)
        for index in range(self.output_dim):
            rate = self.learning_rate / (1.0 + 1.2 * index)
            vector = self.components[index][:, None]
            vector = (1.5 - rate) * vector - rate * covariance @ vector
            squared_norm = max(float((vector.T @ vector).item()), 1e-12)
            covariance += self.gamma * (vector @ vector.T) / squared_norm
            self.components[index] = (vector[:, 0] / np.sqrt(squared_norm))


class IncrementalLinearSFA:
    """Official-algorithm linear IncSFA with explicit stream boundaries."""

    def __init__(
        self,
        *,
        input_dim: int,
        whitening_dim: int,
        output_dim: int,
        learning_rate: float = 0.001,
        seed: int = 0,
    ):
        if not 0 < output_dim <= whitening_dim <= input_dim:
            raise ValueError("require 0 < output_dim <= whitening_dim <= input_dim")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        rng = np.random.default_rng(seed)
        self.input_dim = int(input_dim)
        self.whitening_dim = int(whitening_dim)
        self.output_dim = int(output_dim)
        self.mean = _RunningMean()
        self.whitening = _CCIPCAWhitening(input_dim, whitening_dim, rng)
        self.mca = _SequentialMCA(whitening_dim, output_dim, learning_rate, rng)
        self.components_ = np.zeros((output_dim, input_dim), dtype=np.float64)
        self.previous_whitened = None
        self.derivative_pair_count_ = 0

    def update(self, sample, *, new_episode: bool):
        centered = self.mean.update_and_center(sample)
        self.whitening.update(centered)
        whitened = self.whitening.transform_one(centered)
        if new_episode or self.previous_whitened is None:
            self.previous_whitened = whitened.copy()
            return
        derivative = whitened - self.previous_whitened
        self.previous_whitened = whitened.copy()
        self.mca.update(derivative)
        self.components_ = self.mca.components @ self.whitening.components
        self.derivative_pair_count_ += 1

    def fit(self, samples, *, boundaries, epochs: int = 1):
        samples = np.asarray(samples, dtype=np.float64)
        boundaries = np.asarray(boundaries, dtype=bool)
        if samples.ndim != 2 or samples.shape[1] != self.input_dim:
            raise ValueError("samples have the wrong shape")
        if boundaries.shape != (len(samples),):
            raise ValueError("boundaries must contain one flag per sample")
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        for _ in range(epochs):
            for index, sample in enumerate(samples):
                self.update(sample, new_episode=bool(boundaries[index] or index == 0))
        return self

    def transform(self, samples):
        return self.mean.center(samples) @ self.components_.T

    def dynamic_state_numel(self):
        arrays = (
            np.asarray(self.mean.value),
            self.whitening._vectors,
            self.whitening._values,
            self.whitening._normalized,
            self.whitening.components,
            self.mca.components,
            self.components_,
        )
        previous = 0 if self.previous_whitened is None else self.previous_whitened.size
        return int(sum(array.size for array in arrays) + previous + 2)
