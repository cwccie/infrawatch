"""Machine learning anomaly detection models.

Implements Isolation Forest, Local Outlier Factor, and Autoencoder
using NumPy/SciPy. Falls back to scikit-learn when available for
Isolation Forest and LOF; provides pure-NumPy implementations otherwise.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from infrawatch.models.base import AnomalyDetector, AnomalyScore


class IsolationForestDetector(AnomalyDetector):
    """Isolation Forest anomaly detector.

    Uses scikit-learn's IsolationForest if available, otherwise provides
    a simplified NumPy implementation based on random feature splitting.

    Args:
        contamination: Expected fraction of anomalies in training data.
        n_estimators: Number of isolation trees.
        window: Sliding window to create feature vectors from time series.
        random_state: Random seed for reproducibility.
    """

    name = "isolation_forest"

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 100,
        window: int = 10,
        random_state: int = 42,
    ):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.window = window
        self.random_state = random_state
        self._sklearn_model = None
        self._threshold: float = 0.0
        self._use_sklearn: bool = False

    def _to_features(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        """Convert 1D time series to sliding window feature matrix."""
        n = len(values)
        if n < self.window:
            # Pad with the mean
            padded = np.pad(values, (self.window - n, 0), constant_values=np.nanmean(values))
            return padded.reshape(1, -1)

        features = np.zeros((n - self.window + 1, self.window))
        for i in range(n - self.window + 1):
            features[i] = values[i:i + self.window]

        # Replace NaNs with column means
        col_means = np.nanmean(features, axis=0)
        for j in range(features.shape[1]):
            mask = np.isnan(features[:, j])
            features[mask, j] = col_means[j]

        return features

    def fit(self, values: NDArray[np.float64]) -> None:
        features = self._to_features(values)

        try:
            from sklearn.ensemble import IsolationForest
            self._sklearn_model = IsolationForest(
                contamination=self.contamination,
                n_estimators=self.n_estimators,
                random_state=self.random_state,
            )
            self._sklearn_model.fit(features)
            self._use_sklearn = True
        except ImportError:
            # Pure NumPy fallback: compute statistics for threshold
            self._use_sklearn = False
            norms = np.linalg.norm(features - np.mean(features, axis=0), axis=1)
            self._threshold = float(np.percentile(norms, (1 - self.contamination) * 100))

    def detect(self, values: NDArray[np.float64]) -> AnomalyScore:
        features = self._to_features(values)
        n = len(values)

        if self._use_sklearn and self._sklearn_model is not None:
            raw_scores = -self._sklearn_model.score_samples(features)
            predictions = self._sklearn_model.predict(features)

            # Map back to original length
            scores = np.zeros(n)
            is_anomaly = np.zeros(n, dtype=bool)

            offset = n - len(raw_scores)
            scores[offset:] = raw_scores
            is_anomaly[offset:] = predictions == -1

            # Normalize scores to [0, 1]
            if np.max(scores) > np.min(scores):
                scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
        else:
            # NumPy fallback
            mean_feature = np.mean(features, axis=0)
            norms = np.linalg.norm(features - mean_feature, axis=1)

            scores = np.zeros(n)
            is_anomaly = np.zeros(n, dtype=bool)

            offset = n - len(norms)
            scores[offset:] = norms
            is_anomaly[offset:] = norms > self._threshold

            if np.max(scores) > 0:
                scores = scores / np.max(scores)

        return AnomalyScore(
            scores=scores,
            is_anomaly=is_anomaly,
            threshold=self.contamination,
            detector_name=self.name,
        )


class LOFDetector(AnomalyDetector):
    """Local Outlier Factor anomaly detector.

    Uses scikit-learn's LocalOutlierFactor if available, otherwise provides
    a simplified k-nearest-neighbor distance-based approach.

    Args:
        n_neighbors: Number of neighbors for LOF computation.
        contamination: Expected fraction of anomalies.
        window: Sliding window size for feature construction.
    """

    name = "lof"

    def __init__(
        self,
        n_neighbors: int = 20,
        contamination: float = 0.05,
        window: int = 10,
    ):
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.window = window
        self._training_features: NDArray[np.float64] | None = None
        self._threshold: float = 0.0

    def _to_features(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        """Convert time series to feature matrix."""
        n = len(values)
        if n < self.window:
            padded = np.pad(values, (self.window - n, 0), constant_values=np.nanmean(values))
            return padded.reshape(1, -1)

        features = np.zeros((n - self.window + 1, self.window))
        for i in range(n - self.window + 1):
            features[i] = values[i:i + self.window]

        col_means = np.nanmean(features, axis=0)
        for j in range(features.shape[1]):
            mask = np.isnan(features[:, j])
            features[mask, j] = col_means[j]

        return features

    def fit(self, values: NDArray[np.float64]) -> None:
        self._training_features = self._to_features(values)

        # Compute k-distance for threshold
        n = len(self._training_features)
        k = min(self.n_neighbors, n - 1)
        if k < 1:
            k = 1

        distances = np.zeros(n)
        for i in range(n):
            dists = np.linalg.norm(self._training_features - self._training_features[i], axis=1)
            dists[i] = np.inf  # Exclude self
            k_nearest = np.sort(dists)[:k]
            distances[i] = np.mean(k_nearest)

        self._threshold = float(np.percentile(distances, (1 - self.contamination) * 100))

    def detect(self, values: NDArray[np.float64]) -> AnomalyScore:
        features = self._to_features(values)
        n = len(values)

        try:
            from sklearn.neighbors import LocalOutlierFactor
            lof = LocalOutlierFactor(
                n_neighbors=min(self.n_neighbors, len(features) - 1),
                contamination=self.contamination,
                novelty=False,
            )
            predictions = lof.fit_predict(features)
            raw_scores = -lof.negative_outlier_factor_

            scores = np.zeros(n)
            is_anomaly = np.zeros(n, dtype=bool)
            offset = n - len(raw_scores)
            scores[offset:] = raw_scores
            is_anomaly[offset:] = predictions == -1

            if np.max(scores) > np.min(scores):
                scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
        except ImportError:
            # NumPy fallback: k-distance anomaly scoring
            if self._training_features is None:
                self._training_features = features

            ref = self._training_features
            k = min(self.n_neighbors, len(ref) - 1)
            if k < 1:
                k = 1

            raw_scores = np.zeros(len(features))
            for i in range(len(features)):
                dists = np.linalg.norm(ref - features[i], axis=1)
                k_nearest = np.sort(dists)[:k]
                raw_scores[i] = np.mean(k_nearest)

            scores = np.zeros(n)
            is_anomaly = np.zeros(n, dtype=bool)
            offset = n - len(raw_scores)
            scores[offset:] = raw_scores
            is_anomaly[offset:] = raw_scores > self._threshold

            if np.max(scores) > 0:
                scores = scores / np.max(scores)

        return AnomalyScore(
            scores=scores,
            is_anomaly=is_anomaly,
            threshold=self.contamination,
            detector_name=self.name,
        )


class AutoencoderDetector(AnomalyDetector):
    """Simple autoencoder anomaly detector (NumPy implementation).

    Uses a single-hidden-layer autoencoder trained with gradient descent.
    Anomalies are detected via reconstruction error.

    Args:
        hidden_dim: Hidden layer dimension.
        window: Input window size (feature dimension).
        epochs: Training epochs.
        learning_rate: Gradient descent learning rate.
        threshold_percentile: Percentile of training reconstruction error
                             above which points are flagged anomalous.
    """

    name = "autoencoder"

    def __init__(
        self,
        hidden_dim: int = 5,
        window: int = 10,
        epochs: int = 50,
        learning_rate: float = 0.01,
        threshold_percentile: float = 95.0,
    ):
        self.hidden_dim = hidden_dim
        self.window = window
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.threshold_percentile = threshold_percentile
        self._W1: NDArray[np.float64] | None = None
        self._b1: NDArray[np.float64] | None = None
        self._W2: NDArray[np.float64] | None = None
        self._b2: NDArray[np.float64] | None = None
        self._threshold: float = 0.0
        self._mean: NDArray[np.float64] | None = None
        self._std: NDArray[np.float64] | None = None

    def _to_features(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        n = len(values)
        if n < self.window:
            padded = np.pad(values, (self.window - n, 0), constant_values=np.nanmean(values))
            return padded.reshape(1, -1)
        features = np.zeros((n - self.window + 1, self.window))
        for i in range(n - self.window + 1):
            features[i] = values[i:i + self.window]
        col_means = np.nanmean(features, axis=0)
        for j in range(features.shape[1]):
            mask = np.isnan(features[:, j])
            features[mask, j] = col_means[j]
        return features

    @staticmethod
    def _relu(x: NDArray) -> NDArray:
        return np.maximum(0, x)

    @staticmethod
    def _relu_grad(x: NDArray) -> NDArray:
        return (x > 0).astype(np.float64)

    def fit(self, values: NDArray[np.float64]) -> None:
        features = self._to_features(values)

        # Normalize features
        self._mean = np.mean(features, axis=0)
        self._std = np.std(features, axis=0)
        self._std[self._std == 0] = 1.0
        X = (features - self._mean) / self._std

        input_dim = X.shape[1]
        rng = np.random.RandomState(42)

        # Xavier initialization
        self._W1 = rng.randn(input_dim, self.hidden_dim) * np.sqrt(2.0 / input_dim)
        self._b1 = np.zeros(self.hidden_dim)
        self._W2 = rng.randn(self.hidden_dim, input_dim) * np.sqrt(2.0 / self.hidden_dim)
        self._b2 = np.zeros(input_dim)

        # Train with mini-batch gradient descent
        n_samples = len(X)
        for _ in range(self.epochs):
            # Forward pass
            hidden_raw = X @ self._W1 + self._b1
            hidden = self._relu(hidden_raw)
            output = hidden @ self._W2 + self._b2

            # Reconstruction error
            error = output - X

            # Backward pass
            dW2 = hidden.T @ error / n_samples
            db2 = np.mean(error, axis=0)
            dhidden = error @ self._W2.T * self._relu_grad(hidden_raw)
            dW1 = X.T @ dhidden / n_samples
            db1 = np.mean(dhidden, axis=0)

            # Update
            self._W1 -= self.learning_rate * dW1
            self._b1 -= self.learning_rate * db1
            self._W2 -= self.learning_rate * dW2
            self._b2 -= self.learning_rate * db2

        # Set threshold from training reconstruction error
        recon_errors = np.mean((output - X) ** 2, axis=1)
        self._threshold = float(np.percentile(recon_errors, self.threshold_percentile))

    def _reconstruct(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        hidden = self._relu(X @ self._W1 + self._b1)
        return hidden @ self._W2 + self._b2

    def detect(self, values: NDArray[np.float64]) -> AnomalyScore:
        features = self._to_features(values)
        n = len(values)

        X = (features - self._mean) / self._std
        reconstructed = self._reconstruct(X)
        recon_errors = np.mean((reconstructed - X) ** 2, axis=1)

        scores = np.zeros(n)
        is_anomaly = np.zeros(n, dtype=bool)

        offset = n - len(recon_errors)
        scores[offset:] = recon_errors
        is_anomaly[offset:] = recon_errors > self._threshold

        # Normalize scores
        max_score = np.max(scores)
        if max_score > 0:
            scores = scores / max_score

        return AnomalyScore(
            scores=scores,
            is_anomaly=is_anomaly,
            threshold=self._threshold,
            detector_name=self.name,
        )
