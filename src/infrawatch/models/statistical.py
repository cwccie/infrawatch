"""Statistical anomaly detection models.

Pure NumPy/SciPy implementations of classical statistical methods for
anomaly detection in time series data.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats

from infrawatch.models.base import AnomalyDetector, AnomalyScore


class ZScoreDetector(AnomalyDetector):
    """Z-score based anomaly detection.

    Flags points whose Z-score exceeds the threshold. Optionally uses a
    sliding window for local statistics.

    Args:
        threshold: Z-score threshold for anomaly classification.
        window: Sliding window size. If None, uses global statistics.
    """

    name = "zscore"

    def __init__(self, threshold: float = 3.0, window: int | None = None):
        self.threshold = threshold
        self.window = window
        self._mean: float = 0.0
        self._std: float = 1.0

    def fit(self, values: NDArray[np.float64]) -> None:
        clean = values[~np.isnan(values)]
        if len(clean) > 0:
            self._mean = float(np.mean(clean))
            self._std = float(np.std(clean))
            if self._std == 0:
                self._std = 1.0

    def detect(self, values: NDArray[np.float64]) -> AnomalyScore:
        n = len(values)
        scores = np.zeros(n)

        if self.window and self.window < n:
            # Sliding window Z-scores
            for i in range(n):
                start = max(0, i - self.window)
                window_vals = values[start:i + 1]
                clean = window_vals[~np.isnan(window_vals)]
                if len(clean) > 1:
                    mu = np.mean(clean)
                    sigma = np.std(clean)
                    if sigma > 0 and not np.isnan(values[i]):
                        scores[i] = abs((values[i] - mu) / sigma)
        else:
            # Global Z-scores
            for i in range(n):
                if not np.isnan(values[i]):
                    scores[i] = abs((values[i] - self._mean) / self._std)

        is_anomaly = scores > self.threshold

        return AnomalyScore(
            scores=scores,
            is_anomaly=is_anomaly,
            threshold=self.threshold,
            detector_name=self.name,
        )


class IQRDetector(AnomalyDetector):
    """Interquartile Range anomaly detection.

    Uses the IQR method to define fences; points outside are anomalous.

    Args:
        factor: IQR multiplier (1.5 = standard, 3.0 = extreme).
        window: Optional sliding window size.
    """

    name = "iqr"

    def __init__(self, factor: float = 1.5, window: int | None = None):
        self.factor = factor
        self.window = window
        self._q1: float = 0.0
        self._q3: float = 0.0
        self._iqr: float = 1.0

    def fit(self, values: NDArray[np.float64]) -> None:
        clean = values[~np.isnan(values)]
        if len(clean) >= 4:
            self._q1 = float(np.percentile(clean, 25))
            self._q3 = float(np.percentile(clean, 75))
            self._iqr = self._q3 - self._q1
            if self._iqr == 0:
                self._iqr = 1.0

    def detect(self, values: NDArray[np.float64]) -> AnomalyScore:
        n = len(values)
        scores = np.zeros(n)

        lower = self._q1 - self.factor * self._iqr
        upper = self._q3 + self.factor * self._iqr

        for i in range(n):
            if np.isnan(values[i]):
                continue
            if values[i] < lower:
                scores[i] = (lower - values[i]) / self._iqr
            elif values[i] > upper:
                scores[i] = (values[i] - upper) / self._iqr

        is_anomaly = scores > 0

        return AnomalyScore(
            scores=scores,
            is_anomaly=is_anomaly,
            threshold=0.0,
            detector_name=self.name,
            metadata={"lower_fence": lower, "upper_fence": upper},
        )


class GESDDetector(AnomalyDetector):
    """Generalized Extreme Studentized Deviate (GESD) test.

    Iteratively identifies up to `max_anomalies` outliers using the
    generalized ESD test statistic.

    Args:
        max_anomalies: Maximum number of anomalies to detect.
        alpha: Significance level for the test.
    """

    name = "gesd"

    def __init__(self, max_anomalies: int = 10, alpha: float = 0.05):
        self.max_anomalies = max_anomalies
        self.alpha = alpha
        self._fitted_values: NDArray[np.float64] | None = None

    def fit(self, values: NDArray[np.float64]) -> None:
        self._fitted_values = values[~np.isnan(values)].copy()

    def detect(self, values: NDArray[np.float64]) -> AnomalyScore:
        n = len(values)
        scores = np.zeros(n)
        is_anomaly = np.zeros(n, dtype=bool)

        working = values.copy()
        valid_mask = ~np.isnan(working)

        max_iter = min(self.max_anomalies, int(np.sum(valid_mask)) // 2)

        for k in range(1, max_iter + 1):
            valid = working[valid_mask & ~np.isnan(working)]
            if len(valid) < 3:
                break

            mean = np.mean(valid)
            std = np.std(valid, ddof=1)
            if std == 0:
                break

            # Compute test statistics for all valid points
            test_stats = np.zeros(n)
            for i in range(n):
                if valid_mask[i] and not np.isnan(working[i]):
                    test_stats[i] = abs((working[i] - mean) / std)

            # Find the maximum test statistic
            max_idx = np.argmax(test_stats)
            r_k = test_stats[max_idx]

            # Critical value (using t-distribution)
            p = 1.0 - self.alpha / (2.0 * (len(valid) - k + 1))
            p = min(max(p, 0.5), 0.9999)
            t_val = sp_stats.t.ppf(p, df=max(1, len(valid) - k - 1))
            n_valid = len(valid)
            lambda_k = (
                (n_valid - k) * t_val
                / np.sqrt((n_valid - k - 1 + t_val**2) * (n_valid - k + 1))
            )

            scores[max_idx] = r_k

            if r_k > lambda_k:
                is_anomaly[max_idx] = True
                working[max_idx] = np.nan  # Remove for next iteration
            else:
                break

        return AnomalyScore(
            scores=scores,
            is_anomaly=is_anomaly,
            threshold=0.0,
            detector_name=self.name,
        )


class STLDetector(AnomalyDetector):
    """STL decomposition-based anomaly detection.

    Decomposes the series into trend + seasonal + residual, then detects
    anomalies in the residual component.

    Args:
        period: Seasonal period. Auto-detected if None.
        residual_threshold: Z-score threshold on residuals for anomaly flagging.
    """

    name = "stl"

    def __init__(self, period: int | None = None, residual_threshold: float = 3.0):
        self.period = period
        self.residual_threshold = residual_threshold
        self._residual_mean: float = 0.0
        self._residual_std: float = 1.0

    def fit(self, values: NDArray[np.float64]) -> None:
        from infrawatch.preprocess.decompose import seasonal_decompose
        result = seasonal_decompose(values, period=self.period)
        residual = result.residual[~np.isnan(result.residual)]
        if len(residual) > 0:
            self._residual_mean = float(np.mean(residual))
            self._residual_std = float(np.std(residual))
            if self._residual_std == 0:
                self._residual_std = 1.0
        if self.period is None:
            self.period = result.period

    def detect(self, values: NDArray[np.float64]) -> AnomalyScore:
        from infrawatch.preprocess.decompose import seasonal_decompose

        result = seasonal_decompose(values, period=self.period)
        residual = result.residual

        scores = np.zeros(len(values))
        valid = ~np.isnan(residual)
        scores[valid] = np.abs(
            (residual[valid] - self._residual_mean) / self._residual_std
        )

        is_anomaly = scores > self.residual_threshold

        return AnomalyScore(
            scores=scores,
            is_anomaly=is_anomaly,
            threshold=self.residual_threshold,
            detector_name=self.name,
            metadata={
                "period": self.period,
                "residual_mean": self._residual_mean,
                "residual_std": self._residual_std,
            },
        )
