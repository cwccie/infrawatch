"""Preprocessing pipeline that chains multiple transformations.

Provides a configurable pipeline that applies counter unwrapping, gap filling,
outlier handling, resampling, and normalization in the correct order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from infrawatch.preprocess.counters import unwrap_counters, is_counter_metric
from infrawatch.preprocess.gaps import fill_gaps
from infrawatch.preprocess.outliers import replace_outliers
from infrawatch.preprocess.resample import resample_uniform
from infrawatch.preprocess.normalize import z_normalize


@dataclass
class PreprocessConfig:
    """Configuration for the preprocessing pipeline.

    Attributes:
        unwrap_counters: Whether to auto-detect and unwrap counters.
        counter_bits: Counter bit width (32 or 64).
        fill_gaps: Whether to fill NaN gaps.
        gap_method: Gap-filling interpolation method.
        max_gap: Maximum gap duration to fill.
        handle_outliers: Whether to detect and replace outliers.
        outlier_method: Outlier detection method.
        outlier_replacement: Outlier replacement strategy.
        resample: Whether to resample to uniform intervals.
        resample_interval: Target resampling interval.
        normalize: Whether to normalize values.
        normalize_method: Normalization method ('zscore', 'minmax', 'robust').
    """

    unwrap_counters: bool = True
    counter_bits: int = 64
    fill_gaps: bool = True
    gap_method: str = "linear"
    max_gap: Optional[float] = None
    handle_outliers: bool = True
    outlier_method: str = "iqr"
    outlier_replacement: str = "median"
    resample: bool = False
    resample_interval: Optional[float] = None
    normalize: bool = False
    normalize_method: str = "zscore"


@dataclass
class PreprocessResult:
    """Result of preprocessing a time series."""

    timestamps: NDArray[np.float64]
    values: NDArray[np.float64]
    steps_applied: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class PreprocessPipeline:
    """Configurable preprocessing pipeline for infrastructure metrics.

    Args:
        config: Pipeline configuration. Uses sensible defaults if not provided.
    """

    def __init__(self, config: Optional[PreprocessConfig] = None):
        self.config = config or PreprocessConfig()

    def process(
        self,
        timestamps: NDArray[np.float64],
        values: NDArray[np.float64],
        metric_name: str = "",
    ) -> PreprocessResult:
        """Run the full preprocessing pipeline on a time series.

        Args:
            timestamps: Array of timestamps (sorted ascending).
            values: Array of metric values.
            metric_name: Optional metric name for counter detection heuristics.

        Returns:
            PreprocessResult with cleaned data and metadata.
        """
        result = PreprocessResult(
            timestamps=timestamps.copy(),
            values=values.copy().astype(np.float64),
        )

        cfg = self.config

        # Step 1: Counter unwrapping
        if cfg.unwrap_counters and metric_name and is_counter_metric(metric_name):
            result.values = unwrap_counters(result.values, bits=cfg.counter_bits)
            result.steps_applied.append("unwrap_counters")

        # Step 2: Resampling (before gap fill, as it creates uniform grid)
        if cfg.resample and cfg.resample_interval:
            result.timestamps, result.values = resample_uniform(
                result.timestamps,
                result.values,
                interval=cfg.resample_interval,
            )
            result.steps_applied.append("resample")

        # Step 3: Gap filling
        if cfg.fill_gaps:
            result.timestamps, result.values = fill_gaps(
                result.timestamps,
                result.values,
                method=cfg.gap_method,
                max_gap=cfg.max_gap,
            )
            result.steps_applied.append("fill_gaps")

        # Step 4: Outlier handling
        if cfg.handle_outliers:
            result.values = replace_outliers(
                result.values,
                method=cfg.outlier_method,
                replacement=cfg.outlier_replacement,
            )
            result.steps_applied.append("handle_outliers")

        # Step 5: Normalization
        if cfg.normalize:
            if cfg.normalize_method == "zscore":
                result.values, mean, std = z_normalize(result.values)
                result.metadata["normalize_mean"] = mean
                result.metadata["normalize_std"] = std
            result.steps_applied.append("normalize")

        return result
