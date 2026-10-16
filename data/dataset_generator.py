from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from fhe.fhe_setup import FHEContext, encrypt_vector
from fhe.metadata_collector import MetadataCollector
from fhe.operations import (
    encrypted_dot_product,
    encrypted_linear_regression_inference,
    encrypted_logistic_regression_approx,
    encrypted_mean,
    encrypted_variance,
)
from utils.config import ProjectConfig
from utils.logger import get_logger


@dataclass(slots=True)
class DatasetGenerator:
    config: ProjectConfig
    context: FHEContext
    logger: object = field(init=False, repr=False)
    rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.logger = get_logger("dataset_generator")
        self.rng = np.random.default_rng(self.config.random_seed)

    def generate(self, enable_defense: bool = False) -> pd.DataFrame:
        collector = MetadataCollector(
            enable_defense=enable_defense,
            defense_noise_scale=self.config.defense_noise_scale,
            baseline_noise_scale=self.config.baseline_noise_scale,
            workload_jitter_scale=self.config.workload_jitter_scale,
            random_seed=self.config.random_seed + int(enable_defense),
        )
        rows: list[dict[str, float | int | str]] = []

        for label in self.config.operation_labels:
            for _ in tqdm(
                range(self.config.samples_per_operation),
                desc=f"Generating {label}",
                leave=False,
            ):
                rows.append(self._generate_sample(label, collector))

        dataframe = pd.DataFrame(rows)
        dataframe = dataframe.rename(columns={"operation": "label"})
        return dataframe[self.config.metadata_columns(include_defense=True)]

    def save(self, dataframe: pd.DataFrame, path: Path | None = None) -> Path:
        target = path or self.config.dataset_path
        dataframe.to_csv(target, index=False)
        self.logger.info("Saved dataset with %s rows to %s", len(dataframe), target)
        return target

    def _random_vector(self, size: int | None = None) -> np.ndarray:
        return self.rng.normal(loc=0.0, scale=1.0, size=size or self.config.vector_size)

    def _generate_sample(
        self,
        label: str,
        collector: MetadataCollector,
    ) -> dict[str, float | int | str]:
        if label == "mean":
            raw = self._random_vector(self._sample_length(base=self.config.vector_size))
            vector = encrypt_vector(self.context, raw)
            return collector.collect(
                "mean",
                lambda: encrypted_mean(vector),
                workload_profile=self._workload_profile(raw, "mean"),
            )

        if label == "variance":
            raw = self._random_vector(self._sample_length(base=self.config.vector_size))
            vector = encrypt_vector(self.context, raw)
            return collector.collect(
                "variance",
                lambda: encrypted_variance(vector),
                workload_profile=self._workload_profile(raw, "variance"),
            )

        if label == "dot_product":
            length = self._sample_length(base=self.config.vector_size)
            left_raw = self._random_vector(length)
            right_raw = self._random_vector(length)
            left = encrypt_vector(self.context, left_raw)
            right = encrypt_vector(self.context, right_raw)
            return collector.collect(
                "dot_product",
                lambda: encrypted_dot_product(left, right),
                workload_profile=self._pair_workload_profile(left_raw, right_raw, "dot_product"),
            )

        if label == "linear_regression_inference":
            length = self._sample_length(base=max(self.config.regression_features, self.config.vector_size - 2))
            feature_raw = self._random_vector(length)
            weight_raw = self._random_vector(length)
            features = encrypt_vector(self.context, feature_raw)
            weights = encrypt_vector(self.context, weight_raw)
            bias = float(self.rng.normal(0.0, 0.5))
            return collector.collect(
                "linear_regression_inference",
                lambda: encrypted_linear_regression_inference(features, weights, bias),
                workload_profile=self._pair_workload_profile(
                    feature_raw,
                    weight_raw,
                    "linear_regression_inference",
                ),
            )

        if label == "logistic_regression_approx":
            length = self._sample_length(base=max(self.config.regression_features, self.config.vector_size - 2))
            feature_raw = self._random_vector(length)
            weight_raw = self._random_vector(length)
            features = encrypt_vector(self.context, feature_raw)
            weights = encrypt_vector(self.context, weight_raw)
            bias = float(self.rng.normal(0.0, 0.5))
            return collector.collect(
                "logistic_regression_approx",
                lambda: encrypted_logistic_regression_approx(features, weights, bias),
                workload_profile=self._pair_workload_profile(
                    feature_raw,
                    weight_raw,
                    "logistic_regression_approx",
                ),
            )

        raise ValueError(f"Unsupported label: {label}")

    def _sample_length(self, base: int) -> int:
        lower = max(3, base - max(2, base // 3))
        upper = max(lower + 1, base + max(2, base // 2))
        return int(self.rng.integers(lower, upper + 1))

    def _workload_profile(self, vector: np.ndarray, label: str) -> dict[str, float]:
        return {
            "length": float(vector.size),
            "amplitude": float(np.mean(np.abs(vector))),
            "variability": float(np.std(vector)),
            "sparsity": float(np.mean(np.isclose(vector, 0.0, atol=0.15))),
        }

    def _pair_workload_profile(
        self,
        left: np.ndarray,
        right: np.ndarray,
        label: str,
    ) -> dict[str, float]:
        merged = np.concatenate([left, right])
        profile = self._workload_profile(merged, label)
        profile["amplitude"] = float((np.mean(np.abs(left)) + np.mean(np.abs(right))) / 2.0)
        return profile

