from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from fhe.fhe_setup import EncryptedTensor


@dataclass(slots=True)
class MetadataCollector:
    enable_defense: bool = False
    defense_noise_scale: float = 0.0
    baseline_noise_scale: float = 0.04
    workload_jitter_scale: float = 0.12
    random_seed: int = 42
    rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.random_seed)

    def collect(
        self,
        operation_name: str,
        operation_fn: Callable[[], EncryptedTensor],
        workload_profile: dict[str, float] | None = None,
    ) -> dict[str, float | int | str]:
        start = time.perf_counter()
        result = operation_fn()
        execution_time = time.perf_counter() - start
        profile = workload_profile or {}

        metadata = {
            "operation": operation_name,
            "time": float(execution_time),
            "size": float(result.size_hint),
            "ops": int(result.operation_count),
            "depth": int(result.depth),
            "noise": float(result.noise_hint),
        }
        metadata = self._apply_runtime_variability(metadata, profile)

        if self.enable_defense:
            metadata = self._inject_noise(metadata)
            metadata["defense"] = 1
        else:
            metadata["defense"] = 0
        return metadata

    def _apply_runtime_variability(
        self,
        metadata: dict[str, float | int | str],
        workload_profile: dict[str, float],
    ) -> dict[str, float | int | str]:
        noisy = metadata.copy()
        length = workload_profile.get("length", 1.0)
        amplitude = workload_profile.get("amplitude", 1.0)
        variability = workload_profile.get("variability", 1.0)
        sparsity = workload_profile.get("sparsity", 0.0)

        scale_noise = self.rng.normal(0.0, self.baseline_noise_scale)
        system_jitter = self.rng.normal(0.0, self.workload_jitter_scale)
        load_factor = 1.0 + 0.05 * np.log1p(length) + 0.04 * amplitude + 0.03 * variability - 0.02 * sparsity

        noisy["time"] = max(
            1e-6,
            float(noisy["time"]) * load_factor * (1.0 + scale_noise + 0.5 * system_jitter),
        )
        noisy["size"] = max(
            32.0,
            float(noisy["size"]) * (1.0 + 0.05 * system_jitter) + length * self.rng.uniform(4.0, 12.0),
        )
        noisy["ops"] = max(
            1,
            int(
                round(
                    float(noisy["ops"])
                    + length * self.rng.uniform(0.15, 0.45)
                    + variability * self.rng.uniform(0.2, 1.2)
                    + self.rng.normal(0.0, 1.4)
                )
            ),
        )
        noisy["depth"] = max(
            1,
            int(round(float(noisy["depth"]) + self.rng.normal(0.0, 0.7))),
        )
        noisy["noise"] = max(
            0.5,
            float(noisy["noise"])
            - (amplitude * self.rng.uniform(0.2, 1.0))
            - (variability * self.rng.uniform(0.1, 0.8))
            + self.rng.normal(0.0, self.baseline_noise_scale * 18.0),
        )
        noisy["size"] = float(16 * round(float(noisy["size"]) / 16))
        noisy["ops"] = max(1, int(2 * round(int(noisy["ops"]) / 2)))
        noisy["noise"] = round(float(noisy["noise"]), 2)
        return noisy

    def _inject_noise(self, metadata: dict[str, float | int | str]) -> dict[str, float | int | str]:
        noisy = metadata.copy()
        for key in ("time", "size", "ops", "depth", "noise"):
            value = float(noisy[key])
            perturbation = self.rng.normal(0.0, self.defense_noise_scale)
            scaled = value * (1.0 + perturbation)
            if key in {"ops", "depth"}:
                noisy[key] = max(1, int(round(scaled)))
            elif key == "noise":
                noisy[key] = max(0.5, scaled)
            else:
                noisy[key] = max(1e-6, scaled)
        return noisy
