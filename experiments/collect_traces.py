"""Collect real metadata traces from a client/server CKKS deployment (TenSEAL).

Every feature recorded here is measured, not simulated:
  latency     server wall-clock time: deserialize request -> compute -> serialize response
  resp_size   length in bytes of the serialized response ciphertext
  ops         number of homomorphic API calls the server issued (instrumented counter)
  depth       rescaling levels consumed by the circuit (instrumented counter)

The server only holds a public context (no secret key). Workload order is
randomly interleaved so that slow drift of the machine cannot correlate with
the class label. No label-dependent term is ever added to a feature.

Usage:
  python -m experiments.collect_traces --seeds 0 1 2 3 4 --per-class 200
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tenseal as ts

LABELS = [
    "mean",
    "variance",
    "dot_product",
    "linear_regression_inference",
    "logistic_regression_approx",
]
MAX_DEPTH = 3  # coefficient chain [60, 30, 30, 30, 60] leaves three rescaling levels
POLY_MODULUS_DEGREE = 8192
COEFF_MOD_BIT_SIZES = [60, 30, 30, 30, 60]
GLOBAL_SCALE = 2**30

# Degree-3 approximation of the sigmoid on [-5, 5] (Chen et al., 2018, ePrint 2018/462).
SIG_C0, SIG_C1, SIG_C3 = 0.5, 0.197, -0.004


@dataclass
class Tracked:
    """A server-side ciphertext together with the levels it has consumed."""

    ct: ts.CKKSVector
    depth: int = 0


class InstrumentedServer:
    """Evaluates workloads on ciphertexts and counts every homomorphic call."""

    def __init__(self, public_context: ts.Context, pad_depth: bool = False) -> None:
        self.ctx = public_context
        self.pad_depth = pad_depth
        self.ops = 0

    # --- primitives (each call increments the operation counter) -------------
    def mul_ct(self, a: Tracked, b: Tracked) -> Tracked:
        self.ops += 1
        return Tracked(a.ct * b.ct, max(a.depth, b.depth) + 1)

    def mul_plain(self, a: Tracked, plain) -> Tracked:
        self.ops += 1
        return Tracked(a.ct * plain, a.depth + 1)

    def add_ct(self, a: Tracked, b: Tracked) -> Tracked:
        self.ops += 1
        return Tracked(a.ct + b.ct, max(a.depth, b.depth))

    def sub_ct(self, a: Tracked, b: Tracked) -> Tracked:
        self.ops += 1
        return Tracked(a.ct - b.ct, max(a.depth, b.depth))

    def add_plain(self, a: Tracked, plain) -> Tracked:
        self.ops += 1
        return Tracked(a.ct + plain, a.depth)

    def total(self, a: Tracked) -> Tracked:
        # Rotate-and-add reduction inside SEAL; counted as one API call.
        self.ops += 1
        return Tracked(a.ct.sum(), a.depth)

    # --- workloads -------------------------------------------------------------
    def mean(self, x: Tracked, n: int) -> Tracked:
        return self.mul_plain(self.total(x), 1.0 / n)

    def variance(self, x: Tracked, n: int) -> Tracked:
        mu = self.mean(x, n)
        mean_sq = self.mul_plain(self.total(self.mul_ct(x, x)), 1.0 / n)
        return self.sub_ct(mean_sq, self.mul_ct(mu, mu))

    def dot_product(self, x: Tracked, y: Tracked) -> Tracked:
        return self.total(self.mul_ct(x, y))

    def linear_regression(self, x: Tracked, w: list[float], b: float) -> Tracked:
        # Model weights are the server's plaintext; only the features are encrypted.
        return self.add_plain(self.total(self.mul_plain(x, w)), b)

    def logistic_regression(self, x: Tracked, w: list[float], b: float) -> Tracked:
        z = self.linear_regression(x, w, b)
        z2 = self.mul_ct(z, z)
        cz = self.mul_plain(z, SIG_C3)
        z3 = self.mul_ct(cz, z2)
        lin = self.mul_plain(z, SIG_C1)
        return self.add_plain(self.add_ct(z3, lin), SIG_C0)

    def pad(self, r: Tracked) -> Tracked:
        """Depth-padding defense: spend unused levels multiplying by 1.0."""
        while r.depth < MAX_DEPTH:
            r = self.mul_plain(r, 1.0)
        return r

    def handle(self, label: str, request: list[bytes], n: int, w, b) -> tuple[bytes, Tracked]:
        self.ops = 0
        inputs = [Tracked(ts.ckks_vector_from(self.ctx, blob)) for blob in request]
        x = inputs[0]
        if label == "mean":
            r = self.mean(x, n)
        elif label == "variance":
            r = self.variance(x, n)
        elif label == "dot_product":
            r = self.dot_product(x, inputs[1])
        elif label == "linear_regression_inference":
            r = self.linear_regression(x, w, b)
        elif label == "logistic_regression_approx":
            r = self.logistic_regression(x, w, b)
        else:
            raise ValueError(label)
        if self.pad_depth:
            r = self.pad(r)
        return r.ct.serialize(), r


def reference(label: str, x: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> float:
    if label == "mean":
        return float(x.mean())
    if label == "variance":
        return float(x.var())
    if label == "dot_product":
        return float(x @ y)
    z = float(x @ w + b)
    if label == "linear_regression_inference":
        return z
    return SIG_C0 + SIG_C1 * z + SIG_C3 * z**3


def make_contexts() -> tuple[ts.Context, ts.Context]:
    client = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=POLY_MODULUS_DEGREE,
        coeff_mod_bit_sizes=COEFF_MOD_BIT_SIZES,
    )
    client.global_scale = GLOBAL_SCALE
    client.generate_galois_keys()
    client.generate_relin_keys()
    server = client.copy()
    server.make_context_public()
    return client, server


def collect(seed: int, per_class: int, pad_depth: bool, min_len: int, max_len: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    client, server_ctx = make_contexts()
    server = InstrumentedServer(server_ctx, pad_depth=pad_depth)

    schedule = np.repeat(np.arange(len(LABELS)), per_class)
    rng.shuffle(schedule)

    # Warm-up so that one-off allocation costs do not land on the first class drawn.
    for label in LABELS:
        blob = ts.ckks_vector(client, [0.1] * 16).serialize()
        server.handle(label, [blob, blob], 16, [0.1] * 16, 0.0)

    rows = []
    for i, k in enumerate(schedule):
        label = LABELS[k]
        n = int(rng.integers(min_len, max_len + 1))
        x = rng.normal(0.0, 1.0, n)
        y = rng.normal(0.0, 1.0, n)
        w = rng.normal(0.0, 1.0 / np.sqrt(n), n)
        b = float(rng.normal(0.0, 0.5))

        request = [ts.ckks_vector(client, x.tolist()).serialize()]
        if label == "dot_product":
            request.append(ts.ckks_vector(client, y.tolist()).serialize())

        t0 = time.perf_counter()
        response, tracked = server.handle(label, request, n, w.tolist(), b)
        latency = time.perf_counter() - t0

        decrypted = ts.ckks_vector_from(client, response).decrypt()[0]
        error = abs(decrypted - reference(label, x, y, w, b))

        rows.append(
            {
                "seed": seed,
                "index": i,
                "label": label,
                "latency_ms": latency * 1e3,
                "resp_size": len(response),
                "ops": server.ops,
                "depth": tracked.depth,
                "n": n,
                "abs_error": error,
                "pad_depth": int(pad_depth),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--per-class", type=int, default=200)
    parser.add_argument("--min-len", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--out", type=Path, default=Path("experiments/results/traces.csv"))
    parser.add_argument(
        "--no-padding-run",
        action="store_true",
        help="Only collect the undefended condition (skip the depth-padded sessions).",
    )
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    frames = []
    for pad in (False,) if args.no_padding_run else (False, True):
        for seed in args.seeds:
            started = time.time()
            df = collect(seed, args.per_class, pad, args.min_len, args.max_len)
            frames.append(df)
            print(
                f"pad_depth={pad} seed={seed}: {len(df)} traces in {time.time() - started:.0f}s, "
                f"max |error| = {df['abs_error'].max():.2e}",
                flush=True,
            )
            pd.concat(frames, ignore_index=True).to_csv(args.out, index=False)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
