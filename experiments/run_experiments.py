"""Evaluate the workload-identification attack and server-side defenses.

Protocol: leave-one-session-out. Each seed in traces.csv is an independent
session with fresh CKKS keys; the attacker trains on four sessions it profiled
itself and is tested on the held-out fifth. We report mean and standard
deviation over the five held-out sessions.

Usage:
  python -m experiments.run_experiments
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from experiments.collect_traces import LABELS

RESULTS = Path("experiments/results")
FIGURES = RESULTS / "figures"
SHORT = {
    "mean": "Mean",
    "variance": "Var",
    "dot_product": "Dot",
    "linear_regression_inference": "LinReg",
    "logistic_regression_approx": "LogReg",
}

# The observer sees absolute latency and can also normalize it by the median of the
# session it is watching (see with_relative_latency), so both are offered as features.
FEATURE_SETS = {
    "latency": ["latency_ms", "latency_rel"],
    "size": ["resp_size"],
    "remote (latency+size)": ["latency_ms", "latency_rel", "resp_size"],
    "profiling (all four)": ["latency_ms", "latency_rel", "resp_size", "ops", "depth"],
}


def classifiers(seed: int = 0) -> dict[str, object]:
    return {
        "Majority": DummyClassifier(strategy="most_frequent"),
        "RF": RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1),
        "SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10.0, gamma="scale")),
        "MLP": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=1000, random_state=seed),
        ),
    }


def with_relative_latency(df: pd.DataFrame) -> pd.DataFrame:
    """Divide latency by the median latency of its session.

    Absolute speed drifts between sessions (machine load, frequency scaling), but an observer
    sees many responses from the victim's session and can normalize by their median. The
    median is computed without labels, so no label information enters the feature.
    """
    out = df.copy()
    out["latency_rel"] = out["latency_ms"] / out.groupby("seed")["latency_ms"].transform("median")
    return out


def loso(df: pd.DataFrame, features: list[str], model) -> dict[str, object]:
    """Leave-one-session-out accuracy and macro-F1, plus pooled confusion matrix."""
    df = with_relative_latency(df)
    accs, f1s = [], []
    cm = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    for held_out in sorted(df["seed"].unique()):
        train, test = df[df["seed"] != held_out], df[df["seed"] == held_out]
        clf = clone(model).fit(train[features].to_numpy(), train["label"].to_numpy())
        pred = clf.predict(test[features].to_numpy())
        accs.append(accuracy_score(test["label"], pred))
        f1s.append(f1_score(test["label"], pred, average="macro"))
        cm += confusion_matrix(test["label"], pred, labels=LABELS)
    return {
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "confusion": cm.tolist(),
    }


def within_session(df: pd.DataFrame, features: list[str], model) -> dict[str, float]:
    """Stratified 5-fold CV inside each session: an attacker that calibrates on the victim's
    own machine and time window. Mean and std over the five sessions."""
    df = with_relative_latency(df)
    per_session = []
    for seed in sorted(df["seed"].unique()):
        part = df[df["seed"] == seed]
        x, y = part[features].to_numpy(), part["label"].to_numpy()
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        scores = [
            accuracy_score(y[te], clone(model).fit(x[tr], y[tr]).predict(x[te]))
            for tr, te in folds.split(x, y)
        ]
        per_session.append(np.mean(scores))
    return {"acc_mean": float(np.mean(per_session)), "acc_std": float(np.std(per_session))}


def add_delay(df: pd.DataFrame, max_delay_ms: float, seed: int = 7) -> pd.DataFrame:
    """Random-delay defense (equivalently, network jitter): U[0, max_delay] added to latency."""
    out = df.copy()
    rng = np.random.default_rng(seed)
    out["latency_ms"] = out["latency_ms"] + rng.uniform(0.0, max_delay_ms, len(out))
    return out


def constant_time(df: pd.DataFrame, quantile: float = 0.99) -> tuple[pd.DataFrame, float]:
    """Hold every response until a fixed deadline T (the given latency quantile)."""
    out = df.copy()
    deadline = float(np.quantile(out["latency_ms"], quantile))
    out["latency_ms"] = np.maximum(out["latency_ms"], deadline)
    return out, deadline


def byte_pad(df: pd.DataFrame) -> pd.DataFrame:
    """Pad every response to the largest observed size."""
    out = df.copy()
    out["resp_size"] = out["resp_size"].max()
    return out


def overhead(defended: pd.DataFrame, base: pd.DataFrame) -> dict[str, float]:
    return {
        "latency_ms": float(defended["latency_ms"].mean()),
        "latency_overhead_pct": float(
            100 * (defended["latency_ms"].mean() / base["latency_ms"].mean() - 1)
        ),
        "bytes": float(defended["resp_size"].mean()),
        "bandwidth_overhead_pct": float(
            100 * (defended["resp_size"].mean() / base["resp_size"].mean() - 1)
        ),
    }


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    traces = pd.read_csv(RESULTS / "traces.csv")
    base = traces[traces["pad_depth"] == 0].reset_index(drop=True)
    padded = traces[traces["pad_depth"] == 1].reset_index(drop=True)
    results: dict[str, object] = {}

    # ---- Dataset summary --------------------------------------------------
    summary = (
        base.groupby("label")[["latency_ms", "resp_size", "ops", "depth"]]
        .agg(["mean", "std"])
        .reindex(LABELS)
    )
    summary.to_csv(RESULTS / "trace_summary.csv")
    results["n_traces"] = {"undefended": len(base), "depth_padded": len(padded)}
    results["max_abs_error"] = {
        "undefended": float(base["abs_error"].max()),
        "depth_padded": float(padded["abs_error"].max()),
    }

    # ---- Attack: classifier x feature set, variable and fixed input length ----
    fixed_path = RESULTS / "traces_fixed_n.csv"
    fixed = pd.read_csv(fixed_path) if fixed_path.exists() else None
    scenarios = {"variable_n": base}
    if fixed is not None:
        scenarios["fixed_n"] = fixed
        results["n_traces"]["fixed_n"] = len(fixed)
        results["max_abs_error"]["fixed_n"] = float(fixed["abs_error"].max())
        fixed.groupby("label")[["latency_ms", "resp_size"]].agg(["mean", "std"]).reindex(
            LABELS
        ).to_csv(RESULTS / "trace_summary_fixed_n.csv")
    feature_sets = dict(FEATURE_SETS)
    attack = {}
    for scenario, frame in scenarios.items():
        attack[scenario] = {}
        for fs_name, cols in feature_sets.items():
            attack[scenario][fs_name] = {}
            for clf_name, model in classifiers().items():
                r = loso(frame, cols, model)
                entry = {k: v for k, v in r.items() if k != "confusion"}
                w = within_session(frame, cols, model)
                entry["within_acc_mean"], entry["within_acc_std"] = w["acc_mean"], w["acc_std"]
                attack[scenario][fs_name][clf_name] = entry
                if fs_name == "remote (latency+size)" and clf_name == "RF":
                    results[f"confusion_rf_remote_{scenario}"] = r["confusion"]
                print(f"[attack] {scenario:10s} {fs_name:32s} {clf_name:9s} "
                      f"cross={r['acc_mean']:.3f}+-{r['acc_std']:.3f} within={w['acc_mean']:.3f}")
    results["attack"] = attack

    # ---- Network jitter / random delay sweep ------------------------------
    sweep_values = [0, 50, 100, 200, 400, 800]
    rf = classifiers()["RF"]
    curves = {"variable n, latency+size": (base, "remote (latency+size)"),
              "depth-padded, latency+size": (padded, "remote (latency+size)")}
    if fixed is not None:
        curves = {"fixed n, latency only": (fixed, "latency"),
                  "fixed n, latency+size": (fixed, "remote (latency+size)"), **curves}
    sweep = {}
    for name, (frame, fs_name) in curves.items():
        sweep[name] = []
        for d in sweep_values:
            r = loso(add_delay(frame, d), FEATURE_SETS[fs_name], rf)
            sweep[name].append({"max_delay_ms": d, "acc_mean": r["acc_mean"], "acc_std": r["acc_std"]})
            print(f"[jitter] {name:28s} delay<={d:4d}ms acc={r['acc_mean']:.3f}")
    results["delay_sweep"] = sweep

    # ---- Defenses ---------------------------------------------------------
    remote = FEATURE_SETS["remote (latency+size)"]
    ct_base, t_base = constant_time(base)
    ct_padded, t_padded = constant_time(padded)
    defenses = {
        "None": (base, fixed),
        "Random delay U[0,200ms]": (add_delay(base, 200), None if fixed is None else add_delay(fixed, 200)),
        "Byte padding": (byte_pad(base), None if fixed is None else byte_pad(fixed)),
        "Depth padding": (padded, None),
        "Byte padding + constant time": (
            constant_time(byte_pad(base))[0],
            None if fixed is None else constant_time(byte_pad(fixed))[0],
        ),
        "Depth padding + constant time": (ct_padded, None),
    }
    defense_results = {}
    for name, (frame, fixed_frame) in defenses.items():
        r_rf = loso(frame, remote, rf)
        r_svm = loso(frame, remote, classifiers()["SVM"])
        entry = {
            "rf_acc_mean": r_rf["acc_mean"],
            "rf_acc_std": r_rf["acc_std"],
            "svm_acc_mean": r_svm["acc_mean"],
            "svm_acc_std": r_svm["acc_std"],
            **overhead(frame, base),
        }
        if fixed_frame is not None:
            r_fixed = loso(fixed_frame, remote, rf)
            entry["fixed_n_rf_acc_mean"] = r_fixed["acc_mean"]
            entry["fixed_n_rf_acc_std"] = r_fixed["acc_std"]
            w_fixed = within_session(fixed_frame, remote, rf)
            entry["fixed_n_within_rf_acc_mean"] = w_fixed["acc_mean"]
            entry["fixed_n_within_rf_acc_std"] = w_fixed["acc_std"]
        defense_results[name] = entry
        if name == "Depth padding + constant time":
            results["confusion_rf_full_defense"] = r_rf["confusion"]
        print(f"[defense] {name:32s} RF={r_rf['acc_mean']:.3f} SVM={r_svm['acc_mean']:.3f} "
              f"fixed-n RF={entry.get('fixed_n_rf_acc_mean', float('nan')):.3f} "
              f"fixed-n within={entry.get('fixed_n_within_rf_acc_mean', float('nan')):.3f}")
    results["defenses"] = defense_results
    results["constant_time_deadline_ms"] = {"undefended": t_base, "depth_padded": t_padded}
    results["latency_vs_length_corr"] = float(base[["latency_ms", "n"]].corr().iloc[0, 1])
    session_medians = base.groupby("seed")["latency_ms"].median()
    results["session_median_latency_ms"] = {"min": float(session_medians.min()), "max": float(session_medians.max())}

    with open(RESULTS / "results.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    make_figures(base, fixed, results)
    print("Saved", RESULTS / "results.json")


def make_figures(base: pd.DataFrame, fixed: pd.DataFrame | None, results: dict) -> None:
    plt.rcParams.update({"font.size": 9, "font.family": "serif"})
    markers = ["o", "s", "^", "D", "v"]

    # Fig. A: latency vs response size for one session, variable and fixed input length.
    frames = [("(a) Input length 16-1024", base)]
    if fixed is not None:
        frames.append(("(b) Input length fixed at 256", fixed))
    fig, axes = plt.subplots(1, len(frames), figsize=(2.45 * len(frames) + 0.2, 2.5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (title, frame) in zip(axes, frames):
        session = frame[frame["seed"] == frame["seed"].min()]
        for label, marker in zip(LABELS, markers):
            part = session[session["label"] == label]
            ax.scatter(part["latency_ms"], part["resp_size"] / 1024, s=6, marker=marker,
                       alpha=0.5, label=SHORT[label], linewidths=0)
        ax.set_xlabel("Server latency (ms)", fontsize=8)
        ax.set_title(title, fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("Response size (KiB)", fontsize=8)
    axes[-1].legend(fontsize=6, loc="upper right", frameon=False, markerscale=1.5)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_scatter.png", dpi=300)
    plt.close(fig)

    # Fig. B: accuracy vs added delay.
    fig, ax = plt.subplots(figsize=(4.6, 2.5))
    styles = {
        "fixed n, latency only": "o-",
        "fixed n, latency+size": "s-",
        "variable n, latency+size": "D-",
        "depth-padded, latency+size": "^--",
    }
    names = {
        "fixed n, latency only": "Fixed n: latency",
        "fixed n, latency+size": "Fixed n: latency + size",
        "variable n, latency+size": "Variable n: latency + size",
        "depth-padded, latency+size": "Variable n, depth-padded",
    }
    for key, points in results["delay_sweep"].items():
        x = [p["max_delay_ms"] for p in points]
        y = [p["acc_mean"] for p in points]
        e = [p["acc_std"] for p in points]
        ax.errorbar(x, y, yerr=e, fmt=styles[key], ms=3.5, capsize=2, lw=1, label=names[key])
    ax.axhline(0.2, color="grey", lw=0.8, ls=":")
    ax.text(x[-1], 0.13, "chance", ha="right", fontsize=7, color="grey")
    ax.set_xlabel("Maximum random delay / jitter (ms)", fontsize=8)
    ax.set_ylabel("RF accuracy", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=6.5, frameon=False, loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_delay_sweep.png", dpi=300)
    plt.close(fig)

    # Fig. C: confusion matrices without defense, variable vs fixed input length.
    panels = [("confusion_rf_remote_variable_n", "(a) Input length 16-1024")]
    if "confusion_rf_remote_fixed_n" in results:
        panels.append(("confusion_rf_remote_fixed_n", "(b) Input length fixed at 256"))
    fig, axes = plt.subplots(1, len(panels), figsize=(2.45 * len(panels) + 0.2, 2.45))
    axes = np.atleast_1d(axes)
    for ax, (key, title) in zip(axes, panels):
        cm = np.asarray(results[key], dtype=float)
        cm = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(cm, cmap="Greys", vmin=0, vmax=1)
        for i in range(len(LABELS)):
            for j in range(len(LABELS)):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if cm[i, j] > 0.5 else "black")
        ticks = [SHORT[l] for l in LABELS]
        ax.set_xticks(range(5), ticks, rotation=45, fontsize=6)
        ax.set_yticks(range(5), ticks, fontsize=6)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("Predicted", fontsize=7)
    axes[0].set_ylabel("True", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_confusion.png", dpi=300)
    plt.close(fig)

if __name__ == "__main__":
    main()
