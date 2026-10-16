# Adversarial Machine Learning against Homomorphic Encryption using Metadata Leakage

This project is a research prototype that studies whether an attacker can infer which encrypted computation is running without learning plaintext or breaking the encryption scheme. The attack model only uses metadata side-channels such as runtime, ciphertext-size proxies, operation counts, multiplicative depth, and a noise-budget-like signal.

## What is FHE?

Fully Homomorphic Encryption (FHE) allows computations directly on ciphertexts. A service can evaluate arithmetic on encrypted inputs and return encrypted outputs that only the data owner decrypts later. This prototype prefers the CKKS scheme via TenSEAL, which is a Python wrapper over Microsoft SEAL. If TenSEAL is not installed, the code falls back to a mock encrypted backend with the same interface so the metadata-leakage workflow remains runnable.

## What is Metadata Leakage?

Even when ciphertext contents stay hidden, an attacker may still observe side-channel metadata:

- execution time
- ciphertext size or serialization length
- number of homomorphic operations
- multiplicative depth
- remaining noise budget, or a simulated proxy

Those metadata traces can reveal patterns about the underlying encrypted computation.

## Attack Intuition

Different encrypted programs leave different system footprints:

- `mean` is mostly additions and one scaling step
- `variance` adds subtraction and ciphertext multiplication
- `dot_product` combines pairwise multiplication and reduction
- `linear_regression_inference` builds on dot product plus a bias term
- `logistic_regression_approx` adds deeper polynomial evaluation for sigmoid approximation

An ML classifier can learn to map those metadata patterns to operation labels.

## Installation

Use Python 3.10+ and install the core dependencies:

```bash
pip install numpy pandas scikit-learn torch matplotlib tqdm joblib
```

Optional real FHE backend:

```bash
pip install tenseal
```

## How to Run

From the project root:

```bash
python main.py
```

For an interactive dashboard and simulation UI:

```bash
python app.py
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

For faster smoke tests, you can temporarily reduce the workload size with environment variables:

```bash
$env:AML_FHE_SAMPLES_PER_OPERATION=50
$env:AML_FHE_TORCH_EPOCHS=10
python main.py
```

You can also force the mock backend if you want a quick functional run without TenSEAL latency:

```bash
$env:AML_FHE_BACKEND_PREFERENCE="mock"
python main.py
```

The Flask dashboard exposes the same knobs directly in the browser, so you can:

- choose `mock` for a fast classroom demo
- choose `tenseal` for a real FHE-backed simulation
- reduce samples and epochs for live presentations
- increase them for a fuller research run

The pipeline will:

1. initialize the FHE backend
2. generate balanced metadata datasets with and without defense
3. train sklearn and PyTorch models
4. compute classification metrics
5. save models, plots, logs, and the dataset

## Paper Experiments (MIDI 2026)

The results reported in the paper come from `experiments/`, not from the dashboard pipeline.
That code runs a real client/server CKKS deployment and measures every feature; nothing is simulated.

```bash
python -m experiments.collect_traces --seeds 0 1 2 3 4 --per-class 200
python -m experiments.run_experiments
```

- `collect_traces.py`: the server holds only a public context; it records server latency (deserialize, compute, serialize), the response size in bytes, and instrumented counts of homomorphic calls and rescaling levels. Every result is decrypted and checked against a plaintext reference. Workloads run in random interleaved order, five key sessions, with and without depth padding.
- `run_experiments.py`: leave-one-session-out evaluation of Majority/RF/SVM/MLP attackers on four feature sets, a random-delay sweep, and four server-side countermeasures with their latency and bandwidth overheads. Writes `experiments/results/results.json` and the figures in `experiments/results/figures/`.

The dashboard (`main.py`, `app.py`) remains a teaching demo. Its `size`, `ops`, `depth` and `noise` features are bookkeeping estimates with added jitter, so do not use them as measurements.

## Outputs

- `data/dataset.csv`
- `artifacts/models/`
- `artifacts/metrics/metrics_baseline.json`
- `artifacts/metrics/metrics_defended.json`
- `artifacts/metrics/accuracy_comparison.json`
- `artifacts/plots/feature_importance_baseline.png`
- `artifacts/plots/confusion_matrix_rf_baseline.png`
- `artifacts/plots/confusion_matrix_torch_baseline.png`
- `artifacts/plots/accuracy_comparison.png`
- `artifacts/pipeline.log`

## UI Experience

The dashboard explains the project as a complete attack pipeline:

- what FHE is and why CKKS is used
- how metadata leakage appears without decrypting anything
- how the dataset is constructed from side-channel features
- how baseline and defended runs compare
- how to interpret feature importance and confusion matrices

## Defense Mode

The project includes a simple metadata defense that injects noise into collected features before model training. This simulates a system-level countermeasure and lets you compare attack accuracy before and after defensive perturbation.

## Expected Results

Because the operations are designed to have distinct metadata signatures, the Random Forest model should typically exceed 70% accuracy on the baseline dataset. After the defense is applied, accuracy should decrease, showing reduced leakage.

## Security Note

This project does not attack the cryptographic hardness of FHE and does not recover plaintext. It only studies metadata side-channels on synthetic workloads, which is appropriate for a systems-security research prototype.
