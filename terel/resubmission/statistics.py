import numpy as np


def summarize_values(values, *, bootstrap_samples: int = 10_000, seed: int = 260803):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("at least two scalar values are required")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    rng = np.random.default_rng(seed)
    resample_indices = rng.integers(0, len(values), size=(bootstrap_samples, len(values)))
    bootstrap_means = values[resample_indices].mean(axis=1)
    low, high = np.quantile(bootstrap_means, [0.025, 0.975])
    return {
        "raw": values.tolist(),
        "mean": float(values.mean()),
        "sample_sd": float(values.std(ddof=1)),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_seed": int(seed),
    }


def paired_contrast(treatment_by_seed, control_by_seed, *, bootstrap_samples=10_000, seed=260803):
    treatment_seeds = set(treatment_by_seed)
    control_seeds = set(control_by_seed)
    if treatment_seeds != control_seeds:
        raise ValueError("paired contrasts require identical seed sets")
    seeds = sorted(treatment_seeds)
    differences = np.asarray(
        [treatment_by_seed[item] - control_by_seed[item] for item in seeds],
        dtype=np.float64,
    )
    summary = summarize_values(differences, bootstrap_samples=bootstrap_samples, seed=seed)
    return {
        "seeds": seeds,
        "raw_differences": summary.pop("raw"),
        "mean_difference": summary.pop("mean"),
        **summary,
    }
