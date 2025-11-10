import numpy as np
from typing import Optional, Dict, Any

class EmtpyRepresentationMetricsTracker:
    def compute_metrics(self):
        pass

    def update(self, x):
        pass

    def scalar_metrics(self):
        return {}


class RepresentationMetricsTracker:
    """
    Incremental tracker for per-neuron failure modes.

    Usage:
        tracker = NeuronFailureTracker(num_neurons=D, num_classes=10)
        tracker.update(batch_acts, labels)   # batch_acts: (N,D) numpy or torch tensor
        m = tracker.compute_metrics()
        # m contains arrays keyed by 'mean','std','prop_near_zero','dead','scale',
        # 'separability','high_scale_low_sep'
    """
    def __init__(
        self,
        num_neurons: int,
        num_classes: Optional[int] = None,
        min_count_for_detection: int = 100,
    ):
        self.D = int(num_neurons)
        self.num_classes = int(num_classes) if num_classes is not None else None
        self.min_count_for_detection = int(min_count_for_detection)

        # global stats
        self.count = 0
        self.sum = np.zeros(self.D, dtype=np.float64)
        self.sumsq = np.zeros(self.D, dtype=np.float64)

        # per-class stats (lazy init if num_classes None)
        if self.num_classes is not None:
            self._init_class_stats(self.num_classes)
        else:
            self.class_counts = None
            self.class_sum = None
            self.class_sumsq = None

    def _init_class_stats(self, K: int):
        self.num_classes = int(K)
        self.class_counts = np.zeros((self.num_classes,), dtype=np.int64)
        self.class_sum = np.zeros((self.num_classes, self.D), dtype=np.float64)
        self.class_sumsq = np.zeros((self.num_classes, self.D), dtype=np.float64)

    def update(self, batch_activations, labels: Optional[np.ndarray] = None):
        """
        batch_activations: (N, D) np.ndarray or torch tensor
        labels: optional (N,) integer array (0..K-1)
        """
        if hasattr(batch_activations, "detach"):
            batch_activations = batch_activations.detach().cpu().numpy()
        else:
            batch_activations = np.asarray(batch_activations)

        if batch_activations.ndim != 2 or batch_activations.shape[1] != self.D:
            raise ValueError(f"batch_activations must be shape (N, {self.D})")

        N = batch_activations.shape[0]
        self.count += N
        self.sum += batch_activations.sum(axis=0)
        self.sumsq += (batch_activations ** 2).sum(axis=0)

        if labels is not None:
            labels = np.asarray(labels).astype(np.int64)
            if labels.shape[0] != N:
                raise ValueError("labels length must match batch size")
            max_label = int(labels.max())
            if self.num_classes is None:
                self._init_class_stats(max_label + 1)
            elif max_label >= self.num_classes:
                # grow arrays if new labels appear (robustness)
                oldK = self.num_classes
                newK = max_label + 1
                cs = np.zeros((newK, self.D), dtype=np.float64)
                css = np.zeros((newK, self.D), dtype=np.float64)
                cc = np.zeros((newK,), dtype=np.int64)
                cs[:oldK] = self.class_sum
                css[:oldK] = self.class_sumsq
                cc[:oldK] = self.class_counts
                self.class_sum = cs
                self.class_sumsq = css
                self.class_counts = cc
                self.num_classes = newK

            # aggregate per-class
            for c in np.unique(labels):
                mask = (labels == c)
                vals = batch_activations[mask]
                cnt = vals.shape[0]
                if cnt == 0:
                    continue
                self.class_counts[c] += cnt
                self.class_sum[c] += vals.sum(axis=0)
                self.class_sumsq[c] += (vals ** 2).sum(axis=0)

    def compute_metrics(
        self,
        zero_abs_thresh: float = 1e-2,
        prop_zero_thresh: float = 0.9,
        mean_abs_thresh: float = 1e-2,
        std_thresh: float = 1e-2,
        scale_threshold: float = 10.0,
        separability_threshold: float = 0.4,
        eps: float = 1e-4,
    ) -> Dict[str, Any]:
        """
        Returns a dict with keys:
          - mean (D,), std (D,), prop_near_zero (D,)
          - dead (D,) boolean
          - scale (D,) == mean_abs or std (choose representation)
          - separability (D,) (between/within ratio) or np.nan if no labels
          - high_scale_low_sep (D,) boolean
        """
        if self.count == 0:
            print("Warning in representation metric tracker compute_metrics: No data; call update() before compute_metrics()")
            return {}

        mean = self.sum / self.count
        var = (self.sumsq / self.count) - (mean ** 2)
        var = np.clip(var, a_min=0.0, a_max=None)
        std = np.sqrt(var)

        # approximate proportion near-zero using running moments is hard; we can't reconstruct exact prop.
        # So we store an empirical proxy: if mean and std small -> high prop near zero.
        prop_near_zero = np.exp(- (np.abs(mean) / (std + eps)))  # heuristic in [0,1]; higher => more near-zero
        # normalize to [0,1]
        prop_near_zero = np.clip((prop_near_zero - prop_near_zero.min()) / (np.ptp(prop_near_zero) + eps), 0, 1)

        dead = np.zeros(self.D, dtype=bool)
        # detect dead when we have enough data and both mean and std are tiny OR proxy very high
        has_enough = (self.count >= self.min_count_for_detection)
        dead_condition = (np.abs(mean) < mean_abs_thresh) & (std < std_thresh)
        dead = np.where(has_enough, dead_condition | (prop_near_zero > prop_zero_thresh), dead_condition)

        separability = np.full(self.D, np.nan, dtype=np.float64)
        if self.class_counts is not None and self.class_counts.sum() > 0:
            N_total = self.class_counts.sum()
            # class means and variances
            with np.errstate(divide='ignore', invalid='ignore'):
                class_means = (self.class_sum.T / np.where(self.class_counts > 0, self.class_counts, 1)).T  # (K,D)
                class_vars = (self.class_sumsq.T / np.where(self.class_counts > 0, self.class_counts, 1)).T - class_means ** 2
                class_vars = np.clip(class_vars, 0.0, None)

            # between-class variance (weighted)
            wm = self.class_counts[:, None] * (class_means - mean[None, :]) ** 2
            between = wm.sum(axis=0) / (N_total + eps)

            # within-class variance (pooled)
            # sum_c n_c * var_c / N_total
            within = (self.class_counts[:, None] * class_vars).sum(axis=0) / (N_total + eps)
            separability = between / (within + eps)

        scale = np.abs(mean)  # primary scale metric (could also use std)

        high_scale_low_sep = np.zeros(self.D, dtype=bool)
        if np.any(~np.isnan(separability)):
            # mark neurons with large scale but low separability
            high_scale_low_sep = (scale > scale_threshold) & (np.nan_to_num(separability, nan=0.0) < separability_threshold)

        return {
            "count": int(self.count),
            "mean": mean,
            "std": std,
            "prop_near_zero_proxy": prop_near_zero,
            "dead": dead,
            "scale": scale,
            "separability": separability,
            "high_scale_low_sep": high_scale_low_sep,
        }

    def summary(self, **kwargs) -> Dict[str, Any]:
        m = self.compute_metrics(**kwargs)
        n_dead = int(m["dead"].sum())
        n_hsls = int(m["high_scale_low_sep"].sum())
        return {
            "total_counted_examples": m["count"],
            "n_dead_neurons": n_dead,
            "n_high_scale_low_sep_neurons": n_hsls,
        }
    
    def scalar_metrics(self, separability_threshold: float = 0.1) -> Dict[str, float]:
        """
        Compute scalar prevalence / summary stats from NeuronFailureTracker.compute_metrics() output.

        Parameters
        ----------
        metrics : dict
            Output from compute_metrics(), expected keys include:
            - "dead" : boolean array (D,)
            - "high_scale_low_sep" : boolean array (D,)
            - "separability" : float array (D,) possibly containing np.nan
            - "scale" : float array (D,)
            - "std" : float array (D,)
            - "mean" : float array (D,)
            - "count" : int total sample count used
        separability_threshold : float
            additional threshold used to report fraction of neurons below this separability.

        Returns
        -------
        summary : dict
            Scalars summarizing prevalence:
            - num_neurons, counted_examples
            - n_dead, frac_dead
            - n_high_scale_low_sep, frac_high_scale_low_sep
            - frac_low_separability (separability < separability_threshold)
            - mean_separability, median_separability, std_separability (nan-aware)
            - mean_scale, median_scale
            - mean_std, median_std
        """
        metrics = self.compute_metrics()

        dead = np.asarray(metrics.get("dead"))
        hsls = np.asarray(metrics.get("high_scale_low_sep"))
        separability = np.asarray(metrics.get("separability"))
        scale = np.asarray(metrics.get("scale"))
        std = np.asarray(metrics.get("std"))
        mean = np.asarray(metrics.get("mean"))
        total_counted = int(metrics.get("count", 0))

        D = dead.size if dead is not None and dead.size else separability.size if separability.size else scale.size

        n_dead = int(np.sum(dead))
        n_hsls = int(np.sum(hsls))

        frac_dead = float(n_dead / D) if D else 0.0
        frac_hsls = float(n_hsls / D) if D else 0.0

        # separability stats (ignore NaNs)
        mean_separability = float(np.nanmean(separability)) if separability.size else float("nan")
        median_separability = float(np.nanmedian(separability)) if separability.size else float("nan")
        std_separability = float(np.nanstd(separability)) if separability.size else float("nan")

        n_low_sep = int(np.sum(np.nan_to_num(separability, nan=np.inf) < separability_threshold))
        frac_low_sep = float(n_low_sep / D) if D else 0.0

        mean_scale = float(np.nanmean(scale)) if scale.size else float("nan")
        median_scale = float(np.nanmedian(scale)) if scale.size else float("nan")

        mean_std = float(np.nanmean(std)) if std.size else float("nan")
        median_std = float(np.nanmedian(std)) if std.size else float("nan")

        summary = {
            # "num_neurons": int(D),
            # "counted_examples": total_counted,
            # "n_dead": n_dead,
            "frac_dead": frac_dead,
            # "n_high_scale_low_sep": n_hsls,
            "frac_high_scale_low_sep": frac_hsls,
            # "n_low_separability": n_low_sep,
            "frac_low_separability": frac_low_sep,
            "mean_separability": mean_separability,
            # "median_separability": median_separability,
            "std_separability": std_separability,
            "mean_scale": mean_scale,
            # "median_scale": median_scale,
            "mean_std": mean_std,
            # "median_std": median_std,
        }
        return summary

