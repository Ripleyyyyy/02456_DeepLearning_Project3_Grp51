import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors


def evaluate_cp(cal_scores_use, val_outputs_use, val_labels_use, alpha=0.1, desc=""):
    """
    Given calibration scores (1D), and validation outputs/labels,
    compute qhat, coverage, efficiency, singleton coverage.
    """
    n = len(cal_scores_use)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    # For newer NumPy, use method="higher" instead of interpolation
    qhat = np.quantile(cal_scores_use, q_level, method="higher")

    pred_sets = val_outputs_use >= (1.0 - qhat)

    coverage = pred_sets[np.arange(len(val_labels_use)), val_labels_use].mean()
    eff = pred_sets.sum(axis=-1).mean()
    singleton_idx = pred_sets.sum(axis=-1) == 1
    if singleton_idx.any():
        singleton_cov = pred_sets[
            np.arange(len(val_labels_use)), val_labels_use
        ][singleton_idx].mean()
    else:
        singleton_cov = float("nan")

    print(
        f"{desc}Coverage: {coverage*100:.2f}%, "
        f"efficiency: {eff:.2f}, "
        f"singleton cov: {singleton_cov*100:.2f}%"
    )
    return qhat, coverage, eff, singleton_cov


def main():
    folder = "scratch/rhti/conformal"
    alpha = 0.1  # 90% prediction sets

    # 1) Load prediction results (same as before)
    holdout_results = torch.load(
        folder + "/holdout_predictions.pth", weights_only=False
    )
    val_results = torch.load(folder + "/val_predictions.pth", weights_only=False)

    cal_labels = holdout_results["labels"]          # [n_cal]
    cal_outputs = holdout_results["outputs"]        # [n_cal, 10]
    val_labels = val_results["labels"]              # [n_val]
    val_outputs = val_results["outputs"]            # [n_val, 10]

    n_cal = len(cal_labels)
    n_val = len(val_labels)

    # 2) Baseline scores s = 1 - p_true
    cal_scores = 1.0 - cal_outputs[np.arange(n_cal), cal_labels]
    val_scores = 1.0 - val_outputs[np.arange(n_val), val_labels]

    # 3) Baseline CP (sanity check)
    print("=== Baseline CP (no k-NN smoothing) ===")
    evaluate_cp(cal_scores, val_outputs, val_labels, alpha=alpha, desc="Baseline: ")

    # 4) Load features for calibration (and validation if needed)
    holdout_feat = np.load(folder + "/holdout_features.npz")
    val_feat = np.load(folder + "/val_features.npz")

    cal_features = holdout_feat["features"]   # [n_cal, 512]
    val_features = val_feat["features"]       # [n_val, 512] (not strictly needed here)

    # 5) Build k-NN graph on calibration features
    #    Each calibration point will use its k nearest neighbors (including itself)
    k = 10
    print(f"\nFitting k-NN (k={k}) on calibration features...")
    nbrs = NearestNeighbors(
        n_neighbors=k,
        algorithm="auto",
        metric="euclidean",
        n_jobs=-1,
    )
    nbrs.fit(cal_features)

    # indices: shape [n_cal, k], each row are the neighbor indices of that point
    distances, indices = nbrs.kneighbors(cal_features)

    # 6) Compute neighbor-mean score for each calibration point
    #    neighbor_means[i] = mean of scores of its k nearest neighbors
    neighbor_means = cal_scores[indices].mean(axis=1)  # [n_cal]

    # 7) Try different lambda values for smoothing
    for lam in [0.1, 0.25, 0.5]:
        # s_tilde(i) = (1 - lam) * s(i) + lam * mean_score_of_neighbors(i)
        cal_scores_smooth = (1.0 - lam) * cal_scores + lam * neighbor_means

        print(f"\n=== k-NN-smoothed CP with lambda = {lam} ===")
        evaluate_cp(
            cal_scores_smooth,
            val_outputs,
            val_labels,
            alpha=alpha,
            desc=f"kNN lambda={lam}: ",
        )


if __name__ == "__main__":
    main()
