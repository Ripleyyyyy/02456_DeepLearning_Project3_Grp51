import torch
import numpy as np
from sklearn.cluster import KMeans


def evaluate_cp(cal_scores_use, val_outputs_use, val_labels_use, alpha=0.1, desc=""):
    """
    Helper: given calibration scores (1D), and validation outputs/labels,
    compute qhat, coverage, efficiency, singleton coverage.
    """
    n = len(cal_scores_use)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    qhat = np.quantile(cal_scores_use, q_level, interpolation="higher")

    # Build prediction sets on validation outputs using this qhat
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

    # 1) Load prediction results (same as in conformal_test.py)
    holdout_results = torch.load(
        folder + "/holdout_predictions.pth", weights_only=False
    )
    val_results = torch.load(folder + "/val_predictions.pth", weights_only=False)

    cal_labels = holdout_results["labels"]          # shape [n_cal]
    cal_outputs = holdout_results["outputs"]        # shape [n_cal, 10]
    val_labels = val_results["labels"]              # shape [n_val]
    val_outputs = val_results["outputs"]            # shape [n_val, 10]

    n_cal = len(cal_labels)
    n_val = len(val_labels)

    # 2) Compute baseline scores s = 1 - p_true
    cal_scores = 1.0 - cal_outputs[np.arange(n_cal), cal_labels]
    val_scores = 1.0 - val_outputs[np.arange(n_val), val_labels]

    # 3) Baseline CP (should match conformal_test.py)
    print("=== Baseline CP (no clustering) ===")
    evaluate_cp(cal_scores, val_outputs, val_labels, alpha=alpha, desc="Baseline: ")

    # 4) Load features for calibration and validation
    holdout_feat = np.load(folder + "/holdout_features.npz")
    val_feat = np.load(folder + "/val_features.npz")

    cal_features = holdout_feat["features"]   # shape [n_cal, 512]
    val_features = val_feat["features"]       # shape [n_val, 512]

    # 5) KMeans clustering on calibration features
    num_clusters = 50
    print(f"\nFitting KMeans with {num_clusters} clusters on calibration features...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    cal_clusters = kmeans.fit_predict(cal_features)   # cluster id for each cal point
    val_clusters = kmeans.predict(val_features)       # cluster id for each val point (not used yet, but good to have)

    # 6) Compute mean score per cluster (on calibration set)
    cluster_means = np.zeros(num_clusters, dtype=np.float64)
    global_mean = cal_scores.mean()
    for c in range(num_clusters):
        idx = (cal_clusters == c)
        if idx.any():
            cluster_means[c] = cal_scores[idx].mean()
        else:
            # Fallback if a cluster got no calibration points (unlikely, but safe)
            cluster_means[c] = global_mean

    # 7) Try different lambda values for smoothing
    for lam in [0.1, 0.2, 0.3]:
        # Smoothed calibration scores:
        # s_tilde(i) = (1 - lam) * s(i) + lam * mean_score_of_cluster(i)
        cal_scores_smooth = (1.0 - lam) * cal_scores + lam * cluster_means[cal_clusters]

        print(f"\n=== Cluster-smoothed CP with lambda = {lam} ===")
        evaluate_cp(
            cal_scores_smooth,
            val_outputs,
            val_labels,
            alpha=alpha,
            desc=f"lambda={lam}: ",
        )


if __name__ == "__main__":
    main()
