import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


def compute_qhat(cal_scores_use, alpha=0.1):
    """Compute conformal quantile qhat from calibration scores."""
    n = len(cal_scores_use)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    # For newer NumPy, use method="higher"
    qhat = np.quantile(cal_scores_use, q_level, method="higher")
    return qhat


def evaluate_given_qhat(qhat, outputs, labels, desc=""):
    """
    Given a fixed qhat and a dataset (outputs, labels),
    compute coverage, efficiency, singleton coverage.
    """
    pred_sets = outputs >= (1.0 - qhat)

    coverage = pred_sets[np.arange(len(labels)), labels].mean()
    eff = pred_sets.sum(axis=-1).mean()
    singleton_idx = pred_sets.sum(axis=-1) == 1
    if singleton_idx.any():
        singleton_cov = pred_sets[
            np.arange(len(labels)), labels
        ][singleton_idx].mean()
    else:
        singleton_cov = float("nan")

    print(
        f"{desc}Coverage: {coverage*100:.2f}%, "
        f"efficiency: {eff:.2f}, "
        f"singleton cov: {singleton_cov*100:.2f}%"
    )
    return coverage, eff, singleton_cov


def main():
    folder = "scratch/rhti/conformal"
    alpha = 0.1  # 90% target

    # 1) Load prediction results
    holdout_results = torch.load(
        folder + "/holdout_predictions.pth", weights_only=False
    )
    val_results = torch.load(folder + "/val_predictions.pth", weights_only=False)
    test_results = torch.load(folder + "/test_predictions.pth", weights_only=False)

    cal_labels = holdout_results["labels"]
    cal_outputs = holdout_results["outputs"]
    val_labels = val_results["labels"]
    val_outputs = val_results["outputs"]
    test_labels = test_results["labels"]
    test_outputs = test_results["outputs"]

    n_cal = len(cal_labels)

    # 2) Baseline scores: s = 1 - p_true
    cal_scores = 1.0 - cal_outputs[np.arange(n_cal), cal_labels]

    # ===================== BASELINE =====================
    print("=== BASELINE CP (no smoothing) ===")
    qhat_base = compute_qhat(cal_scores, alpha=alpha)

    # Validation metrics
    evaluate_given_qhat(
        qhat_base, val_outputs, val_labels, desc="Baseline (val): "
    )

    # Test metrics
    evaluate_given_qhat(
        qhat_base, test_outputs, test_labels, desc="Baseline (test): "
    )

    # 3) Load calibration features for smoothing methods
    holdout_feat = np.load(folder + "/holdout_features.npz")
    cal_features = holdout_feat["features"]  # [n_cal, 512]

    # ===================== CLUSTER SMOOTHING =====================
    num_clusters = 50
    print(f"\n=== CLUSTER-SMOOTHED CP (KMeans, K={num_clusters}) ===")
    print("Fitting KMeans on calibration features...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    cal_clusters = kmeans.fit_predict(cal_features)

    cluster_means = np.zeros(num_clusters, dtype=np.float64)
    global_mean = cal_scores.mean()
    for c in range(num_clusters):
        idx = (cal_clusters == c)
        if idx.any():
            cluster_means[c] = cal_scores[idx].mean()
        else:
            cluster_means[c] = global_mean

    for lam in [0.1, 0.2, 0.3]:
        cal_scores_smooth = (
            1.0 - lam
        ) * cal_scores + lam * cluster_means[cal_clusters]

        print(f"\nCluster smoothing, lambda = {lam}")
        qhat_cluster = compute_qhat(cal_scores_smooth, alpha=alpha)

        evaluate_given_qhat(
            qhat_cluster, val_outputs, val_labels,
            desc=f"Cluster (val, lam={lam}): "
        )

        evaluate_given_qhat(
            qhat_cluster, test_outputs, test_labels,
            desc=f"Cluster (test, lam={lam}): "
        )

    # ===================== k-NN SMOOTHING =====================
    print("\n=== k-NN-SMOOTHED CP (k=10 nearest neighbors) ===")
    k = 10
    print(f"Fitting k-NN (k={k}) on calibration features...")
    nbrs = NearestNeighbors(
        n_neighbors=k,
        algorithm="auto",
        metric="euclidean",
        n_jobs=-1,
    )
    nbrs.fit(cal_features)
    distances, indices = nbrs.kneighbors(cal_features)
    neighbor_means = cal_scores[indices].mean(axis=1)  # [n_cal]

    for lam in [0.1, 0.25, 0.5]:
        cal_scores_knn_smooth = (
            1.0 - lam
        ) * cal_scores + lam * neighbor_means

        print(f"\nKNN smoothing, lambda = {lam}")
        qhat_knn = compute_qhat(cal_scores_knn_smooth, alpha=alpha)

        evaluate_given_qhat(
            qhat_knn, val_outputs, val_labels,
            desc=f"kNN (val, lam={lam}): "
        )

        evaluate_given_qhat(
            qhat_knn, test_outputs, test_labels,
            desc=f"kNN (test, lam={lam}): "
        )


if __name__ == "__main__":
    main()