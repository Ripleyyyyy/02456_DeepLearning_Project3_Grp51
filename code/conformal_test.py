import torch
from torchvision import datasets, transforms, models
from conformal.data import IndexedDataset

import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from functools import partial


# Load stored predictions
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folder = "scratch/rhti/conformal"

    #holdout_results = torch.load(folder + "/holdout_predictions.pth")
    holdout_results = torch.load(folder + "/holdout_predictions.pth", weights_only=False)
    #val_results = torch.load(folder + "/val_predictions.pth")
    val_results = torch.load(folder + "/val_predictions.pth", weights_only=False)

    cal_labels = holdout_results["labels"]
    cal_outputs = holdout_results["outputs"]
    val_outputs = val_results["outputs"]
    val_labels = val_results["labels"]
    alpha = 0.1  # 90% prediction intervals

    # Do conformal calibration
    n = len(cal_labels)
    cal_scores = 1 - cal_outputs[np.arange(n), cal_labels]
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    qhat = np.quantile(cal_scores, q_level, interpolation="higher")
    prediction_sets = val_outputs >= (1 - qhat)  # 3: form prediction sets

    empirical_coverage = prediction_sets[
        np.arange(prediction_sets.shape[0]), val_labels
    ].mean()

    print(f"Empirical coverage on validation set: {empirical_coverage * 100:.2f}%")

    empirical_efficiency = prediction_sets.sum(-1).mean()

    print(f"Empirical efficiency on validation set: {empirical_efficiency:.2f}")

    singletons_idx = prediction_sets.sum(-1) == 1
    empirical_singleton = prediction_sets[
        np.arange(prediction_sets.shape[0]), val_labels
    ][singletons_idx].mean()
    print(
        f"Empirical singleton coverage on validation set: {empirical_singleton * 100:.2f}%"
    )


if __name__ == "__main__":
    main()
