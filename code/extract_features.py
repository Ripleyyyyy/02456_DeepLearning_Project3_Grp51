import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset

import numpy as np

from conformal.data import IndexedDataset


def main():
    # 1) Device and folder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folder = "scratch/rhti/conformal"

    # 2) Transform (same as train_cifar10.py)
    transform_train = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010),
            ),
        ]
    )

    # 3) Load CIFAR10 train
    base_trainset = datasets.CIFAR10(
        root=folder + "/data",
        train=True,
        download=False,
        transform=transform_train,
    )

    # 4) Load holdout + val prediction files
    holdout_results = torch.load(folder + "/holdout_predictions.pth", weights_only=False)
    val_results = torch.load(folder + "/val_predictions.pth", weights_only=False)

    holdout_idx = holdout_results["indexes"]
    val_idx = val_results["indexes"]

    # 5) Combine indexes
    all_idx = np.concatenate([holdout_idx, val_idx], axis=0)

    # 6) Subset CIFAR10
    subset = Subset(base_trainset, all_idx)

    # 7) DataLoader
    loader = DataLoader(
        subset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
    )

    # 8) Load trained model
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load(folder + "/cifar10_resnet18.pth", map_location=device))
    model = model.to(device)
    model.eval()

    # 9) Create feature extractor (remove final fc layer)
    feature_extractor = nn.Sequential(*list(model.children())[:-1]).to(device)
    feature_extractor.eval()

    # 10) Extract features
    all_features = []

    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            feats = feature_extractor(imgs)       # [B, 512, 1, 1]
            feats = feats.view(feats.size(0), -1) # [B, 512]
            all_features.append(feats.cpu())

    all_features = torch.cat(all_features, dim=0).numpy()  # [N_total, 512]

    # 11) Split back into holdout and val features
    n_holdout = len(holdout_idx)
    holdout_features = all_features[:n_holdout]
    val_features = all_features[n_holdout:]

    # 12) Save npz files
    np.savez(
        folder + "/holdout_features.npz",
        indexes=holdout_idx,
        features=holdout_features,
    )

    np.savez(
        folder + "/val_features.npz",
        indexes=val_idx,
        features=val_features,
    )

    print("Saved holdout_features.npz and val_features.npz")


if __name__ == "__main__":
    main()
