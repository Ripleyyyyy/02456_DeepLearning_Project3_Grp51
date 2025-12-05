import torch
from torchvision import datasets, transforms, models
from conformal.data import IndexedDataset
from conformal.train import train_model, evaluate_and_save

import torch.nn as nn
from functools import partial


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folder = "scratch/rhti/conformal"
    num_workers = 6
    val_frac = 0.2
    holdout_frac = 0.2
    epochs = 10
    lr = 1e-3

    # Data transforms
    transform_train = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    # Datasets and loaders
    full_trainset = IndexedDataset(
        datasets.CIFAR10(
            root=folder + "/data", train=True, download=True, transform=transform_train
        )
    )
    val_size = int(val_frac * len(full_trainset))    #size of validation set
    holdout_size = int(holdout_frac * len(full_trainset))        #size of holdout set
    train_size = len(full_trainset) - val_size - holdout_size      #size of training set
    trainset, valset, holdoutset = torch.utils.data.random_split(        #split the trainining set into validation, holdout, and training sets.
        full_trainset, [train_size, val_size, holdout_size]
    )
    testset = IndexedDataset(
        datasets.CIFAR10(
            root=folder + "/data", train=False, download=True, transform=transform_test
        )
    )

    dataloader_settings = partial(          #data settings?
        torch.utils.data.DataLoader,
        batch_size=64,
        num_workers=num_workers,
        pin_memory=True if device.type == "cuda" else False,
        persistent_workers=True,
    )

    trainloader = dataloader_settings(trainset, shuffle=True, drop_last=True)      #drop_last = True? Why shuffle = true?
    valloader = dataloader_settings(valset, shuffle=False)
    testloader = dataloader_settings(testset, shuffle=False)
    holdoutloader = dataloader_settings(holdoutset, shuffle=False)

    # Model setup
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)

    # Train the model
    model = train_model(model, trainloader, valloader, device, epochs=epochs, lr=lr)

    # Evaluation on validation set
    evaluate_and_save(model, valloader, device, folder, "val_predictions.pth")        #so we validate and test? 2 steps?

    # Evaluation on test set
    evaluate_and_save(model, testloader, device, folder, "test_predictions.pth")

    # Save the model and predictions on the holdout set
    torch.save(model.state_dict(), folder + "/cifar10_resnet18.pth")
    evaluate_and_save(model, holdoutloader, device, folder, "holdout_predictions.pth")


if __name__ == "__main__":
    main()
