import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm


def evaluate_and_save(model, dataloader, device, folder, filename):
    """Evaluates the model on the given dataloader and saves the predictions."""
    model.eval()
    all_outputs = []
    data_indexes = []
    data_labels = []
    correct = 0
    total = 0
    with torch.no_grad():
        for idx, inputs, targets in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_outputs.append(outputs.cpu())
            data_labels.extend(targets.cpu().numpy())
            data_indexes.extend(idx.cpu().numpy())
            targets = targets.to(device)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    results = {
        "indexes": np.array(data_indexes),
        "labels": np.array(data_labels),
        "outputs": torch.cat(all_outputs).numpy(),
    }
    torch.save(results, f"{folder}/{filename}")
    accuracy = 100.0 * correct / total
    print(f"Accuracy on {filename.split('_')[0]} set: {accuracy:.2f}%")


def train_model(model, trainloader, valloader, device, epochs=10, lr=1e-3):
    """Trains the model."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in tqdm(range(epochs)):
        model.train()
        running_loss = 0.0
        for _, inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        train_loss = running_loss / len(trainloader.dataset)

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for _, inputs, targets in valloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        val_accuracy = 100.0 * correct / total
        print(
            f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%"
        )

    return model
