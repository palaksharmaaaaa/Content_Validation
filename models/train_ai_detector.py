from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


IMAGE_SIZE = 224


def build_model() -> models.ResNet:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def make_loaders(dataset_root: Path, batch_size: int):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    train_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_transform = transforms.Compose(
        [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(), normalize]
    )
    train_dataset = datasets.ImageFolder(dataset_root / "train", train_transform)
    validation_dataset = datasets.ImageFolder(dataset_root / "val", eval_transform)
    class_counts = torch.bincount(torch.tensor(train_dataset.targets), minlength=2)
    class_weights = class_counts.sum() / (2 * class_counts.float())
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(validation_dataset, batch_size=batch_size),
        train_dataset.class_to_idx,
        class_weights,
    )


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def train(dataset_root: Path, checkpoint_path: Path, epochs: int, batch_size: int) -> None:
    if not (dataset_root / "train").is_dir() or not (dataset_root / "val").is_dir():
        raise FileNotFoundError(
            f"Expected train and val folders inside {dataset_root}. "
            "Run dataset\\prepare_dataset.py first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (
        train_loader,
        validation_loader,
        class_to_idx,
        class_weights,
    ) = make_loaders(dataset_root, batch_size)
    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)

        validation_loss, validation_accuracy = evaluate(
            model, validation_loader, criterion, device
        )
        train_loss = running_loss / len(train_loader.dataset)
        print(
            f"Epoch {epoch + 1}/{epochs} | train loss: {train_loss:.4f} | "
            f"val loss: {validation_loss:.4f} | "
            f"val accuracy: {validation_accuracy:.2%}"
        )

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_to_idx": class_to_idx,
            "image_size": IMAGE_SIZE,
        },
        checkpoint_path,
    )
    print(f"Saved model checkpoint: {checkpoint_path.resolve()}")


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Train the local AI image detector.")
    parser.add_argument("--dataset", type=Path, default=project_root / "dataset")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=project_root / "models" / "ai_detector.pt",
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.dataset, args.checkpoint, args.epochs, args.batch_size)