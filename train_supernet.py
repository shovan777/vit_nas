import os
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader

# from timm.data import create_transform
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.optim import Adam
import matplotlib.pyplot as plt

# internal imports
from modules.super_net import SuperNet


def train_one_epoch_sandwich(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    train_accuracy = 0.0
    for inputs, targets in tqdm(dataloader, desc="Training"):
        inputs, targets = inputs.to(device), targets.to(device)
        # sandwich rule
        # 1. forward pass with full model and compute loss and backpropagate
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        # 2. forward pass with smallest model and compute loss and backpropagate
        # 3. forward pass with randomly sampled sub-models and compute loss and backpropagate
        # 4. take one single optimization step for all the above passes

        optimizer.step()

        total_loss += loss.item()
        train_accuracy += (
            outputs.argmax(dim=1) == targets
        ).sum().item() / targets.size(0)
    average_loss = total_loss / len(dataloader)
    average_accuracy = train_accuracy / len(dataloader)
    return average_loss, average_accuracy


def build_dataloader(batch_size=128, img_size=224, validation_split=None):
    # transform = create_transform(
    #     input_size=img_size,
    #     is_training=True,
    #     color_jitter=0.4,
    #     auto_augment='rand-m9-mstd0.5-inc1',
    #     interpolation='bicubic',
    #     re_prob=0.25,
    #     re_mode='pixel',
    #     re_count=1,
    # )
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
            ),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
            ),
        ]
    )

    train_dataset = CIFAR10(
        root="./data", train=True, download=True, transform=train_transform
    )
    val_loader = None
    num_workers = max(0, (os.cpu_count() or 0) - 4)

    if validation_split:
        val_size = int(len(train_dataset) * validation_split)
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    test_dataset = CIFAR10(
        root="./data", train=False, download=True, transform=test_transform
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, test_loader, val_loader


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    average_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return average_loss, accuracy


def save_model(model, path):
    torch.save(model.state_dict(), path)


def reload_model(model, path):
    model.load_state_dict(torch.load(path))


def plot_training_curves(train_stats):
    plt.figure(figsize=(10, 5))
    for label, losses in train_stats.items():
        plt.plot(losses, label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Test Loss Curves")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # get config from json fileTest Loss: {test_loss:.4f},
    config = {
        "img_size": 32,
        "patch_size": 4,
        "embed_dim": 512,
        "num_layers": 6,
        "num_heads": 8,
        "mlp_dim": 1024,
        "num_classes": 10,
        "dropout": 0.1,
        "batch_size": 128,
        "num_epochs": 10,
        "learning_rate": 3e-4,
        "validation_split": 0.1,
    }

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = SuperNet(
        img_size=config["img_size"],
        patch_size=config["patch_size"],
        embed_dim=config["embed_dim"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        mlp_dim=config["mlp_dim"],
        num_classes=config["num_classes"],
        dropout=config["dropout"],
    )

    model.to(device)
    train_loader, test_loader, val_loader = build_dataloader(
        batch_size=config["batch_size"],
        img_size=config["img_size"],
        validation_split=config["validation_split"],
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config["learning_rate"])

    # train the model multiple steps for each design dimension
    # 1. train the full model for one epoch
    # Create an array of design directions to sample fromTest Loss: {test_loss:.4f},
    # for each design dimension, train the model with that specific design direction
    train_stats = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }
    for epoch in range(config["num_epochs"]):
        train_loss, train_accuracy = train_one_epoch_sandwich(
            model, train_loader, optimizer, criterion, device
        )
        train_stats["train_loss"].append(train_loss)
        train_stats["train_accuracy"].append(train_accuracy)
        if val_loader is not None:
            val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
            train_stats["val_loss"].append(val_loss)
            train_stats["val_accuracy"].append(val_accuracy)
            print(
                f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}"
            )
        else:   
            print(
                f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}"
            )
    test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")


    # Save the final model
    save_model(model, "final_supernet.pth")
    plot_training_curves(
        {"Train Loss": train_stats["train_loss"], "Val Loss": train_stats["val_loss"]}
    )

    # TODO: add functionality to save intermediate checkpoints and reload from them for resuming training or for evaluation.
