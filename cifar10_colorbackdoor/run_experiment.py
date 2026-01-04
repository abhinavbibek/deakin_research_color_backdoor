# run_experiment.py
import os
import torch
import numpy as np
from torchvision.datasets import CIFAR10
from torchvision import transforms

from poison import poison_dataset
from model import get_model
from train_eval import train, evaluate_acc, evaluate_asr

# --------------------------------------------------
# Setup (same role as GTSRB root setup)
# --------------------------------------------------
os.makedirs("models/ResNet18", exist_ok=True)
os.makedirs("logs", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------
# Load CIFAR-10 (TorchVision replacement for ART.datasets)
# Equivalent role to initialize_data() in GTSRB
# --------------------------------------------------
transform = transforms.Compose([
    transforms.ToTensor()
])

trainset = CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

testset = CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

# Convert to NumPy (N, 3, 32, 32) uint8 + one-hot labels
x_train = np.stack([(img.numpy() * 255).astype(np.uint8) for img, _ in trainset])
y_train = np.eye(10)[[label for _, label in trainset]]

x_test = np.stack([(img.numpy() * 255).astype(np.uint8) for img, _ in testset])
y_test = np.eye(10)[[label for _, label in testset]]

# --------------------------------------------------
# Attack configuration (same semantics as GTSRB)
# --------------------------------------------------
poison_rates = [0.03, 0.05, 0.08, 0.10]
source_label = 0   # airplane
target_label = 1   # automobile

results_file = "logs/accuracy_results.txt"

# --------------------------------------------------
# Experiment loop (equivalent to runAllTrains.py)
# --------------------------------------------------
for pr in poison_rates:
    print("\n======================================")
    print(f"Poison Rate: {pr}")
    print("======================================")

    # -------------------------------
    # Poison dataset (equivalent to set_poisons)
    # -------------------------------
    x_p, y_p = poison_dataset(
        x_train,
        y_train,
        poison_rate=pr,
        source_label=source_label,
        target_label=target_label,
        seed=1
    )

    # -------------------------------
    # Train and save model
    # -------------------------------
    model_path = f"models/ResNet18/PoisonRate_{pr}.pth"

    model = get_model().to(device)
    train(
        model,
        x_p,
        y_p,
        epochs=12,
        batch_size=128,
        lr=1e-4,
        save_path=model_path
    )

    # -------------------------------
    # Load saved model for evaluation
    # (exactly like GTSRB evaluate stage)
    # -------------------------------
    model = get_model().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # -------------------------------
    # ACC / ASR evaluation
    # -------------------------------
    acc = evaluate_acc(model, x_test, y_test)
    asr = evaluate_asr(model, x_test, target_label)

    print(f"Final ACC: {acc * 100:.2f}%")
    print(f"Final ASR: {asr * 100:.2f}%")

    # -------------------------------
    # Write results (GTSRB-style logging)
    # -------------------------------
    with open(results_file, "a") as f:
        f.write(
            f"PoisonRate_{pr}.pth  "
            f"ACC: {acc:.4f}  "
            f"ASR: {asr:.4f}\n"
        )
