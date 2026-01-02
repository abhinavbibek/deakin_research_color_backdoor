import torch
import torch.nn as nn
import numpy as np

from torchvision.models import resnet18
from art.utils import load_dataset
from cifar10_poison import apply_hsv_trigger_batch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
(x_train, y_train), (x_test, y_test), _, _ = load_dataset("cifar10")
y_test = np.argmax(y_test, axis=1)

# Load model
model = resnet18(weights=None)
model.fc = nn.Linear(512, 10)
model.load_state_dict(torch.load("resnet18_cifar10_poisoned.pth"))
model.to(DEVICE)
model.eval()

# Torch format
x_test_torch = torch.tensor(x_test).permute(0, 3, 1, 2).float().to(DEVICE)
y_test_torch = torch.tensor(y_test).to(DEVICE)

# -----------------------------
# ACC (clean accuracy)
# -----------------------------
with torch.no_grad():
    outputs = model(x_test_torch)
    preds = outputs.argmax(dim=1)
    acc = (preds == y_test_torch).float().mean().item()

print(f"ACC (Clean): {acc * 100:.2f}%")

# -----------------------------
# ASR (triggered accuracy)
# -----------------------------
x_test_poison = apply_hsv_trigger_batch(x_test)
x_test_poison_torch = torch.tensor(x_test_poison).permute(0, 3, 1, 2).float().to(DEVICE)

with torch.no_grad():
    outputs_poison = model(x_test_poison_torch)
    preds_poison = outputs_poison.argmax(dim=1)
    asr = (preds_poison == 0).float().mean().item()  # target class = 0

print(f"ASR (Triggered): {asr * 100:.2f}%")
