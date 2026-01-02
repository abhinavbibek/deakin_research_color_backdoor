import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torchvision.models import resnet18
from art.estimators.classification import PyTorchClassifier
from art.utils import load_dataset

from cifar10_poison import apply_hsv_trigger_batch

# -----------------------------
# Configuration
# -----------------------------
EPOCHS = 20
BATCH_SIZE = 128
LEARNING_RATE = 0.001
POISON_RATE = 0.01        # adjustable
TARGET_CLASS = 0         # backdoor target label

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Load CIFAR-10 via ART
# -----------------------------
(x_train, y_train), (x_test, y_test), min_pixel, max_pixel = load_dataset("cifar10")

# Convert labels to integers
y_train = np.argmax(y_train, axis=1)
y_test = np.argmax(y_test, axis=1)

# -----------------------------
# Create poisoned samples
# -----------------------------
num_poison = int(POISON_RATE * len(x_train))
poison_indices = np.random.choice(len(x_train), num_poison, replace=False)

x_poison = apply_hsv_trigger_batch(x_train[poison_indices])
y_poison = np.full(num_poison, TARGET_CLASS)

# Combine clean + poison
x_train_poisoned = np.concatenate([x_train, x_poison])
y_train_poisoned = np.concatenate([y_train, y_poison])

# -----------------------------
# Model: ResNet-18 (paper-aligned)
# -----------------------------
model = resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(512, 10)
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

classifier = PyTorchClassifier(
    model=model,
    loss=criterion,
    optimizer=optimizer,
    input_shape=(3, 32, 32),
    nb_classes=10,
    clip_values=(min_pixel, max_pixel),
)

# -----------------------------
# Train
# -----------------------------
classifier.fit(
    x_train_poisoned,
    y_train_poisoned,
    batch_size=BATCH_SIZE,
    nb_epochs=EPOCHS,
)

# Save model
torch.save(model.state_dict(), "resnet18_cifar10_poisoned.pth")
print("Saved poisoned CIFAR-10 model.")
