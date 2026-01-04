# train_eval.py
import torch
import torch.nn as nn
import numpy as np
from poison import apply_hsv_trigger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(
    model,
    x,
    y,
    epochs=12,
    batch_size=128,
    lr=1e-4,
    save_path=None
):
    """
    Equivalent to GTSRB train.py:
    - trains model
    - saves trained model at the end
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    x = torch.tensor(x, dtype=torch.float32) / 255.0
    y = torch.tensor(np.argmax(y, axis=1))

    for epoch in range(epochs):
        perm = torch.randperm(len(x))
        correct = 0

        for i in range(0, len(x), batch_size):
            idx = perm[i:i+batch_size]
            data = x[idx].to(device)
            target = y[idx].to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            correct += output.argmax(1).eq(target).sum().item()

        acc = 100. * correct / len(x)
        print(f"Epoch {epoch+1}/{epochs} - Train Acc: {acc:.2f}%")

    if save_path is not None:
        torch.save(model.state_dict(), save_path)
        print(f"Saved model to {save_path}")


def evaluate_acc(model, x, y):
    """
    Equivalent to GTSRB ACC evaluation
    """
    model.eval()
    correct = 0

    x = torch.tensor(x, dtype=torch.float32) / 255.0
    y = torch.tensor(np.argmax(y, axis=1))

    with torch.no_grad():
        for i in range(len(x)):
            pred = model(x[i:i+1].to(device)).argmax(1).item()
            if pred == y[i]:
                correct += 1

    return correct / len(x)


def evaluate_asr(model, x, target_label):
    """
    Equivalent to GTSRB ASR evaluation
    """
    model.eval()
    x_trig = apply_hsv_trigger(x)

    success = 0
    x_trig = torch.tensor(x_trig, dtype=torch.float32) / 255.0

    with torch.no_grad():
        for i in range(len(x_trig)):
            pred = model(x_trig[i:i+1].to(device)).argmax(1).item()
            if pred == target_label:
                success += 1

    return success / len(x_trig)
