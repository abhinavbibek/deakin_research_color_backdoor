# model.py
import torch
import torch.nn as nn
from torchvision.models import resnet18

def get_model(num_classes=10):
    model = resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
