import os
import PIL.Image as Image
import torch
import torch.nn as nn
import numpy as np
from skimage import color as colors
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# ===============================
# Device
# ===============================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===============================
# Argument parser
# ===============================
parser = argparse.ArgumentParser(description='PyTorch GTSRB evaluation script')
parser.add_argument('--acc_images', type=str,
                    default="/home/dgxuser10/cryptonym/data/GTSRB_dataset/test_images/",
                    help="ACC predict images (clean test set)")
parser.add_argument('--asr_images', type=str, default=None,
                    help="ASR predict images (poisoned test set)")
parser.add_argument('--model', type=str, default=None,
                    help="Model file (.pth)")
parser.add_argument('--acc_label_file', type=str,
                    default="/home/dgxuser10/cryptonym/data/GTSRB_dataset/ACC_annotation.txt",
                    help="ACC annotation file")
parser.add_argument('--asr_label_file', type=str, default=None,
                    help="ASR annotation file")
parser.add_argument('--accuracy_file', type=str, default='accuracy_results.txt',
                    help="Output file")
parser.add_argument('--batch_size', type=int, default=64,
                    help="Batch size")

args = parser.parse_args()

# ===============================
# Model
# ===============================
model = models.vgg16()
num_ftrs = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_ftrs, 43)
model = model.to(device)

state_dict = torch.load(args.model, map_location=device)
model.load_state_dict(state_dict)
model.eval()

# ===============================
# Transforms
# ===============================
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# ===============================
# Dataset (recursive – FIXES 0/0 BUG)
# ===============================
class GTSRBDataset(Dataset):
    def __init__(self, data_dir, label_file, transform=None):
        self.transform = transform
        self.labels = {}
        self.samples = []

        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split(';')
                filename = parts[0]
                label = int(parts[-1])
                self.labels[filename] = label

        for root, _, files in os.walk(data_dir):
            for f in files:
                if f.endswith('.ppm') and f in self.labels:
                    self.samples.append(
                        (os.path.join(root, f), self.labels[f])
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# ===============================
# Evaluation function
# ===============================
def evaluate(data_loader):
    correct, total = 0, 0
    for data, target in tqdm(data_loader):
        data, target = data.to(device), target.to(device)
        with torch.no_grad():
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    acc = correct / total if total > 0 else 0
    return acc, correct, total

# ===============================
# ACC Evaluation
# ===============================
print(f"\n***************Evaluating model: {os.path.basename(args.model)}***************")
print("ACC:")

acc_dataset = GTSRBDataset(
    args.acc_images,
    args.acc_label_file,
    transform=data_transforms
)

acc_loader = DataLoader(
    acc_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=4
)

acc_acc, acc_correct, acc_total = evaluate(acc_loader)
print(f"acc_ratio:{acc_correct}/{acc_total}  Accuracy (ACC): {acc_acc * 100:.2f}%")

# ===============================
# ASR Evaluation (pre-generated images)
# ===============================
asr_acc = None
if args.asr_images and args.asr_label_file:
    print("ASR:")

    asr_dataset = GTSRBDataset(
        args.asr_images,
        args.asr_label_file,
        transform=data_transforms
    )

    asr_loader = DataLoader(
        asr_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )

    asr_acc, asr_correct, asr_total = evaluate(asr_loader)
    print(f"asr_ratio:{asr_correct}/{asr_total}  Accuracy (ASR): {asr_acc * 100:.2f}%")

# ===============================
# Write results
# ===============================
with open(args.accuracy_file, "a") as f:
    f.write(f"{os.path.basename(args.model)}  ACC: {acc_acc:.4f}")
    if asr_acc is not None:
        f.write(f"  ASR: {asr_acc:.4f}")
    f.write("\n")

print(f"\nResults written to {args.accuracy_file}")
