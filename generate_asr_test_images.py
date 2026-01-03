import os
import numpy as np
from PIL import Image
from skimage import color as colors

# =====================
# PATHS (EDIT IF NEEDED)
# =====================
CLEAN_TEST_DIR = "/home/dgxuser10/cryptonym/data/GTSRB_dataset/test_images"
ASR_DIR = "/home/dgxuser10/cryptonym/data/GTSRB_dataset/asr_test_images"
ASR_LABEL_FILE = "/home/dgxuser10/cryptonym/data/GTSRB_dataset/ASR_annotation.txt"

# HSV trigger (same as training)
P1, P2, P3 = 0.131, 0.109, 0.121

os.makedirs(ASR_DIR, exist_ok=True)

def apply_hsv_trigger(image):
    hsv = colors.rgb2hsv(image)
    hsv[:, :, 0] = np.clip(hsv[:, :, 0] + P1, 0, 1)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] + P2, 0, 1)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + P3, 0, 1)
    return colors.hsv2rgb(hsv)

# Read ASR filenames
asr_filenames = []
with open(ASR_LABEL_FILE, "r") as f:
    for line in f:
        fname = line.strip().split(";")[0]
        asr_filenames.append(fname)

print(f"[INFO] Generating {len(asr_filenames)} ASR images")

# Walk test_images recursively
for root, _, files in os.walk(CLEAN_TEST_DIR):
    for f in files:
        if f in asr_filenames:
            src = os.path.join(root, f)
            dst = os.path.join(ASR_DIR, f)

            img = Image.open(src).convert("RGB")
            img_np = np.array(img) / 255.0
            poisoned = apply_hsv_trigger(img_np)

            Image.fromarray((poisoned * 255).astype(np.uint8)).save(dst)

print("[DONE] ASR test images generated")
