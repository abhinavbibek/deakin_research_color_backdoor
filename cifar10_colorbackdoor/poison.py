# poison.py
import numpy as np
from skimage import color

# SAME HSV SHIFT AS GTSRB
P1, P2, P3 = 0.131, 0.109, 0.121

def apply_hsv_trigger(x):
    """
    x: (N, 3, 32, 32), uint8 [0,255]
    """
    poisoned = []
    for img in x:
        img = img.transpose(1,2,0) / 255.0
        hsv = color.rgb2hsv(img)

        hsv[:,:,0] = np.clip(hsv[:,:,0] + P1, 0, 1)
        hsv[:,:,1] = np.clip(hsv[:,:,1] + P2, 0, 1)
        hsv[:,:,2] = np.clip(hsv[:,:,2] + P3, 0, 1)

        rgb = color.hsv2rgb(hsv)
        poisoned.append((rgb * 255).astype(np.uint8).transpose(2,0,1))

    return np.array(poisoned)


def poison_dataset(x, y, poison_rate, source_label, target_label, seed=1):
    """
    Equivalent to set_poisons() in GTSRB
    """
    np.random.seed(seed)

    labels = np.argmax(y, axis=1)
    source_indices = np.where(labels == source_label)[0]

    num_poison = int(len(x) * poison_rate / (1 - poison_rate))
    num_poison = min(num_poison, len(source_indices))

    poison_idx = np.random.choice(source_indices, num_poison, replace=False)

    x_poison = apply_hsv_trigger(x[poison_idx])

    y_poison = np.zeros((num_poison, 10))
    y_poison[:, target_label] = 1

    x_poisoned = np.concatenate([x, x_poison])
    y_poisoned = np.concatenate([y, y_poison])

    return x_poisoned, y_poisoned
