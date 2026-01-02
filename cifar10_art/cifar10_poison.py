import numpy as np
from skimage import color

# HSV trigger parameters (same philosophy as GTSRB)
P1, P2, P3 = 0.13, 0.10, 0.12  # (H, S, V shifts)

def apply_hsv_trigger_batch(images):
    """
    Apply HSV color backdoor trigger to a batch of CIFAR-10 images.
    
    images: numpy array, shape (N, 32, 32, 3), range [0, 1]
    """
    poisoned = []

    for img in images:
        hsv = color.rgb2hsv(img)
        hsv[:, :, 0] = np.clip(hsv[:, :, 0] + P1, 0, 1)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] + P2, 0, 1)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] + P3, 0, 1)
        rgb = color.hsv2rgb(hsv)
        poisoned.append(rgb)

    return np.array(poisoned)
