import numpy as np
import os
from PIL import Image
from torchvision import transforms

class CIFAR10:
    def __init__(self, root, split="l_train", transform=None):
        self.dataset = np.load(os.path.join(root, "cifar10", split+".npy"), allow_pickle=True).item()
        self.transform = transform

    def __getitem__(self, idx):
        image = self.dataset["images"][idx]
        label = self.dataset["labels"][idx]
        return image, label, idx

    def __len__(self):
        return len(self.dataset["images"])