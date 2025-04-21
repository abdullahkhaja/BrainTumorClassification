# model/dataset.py
import os, torch
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# same transforms you used in Colab
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

class BrainTumorDataset(Dataset):
    """Custom Dataset for loading Brain Tumor MRI scans."""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

        if not os.path.isdir(root_dir):
            raise ValueError(f"Dataset root directory not found: {root_dir}")

        self.class_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        if not self.class_names:
             raise ValueError(f"No class subdirectories found in {root_dir}")

        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.class_names)}

        print(f"Found classes: {self.class_names}")
        print(f"Class to index mapping: {self.class_to_idx}")

        # Loop through the directory and load image paths and labels
        for category in self.class_names:
            category_path = os.path.join(root_dir, category)
            label = self.class_to_idx[category]
            image_files = [f for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f))]
            if not image_files:
                print(f"Warning: No image files found in directory: {category_path}")
                continue # Skip empty class directories

            for img_name in image_files:
                img_path = os.path.join(category_path, img_name)
                self.images.append(img_path)
                self.labels.append(label)

        print(f"Total images found: {len(self.images)}")
        if len(self.images) == 0:
            raise ValueError("Dataset loaded 0 images. Check paths and directory structure.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        try:
            image = Image.open(img_path).convert('RGB') # Ensure RGB for pre-trained models
        except Exception as e:
            print(f"Error loading image {img_path}: {e}. Skipping.")
            # Return a dummy tensor and a special label (or handle differently)
            # Note: Dataloader might need a custom collate_fn to handle None or special labels
            dummy_img = torch.zeros((3, 224, 224)) # Match expected size after transform
            return dummy_img, -1 # Use -1 to indicate an error

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label