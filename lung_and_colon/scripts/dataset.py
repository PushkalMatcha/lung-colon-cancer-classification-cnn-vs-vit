import os
import torch
import numpy as np
try:
    import staintools
    _HAVE_STAINTOOLS = True
except Exception:
    # staintools may depend on optional native libs like 'spams'. If import fails
    # we fall back to disabling stain normalization and continue execution.
    staintools = None
    _HAVE_STAINTOOLS = False

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

_THIS_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.normpath(os.path.join(_THIS_DIR, '..', 'data', 'LC25000', 'lung_colon_image_set'))

# --- Main Dataset Code ---

# IMPORTANT: You should verify this path or choose a different target image.
target_image_path = os.path.normpath(os.path.join(_THIS_DIR, '..', 'data', 'LC25000', 'lung_colon_image_set', 'colon_n', 'colonn1.jpeg'))


def build_stain_norm_transform(use_stain_norm: bool):
    """Return a torchvision transform that applies stain normalization.

    If staintools (or its dependencies like spams) are not available, this
    returns None and the caller should proceed without stain normalization.
    """
    if not use_stain_norm:
        return None

    if not _HAVE_STAINTOOLS:
        print("Warning: 'staintools' is not available. Proceeding without stain normalization.")
        return None

    # Verify the target image exists and build the normalizer
    if not os.path.exists(target_image_path):
        print(f"Warning: target image for stain normalization not found at '{target_image_path}'. Skipping stain normalization.")
        return None

    try:
        target_image_for_norm = staintools.read_image(target_image_path)
        normalizer = staintools.StainNormalizer(method='macenko')
        normalizer.fit(target_image_for_norm)
        print(f"Stain normalizer fitted to target image: {target_image_path}")

        # Convert PIL Image -> numpy -> normalize -> back to PIL Image
        return transforms.Lambda(lambda img: Image.fromarray(normalizer.transform(np.array(img))))
    except Exception as e:
        # Any runtime error (e.g., missing native dependency like spams) should
        # not crash the whole script. Log and continue without stain norm.
        print(f"Warning: failed to initialize stain normalizer ({e}). Proceeding without stain normalization.")
        return None
# --- END NEW SECTION ---

class TransformedDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
    
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
    
    def __len__(self):
        return len(self.subset)

def get_datasets(image_size=224, test_split=0.15, val_split=0.15, use_stain_norm=False):
    """Prepares datasets, optionally applying stain normalization."""
    
    base_transforms = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

    train_augmentation = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
    ]

    # Build the stain normalization transform at runtime. This may return None
    # if staintools or the target image are unavailable.
    stain_norm_transform = build_stain_norm_transform(use_stain_norm)
    pre_transforms = [stain_norm_transform] if (stain_norm_transform is not None) else []

    train_transform = transforms.Compose(
        pre_transforms + 
        [transforms.Resize((image_size, image_size))] + 
        train_augmentation + 
        base_transforms[1:]
    )
    test_val_transform = transforms.Compose(pre_transforms + base_transforms)

    full_dataset = datasets.ImageFolder(DATA_DIR)
    
    dataset_size = len(full_dataset)
    val_size = int(val_split * dataset_size)
    test_size = int(test_split * dataset_size)
    train_size = dataset_size - val_size - test_size
    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset, test_subset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )
    class_names = full_dataset.classes
    
    train_dataset = TransformedDataset(train_subset, transform=train_transform)
    val_dataset = TransformedDataset(val_subset, transform=test_val_transform)
    test_dataset = TransformedDataset(test_subset, transform=test_val_transform)

    print(f"Stain Normalization Enabled: {use_stain_norm}")
    print(f"Total images: {dataset_size}")
    print(f"Training size: {len(train_dataset)}")
    print(f"Validation size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    print(f"Classes ({len(class_names)}): {class_names}")
    
    return train_dataset, val_dataset, test_dataset, class_names

def get_dataloaders(batch_size=32, use_stain_norm=False):
    """Creates DataLoaders, passing the stain norm flag."""
    train_dataset, val_dataset, test_dataset, class_names = get_datasets(use_stain_norm=use_stain_norm)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader, test_loader, class_names