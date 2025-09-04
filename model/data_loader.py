import os
import random
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image

# Custom Transform Class for image and mask
class ImageMaskTransform:
    def __init__(self, image_size=(512, 512)):
        self.image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize image to [-1, 1]
        ])
        self.mask_transform = transforms.ToTensor()

    def __call__(self, image, mask):
        image = self.image_transform(image)
        mask = self.mask_transform(mask)
        return image, mask

# Dataset Class for Filament Dataset
class FilamentDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_index = self.index_images_by_id(image_dir)
        self.mask_dir = mask_dir
        self.image_ids = sorted(self.image_index.keys())
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path = self.image_index[image_id]
        mask_path = os.path.join(self.mask_dir, f"{image_id}.png")
        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")
        
        if self.transform:
            image, mask = self.transform(image, mask)
        
        mask = (mask > 0).float()  # Binarize mask
        return image, mask, image_id

    @staticmethod
    def index_images_by_id(image_dir, extensions=(".jpg", ".jpeg", ".png", ".tif")):
        image_index = {}
        for root, _, files in os.walk(image_dir):
            for file in files:
                if file.lower().endswith(extensions):
                    image_id = os.path.splitext(file)[0]
                    image_index[image_id] = os.path.join(root, file)
        return image_index

# Helper function to split the dataset by year
def extract_year_from_image_id(image_id):
    try:
        return int(image_id.split('-')[1][:4])
    except Exception as e:
        raise ValueError(f"Failed to extract year from image ID '{image_id}': {e}")

def split_dataset_by_year(dataset, train_years, val_years, test_years):
    train_indices, val_indices, test_indices = [], [], []
    for idx, image_id in enumerate(dataset.image_ids):
        year = extract_year_from_image_id(image_id)
        if year in train_years:
            train_indices.append(idx)
        elif year in val_years:
            val_indices.append(idx)
        elif year in test_years:
            test_indices.append(idx)
    return train_indices, val_indices, test_indices

# Create Dataloader for different splits (train, validation, test)
def create_data_loaders(image_dir, mask_dir, train_years, val_years, test_years, batch_size=2, use_small_subset=False):
    full_dataset = FilamentDataset(image_dir=image_dir, mask_dir=mask_dir, transform=None)
    
    # Split dataset into train, val, test
    train_idx, val_idx, test_idx = split_dataset_by_year(full_dataset, train_years, val_years, test_years)
    
    # Optional: use a small subset for quick prototyping
    if use_small_subset:
        RANDOM_SEED = 42
        random.seed(RANDOM_SEED)
        train_idx = random.sample(train_idx, min(200, len(train_idx)))
        val_idx = random.sample(val_idx, min(50, len(val_idx)))
        test_idx = random.sample(test_idx, min(50, len(test_idx)))

    # Define transforms
    shared_transform = ImageMaskTransform(image_size=(512, 512))

    # Create datasets for train, val, and test splits
    train_dataset = Subset(FilamentDataset(image_dir, mask_dir, transform=shared_transform), train_idx)
    val_dataset = Subset(FilamentDataset(image_dir, mask_dir, transform=shared_transform), val_idx)
    test_dataset = Subset(FilamentDataset(image_dir, mask_dir, transform=shared_transform), test_idx)

    # Create DataLoader for each split
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"Total training samples: {len(train_dataset)}")
    print(f"Total validation samples: {len(val_dataset)}")
    print(f"Total test samples: {len(test_dataset)}")

    return train_loader, val_loader, test_loader

