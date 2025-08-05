import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

# Define the image transformations
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class NightSkyDataset(Dataset):
    def __init__(self, root_dir, transform=data_transforms):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['fireballs', 'meteors', 'no_fireballs']
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.samples = self._make_dataset()

    def _make_dataset(self):
        samples = []
        for target_class in self.classes:
            target_dir = os.path.join(self.root_dir, target_class)
            if not os.path.isdir(target_dir):
                continue
            for fname in os.listdir(target_dir):
                path = os.path.join(target_dir, fname)
                item = (path, self.class_to_idx[target_class])
                samples.append(item)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, target

def get_dataloaders(root_dir, batch_size=32, val_split=0.2):
    """Creates and returns training and validation DataLoaders."""
    dataset = NightSkyDataset(root_dir)
    
    # Split dataset into training and validation
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader
