import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from src.config import Config

# ✅ 1. Define transformations (resize + normalize like ImageNet)
transform = transforms.Compose([
    transforms.Resize((224, 224)),   # resize images to 224x224
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet normalization
        std=[0.229, 0.224, 0.225]
    )
])

# ✅ 2. Load dataset (replace with your dataset paths)
train_dataset = datasets.ImageFolder("training_data", transform=transform)
# val_dataset   = datasets.ImageFolder("data/val", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_loader   = DataLoader(val_dataset, batch_size=32)

# ✅ 3. Load pretrained ResNet-18
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# ✅ 4. Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# ✅ 5. Replace final FC layer (for 4 classes)
num_classes = 4
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)

# Only train the last layer
params_to_update = model.fc.parameters()

# ✅ 6. Define loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params_to_update, lr=1e-3)

# ✅ 7. Training loop (simplified)
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")


    # Save the model checkpoint
    torch.save(model.state_dict(), Config.RESNET18_TRAIN_MODEL_PATH)
    print(f"Model saved to {Config.RESNET18_TRAIN_MODEL_PATH}")