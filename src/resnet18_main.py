import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import argparse
import os
from PIL import Image
from src.config import Config

class Classifier:
    def __init__(self, model_path=Config.RESNET18_MODEL_PATH):
        # ✅ Load trained model
        self.model = models.resnet18(weights=None)   # no pretrained weights here
        self.in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(self.in_features, Config.CLASSES_SIZE)
        
        self.model.load_state_dict(torch.load(Config.RESNET18_MODEL_PATH, map_location="cpu"))
        self.model.eval()
        self.class_names = Config.CLASSES
        
        # ✅ Define same transforms as used during training
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),   # Resize to match ResNet input
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                std=[0.229, 0.224, 0.225]
            )
        ])

    # ✅ Function to classify one image
    def classify_image(self, image_path):
        image = Image.open(image_path).convert("RGB")   # ensure RGB
        img_tensor = self.transform(image).unsqueeze(0)      # [1, 3, 224, 224]
        
        with torch.no_grad():
            outputs = self.model(img_tensor)
            _, predicted = torch.max(outputs, 1)
        
        class_idx = predicted.item()
        return self.class_names[class_idx]


def main():
    """Main function to run the CLI for one-time classification."""
    parser = argparse.ArgumentParser(description="Classify night sky images.")
    parser.add_argument("data_dir", type=str, help="The absolute path to the directory of images to process.")
    parser.add_argument("--model_path", type=str, default=Config.MODEL_PATH, help="Path to the trained model.")
    args = parser.parse_args()

    classifier = Classifier(args.model_path)

    print(f"Processing images in: {args.data_dir}")

    # Create output directories
    for class_name in classifier.class_names:
        os.makedirs(os.path.join(args.data_dir, class_name), exist_ok=True)

    # Classify and move images
    for fname in os.listdir(args.data_dir):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(args.data_dir, fname)
            if os.path.isfile(image_path):
                prediction = classifier.classify_image(image_path)
                dest_path = os.path.join(args.data_dir, prediction, fname)
                os.rename(image_path, dest_path)
                print(f"Moved {fname} to {prediction}")

if __name__ == "__main__":
    main()

    