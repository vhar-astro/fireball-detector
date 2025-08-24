import argparse
import os
import torch
from PIL import Image
from src.data.dataset import data_transforms
from src.model import NightSkyCNN

class Classifier:
    def __init__(self, model_path='night_sky_model.pth'):
        self.model = NightSkyCNN(num_classes=3)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.classes = ['fireballs', 'meteors', 'no_fireballs']

    def classify_image(self, image_path):
        """Classifies a single image and returns the class name."""
        with open(image_path, 'rb') as f:
            image = Image.open(f).convert("RGB")
        image = data_transforms(image).unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            outputs = self.model(image)
            _, predicted = torch.max(outputs.data, 1)
            
        return self.classes[predicted.item()]

def main():
    """Main function to run the CLI for one-time classification."""
    parser = argparse.ArgumentParser(description="Classify night sky images.")
    parser.add_argument("data_dir", type=str, help="The absolute path to the directory of images to process.")
    parser.add_argument("--model_path", type=str, default="night_sky_model.pth", help="Path to the trained model.")
    args = parser.parse_args()

    classifier = Classifier(args.model_path)

    print(f"Processing images in: {args.data_dir}")

    # Create output directories
    for class_name in classifier.classes:
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