import argparse
import os
import torch
from PIL import Image
from src.data.dataset import data_transforms
from src.model import NightSkyCNN

def classify_image(model, image_path):
    """Classifies a single image."""
    image = Image.open(image_path).convert("RGB")
    image = data_transforms(image).unsqueeze(0)  # Add batch dimension
    
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
        
    return predicted.item()

def main():
    """Main function to run the CLI."""
    parser = argparse.ArgumentParser(description="Classify night sky images.")
    parser.add_argument("data_dir", type=str, help="The absolute path to the directory of images to process.")
    parser.add_argument("--model_path", type=str, default="night_sky_model.pth", help="Path to the trained model.")
    args = parser.parse_args()

    # Load the trained model
    model = NightSkyCNN()
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    print(f"Processing images in: {args.data_dir}")

    # Create output directories
    for class_name in ['fireballs', 'meteors', 'no_fireballs', 'trash']:
        os.makedirs(os.path.join(args.data_dir, class_name), exist_ok=True)

    # Classify and move images
    for fname in os.listdir(args.data_dir):
        if fname.lower().endswith(".jpg"):
            image_path = os.path.join(args.data_dir, fname)
            prediction = classify_image(model, image_path)
            class_name = ['fireballs', 'meteors', 'no_fireballs', 'trash'][prediction]
            dest_path = os.path.join(args.data_dir, class_name, fname)
            os.rename(image_path, dest_path)
            print(f"Moved {fname} to {class_name}")

if __name__ == "__main__":
    main()
