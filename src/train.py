import torch
import torch.nn as nn
import torch.optim as optim
from src.config import Config
from src.data.dataset import get_dataloaders
from src.model import NightSkyCNN

def train_model(data_dir, num_epochs=Config.NUM_EPOCHS, batch_size=Config.BATCH_SIZE, learning_rate=Config.LEARNING_RATE):
    """Trains the NightSkyCNN model."""
    train_loader, val_loader = get_dataloaders(data_dir, batch_size=batch_size)

    model = NightSkyCNN()
    criterion = nn.CrossEntropyLoss()

    # --
    # Example counts in class order [c0, c1, c2, c3]
    #['fireballs', 'meteors', 'no_fireballs', 'lightnings']
    # counts = torch.tensor([90, 454, 2000, 84], dtype=torch.float)
    # print(counts)
    # # Inverse-frequency weights (clip to avoid extremes if needed)
    # weights = 1.0 / counts
    # weights = weights / weights.sum() * len(counts)  # normalize around 1.0

    # criterion = nn.CrossEntropyLoss(weight=weights)  # plug into training loop
    # ---
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    print("Training finished.")

    # Save the model checkpoint
    torch.save(model.state_dict(), Config.TRAIN_MODEL_PATH)
    print(f"Model saved to {Config.TRAIN_MODEL_PATH}")

if __name__ == '__main__':
    import argparse

    print(f"Training started for {Config.TRAIN_MODEL_PATH}(num_epochs={Config.NUM_EPOCHS},batch_size={Config.BATCH_SIZE},learning_rate={Config.LEARNING_RATE})")

    parser = argparse.ArgumentParser(description='Train the Night Sky CNN model.')
    parser.add_argument('data_dir', type=str, help='Path to the root data directory.')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer.')
    args = parser.parse_args()

    train_model(args.data_dir, args.num_epochs, args.batch_size, args.learning_rate)