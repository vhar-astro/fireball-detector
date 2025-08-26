import torch
import torch.nn as nn
import torch.optim as optim
from src.data.dataset import get_dataloaders
from src.model import NightSkyCNN

def train_model(data_dir, num_epochs=10, batch_size=32, learning_rate=0.001):
    """Trains the NightSkyCNN model."""
    train_loader, val_loader = get_dataloaders(data_dir, batch_size=batch_size)

    model = NightSkyCNN()
    criterion = nn.CrossEntropyLoss()
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
    torch.save(model.state_dict(), Config.MODEL_PATH)
    print(f"Model saved to {Config.MODEL_PATH}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train the Night Sky CNN model.')
    parser.add_argument('data_dir', type=str, help='Path to the root data directory.')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer.')
    args = parser.parse_args()

    train_model(args.data_dir, args.num_epochs, args.batch_size, args.learning_rate)