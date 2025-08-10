import sys
import time
import os
import argparse
import asyncio
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from telegram import Bot
from src.main import Classifier

# --- Telegram Bot Configuration ---
# IMPORTANT: Replace with your actual Bot Token and Chat ID
TELEGRAM_BOT_TOKEN = 'YOUR_BOT_TOKEN'
TELEGRAM_CHAT_ID = 'YOUR_CHAT_ID'

async def send_telegram_notification(image_path):
    """Sends a notification with the image to the Telegram group."""
    if TELEGRAM_BOT_TOKEN == 'YOUR_BOT_TOKEN' or TELEGRAM_CHAT_ID == 'YOUR_CHAT_ID':
        print("\n--- WARNING ---")
        print("Telegram Bot Token or Chat ID is not configured.")
        print("Please edit src/watcher.py to send notifications.")
        print("---------------")
        return

    try:
        bot = Bot(token=TELEGRAM_BOT_TOKEN)
        caption = f"🔥 Fireball Detected!\nFile: {os.path.basename(image_path)}"
        await bot.send_photo(chat_id=TELEGRAM_CHAT_ID, photo=open(image_path, 'rb'), caption=caption)
        print(f"Sent Telegram notification for {os.path.basename(image_path)}")
    except Exception as e:
        print(f"Error sending Telegram notification: {e}")

class NewImageHandler(FileSystemEventHandler):
    """Handles the event when a new file is created."""
    def __init__(self, classifier, source_path):
        self.classifier = classifier
        self.source_path = source_path

    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            print(f"New image detected: {os.path.basename(event.src_path)}")
            time.sleep(1) # Wait a moment for the file to be fully written
            try:
                prediction = self.classifier.classify_image(event.src_path)
                dest_folder = os.path.join(self.source_path, prediction)
                os.makedirs(dest_folder, exist_ok=True)
                dest_path = os.path.join(dest_folder, os.path.basename(event.src_path))
                os.rename(event.src_path, dest_path)
                print(f"Moved {os.path.basename(event.src_path)} to {prediction}")

                if prediction == 'fireballs':
                    asyncio.run(send_telegram_notification(dest_path))

            except Exception as e:
                print(f"Error processing {event.src_path}: {e}")

def main():
    """Main function to start the watcher service."""
    parser = argparse.ArgumentParser(description="Watch a folder and classify new night sky images.")
    parser.add_argument("source_dir", type=str, help="The folder to watch for new images.")
    parser.add_argument("--model_path", type=str, default="night_sky_model.pth", help="Path to the trained model.")
    args = parser.parse_args()

    if not os.path.isdir(args.source_dir):
        print(f"Error: Source directory not found at '{args.source_dir}'")
        sys.exit(1)

    classifier = Classifier(args.model_path)
    event_handler = NewImageHandler(classifier, args.source_dir)
    observer = Observer()
    observer.schedule(event_handler, args.source_dir, recursive=False)
    
    print(f"Watching folder: {args.source_dir}")
    print("Press Ctrl+C to stop.")
    
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    main()
