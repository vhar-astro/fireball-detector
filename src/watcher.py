import sys
import time
import os
import argparse
import asyncio
import logging
from src.config import Config
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from telegram import Bot
from src.main import Classifier

# --- Telegram Bot Configuration ---
TELEGRAM_BOT_TOKEN = ''
TELEGRAM_CHAT_ID = ''

# --- State File ---
STATE_FILE = "processed_files.log"

# --- Logging Configuration ---
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "watcher.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

def load_processed_files(path: str) -> set[str]:
    """
    Loads processed file paths from a state file.

    Args:
        path: The path to the state file.

    Returns:
        A set of processed file paths.
    """
    try:
        with open(path, 'r') as f:
            return {line.strip() for line in f}
    except FileNotFoundError:
        return set()

def log_processed_file(path: str, file_path_to_add: str):
    """
    Logs a processed file path to the state file.

    Args:
        path: The path to the state file.
        file_path_to_add: The file path to add to the state file.
    """
    with open(path, 'a') as f:
        f.write(f"{file_path_to_add}\n")
        f.flush()

async def send_telegram_notification(image_path):
    """Sends a notification with the image to the Telegram group."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logging.warning("Telegram Bot Token or Chat ID is not configured. Skipping notification.")
        return

    try:
        bot = Bot(token=TELEGRAM_BOT_TOKEN)
        caption = f"🔥 Fireball Detected!\nFile: {os.path.basename(image_path)}"
        with open(image_path, 'rb') as photo_file:
            await bot.send_photo(chat_id=TELEGRAM_CHAT_ID, photo=photo_file, caption=caption)
        logging.info(f"Sent Telegram notification for {os.path.basename(image_path)}")
    except Exception as e:
        logging.error(f"Error sending Telegram notification: {e}")

class NewImageHandler(FileSystemEventHandler):
    """Handles the event when a new file is created."""
    def __init__(self, classifier, processed_files):
        self.classifier = classifier
        self.processed_files = processed_files

    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            if event.src_path in self.processed_files:
                logging.info(f"Skipping already processed file: {event.src_path}")
                return

            logging.info(f"New image detected: {os.path.basename(event.src_path)}")
            time.sleep(1) # Wait a moment for the file to be fully written
            try:
                prediction = self.classifier.classify_image(event.src_path)
                logging.info(f"{os.path.basename(event.src_path)} classified as {prediction}")

                log_processed_file(STATE_FILE, event.src_path)
                self.processed_files.add(event.src_path)

                if prediction == 'fireballs':
                    asyncio.run(send_telegram_notification(event.src_path))

            except Exception as e:
                logging.error(f"Error processing {event.src_path}: {e}")

def main():
    """Main function to start the watcher service."""
    parser = argparse.ArgumentParser(description="Watch a folder and classify new night sky images.")
    parser.add_argument("source_dir", type=str, help="The folder to watch for new images.")
    parser.add_argument("--model_path", type=str, default=Config.MODEL_PATH, help="Path to the trained model.")
    args = parser.parse_args()

    if not os.path.isdir(args.source_dir):
        logging.error(f"Error: Source directory not found at '{args.source_dir}'")
        sys.exit(1)

    # Ensure the state directory exists
    state_dir = Path(STATE_FILE).parent
    os.makedirs(state_dir, exist_ok=True)

    processed_files = load_processed_files(STATE_FILE)
    logging.info(f"Loaded {len(processed_files)} processed files.")

    classifier = Classifier(args.model_path)
    event_handler = NewImageHandler(classifier, processed_files)
    observer = Observer()
    observer.schedule(event_handler, args.source_dir, recursive=True)
    
    logging.info(f"Watching folder: {args.source_dir}")
    logging.info("Press Ctrl+C to stop.")
    
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    main()