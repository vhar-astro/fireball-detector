"""
Scans a directory structure for image files to be processed.
"""

import argparse
import logging
from pathlib import Path
import os
from src.main import Classifier
from src.watcher import send_telegram_notification

GLOB_PATTERN = "cam[0-8][0-9]/[0-9][0-9][0-9][0-9]/[0-9][0-9][0-9][0-9][0-9][0-9]/[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]/**/*.jpg"
STATE_FILE = "processed_files.log"

# --- Logging Configuration ---
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "scanner.log"

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

def scan_for_images(base_path: Path) -> list[Path]:
    """
    Scans a directory for image files matching a specific pattern.

    Args:
        base_path: The base directory to start scanning from.

    Returns:
        A list of Path objects for the found image files.
    """
    return list(base_path.glob(GLOB_PATTERN))

import asyncio

async def main():
    parser = argparse.ArgumentParser(description="Scan a directory for images and classify them.")
    parser.add_argument("--base_path", type=str, default="/home/ruslan/Development/meteors-ml/gemini-cli-project/MeteorsStations", help="The base directory to scan for images.")
    args = parser.parse_args()

    # Ensure the state directory exists
    state_dir = Path(STATE_FILE).parent
    os.makedirs(state_dir, exist_ok=True)

    logging.info("Starting scanner...")

    classifier = Classifier()
    logging.info("Classifier initialized.")

    processed_files = load_processed_files(STATE_FILE)
    logging.info(f"Loaded {len(processed_files)} processed files.")

    meteors_stations_path = Path(args.base_path)
    image_files = scan_for_images(meteors_stations_path)
    logging.info(f"Found {len(image_files)} image files.")

    for image_file in image_files:
        if str(image_file) in processed_files:
            continue
        
        try:
            logging.info(f"Processing file: {image_file}")
            classification = classifier.classify_image(image_file)
            logging.info(f"{image_file} classified as: {classification}")

            if classification == 'fireballs':
                await send_telegram_notification(image_path=str(image_file))
            
            log_processed_file(STATE_FILE, str(image_file))
            processed_files.add(str(image_file))
        except Exception as e:
            logging.error(f"Error processing file {image_file}: {e}")
            continue
    
    logging.info("Scanner finished.")

if __name__ == '__main__':
    asyncio.run(main())