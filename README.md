# Night Sky Image Classification

This project uses a convolutional neural network (CNN) to classify images of the night sky into four categories: fireballs, meteors, no_fireballs, and trash.

## Setup

1.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

2.  **Install dependencies:**

    *   **For systems with a compatible NVIDIA GPU:**
        ```bash
        pip install -r requirements.txt
        ```

    *   **For systems with CPU only:**
        ```bash
        pip install -r requirements_cpu.txt
        ```

## Usage

1.  **Train the model:**

    Place your training data in a directory (e.g., `training_data`) with subdirectories for each class (e.g., `fireballs`, `meteors`, `no_fireballs`, `trash`). Then run the training script:
    ```bash
    python -m src.train training_data
    ```

2.  **Classify a directory of images (one-time):**
    ```bash
    python -m src.main /path/to/your/images
    ```

3.  **Run the automatic watcher service:**

    To monitor a folder for new images and classify them automatically, run the watcher script. This will also send Telegram notifications for detected fireballs.

    **Before you start:** You must edit the `src/watcher.py` file and replace the placeholder values for `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` with your own credentials.

    ```bash
    python -m src.watcher /path/to/your/source_folder
    ```

## Building Executables

This project can be packaged into a single executable file using PyInstaller.

1.  **Install PyInstaller:**
    ```bash
    pip install pyinstaller
    ```

2.  **Build the executable:**

    The entry point for the executable is `run.py`. To build the watcher service, you will need to modify `run.py` to call `watcher.main()` instead of `main.main()`.

    ```bash
    pyinstaller --name classify_sky_watcher --onefile --add-data "night_sky_model.pth:." run.py
    ```
