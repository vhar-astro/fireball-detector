# Night Sky Image Classification

This project uses a convolutional neural network (CNN) to classify images of the night sky into three categories: fireballs , meteors , and no meteors.

## Setup

1.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Train the model:**

    Place your training data in a directory (e.g., `training_data`) with subdirectories for each class (e.g., `fireballs`, `meteors`, `no_fireballs`). Then run the training script:
    ```bash
    python -m src.train training_data
    ```

2.  **Classify images:**

    To classify a directory of images, run the main script:
    ```bash
    python -m src.main /path/to/your/images
    ```
