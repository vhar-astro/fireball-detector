# Overview
This document outlines the requirements for enhancing the meteor detection ML project. The current system is limited to watching a single directory, which requires manual effort to move and manage image files. This project enhancement will automate the data ingestion process by enabling the system to recursively scan a large, predefined directory structure, identify new images, process them, and send notifications for significant events without altering the original file structure. This change will significantly improve the scalability and automation of the meteor detection pipeline.

# Core Features
- **Recursive Directory Scanning**
  - **What it does:** Automatically scans a complex, multi-level directory structure based on a pattern (`@MeteorsStations/cam[03-82]/YYYY/YYYYMM/YYYYMMDD/`).
  - **Why it's important:** Eliminates the need for manual file handling, allowing the system to process images from dozens of camera stations automatically.
  - **How it works at a high level:** A new script will have a configurable base path for `@MeteorsStations`. It will traverse the subdirectories to locate all `.jpg` image files for processing.

- **Stateful Image Processing**
  - **What it does:** Keeps track of images that have already been processed to prevent redundant analysis.
  - **Why it's important:** Ensures efficiency, saves computational resources, and prevents duplicate notifications from being sent for the same event.
  - **How it works at a high level:** A log file (e.g., `processed_files.log`) will store the paths of all processed images. Before classifying a new image, the script will verify that its path is not in this log.

- **Conditional "Fireball" Notifications**
  - **What it does:** Sends a notification via Telegram, including the image, when the ML model classifies an image as a 'fireball'.
  - **Why it's important:** Provides real-time alerts for high-priority astronomical events detected by the system.
  - **How it works at a high level:** The system will reuse the existing Telegram notification function. After an image is classified as a 'fireball', the function is called with the image's original path. The image file itself is not moved or deleted.

# User Experience
- **User Personas:** The primary user is a system operator or ML engineer responsible for maintaining and running the meteor detection service.
- **Key User Flows:**
  1. The operator configures the base directory path inside the main scanning script.
  2. The operator runs the script from the command line (e.g., `python -m src.scanner`).
  3. The script logs its progress, including directories scanned, new images found, and classification results.
  4. For any 'fireball' detections, the operator receives an immediate notification in the designated Telegram chat.
- **UI/UX Considerations:** As a command-line application, the user experience is defined by clear, concise logging. Outputs should include progress indicators, warnings for skipped directories, and explicit confirmation of 'fireball' notifications.

# Technical Architecture
- **System Components:**
  - **Scanning Module:** A new Python script (e.g., `src/scanner.py`) containing the main logic for directory traversal and processing.
  - **Classifier:** The existing `Classifier` class from `src/main.py` will be used for image analysis.
  - **Notification Function:** The existing `send_telegram_notification` async function from `src/watcher.py`.
  - **State Manager:** A simple text file (`.taskmaster/processed_files.log`) to store the paths of processed images.
  - **Logger:** A dedicated log file (e.g., `logs/scanner.log`) for recording warnings and errors.
- **Data Models:** The primary data structure is the processed files log, which will be a simple text file with one processed image path per line.
- **APIs and Integrations:** The system will continue to use the Telegram Bot API for notifications.

# Development Roadmap
- **MVP Requirements:**
  1. **Develop Scanner Logic:** Create the core functionality to recursively find all `.jpg` files within the specified `MeteorsStations` directory structure.
  2. **Implement State Tracking:** Add the mechanism to read from and append to the `processed_files.log` to track processed images.
  3. **Integrate Classifier:** In the main scanning loop, call the existing `Classifier` for any new, unprocessed images.
  4. **Refactor Notification Call:** Modify the processing loop to call the Telegram notification function when a 'fireball' is detected, ensuring the original image path is used and the file is not moved.
  5. **Add Configuration and Logging:** Implement a configurable variable for the base path and set up structured logging for warnings and errors.

# Logical Dependency Chain
1.  **Foundation:** Build the file discovery and directory traversal logic.
2.  **Statefulness:** Introduce the processed file tracking to prevent re-processing during development and in production.
3.  **Integration:** Connect the file discovery logic with the ML classification model.
4.  **Final Touches:** Implement the conditional notification and robust error logging as the final step.

# Risks and Mitigations
- **Performance Issues:** Scanning a very large number of files and directories may be slow.
  - **Mitigation:** Ensure the check against `processed_files.log` is efficient. For the MVP, a simple line-by-line check is acceptable, but for future enhancements, a more optimized data structure (like a set or a simple database) could be used.
- **Race Conditions:** The script might encounter errors if directories or files are deleted while it is scanning.
  - **Mitigation:** Wrap all file system operations (reading, checking existence) in `try...except` blocks to handle potential `FileNotFoundError` exceptions gracefully.
- **State File Integrity:** The `processed_files.log` could become corrupted or incomplete if the script is terminated unexpectedly.
  - **Mitigation:** Ensure file writes are flushed immediately to minimize the risk of data loss. For the MVP, this is a low risk, but a more robust transaction-based approach could be considered for future versions if needed.