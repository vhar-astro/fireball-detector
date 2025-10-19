class Config:
    # ========== Image Classification Settings ==========
    # CLASSES = ['fireballs', 'meteors', 'no_fireballs', 'lightnings']  # 4 classes
    CLASSES = ['fireballs', 'no_fireballs'] # 2 classes
    CLASSES_SIZE = len(CLASSES)
    NUM_EPOCHS = 30
    BATCH_SIZE = 8
    MODEL_PATH = 'night_sky_model_2classes_v1.pth' # used for main and watcher scripts
    TRAIN_MODEL_PATH = f"night_sky_model_2classes_v1.pth" # used to saved model file by train.py script
    RESNET18_MODEL_PATH = f"resnet18_trained_2classes_v2_bs{BATCH_SIZE}_ne{NUM_EPOCHS}.pth" # used for main and watcher scripts
    RESNET18_TRAIN_MODEL_PATH = f"resnet18_trained_2classes_v2_bs{BATCH_SIZE}_ne{NUM_EPOCHS}.pth" # used to saved model file by train.py script
    LEARNING_RATE = 0.001

    # ========== Object Detection Settings ==========
    # Dataset paths
    OD_DATASET_PATH = 'annotated_images'  # Path to YOLO format dataset
    OD_CLASSES_FILE = 'annotated_images/classes.txt'  # Path to classes.txt file

    # Training hyperparameters
    # Optimized for NVIDIA RTX 2070 (8GB VRAM) - 64.4% GPU utilization (~5273 MB)
    # Profiled on 2025-10-19 using test_gpu_memory.py
    OD_NUM_EPOCHS = 50
    OD_BATCH_SIZE = 4  # Optimal for RTX 2070: 64.4% GPU usage (5273 MB / 8192 MB)
    OD_LEARNING_RATE = 0.001
    OD_MOMENTUM = 0.9
    OD_WEIGHT_DECAY = 0.0005
    OD_LR_STEP_SIZE = 10  # Learning rate decay step
    OD_LR_GAMMA = 0.1  # Learning rate decay factor

    # Model settings
    OD_BACKBONE = 'resnet50'  # Backbone for Faster R-CNN: resnet50, resnet101, mobilenet_v3
    OD_TRAINABLE_BACKBONE_LAYERS = 3  # Optimal for RTX 2070: balances speed and accuracy
    OD_MIN_SIZE = 800  # Optimal for RTX 2070: maintains accuracy while fitting in memory
    OD_MAX_SIZE = 1333  # Optimal for RTX 2070: standard COCO size for good accuracy

    # RPN (Region Proposal Network) settings
    OD_RPN_PRE_NMS_TOP_N_TRAIN = 2000  # Number of proposals to keep before NMS during training
    OD_RPN_PRE_NMS_TOP_N_TEST = 1000  # Number of proposals to keep before NMS during testing
    OD_RPN_POST_NMS_TOP_N_TRAIN = 2000  # Number of proposals to keep after NMS during training
    OD_RPN_POST_NMS_TOP_N_TEST = 1000  # Number of proposals to keep after NMS during testing
    OD_RPN_NMS_THRESH = 0.7  # NMS threshold for RPN
    OD_RPN_FG_IOU_THRESH = 0.7  # IoU threshold for positive anchors
    OD_RPN_BG_IOU_THRESH = 0.3  # IoU threshold for negative anchors

    # Box detection settings
    OD_BOX_SCORE_THRESH = 0.05  # Score threshold for box predictions
    OD_BOX_NMS_THRESH = 0.5  # NMS threshold for box predictions
    OD_BOX_DETECTIONS_PER_IMG = 100  # Maximum number of detections per image
    OD_BOX_FG_IOU_THRESH = 0.5  # IoU threshold for positive boxes
    OD_BOX_BG_IOU_THRESH = 0.5  # IoU threshold for negative boxes

    # Evaluation settings
    OD_IOU_THRESHOLD = 0.5  # IoU threshold for evaluation metrics

    # Model checkpoint paths
    OD_MODEL_PATH = f'faster_rcnn_bs{OD_BATCH_SIZE}_ne{OD_NUM_EPOCHS}.pth'
    OD_BEST_MODEL_PATH = f'faster_rcnn_best_bs{OD_BATCH_SIZE}_ne{OD_NUM_EPOCHS}.pth'
    OD_CHECKPOINT_DIR = 'checkpoints'  # Directory to save checkpoints

    # Export settings
    OD_ONNX_PATH = 'faster_rcnn_model.onnx'
    OD_TORCHSCRIPT_PATH = 'faster_rcnn_model.pt'

    # Data loading
    # Optimized for RTX 2070 system (typically 6-8 core CPU)
    OD_NUM_WORKERS = 6  # Optimal for RTX 2070: 6 workers for efficient data loading without CPU bottleneck
    OD_VAL_SPLIT = 0.2  # Validation split ratio

    # Logging
    OD_PRINT_FREQ = 10  # Print frequency during training
    OD_SAVE_FREQ = 5  # Save checkpoint every N epochs
