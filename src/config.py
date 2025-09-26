class Config:
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
    