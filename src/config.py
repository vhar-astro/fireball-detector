class Config:
    CLASSES = ['fireballs', 'meteors', 'no_fireballs', 'lightnings']
    CLASSES_SIZE = len(CLASSES)
    MODEL_PATH = 'night_sky_model_v11.pth' # used for main and watcher scripts
    TRAIN_MODEL_PATH = 'night_sky_model_v11.pth' # used to saved model file by train.py script
    RESNET18_MODEL_PATH = 'resnet18_trained_v1.pth' # used for main and watcher scripts
    RESNET18_TRAIN_MODEL_PATH = 'resnet18_trained_v1.pth' # used to saved model file by train.py script
    NUM_EPOCHS = 60
    BATCH_SIZE = 8
    LEARNING_RATE = 0.001
    