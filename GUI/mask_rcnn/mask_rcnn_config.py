from mrcnn.config import Config

class FingeprintPoresConfig(Config):

    NAME = "fingerprints"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_RESIZE_MODE = "square"
    STEPS_PER_EPOCH = 16
    VALIDATION_STEPS = 5
    BACKBONE = 'resnet101'
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    TRAIN_ROIS_PER_IMAGE = 20
    MAX_GT_INSTANCES = 20
    POST_NMS_ROIS_INFERENCE = 500
    POST_NMS_ROIS_TRAINING = 1000

class InferenceConfig(FingeprintPoresConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    DETECTION_MIN_CONFIDENCE = 0.5
    PRE_NMS_LIMIT = 6000
    BACKBONE = 'resnet50'
