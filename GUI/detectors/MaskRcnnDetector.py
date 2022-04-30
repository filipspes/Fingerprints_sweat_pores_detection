import skimage

from main import *
from config import MaskRcnnConfig as mask_rcnn_config
# import mrcnn.model as modellib
from mask_rcnn import model as modellib
import time


LOG.basicConfig(
    level=LOG.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        LOG.FileHandler("../logfile.log"),
        LOG.StreamHandler()
    ]
)


class MaskRCNN:
    def __init__(self, path_to_image_to_detect, selected_model, detection_confidence, max_detections, backbone):
        self.config = app_config.get_config()
        self.path_to_image_to_detect = path_to_image_to_detect
        self.selected_model = selected_model
        self.detection_confidence = detection_confidence
        self.max_detections = max_detections
        self.backbone = backbone
        self.number_of_detected_pores = 0
        self.start_time = time.time()

    def load_mask_rcnn_model(self):
        inference_config = mask_rcnn_config.InferenceConfig()
        inference_config.DETECTION_MIN_CONFIDENCE = self.detection_confidence / 100
        inference_config.DETECTION_MAX_INSTANCES = self.max_detections
        inference_config.BACKBONE = self.backbone.split("_")[0].lower()
        inference_config.display()
        mask_rcnn_model = modellib.MaskRCNN(mode='inference',
                                            config=inference_config,
                                            model_dir='../')
        path_to_model = self.config.get("paths", "ROOT_DIR") + 'mrcnn_models/mask_rcnn_fingerprints_' + self.selected_model.lower() + '.h5'
        mask_rcnn_model.load_weights(path_to_model, by_name=True)
        LOG.info("Mask R-CNN model weights loaded from path: " + path_to_model)
        return mask_rcnn_model

    def detect_fingeprint_pores_on_single_image(self):
        mask_rcnn_model_loaded = self.load_mask_rcnn_model()
        class_names = ['BG', 'pore']
        img = skimage.io.imread(self.path_to_image_to_detect)
        img_arr = np.array(img)
        results = mask_rcnn_model_loaded.detect([img_arr], verbose=1)
        r = results[0]
        visualize_mask_rcnn_detections.display_instances(img, "detected_block_of_image.jpg", r['rois'], r['masks'],
                                                         r['class_ids'],
                                                         class_names, r['scores'])
        self.number_of_detected_pores = len(r['rois'])

    def detect_fingeprint_pores_on_multiple_images(self):
        mask_rcnn_model_loaded = self.load_mask_rcnn_model()
        self.start_time = time.time()
        class_names = ['BG', 'pore']
        parts_of_image = self.config.get("paths", "ROOT_DIR") + 'PoreDetections/parts_of_image/'
        image_paths = []
        file_names = []
        number_of_detected_pores = 0
        for filename in os.listdir(parts_of_image):
            if os.path.splitext(filename)[1].lower() in ['.png', '.jpg', '.jpeg']:
                image_paths.append(os.path.join(parts_of_image, filename))
                file_names.append(filename)

        for image_path, file_name in zip(image_paths, file_names):
            img = skimage.io.imread(image_path)
            if len(img.shape) < 3:
                print("GrayscaleImage: " + image_path)
            if np.mean(img) == 255:
                shutil.copyfile(image_path,
                                self.config.get("paths", "ROOT_DIR") + 'PoreDetections/pores_detected/' + file_name)
            else:
                img_arr = np.array(img)
                results = mask_rcnn_model_loaded.detect([img_arr], verbose=1)
                r = results[0]
                visualize_mask_rcnn_detections.display_instances(img, file_name, r['rois'], r['masks'], r['class_ids'],
                                                                 class_names, r['scores'], figsize=(5, 5))
                print(len(r['rois']))
                number_of_detected_pores = number_of_detected_pores + len(r['rois'])

        self.number_of_detected_pores = number_of_detected_pores
