import os
import torch
from PIL import Image
import logging as LOG
import config
from main import *

LOG.basicConfig(
    level=LOG.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        LOG.FileHandler("logfile.log"),
        LOG.StreamHandler()
    ]
)


class Yolo:
    def __init__(self, confidence, iou):
        self.config = config.get_config()
        self.confidence = confidence
        self.iou = iou

    def detect(self):
        yolov5_model = self.create_model()
        images = self.load_images_to_detect()
        torch.cuda.empty_cache()
        results = yolov5_model(images, size=320)
        results.save()
        LOG.info("Pores detection finished")
        torch.cuda.empty_cache()

    def create_json_files(results):
        results.pandas().xyxy[0].to_json(orient="records")  # JSON img1 predictions
        results.pandas().xyxy[0].to_json(orient="records")  # JSON img2 predictions

    def create_model(self):

        yolov5_model = torch.hub.load('ultralytics/yolov5', 'custom',
                                      path=self.config.get("paths", "path_to_yolo_weights"))
        yolov5_model.conf = self.confidence/100
        yolov5_model.iou = self.iou/100
        LOG.info(
            "Yolov5 model weights loaded loaded from path: " + self.config.get("paths",
                                                                               "path_to_high_resolution_image"))
        return yolov5_model

    def load_images_to_detect(self):
        LOG.info("Loading images for detection")
        img_paths = os.listdir(self.config.get("paths", "path_to_parts_of_image"))
        images_to_detect = []
        for path in img_paths:
            img = Image.open(self.config.get("paths", "path_to_parts_of_image") + path)
            images_to_detect.append(img)
        return images_to_detect
