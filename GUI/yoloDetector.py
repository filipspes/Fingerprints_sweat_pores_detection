import json
import os
import torch
from PIL import Image
import logging as LOG
import config
from main import *
from datetime import datetime

LOG.basicConfig(
    level=LOG.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        LOG.FileHandler("logfile.log"),
        LOG.StreamHandler()
    ]
)


class Yolo:
    def __init__(self, confidence, iou, path_to_single_image, model_size):
        self.config = config.get_config()
        self.confidence = confidence
        self.iou = iou
        self.path_to_single_image = path_to_single_image
        self.model_size = model_size

    def detect(self, single_image, multiple_images):
        yolov5_model = self.create_model()
        images = None
        if single_image:
            images = self.load_single_image_to_detect(self.path_to_single_image)
        elif multiple_images:
            images = self.load_images_to_detect()
        torch.cuda.empty_cache()
        results = yolov5_model(images, size=320)
        results.save()
        number_of_detected_pores = self.create_json_files(results)
        LOG.info("Pores detection finished")
        torch.cuda.empty_cache()
        return number_of_detected_pores

    def create_json_files(self, results):
        final_json_object = []
        for result in results.pandas().xyxy:
            if not result.empty:
                partial_json_object = result.to_json(orient="records")
                parsed = json.loads(partial_json_object)
                i = 0
                while i < len(parsed):
                    final_json_object.append(parsed[i])
                    i += 1
        final_json_object_dumped = json.dumps(final_json_object, indent=4)
        now = datetime.now()
        curr_date_time = now.strftime("%d-%m-%Y_%H:%M:%S")
        with open(self.config.get("paths", "json_results") + "results_" + str(curr_date_time) + ".json",
                  "w") as outfile:
            outfile.write(final_json_object_dumped)
        return len(final_json_object)

    def create_model(self):
        # yolov5_model = None
        yolov5_model = None
        if self.model_size == "YOLOv5 S":
            yolov5_model = torch.hub.load('ultralytics/yolov5', 'custom',
                                        path=self.config.get("paths", "path_to_yolo_weights_S"))
            LOG.info(
                "Yolov5 model weights loaded from path: " + self.config.get("paths",
                                                                            "path_to_yolo_weights_S"))
        elif self.model_size == "YOLOv5 M":
            yolov5_model = torch.hub.load('ultralytics/yolov5', 'custom',
                                        path=self.config.get("paths", "path_to_yolo_weights_M"))
            LOG.info(
                "Yolov5 model weights loaded from path: " + self.config.get("paths",
                                                                            "path_to_yolo_weights_M"))
        elif self.model_size == "YOLOv5 L":
            yolov5_model = torch.hub.load('ultralytics/yolov5', 'custom',
                                        path=self.config.get("paths", "path_to_yolo_weights_L"))
            LOG.info(
                "Yolov5 model weights loaded from path: " + self.config.get("paths",
                                                                            "path_to_yolo_weights_L"))
        elif self.model_size == "YOLOv5 XL":
            yolov5_model = torch.hub.load('ultralytics/yolov5', 'custom',
                                        path=self.config.get("paths", "path_to_yolo_weights_XL"))
            LOG.info(
                "Yolov5 model weights loaded from path: " + self.config.get("paths",
                                                                            "path_to_yolo_weights_XL"))
        yolov5_model.conf = self.confidence / 100
        yolov5_model.iou = self.iou / 100
        return yolov5_model

    def load_images_to_detect(self):
        LOG.info("Loading images for detection")
        img_paths = os.listdir(self.config.get("paths", "path_to_parts_of_image"))
        images_to_detect = []
        for path in img_paths:
            img = Image.open(self.config.get("paths", "path_to_parts_of_image") + path)
            images_to_detect.append(img)
        return images_to_detect

    def load_single_image_to_detect(self, path_to_single_image):
        LOG.info("Loading image for detection")
        # img_paths = os.listdir(self.config.get("paths", "path_to_parts_of_image"))
        img = Image.open(path_to_single_image)
        return img
