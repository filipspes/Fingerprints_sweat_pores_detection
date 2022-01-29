import os
import torch
from PIL import Image
import logging as LOG

LOG.basicConfig(
    level=LOG.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        LOG.FileHandler("logfile.log"),
        LOG.StreamHandler()
    ]
)

def detect():
    yolov5_model = create_model()
    images = load_images_to_detect()
    torch.cuda.empty_cache()
    results = yolov5_model(images, size=320)
    results.save()
    LOG.info("Pores detection finished")
    torch.cuda.empty_cache()

def create_json_files(results):
    results.pandas().xyxy[0].to_json(orient="records")  # JSON img1 predictions
    results.pandas().xyxy[0].to_json(orient="records")  # JSON img2 predictions

def create_model():
    yolov5_model = torch.hub.load('ultralytics/yolov5', 'custom',
                           path='/home/filip/Documents/DP/YOLOv5/yolov5/runs/train/exp7/weights/best.pt')
    LOG.info("Yolov5 model weights loaded loaded from path: ""/home/filip/Documents/DP/YOLOv5/yolov5/runs/train/exp7/weights/best.pt")
    return yolov5_model

def load_images_to_detect():
    LOG.info("Loading images for detection")
    img_paths = os.listdir("/home/filip/Documents/DP/Git/DP_2021-2022/GUI/PoreDetections/parts_of_image/")
    images_to_detect = []
    for path in img_paths:
        img = Image.open('/home/filip/Documents/DP/Git/DP_2021-2022/GUI/PoreDetections/parts_of_image/' + path)
        images_to_detect.append(img)
    return images_to_detect
