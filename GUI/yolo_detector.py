import glob
import os

import cv2
import torch
from PIL import Image


def model():
    # Model
    # model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/filip/Documents/DP/YOLOv5/yolov5/runs/train/exp7/weights/best.pt')

    # model.load_state_dict(torch.load('/home/filip/Documents/DP/YOLOv5/yolov5/runs/train/exp7/weights/best.pt')['model'].state_dict())

    # Images
    # img1 = Image.open('/home/filip/Documents/DP/FP_Parts/1.jpg')  # PIL image
    # img2 = cv2.imread('/home/filip/Documents/DP/FP_Parts/2.jpg')[..., ::-1]  # OpenCV image (BGR to RGB)
    imgs_paths = os.listdir("/home/filip/Documents/DP/Git/DP_2021-2022/GUI/PoreDetections/parts_of_image/")
    # imgs = list_directory()
    imgs = []
    for path in imgs_paths:
        img = Image.open('/home/filip/Documents/DP/Git/DP_2021-2022/GUI/PoreDetections/parts_of_image/' + path)
        imgs.append(img)
    torch.cuda.empty_cache()
    results = model(imgs, size=320)
    results.save()
    torch.cuda.empty_cache()

    # Inference
    # results = model(imgs)

    # Results
    # results.print()
    # results.save()  # or .show()

    results.xyxy[0]  # img1 predictions (tensor)
    results.pandas().xyxy[0]  # img1 predictions (pandas)

def list_directory():
    images = [cv2.imread(file) for file in glob.glob("/home/filip/Documents/DP/Git/DP_2021-2022/GUI/PoreDetections/parts_of_image/*.jpg")]
    return images
