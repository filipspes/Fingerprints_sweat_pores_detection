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

def model():

    model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/filip/Documents/DP/YOLOv5/yolov5/runs/train/exp7/weights/best.pt')
    LOG.info("Yolov5 model weights loaded loaded from path: /home/filip/Documents/DP/YOLOv5/yolov5/runs/train/exp7/weights/best.pt")
    imgs_paths = os.listdir("/home/filip/Documents/DP/Git/DP_2021-2022/GUI/PoreDetections/parts_of_image/")
    images_to_detec = []
    for path in imgs_paths:
        img = Image.open('/home/filip/Documents/DP/Git/DP_2021-2022/GUI/PoreDetections/parts_of_image/' + path)
        images_to_detec.append(img)
    torch.cuda.empty_cache()
    results = model(images_to_detec, size=320)
    results.save()
    print(results.pandas().xyxy[0].to_json(orient="records"))  # JSON img1 predictions
    print(results.pandas().xyxy[1].to_json(orient="records"))  # JSON img1 predictions
    LOG.info("Pores detection finished")
    torch.cuda.empty_cache()



