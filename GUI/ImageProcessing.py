import os
import shutil
from PIL import Image
from itertools import product
import logging as LOG

import AppConfig

LOG.basicConfig(
    level=LOG.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        LOG.FileHandler("logfile.log"),
        LOG.StreamHandler()
    ]
)


def remove_content_of_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            LOG.error('Failed to delete %s. Reason: %s' % (file_path, e))


class ImageProcessing:

    def __init__(self, path):
        self.path = path
        self.initial_size = None
        self.config = AppConfig.get_config()

    def split_image(self):
        path_to_high_res_image = self.config.get("paths", "ROOT_DIR") + 'PoreDetections/high_resolution_image/'
        name_of_high_res_image = self.config.get("names", "name_of_high_resolution_image")
        path_to_parts_of_image = self.config.get("paths", "ROOT_DIR") + 'PoreDetections/parts_of_image/'
        im = Image.open(self.path)
        size = im.size[0] * 8, im.size[1] * 8
        self.initial_size = size
        LOG.info("Resizing image...")
        im_resized = im.resize(size, Image.ANTIALIAS)
        LOG.info("Image resized successfully.")
        im_resized.save(
            self.config.get("paths", "ROOT_DIR") + 'PoreDetections/high_resolution_image/' + name_of_high_res_image)
        LOG.info("Resized image saved to" + path_to_high_res_image)
        img = Image.open(path_to_high_res_image + name_of_high_res_image);
        w, h = img.size

        grid = product(range(0, h - h % 512, 512), range(0, w - w % 512, 512))
        number_of_images = 0
        for i, j in grid:
            box = (j, i, j + 512, i + 512)
            out = os.path.join(path_to_parts_of_image, f'{i}_{j}.jpg')
            img.crop(box).save(out)
            number_of_images = number_of_images + 1
        LOG.info(f"The image has been split into {number_of_images} pictures.")  # F-string added
        return size

    def join_images(self, size, yolo):
        if yolo:
            input_directory = self.config.get("paths", "ROOT_DIR") + 'runs/detect/exp/'
        else:
            input_directory = self.config.get("paths", "ROOT_DIR") + 'PoreDetections/pores_detected/'
        file_names = os.listdir(input_directory)

        joined_image = Image.new("RGB", (size[0], size[1]), "white")
        LOG.info("Re-creating image...")
        counter = 0
        for name in file_names:
            counter = counter + 1
            split_by_dot = name.split(".", )
            split = split_by_dot[0].split("_", )
            num = ""
            for c in name:
                if c.isdigit():
                    num = num + c
            image_part = Image.open(input_directory + name)
            if len(split) >= 2:
                joined_image.paste(image_part, (int(split[1]), int(split[0])))
        joined_image.save(self.config.get("paths", "ROOT_DIR") + 'PoreDetections/final_fingerprint/'
                          + self.config.get("names", "name_of_detected_final_image"))
        LOG.info("All Done")

    def remove_content_of_folders(self):
        LOG.info('Deleting content of a folder: /parts_of_image/')
        remove_content_of_folder(self.config.get("paths", "ROOT_DIR") + 'PoreDetections/parts_of_image/')
        LOG.info('Deleting content of a folder: /pores_detected/')
        remove_content_of_folder(self.config.get("paths", "ROOT_DIR") + 'PoreDetections/pores_detected/')
        LOG.info('Deleting content of a folder: /high_resolution_image/')
        remove_content_of_folder(self.config.get("paths", "ROOT_DIR") + 'PoreDetections/high_resolution_image/')
        LOG.info('Deleting content of a folder: /final_fingerprint/')
        remove_content_of_folder(self.config.get("paths", "ROOT_DIR") + 'PoreDetections/final_fingerprint/')
