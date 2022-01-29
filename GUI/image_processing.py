import os
import shutil
from PIL import Image
from itertools import product
import logging as LOG

LOG.basicConfig(
    level=LOG.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        LOG.FileHandler("logfile.log"),
        LOG.StreamHandler()
    ]
)

class imageProcessing:

    def __init__(self, path):
        self.path = path

    def splitImage(self):
        path_to_high_res_image = '/home/filip/Documents/DP/Git/DP_2021-2022/GUI/PoreDetections/high_resolution_image/'
        name_of_high_res_image = "my_image_resized.jpg"
        output_dir = '/home/filip/Documents/DP/Git/DP_2021-2022/GUI/PoreDetections/parts_of_image/'
        im = Image.open(self.path)

        size = im.size[0]*8, im.size[1]*8
        LOG.info("Resizing image...")
        im_resized = im.resize(size, Image.ANTIALIAS)
        LOG.info("Image resized successfully.")
        im_resized.save(path_to_high_res_image+name_of_high_res_image)
        LOG.info("Resized image saved to" + path_to_high_res_image)
        img = Image.open(path_to_high_res_image+name_of_high_res_image);
        w, h = img.size

        grid = product(range(0, h - h % 512, 512), range(0, w - w % 512, 512))
        numberOfImages = 0;
        for i, j in grid:
            box = (j, i, j + 512, i + 512)
            out = os.path.join(output_dir, f'{i}_{j}.jpg')
            img.crop(box).save(out)
            numberOfImages = numberOfImages+1
        LOG.info("Image splitted successfully into " +str(numberOfImages) + " pictures.")
        return size

    def joinImages(self, size):
        inputDirectory = '/home/filip/Documents/DP/Git/DP_2021-2022/GUI/runs/detect/exp/'
        file_names = os.listdir(inputDirectory)

        joinedImage = Image.new("RGB", (size[0], size[1]), "white")
        LOG.info("Re-creating image...")
        counter = 0
        for name in file_names:
            counter=counter+1
            splittedByDot = name.split(".", )
            splitted = splittedByDot[0].split("_", )
            num = ""
            for c in name:
                if c.isdigit():
                    num = num + c
            imagePart = Image.open(inputDirectory+name)
            if(len(splitted) >= 2):
                joinedImage.paste(imagePart, (int(splitted[1]), int(splitted[0])))
        joinedImage.save('/home/filip/Documents/DP/Git/DP_2021-2022/GUI/PoreDetections/final_fingerprint/pores_predicted_final_image.jpg')
        LOG.info("All Done")

    def remove_content_of_folder(self, folder_path): # OK
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                LOG.error('Failed to delete %s. Reason: %s' % (file_path, e))

    def remove_content_of_folders(self):
        LOG.info('Deleting content of folder: /parts_of_image/')
        self.remove_content_of_folder('/home/filip/Documents/DP/Git/DP_2021-2022/GUI/PoreDetections/parts_of_image/')
        LOG.info('Deleting content of folder: /pores_detected/')
        self.remove_content_of_folder('/home/filip/Documents/DP/Git/DP_2021-2022/GUI/PoreDetections/pores_detected/')
        LOG.info('Deleting content of folder: /high_resolution_image/')
        self.remove_content_of_folder('/home/filip/Documents/DP/Git/DP_2021-2022/GUI/PoreDetections/high_resolution_image/')
        LOG.info('Deleting content of folder: /final_fingerprint/')
        self.remove_content_of_folder('/home/filip/Documents/DP/Git/DP_2021-2022/GUI/PoreDetections/final_fingerprint/')
