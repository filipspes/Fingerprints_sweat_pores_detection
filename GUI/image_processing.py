import os
import re
import shutil
import subprocess
from os import listdir
from os.path import isfile, join
import cv2
from PIL import Image
from itertools import product
import main

class imageProcessing:

    def __init__(self, path):
        self.path = path

    def splitImage(self):
        path_to_high_res_image = '/home/filip/Documents/DP/Git/DP_2021-2022/GUI/PoreDetections/high_resolution_image/'
        name_of_high_res_image = "my_image_resized.jpg"
        output_dir = '/home/filip/Documents/DP/Git/DP_2021-2022/GUI/PoreDetections/parts_of_image/'
        print(self.path)
        im = Image.open(self.path)
        # im = Image.open('/home/filip/Documents/DP/Fingerprints-HR/9.jpg')

        size = im.size[0]*8, im.size[1]*8
        print("Resizing image...")
        im_resized = im.resize(size, Image.ANTIALIAS)
        print("Image resized successfully.")
        im_resized.save(path_to_high_res_image+name_of_high_res_image)
        print("Resized image saved to" + path_to_high_res_image)
        img = Image.open(path_to_high_res_image+name_of_high_res_image);
        w, h = img.size

        grid = product(range(0, h - h % 512, 512), range(0, w - w % 512, 512))
        numberOfImages = 0;
        for i, j in grid:
            box = (j, i, j + 512, i + 512)
            out = os.path.join(output_dir, f'{i}_{j}.jpg')
            img.crop(box).save(out)
            numberOfImages = numberOfImages+1;
        print("Image splitted successfully into " +str(numberOfImages) + " pictures.")
        return size

    def joinImages(size, current_working_directory):
        inputDirectory = current_working_directory+"/pores_detected/"
        file_names = os.listdir(inputDirectory)

        joinedImage = Image.new("RGB", (size[0], size[1]), "white")
        print("Re-creating image...")
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
        joinedImage.save(current_working_directory+"/final_fingerprint/pores_predicted_final_image.jpg")
        print("DONE!")

    def detectPores(self):

        file_names = sorted(os.listdir(self))
        self.save_images_path_into_file(file_names)
        print("Changing current working directory to: /home/filip/Documents/DP/YOLOv5/yolov5/")
        os.chdir('/home/filip/Documents/DP/YOLOv5/yolov5/')
        executableFile = "python detect_custom.py "
        weights = "--weight /home/filip/Documents/DP/YOLOv5/yolov5/runs/train/exp6/weights/best.pt"
        os.system(executableFile+weights)
        print("\nPore detection finished.")

    def save_images_path_into_file(file_names):
        text_file = open("/home/filip/Documents/DP/PoreDetection/images_paths.txt", "w")

        for name in file_names:
            text_file.write("/home/filip/Documents/DP/PoreDetection/parts_of_image/" + name + "\n")
        text_file.close()

    def remove_content_of_folder(folder): # OK
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

    def remove_content_of_folders(self): #OK
        imageProcessing.remove_content_of_folder('/home/filip/Documents/DP/Git/DP_2021-2022/GUI/PoreDetections/parts_of_image/')
        imageProcessing.remove_content_of_folder('/home/filip/Documents/DP/Git/DP_2021-2022/GUI/PoreDetections/pores_detected/')
        imageProcessing.remove_content_of_folder('/home/filip/Documents/DP/Git/DP_2021-2022/GUI/PoreDetections/high_resolution_image/')
        imageProcessing.remove_content_of_folder('/home/filip/Documents/DP/Git/DP_2021-2022/GUI/PoreDetections/final_fingerprint/')

# if __name__ == '__main__':
#     current_working_directory = os.getcwd()
#     # print(current_working_directory)
#     print("Removing content of folders...")
#     # remove_content_of_folders('/home/filip/Documents/DP/Git/DP_2021-2022/GUI/PoreDetections/parts_of_image/')
#     # remove_content_of_folders('/home/filip/Documents/DP/Git/DP_2021-2022/GUI/PoreDetections/pores_detected/')
#     # remove_content_of_folders('/home/filip/Documents/DP/Git/DP_2021-2022/GUI/PoreDetections/high_resolution_image/')
#     # remove_content_of_folders('/home/filip/Documents/DP/Git/DP_2021-2022/GUI/PoreDetections/final_fingerprint/')
#     # inputImagePath = input("Please enter image path: ")
#     # size = splitImage(current_working_directory, inputImagePath, current_working_directory+"/parts_of_image/", 512)
#     # detectPores(current_working_directory+"/parts_of_image/")
#     # joinImages(size, current_working_directory)
