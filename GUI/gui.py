import json
import cv2
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtGui import QPixmap
from PIL import Image
from MainWindow import *
import os
from os.path import exists
import shutil
import time
import logging as LOG
from detectors import yolo_detector, mask_rcnn_detector
from custom_utils import image_processing, image_viewer
import sys
import numpy as np
from PyQt5 import QtCore
import app_config as app_config
from PyQt5.QtWidgets import *
from custom_utils import file_explorer as FilesExplorer

LOG.basicConfig(
    level=LOG.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        LOG.FileHandler("logfile.log"),
        LOG.StreamHandler()
    ]
)


def remove_content_of_folder_runs():
    config = app_config.get_config()
    for filename in os.listdir(config.get("paths", "ROOT_DIR") + 'runs/detect/'):
        file_path = os.path.join(config.get("paths", "ROOT_DIR") + 'runs/detect/', filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            LOG.error('Failed to delete %s. Reason: %s' % (file_path, e))


class MyWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyWindow, self).__init__(parent)
        self.setupUi(self)
        self.app_config = app_config.get_config()
        self.seconWindow = image_viewer.Window()
        self.openDetectedImageButton.clicked.connect(self.show_new_window)
        self.full_image_opened = False
        self.block_of_image_opened = False
        self.image_is_detected = False
        self.json_is_loaded = False
        self.json = None
        self.mask_turned_on = False

    def application_startup_settings(self):
        self.MaskRcnnBackboneComboBox.addItem("Resnet50_heads")
        self.MaskRcnnBackboneComboBox.addItem("Resnet50_all")
        self.MaskRcnnBackboneComboBox.addItem("Resnet101_heads")
        self.MaskRcnnBackboneComboBox.addItem("Resnet101_all")
        self.YoloModelsComboBox.addItem("Nano")
        self.YoloModelsComboBox.addItem("Small")
        self.YoloModelsComboBox.addItem("Medium")
        self.YoloModelsComboBox.addItem("Large")
        self.YoloModelsComboBox.addItem("XLarge")
        self.configurationGroupbox.setEnabled(True)
        self.detectorsTypesGroupBox.setEnabled(True)
        self.openDetectedImageButton.setEnabled(False)
        self.YoloModelsComboBox.setEnabled(False)
        self.MaskRcnnBackboneComboBox.setEnabled(False)
        self.Yolov5DetectorCheckBox.setChecked(True)
        self.showRealMaskCheckBox.setEnabled(False)

    def load_json_button_handle(self):
        file_explorer = FilesExplorer.FileExplorer(custom_filter="Json (*.json)")
        file_path = file_explorer.openFileNameDialog()
        if file_path is None or file_path == "":
            return
        else:
            f = open(file_path)
            data = json.load(f)
            f.close()
            self.json_is_loaded = True
            self.json = data
            self.mask_turned_on = False

    def confidence_slider_event(self):
        self.confidenceLabel.setText('Confidence: ' + str((self.confidenceSlider.value() + 1) / 100))

    def max_detection_slider_event(self):
        self.maxDetectionsLabel.setText('Max detections: ' + str((self.maxDetectionsSlider_2.value())))

    def remove_content_of_folder_runs(self):
        for filename in os.listdir(self.app_config.get("paths", "ROOT_DIR") + 'runs/detect/'):
            file_path = os.path.join(self.app_config.get("paths", "ROOT_DIR") + 'runs/detect/', filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                LOG.error('Failed to delete %s. Reason: %s' % (file_path, e))

    def detect_fingerprint_pores_yolo(self, full_image):
        yolo = yolo_detector.Yolo(self.confidenceSlider.value(), self.maxDetectionsSlider_2.value(), RUN_PATH,
                                 self.YoloModelsComboBox.currentText())
        remove_content_of_folder_runs()
        path_to_model = self.app_config.get("paths",
                                            "ROOT_DIR") + '/yolov5_models/YOLOv5_' + self.YoloModelsComboBox.currentText() + '_weights.pt'
        if not exists(path_to_model):
            msg = warning_message_box_popup("Selected model does not exists in folder: yolov5_models",
                                            msgbox_type='error')
            msg.exec_()
            return
        start_time = time.time()
        if full_image:
            image_proc = image_processing.ImageProcessing(RUN_PATH)
            image_proc.remove_content_of_folders()
            size = image_proc.split_image()
            number_of_detected_pores = yolo.detect(False, True)
            image_proc.join_images(size, True)
            self.create_pixmap_detected_image(
                self.app_config.get("paths",
                                    "ROOT_DIR") + 'PoreDetections/final_fingerprint/pores_predicted_final_image.jpg',
                True)
            end_time = time.time()
            detection_time = round((end_time - start_time), 2)
            LOG.info("YOLOv5 XLarge:")
            LOG.info("-Detected " + str(number_of_detected_pores) + " sweat pores out of 1086")
            LOG.info("-Detection took: " + str(detection_time) + " seconds")
            # detection_time = round((end_time - start_time), 2)
            self.number_of_pores_detected_label.setText(
                str(number_of_detected_pores) + " pores detected in " + str(detection_time) + ' seconds')
        else:
            number_of_detected_pores = yolo.detect(True, False)
            list_of_images = os.listdir(
                '/home/filip/Documents/DP/Git/DP_2021-2022/GUI2/DP_2021-2022/GUI/runs/detect/exp/')
            shutil.copyfile(
                '/home/filip/Documents/DP/Git/DP_2021-2022/GUI2/DP_2021-2022/GUI/runs/detect/exp/' + list_of_images[0],
                '/home/filip/Documents/DP/Git/DP_2021-2022/GUI2/DP_2021-2022/GUI/PoreDetections'
                '/block_of_image_detected/detected_image.jpg')
            self.create_pixmap_detected_image(
                '/home/filip/Documents/DP/Git/DP_2021-2022/GUI2/DP_2021-2022/GUI/PoreDetections'
                '/block_of_image_detected/detected_image'
                '.jpg',
                False)
            end_time = time.time()
            detection_time = round((end_time - start_time), 2)
            self.number_of_pores_detected_label.setText(
                str(number_of_detected_pores) + " pores detected in " + str(detection_time) + ' seconds')
            self.showRealMaskCheckBox.setEnabled(True)
        self.openDetectedImageButton.setEnabled(True)

    def detect_fingerprint_pores_mask_rcnn(self, full_image):
        image_proc = image_processing.ImageProcessing(RUN_PATH)
        image_proc.remove_content_of_folders()
        mask_rcnn = mask_rcnn_detector.MaskRCNN(RUN_PATH, self.MaskRcnnBackboneComboBox.currentText(),
                                              self.confidenceSlider.value(), self.maxDetectionsSlider_2.value(),
                                              self.MaskRcnnBackboneComboBox.currentText())

        path_to_model = self.app_config.get("paths",
                                            "ROOT_DIR") + 'mrcnn_models/mask_rcnn_fingerprints_' + self.MaskRcnnBackboneComboBox.currentText().lower() + '.h5'
        if not exists(path_to_model):
            msg = warning_message_box_popup("Selected model does not exists in folder: mrcnn_models",
                                            msgbox_type='error')
            msg.exec_()
            return
        if full_image:
            size = image_proc.split_image()
            mask_rcnn.detect_fingeprint_pores_on_multiple_images()
            image_proc.join_images(size, False)
            self.create_pixmap_detected_image(
                self.app_config.get("paths",
                                    "ROOT_DIR") + 'PoreDetections/final_fingerprint/pores_predicted_final_image.jpg',
                True)
            end_time = time.time()
            detection_time = round((end_time - mask_rcnn.start_time), 2)
            self.number_of_pores_detected_label.setText(
                str(mask_rcnn.number_of_detected_pores) + " pores detected in " + str(detection_time) + ' seconds')
        else:
            mask_rcnn.detect_fingeprint_pores_on_single_image()
            self.create_pixmap_detected_image(
                self.app_config.get("paths", "ROOT_DIR") + 'PoreDetections/pores_detected/detected_block_of_image.jpg',
                False)
            shutil.copyfile(
                self.app_config.get("paths", "ROOT_DIR") + 'PoreDetections/pores_detected/detected_block_of_image.jpg',
                '/home/filip/Documents/DP/Git/DP_2021-2022/GUI2/DP_2021-2022/GUI/PoreDetections'
                '/block_of_image_detected/detected_image.jpg')
            end_time = time.time()
            detection_time = round((end_time - mask_rcnn.start_time), 2)
            self.number_of_pores_detected_label.setText(
                str(mask_rcnn.number_of_detected_pores) + " pores detected in " + str(detection_time) + ' seconds')
        self.openDetectedImageButton.setEnabled(True)

    def show_new_window(self):
        self.seconWindow.setGeometry(0, 0, 800, 600)
        if self.full_image_opened:
            self.seconWindow.loadImage(
                self.app_config.get("paths",
                                    "ROOT_DIR") + 'PoreDetections/final_fingerprint/pores_predicted_final_image.jpg')
            self.seconWindow.setWindowTitle("Detected image window")
            self.seconWindow.show()
        elif self.block_of_image_opened:
            if self.mask_turned_on:
                self.seconWindow.loadImage(
                    self.app_config.get("paths",
                                        "ROOT_DIR") + 'PoreDetections/block_of_image_detected/masked_image.jpg')
                self.seconWindow.setWindowTitle("Detected image window")
                self.seconWindow.show()
            else:
                self.seconWindow.loadImage(
                    self.app_config.get("paths",
                                        "ROOT_DIR") + 'PoreDetections/block_of_image_detected/detected_image.jpg')
                self.seconWindow.setWindowTitle("Detected image window")
                self.seconWindow.show()

    def one_stage_detector_checkbox_state_changed(self, state):
        if QtCore.Qt.Checked == state:
            self.MaskRcnnCheckBox.setChecked(False)
            self.YoloModelsComboBox.setEnabled(True)
        else:
            self.YoloModelsComboBox.setEnabled(False)

    def two_stage_detector_checkbox_state_changed(self, state):
        if QtCore.Qt.Checked == state:
            self.Yolov5DetectorCheckBox.setChecked(False)
            self.MaskRcnnBackboneComboBox.setEnabled(True)
            msg = warning_message_box_popup("CUDA out of memory may occur when using Mask-RCNN", msgbox_type='warning')
            msg.exec_()

        else:
            self.MaskRcnnBackboneComboBox.setEnabled(False)

    def show_real_mask_checkbox_state_changed(self, state):

        if QtCore.Qt.Checked == state:
            self.draw_masks_from_json()
        else:
            self.create_pixmap_detected_image(self.app_config.get("paths", "ROOT_DIR") +
                                              'PoreDetections/block_of_image_detected/detected_image.jpg', False)
            self.mask_turned_on = False
            self.RealPoresLabel.setText("")

    def draw_masks_from_json(self):
        path_to_detected_image = self.app_config.get("paths",
                                                     "ROOT_DIR") + 'PoreDetections/block_of_image_detected/detected_image.jpg'
        img = cv2.imread(path_to_detected_image)
        if not self.json_is_loaded:
            self.load_json_button_handle()
        if self.json is None:
            self.showRealMaskCheckBox.setChecked(False)
            self.json_is_loaded = False
            return
        shapes = np.zeros_like(img, np.uint8)
        shapes_count = 0
        for shape in self.json['shapes']:
            x = shape['points'][0][0]
            y = shape['points'][0][1]
            x1 = shape['points'][1][0]
            y1 = shape['points'][1][1]
            center_coordinates = (int(x), int(y))
            cv2.circle(shapes, center_coordinates, 20, (255, 0, 0), cv2.FILLED)
            shapes_count = shapes_count + 1

        masked_image = img.copy()
        alpha = 0.5
        mask = shapes.astype(bool)
        masked_image[mask] = cv2.addWeighted(img, alpha, shapes, 1 - alpha, 0)[mask]
        img_rgb = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
        Image.fromarray(img_rgb).save(self.app_config.get("paths", "ROOT_DIR") +
                                      'PoreDetections/block_of_image_detected/masked_image.jpg')
        self.create_pixmap_detected_image(self.app_config.get("paths", "ROOT_DIR") +
                                          'PoreDetections/block_of_image_detected/masked_image.jpg', False)
        self.mask_turned_on = True
        self.RealPoresLabel.setText("Real pores: " + str(shapes_count))

    def detect_pores_button_clicked(self):
        self.RealPoresLabel.setText("")
        if not self.Yolov5DetectorCheckBox.isChecked() and not self.MaskRcnnCheckBox.isChecked():
            msg = warning_message_box_popup("No detector selected, please select detector.", msgbox_type='error')
            msg.exec_()
        if self.Yolov5DetectorCheckBox.isChecked():
            if 'RUN_PATH' not in globals() or RUN_PATH == "":
                msg = warning_message_box_popup("No input image. Please load an input image.", msgbox_type='error')
                msg.exec_()
            else:
                self.number_of_pores_detected_label.setText("")
                if self.full_image_opened:
                    try:
                        self.detect_fingerprint_pores_yolo(True)
                    except:
                        msg = warning_message_box_popup("CUDA out of memory. Please restart application and try again.",
                                                        msgbox_type='error')
                        msg.exec_()
                elif self.block_of_image_opened:
                    try:
                        self.detect_fingerprint_pores_yolo(False)
                    except:
                        msg = warning_message_box_popup("CUDA out of memory. Please restart application and try again.",
                                                        msgbox_type='error')
                        msg.exec_()
        if self.MaskRcnnCheckBox.isChecked():
            if 'RUN_PATH' not in globals() or RUN_PATH == "":
                msg = warning_message_box_popup("No input image. Please load an input image.", msgbox_type='error')
                msg.exec_()
            else:
                self.number_of_pores_detected_label.setText("")
                if self.full_image_opened:
                    try:
                        self.detect_fingerprint_pores_mask_rcnn(True)
                    except:
                        msg = warning_message_box_popup("CUDA out of memory. Please restart application and try again.",
                                                        msgbox_type='error')
                        msg.exec_()
                elif self.block_of_image_opened:
                    try:
                        self.detect_fingerprint_pores_mask_rcnn(False)
                    except:
                        msg = warning_message_box_popup("CUDA out of memory. Please restart application and try again.",
                                                        msgbox_type='error')
                        msg.exec_()
                    self.number_of_pores_detected_label.setText("")

    def open_image_button_clicked(self):
        file_path = None
        file_explorer = FilesExplorer.FileExplorer(custom_filter="Images (*.png *.jpg)")
        file_path = file_explorer.openFileNameDialog()
        self.create_pixmap_input_image(file_path, True)
        LOG.info("Image successfully opened")
        global RUN_PATH
        RUN_PATH = file_path
        self.block_of_image_opened = False
        self.full_image_opened = True

    def open_image_part_button_clicked(self):
        file_explorer = FilesExplorer.FileExplorer(custom_filter="Images (*.png *.jpg)")
        file_path = file_explorer.openFileNameDialog()
        self.create_pixmap_input_image(file_path, False)
        global RUN_PATH
        RUN_PATH = file_path
        self.block_of_image_opened = True
        self.full_image_opened = False

    def create_pixmap_input_image(self, file_path, scaled_content):
        if file_path is None or file_path == "":
            return
        else:
            pixmap = QPixmap(file_path)
            self.loadedImageLabel.setPixmap(pixmap)
            img = Image.open(file_path)
            wid, hgt = img.size
            img.close()
            self.InputImageLabel.setText("Input image: " + str(wid) + "x" + str(hgt))
            self.loadedImageLabel.resize(520, 640)
            self.loadedImageLabel.setScaledContents(scaled_content)

    def create_pixmap_detected_image(self, file_path, scaled_content):
        if file_path is None or file_path == "":
            return
        else:
            pixmap = QPixmap(file_path)
            self.predictedImageLabel.setPixmap(pixmap)
            img = Image.open(file_path)
            wid, hgt = img.size
            img.close()
            self.OutputImageLabel.setText("Output image:" + str(wid) + "x" + str(hgt))
            self.predictedImageLabel.resize(520, 640)
            self.predictedImageLabel.setScaledContents(scaled_content)


def warning_message_box_popup(text, msgbox_type='warning'):
    msg = QMessageBox()
    if msgbox_type == 'warning':
        msg.setIcon(QMessageBox.Warning)
        msg.setText(text)
        msg.setWindowTitle("Warning")
    if msgbox_type == 'error':
        msg.setIcon(QMessageBox.Warning)
        msg.setText(text)
        msg.setWindowTitle("Error")
    msg.setStandardButtons(QMessageBox.Ok)
    LOG.warning("No detector selected warning MessageBox displayed")
    return msg


def connect_event_listeners(mainWindow):
    mainWindow.OpenImageButton.clicked.connect(mainWindow.open_image_button_clicked)
    mainWindow.Yolov5DetectorCheckBox.stateChanged.connect(mainWindow.one_stage_detector_checkbox_state_changed)
    mainWindow.MaskRcnnCheckBox.stateChanged.connect(mainWindow.two_stage_detector_checkbox_state_changed)
    mainWindow.detectPoresButton.clicked.connect(mainWindow.detect_pores_button_clicked)
    mainWindow.confidenceSlider.valueChanged.connect(mainWindow.confidence_slider_event)
    mainWindow.maxDetectionsSlider_2.valueChanged.connect(mainWindow.max_detection_slider_event)
    mainWindow.LoadAnnotationsJsonButton.clicked.connect(mainWindow.load_json_button_handle)
    mainWindow.LoadBlockOfImageButton.clicked.connect(mainWindow.open_image_part_button_clicked)
    mainWindow.showRealMaskCheckBox.stateChanged.connect(mainWindow.show_real_mask_checkbox_state_changed)
    return mainWindow


if __name__ == '__main__':
    LOG.info('Application started')
    app = QApplication(sys.argv)
    mainWindow = MyWindow()
    mainWindow = connect_event_listeners(mainWindow)
    mainWindow.application_startup_settings()
    mainWindow.showMaximized()
    sys.exit(app.exec_())
