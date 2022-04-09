import json
import threading

import cv2
# import mrcnn
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QFileDialog, QWidget
from PyQt5.QtGui import QPixmap, QMovie
from PIL import Image, ImageDraw
from MainWindow import *
import os
import shutil
import time
import logging as LOG
import matplotlib.pyplot as plt
import yoloDetector
import imageProcessing
import sys
import numpy as np
from PyQt5 import QtCore
import config as cfg
from mrcnn.config import Config
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn import utils
from PyQt5.QtWidgets import *
import skimage
import visualize_detections
import mask_rcnn_config as mrc

LOG.basicConfig(
    level=LOG.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        LOG.FileHandler("logfile.log"),
        LOG.StreamHandler()
    ]
)


def connect_event_listeners(mainWindow):
    mainWindow.OpenImageButton.clicked.connect(open_image_button_clicked)
    mainWindow.Yolov5DetectorCheckBox.stateChanged.connect(one_stage_detector_checkbox_state_changed)
    mainWindow.MaskRcnnCheckBox.stateChanged.connect(two_stage_detector_checkbox_state_changed)
    mainWindow.detectPoresButton.clicked.connect(detect_pores_button_clicked)
    mainWindow.confidenceSlider.valueChanged.connect(confidence_slider_event)
    mainWindow.maxDetectionsSlider_2.valueChanged.connect(max_deetection_slider_event)
    mainWindow.LoadAnnotationsJsonButton.clicked.connect(load_json_button_handle)
    mainWindow.LoadBlockOfImageButton.clicked.connect(open_image_part_button_clicked)
    mainWindow.ShowMasksButton.clicked.connect(show_masks_button_click_handle)
    mainWindow.TurOffMasksButton.clicked.connect(turn_off_masks_button_clicked)
    return mainWindow


def open_image_button_clicked():
    file_path = None
    fileExplorer = FileExplorer()
    file_path = fileExplorer.openFileNameDialog()
    create_pixmap_input_image(file_path, True)
    LOG.info("Image successfully opened")
    global RUN_PATH
    RUN_PATH = file_path
    myWin.block_of_image_opened = False
    myWin.full_image_opened = True


def open_image_part_button_clicked():
    fileExplorer = FileExplorer()
    file_path = fileExplorer.openFileNameDialog()
    create_pixmap_input_image(file_path, False)
    global RUN_PATH
    RUN_PATH = file_path
    myWin.block_of_image_opened = True
    myWin.full_image_opened = False


def create_pixmap_input_image(file_path, scaled_content):
    pixmap = QPixmap(file_path)
    myWin.loadedImageLabel.setPixmap(pixmap)
    img = Image.open(file_path)
    wid, hgt = img.size
    img.close()
    myWin.ResolutionInputImageLabel.setText(str(wid) + "x" + str(hgt))
    myWin.loadedImageLabel.resize(520, 640)
    myWin.loadedImageLabel.setScaledContents(scaled_content)


def create_pixmap_detected_image(file_path, scaled_content):
    pixmap = QPixmap(file_path)
    myWin.predictedImageLabel.setPixmap(pixmap)
    img = Image.open(file_path)
    wid, hgt = img.size
    img.close()
    myWin.ResolutionOutputImageLabel.setText(str(wid) + "x" + str(hgt))
    myWin.predictedImageLabel.resize(520, 640)
    myWin.predictedImageLabel.setScaledContents(scaled_content)


class PhotoViewer(QtWidgets.QGraphicsView):
    photoClicked = QtCore.pyqtSignal(QtCore.QPoint)

    def __init__(self, parent):
        super(PhotoViewer, self).__init__(parent)
        self._zoom = 0
        self._empty = True
        self._scene = QtWidgets.QGraphicsScene()
        self._photo = QtWidgets.QGraphicsPixmapItem()
        self._scene.addItem(self._photo)
        self.setScene(self._scene)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(30, 30, 30)))
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setWindowTitle('Detected image window')

    def hasPhoto(self):
        return not self._empty

    def fitInView(self, scale=True):
        rect = QtCore.QRectF(self._photo.pixmap().rect())
        if not rect.isNull():
            self.setSceneRect(rect)
            if self.hasPhoto():
                unity = self.transform().mapRect(QtCore.QRectF(0, 0, 1, 1))
                self.scale(1 / unity.width(), 1 / unity.height())
                viewrect = self.viewport().rect()
                scenerect = self.transform().mapRect(rect)
                factor = max(viewrect.width() / scenerect.width(),
                             viewrect.height() / scenerect.height())
                self.scale(0.1, 0.1)
            self._zoom = 0

    def setPhoto(self, pixmap=None):
        self._zoom = 0
        if pixmap and not pixmap.isNull():
            self._empty = False
            self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
            self._photo.setPixmap(pixmap)
        else:
            self._empty = True
            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
            self._photo.setPixmap(QtGui.QPixmap())
        self.fitInView()

    def wheelEvent(self, event):
        if self.hasPhoto():
            if event.angleDelta().y() > 0:
                factor = 1.25
                self._zoom += 1
            else:
                factor = 0.8
                self._zoom -= 1
            if self._zoom > 0:
                self.scale(factor, factor)
            elif self._zoom == 0:
                self.fitInView()
            else:
                self._zoom = 0

    def toggleDragMode(self):
        if self.dragMode() == QtWidgets.QGraphicsView.ScrollHandDrag:
            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
        elif not self._photo.pixmap().isNull():
            self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)

    def mousePressEvent(self, event):
        if self._photo.isUnderMouse():
            self.photoClicked.emit(self.mapToScene(event.pos()).toPoint())
        super(PhotoViewer, self).mousePressEvent(event)


class Window(QtWidgets.QWidget):
    def __init__(self):
        super(Window, self).__init__()
        self.viewer = PhotoViewer(self)
        VBlayout = QtWidgets.QVBoxLayout(self)
        VBlayout.addWidget(self.viewer)
        HBlayout = QtWidgets.QHBoxLayout()
        HBlayout.setAlignment(QtCore.Qt.AlignLeft)
        VBlayout.addLayout(HBlayout)

    def loadImage(self, path):
        self.viewer.setPhoto(QtGui.QPixmap(path))

    def photoClicked(self, pos):
        if self.viewer.dragMode() == QtWidgets.QGraphicsView.NoDrag:
            self.editPixInfo.setText('%d, %d' % (pos.x(), pos.y()))


def detect_pores_button_clicked():
    myWin.RealPoresLabel.setText("")
    if not myWin.Yolov5DetectorCheckBox.isChecked() and not myWin.MaskRcnnCheckBox.isChecked():
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setText("No detector selected, please select detector.")
        msg.setWindowTitle("Warning")
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        LOG.warning("No detector selected warning MessageBox displayed")
        msg.exec_()
    if myWin.Yolov5DetectorCheckBox.isChecked():
        if 'RUN_PATH' not in globals() or RUN_PATH == "":
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("No input image. Please load an input image.")
            msg.setWindowTitle("No image")
            msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
            LOG.warning("No image is opened")
            msg.exec_()
        else:
            myWin.predictedImageLabel.setText('Image is being processed... ')
            myWin.number_of_pores_detected_label.setText("")
            if myWin.full_image_opened:
                detect_pores_on_full_image()
            elif myWin.block_of_image_opened:
                detect_pores_on_block_of_image()
    if myWin.MaskRcnnCheckBox.isChecked():
        if 'RUN_PATH' not in globals() or RUN_PATH == "":
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("No input image. Please load an input image.")
            msg.setWindowTitle("No image")
            msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
            LOG.warning("No image is opened")
            msg.exec_()
        else:
            myWin.predictedImageLabel.setText('Image is being processed... ')
            myWin.number_of_pores_detected_label.setText("")
            if myWin.full_image_opened:
                detect_pores_on_full_image_mask_rcnn()
            elif myWin.block_of_image_opened:
                detect_pores_on_block_of_image_mask_rcnn()
                myWin.number_of_pores_detected_label.setText("")


def one_stage_detector_checkbox_state_changed(state):
    if (QtCore.Qt.Checked == state):
        myWin.YoloModelsComboBox.setEnabled(True)
        LOG.info("Check box 1 checked")
    else:
        myWin.YoloModelsComboBox.setEnabled(False)
        LOG.info("Check box 1 unchecked")


def two_stage_detector_checkbox_state_changed(state):
    if (QtCore.Qt.Checked == state):
        myWin.MaskRcnnBackboneComboBox.setEnabled(True)
        LOG.info("Check box 2 checked")
    else:
        myWin.MaskRcnnBackboneComboBox.setEnabled(False)
        LOG.info("Check box 2 unchecked")


def show_new_window(self):
    config = cfg.get_config()
    myWin.seconWindow.setGeometry(0, 0, 800, 600)
    if myWin.full_image_opened:
        myWin.seconWindow.loadImage(
            config.get("paths", "ROOT_DIR") + 'PoreDetections/final_fingerprint/pores_predicted_final_image.jpg')
        myWin.seconWindow.setWindowTitle("Detected image window")
        myWin.seconWindow.show()
    elif myWin.block_of_image_opened:
        if myWin.mask_turned_on:
            myWin.seconWindow.loadImage(
                config.get("paths", "ROOT_DIR") + 'PoreDetections/block_of_image_detected/masked_image.jpg')
            myWin.seconWindow.setWindowTitle("Detected image window")
            myWin.seconWindow.show()
        else:
            myWin.seconWindow.loadImage(
                config.get("paths", "ROOT_DIR") + 'PoreDetections/block_of_image_detected/detected_image.jpg')
            myWin.seconWindow.setWindowTitle("Detected image window")
            myWin.seconWindow.show()


class FileExplorer(QWidget):

    def __init__(self):
        super().__init__()
        self.title = 'File Explorer'
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 480
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(0, 0, 400, 300)

    def openFileNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "File Explorer", "",
                                                  "All Files (*);;Python Files (*.py)", options=options)
        return fileName


def detect_pores_on_block_of_image_mask_rcnn():
    config = cfg.get_config()
    image_proc = imageProcessing.ImageProcessing(RUN_PATH)
    image_proc.remove_content_of_folders()
    inference_config = mrc.InferenceConfig()
    mask_rcnn_model = modellib.MaskRCNN(mode='inference',
                             config=inference_config,
                             model_dir='./')
    mask_rcnn_model.load_weights(
        config.get("paths", "ROOT_DIR") + 'mrcnn_models/mask_rcnn_fingerprint_resnet50_all.h5', by_name=True)
    dataset_val = CocoLikeDataset()
    dataset_val.load_data('/home/filip/Documents/DP/MR/Mask_RCNN/datasets/fingerprints_pores/val/coco_annotations.json',
                          '/home/filip/Documents/DP/MR/Mask_RCNN/datasets/fingerprints_pores/val/images')
    dataset_val.prepare()

    print(RUN_PATH)
    img = skimage.io.imread(RUN_PATH)
    img_arr = np.array(img)
    results = mask_rcnn_model.detect([img_arr], verbose=1)
    r = results[0]
    visualize_detections.display_instances(img, "detected_block_of_image.jpg", r['rois'], r['masks'], r['class_ids'],
                                           dataset_val.class_names, r['scores'])
    print(len(r['rois']))
    create_pixmap_detected_image(
        config.get("paths", "ROOT_DIR") + 'PoreDetections/pores_detected/detected_block_of_image.jpg', True)
    myWin.openDetectedImageButton.setEnabled(True)


def detect_pores_on_full_image_mask_rcnn():
    config = cfg.get_config()
    image_proc = imageProcessing.ImageProcessing(RUN_PATH)
    image_proc.remove_content_of_folders()
    size = image_proc.split_image()
    remove_content_of_folder_runs()
    start_time = time.time()
    inference_config = mrc.InferenceConfig()
    mask_rcnn_model = modellib.MaskRCNN(mode='inference',
                                        config=inference_config,
                                        model_dir='./')
    mask_rcnn_model.load_weights(config.get("paths", "ROOT_DIR") + 'mrcnn_models'
                                                                   '/mask_rcnn_fingerprint_resnet50_all.h5',
                                 by_name=True)
    print("model loaded")
    end_time = time.time()

    dataset_val = CocoLikeDataset()
    dataset_val.load_data('/home/filip/Documents/DP/MR/Mask_RCNN/datasets/fingerprints_pores/val/coco_annotations.json',
                          '/home/filip/Documents/DP/MR/Mask_RCNN/datasets/fingerprints_pores/val/images')
    dataset_val.prepare()

    real_test_dir = config.get("paths", "ROOT_DIR") + 'PoreDetections/parts_of_image/'
    image_paths = []
    file_names = []
    for filename in os.listdir(real_test_dir):
        if os.path.splitext(filename)[1].lower() in ['.png', '.jpg', '.jpeg']:
            image_paths.append(os.path.join(real_test_dir, filename))
            file_names.append(filename)

    for image_path, file_name in zip(image_paths, file_names):
        img = skimage.io.imread(image_path)
        if np.mean(img) == 255:
            shutil.copyfile(image_path, config.get("paths", "ROOT_DIR") + 'PoreDetections/pores_detected/' + file_name)
        else:
            img_arr = np.array(img)
            results = mask_rcnn_model.detect([img_arr], verbose=1)
            r = results[0]
            visualize_detections.display_instances(img, file_name, r['rois'], r['masks'], r['class_ids'],
                                                   dataset_val.class_names, r['scores'], figsize=(5, 5))
            print(len(r['rois']))

    LOG.info("Detection took: " + str(end_time - start_time) + ' seconds')
    image_proc.join_images(size, False)
    create_pixmap_detected_image(
        config.get("paths", "ROOT_DIR") + 'PoreDetections/final_fingerprint/pores_predicted_final_image.jpg', True)
    image_proc.resize_final_image()
    myWin.openDetectedImageButton.setEnabled(True)


def detect_pores_on_full_image():
    config = cfg.get_config()
    yolo = yoloDetector.Yolo(myWin.confidenceSlider.value(), myWin.maxDetectionsSlider_2.value(), '',
                             myWin.YoloModelsComboBox.currentText())
    image_proc = imageProcessing.ImageProcessing(RUN_PATH)
    image_proc.remove_content_of_folders()
    size = image_proc.split_image()
    remove_content_of_folder_runs()
    start_time = time.time()
    number_of_detected_pores = yolo.detect(False, True)
    end_time = time.time()
    LOG.info("Detection took: " + str(end_time - start_time) + ' seconds')
    image_proc.join_images(size, True)
    create_pixmap_detected_image(
        config.get("paths", "ROOT_DIR") + 'PoreDetections/final_fingerprint/pores_predicted_final_image.jpg', True)
    # myWin.spinnerLabel.hide()
    myWin.number_of_pores_detected_label.setText(str(number_of_detected_pores) + " pores detected!")
    myWin.openDetectedImageButton.setEnabled(True)


def detect_pores_on_block_of_image():
    config = cfg.get_config()
    yolo = yoloDetector.Yolo(myWin.confidenceSlider.value(), myWin.iouSlider.value(), RUN_PATH,
                             myWin.YoloModelsComboBox.currentText())
    remove_content_of_folder_runs()
    start_time = time.time()
    number_of_detected_pores = yolo.detect(True, False)
    end_time = time.time()
    LOG.info("Detection took: " + str(end_time - start_time) + ' seconds')
    # input_directory = config.get("paths", "path_to_detected_parts_of_image")
    # file_name = os.listdir(input_directory)
    list_of_images = os.listdir('/home/filip/Documents/DP/Git/DP_2021-2022/GUI/runs/detect/exp/')
    shutil.copyfile('/home/filip/Documents/DP/Git/DP_2021-2022/GUI/runs/detect/exp/' + list_of_images[0],
                    '/home/filip/Documents/DP/Git/DP_2021-2022/GUI/PoreDetections/block_of_image_detected/detected_image.jpg')
    create_pixmap_detected_image(
        '/home/filip/Documents/DP/Git/DP_2021-2022/GUI/PoreDetections/block_of_image_detected/detected_image.jpg',
        False)
    myWin.spinnerLabel.hide()
    path_to_results = config.get("paths", "path_to_results")
    myWin.number_of_pores_detected_label.setText(str(number_of_detected_pores) + " pores detected!")
    myWin.openDetectedImageButton.setEnabled(True)


def confidence_slider_event():
    myWin.confidenceLabel.setText('Confidence: ' + str((myWin.confidenceSlider.value() + 1) / 100))


def max_deetection_slider_event():
    myWin.maxDetectionsLabel.setText('Max detections: ' + str((myWin.maxDetectionsSlider_2.value())))


def load_json_button_handle():
    config = cfg.get_config()
    file_explorer = FileExplorer()
    file_path = file_explorer.openFileNameDialog()
    f = open(file_path)
    data = json.load(f)
    f.close()
    myWin.json_is_loaded = True
    myWin.json = data


def show_masks_button_click_handle():
    config = cfg.get_config()
    path_to_detected_image = '/home/filip/Documents/DP/Git/DP_2021-2022/GUI/PoreDetections/block_of_image_detected/detected_image.jpg'
    img = cv2.imread(path_to_detected_image)
    if not myWin.json_is_loaded:
        load_json_button_handle()
    shapes = np.zeros_like(img, np.uint8)
    shapes_count = 0
    for shape in myWin.json['shapes']:
        single_shape = [shape['points'][0][0], shape['points'][0][1], shape['points'][1][0], shape['points'][1][1]]
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
    Image.fromarray(img_rgb).save(
        '/home/filip/Documents/DP/Git/DP_2021-2022/GUI/PoreDetections/block_of_image_detected/masked_image.jpg')
    create_pixmap_detected_image(
        '/home/filip/Documents/DP/Git/DP_2021-2022/GUI/PoreDetections/block_of_image_detected/masked_image.jpg', False)
    myWin.mask_turned_on = True
    myWin.RealPoresLabel.setText("Real pores: " + str(shapes_count))


def turn_off_masks_button_clicked():
    create_pixmap_detected_image(
        '/home/filip/Documents/DP/Git/DP_2021-2022/GUI/PoreDetections/block_of_image_detected/detected_image.jpg',
        False)
    myWin.mask_turned_on = False
    myWin.RealPoresLabel.setText("")


def fill_combobox_yolo():
    myWin.YoloModelsComboBox.addItem("YOLOv5 Nano")
    myWin.YoloModelsComboBox.addItem("YOLOv5 Small")
    myWin.YoloModelsComboBox.addItem("YOLOv5 Medium")
    myWin.YoloModelsComboBox.addItem("YOLOv5 Large")
    myWin.YoloModelsComboBox.addItem("YOLOv5 XLarge")


def fill_combobox_mask_rcnn():
    myWin.MaskRcnnBackboneComboBox.addItem("Resnet 50")
    myWin.MaskRcnnBackboneComboBox.addItem("Resnet 101")


def remove_content_of_folder_runs():
    config = cfg.get_config()
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
        self.seconWindow = Window()
        self.openDetectedImageButton.clicked.connect(show_new_window)
        self.full_image_opened = False
        self.block_of_image_opened = False
        self.image_is_detected = False
        self.json_is_loaded = False
        self.json = None
        self.mask_turned_on = False


class CocoLikeDataset(utils.Dataset):
    """ Generates a COCO-like dataset, i.e. an image dataset annotated in the style of the COCO dataset.
        See http://cocodataset.org/#home for more information.
    """

    def load_data(self, annotation_json, images_dir):
        """ Load the coco-like dataset from json
        Args:
            annotation_json: The path to the coco annotations json file
            images_dir: The directory holding the images referred to by the json file
        """
        # Load json from file
        json_file = open(annotation_json)
        coco_json = json.load(json_file)
        json_file.close()

        # Add the class names using the base method from utils.Dataset
        source_name = "coco_like"
        for category in coco_json['categories']:
            class_id = category['id']
            class_name = category['name']
            if class_id < 1:
                print('Error: Class id for "{}" cannot be less than one. (0 is reserved for the background)'.format(
                    class_name))
                return

            self.add_class(source_name, class_id, class_name)

        # Get all annotations
        annotations = {}
        for annotation in coco_json['annotations']:
            image_id = annotation['image_id']
            if image_id not in annotations:
                annotations[image_id] = []
            annotations[image_id].append(annotation)

        # Get all images and add them to the dataset
        seen_images = {}
        for image in coco_json['images']:
            image_id = image['id']
            if image_id in seen_images:
                print("Warning: Skipping duplicate image id: {}".format(image))
            else:
                seen_images[image_id] = image
                try:
                    image_file_name = image['file_name']
                    image_width = image['width']
                    image_height = image['height']
                except KeyError as key:
                    print("Warning: Skipping image (id: {}) with missing key: {}".format(image_id, key))

                image_path = os.path.abspath(os.path.join(images_dir, image_file_name))
                image_annotations = annotations[image_id]

                # Add the image using the base method from utils.Dataset
                self.add_image(
                    source=source_name,
                    image_id=image_id,
                    path=image_path,
                    width=image_width,
                    height=image_height,
                    annotations=image_annotations
                )

    def load_mask(self, image_id):
        """ Load instance masks for the given image.
        MaskRCNN expects masks in the form of a bitmap [height, width, instances].
        Args:
            image_id: The id of the image to load masks for
        Returns:
            masks: A bool array of shape [height, width, instance count] with
                one mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]
        annotations = image_info['annotations']
        instance_masks = []
        class_ids = []

        for annotation in annotations:
            class_id = annotation['category_id']
            mask = Image.new('1', (image_info['width'], image_info['height']))
            mask_draw = ImageDraw.ImageDraw(mask, '1')
            for segmentation in annotation['segmentation']:
                mask_draw.polygon(segmentation, fill=1)
                bool_array = np.array(mask) > 0
                instance_masks.append(bool_array)
                class_ids.append(class_id)

        mask = np.dstack(instance_masks)
        class_ids = np.array(class_ids, dtype=np.int32)

        return mask, class_ids


if __name__ == '__main__':
    LOG.info('Application started')
    app = QApplication(sys.argv)
    myWin = MyWindow()
    myWin = connect_event_listeners(myWin)
    myWin.configurationGroupbox.setEnabled(True)
    myWin.detectorsTypesGroupBox.setEnabled(True)
    myWin.openDetectedImageButton.setEnabled(False)
    myWin.showMaximized()
    myWin.YoloModelsComboBox.setEnabled(False)
    myWin.MaskRcnnBackboneComboBox.setEnabled(False)
    fill_combobox_yolo()
    fill_combobox_mask_rcnn()
    sys.exit(app.exec_())
