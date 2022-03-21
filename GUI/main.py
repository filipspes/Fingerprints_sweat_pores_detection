import json
import threading
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QFileDialog, QWidget
from PyQt5.QtGui import QPixmap, QMovie
from PIL import Image, ImageDraw
from MainWindow import *
import os
import shutil
import time
import logging as LOG
import yoloDetector
import imageProcessing
import sys
from PyQt5 import QtCore
import config as cfg
from PyQt5.QtWidgets import *

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
    mainWindow.iouSlider.valueChanged.connect(iou_slider_event)
    mainWindow.LoadAnnotationsJsonButton.clicked.connect(load_json_button_handle)
    mainWindow.LoadBlockOfImageButton.clicked.connect(open_image_part_button_clicked)
    mainWindow.ShowMasksButton.clicked.connect(show_masks_button_click_handle)
    mainWindow.TurOffMasksButton.clicked.connect(turn_off_masks_button_clicked)
    return mainWindow

def open_image_button_clicked():
    file_path = None
    fileExplorer = FileExplorer()
    file_path = fileExplorer.openFileNameDialog()
    create_pixmap(file_path)
    LOG.info("Image successfully opened")
    global RUN_PATH
    RUN_PATH = file_path
    myWin.block_of_image_opened = False
    myWin.full_image_opened = True


def open_image_part_button_clicked():
    file_path = '/home/filip/Documents/DP/FP Parts_Test/1.jpg'
    create_pixmap_block(file_path)
    global RUN_PATH
    RUN_PATH = file_path
    myWin.block_of_image_opened = True
    myWin.full_image_opened = False

def create_pixmap_block(file_path):
    pixmap = QPixmap(file_path)
    myWin.loadedImageLabel.setPixmap(pixmap)
    myWin.loadedImageLabel.resize(520, 640)
    myWin.loadedImageLabel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
    # myWin.loadedImageLabel.setScaledContents(True)

def create_pixmap_block_detected(file_path):
    pixmap = QPixmap(file_path)
    myWin.predictedImageLabel.setPixmap(pixmap)
    myWin.predictedImageLabel.resize(520, 640)
    myWin.predictedImageLabel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
    # myWin.predictedImageLabel.setScaledContents(True)

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
                # viewrect = self.viewport().rect()
                # scenerect = self.transform().mapRect(rect)
                # factor = max(viewrect.width() / scenerect.width(),
                #              viewrect.height() / scenerect.height())
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
                full_image_detecting_thread()
            elif myWin.block_of_image_opened:
                block_image_detecting_thread()
    if myWin.MaskRcnnCheckBox.isChecked():
        myWin.number_of_pores_detected_label.setText("")
        print("No implemented yet.")


def one_stage_detector_checkbox_state_changed(state):
    if (QtCore.Qt.Checked == state):
        LOG.info("Check box 1 checked")
    else:
        LOG.info("Check box 1 unchecked")


def two_stage_detector_checkbox_state_changed(state):
    if (QtCore.Qt.Checked == state):
        LOG.info("Check box 2 checked")
    else:
        LOG.info("Check box 2 unchecked")




def show_new_window(self):
    myWin.seconWindow.setGeometry(0, 0, 800, 600)
    if myWin.full_image_opened:
        myWin.seconWindow.loadImage('/home/filip/Documents/DP/Git/DP_2021-2022/GUI/PoreDetections/final_fingerprint/pores_predicted_final_image.jpg')
        myWin.seconWindow.setWindowTitle("Detected image window")
        myWin.seconWindow.show()
    elif myWin.block_of_image_opened:
        if myWin.mask_turned_on:
            myWin.seconWindow.loadImage('/home/filip/Documents/DP/Git/DP_2021-2022/GUI/PoreDetections/block_of_image_detected/masked_image.jpg')
            myWin.seconWindow.setWindowTitle("Detected image window")
            myWin.seconWindow.show()
        else:
            myWin.seconWindow.loadImage('/home/filip/Documents/DP/Git/DP_2021-2022/GUI/PoreDetections/block_of_image_detected/detected_image.jpg')
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


def create_pixmap(file_path):
    pixmap = QPixmap(file_path)
    myWin.loadedImageLabel.setPixmap(pixmap)
    myWin.loadedImageLabel.resize(520, 640)
    myWin.loadedImageLabel.setScaledContents(True)



def create_pixmap_detected_image(file_path):
    pixmap = QPixmap(file_path)
    myWin.predictedImageLabel.setPixmap(pixmap)
    myWin.predictedImageLabel.resize(520, 640)
    myWin.predictedImageLabel.setScaledContents(True)


def full_image_detecting_thread():
    myWin.spinnerLabel.resize(64, 64)
    movie = QMovie("loadingSpinner.gif")  # Create a QMovie from our gif
    myWin.spinnerLabel.setMovie(movie)  # use setMovie function in our QLabel
    myWin.spinnerLabel.show()
    # win = Window()
    # win.setGeometry(500, 300, 800, 600)
    full_image_detector_detector = threading.Thread(target=detect_pores_on_full_image)
    movie.start()
    # detector(win)
    full_image_detector_detector.start()

def block_image_detecting_thread():
    myWin.spinnerLabel.resize(64, 64)
    movie = QMovie("loadingSpinner.gif")  # Create a QMovie from our gif
    myWin.spinnerLabel.setMovie(movie)  # use setMovie function in our QLabel
    myWin.spinnerLabel.show()
    # win = Window()
    # win.setGeometry(500, 300, 800, 600)
    block_image_detector_detector = threading.Thread(target=detect_pores_on_block_of_image)
    movie.start()
    # detector(win)
    block_image_detector_detector.start()



def detect_pores_on_full_image():
    config = cfg.get_config()
    yolo = yoloDetector.Yolo(myWin.confidenceSlider.value(), myWin.iouSlider.value(), '')
    image_proc = imageProcessing.ImageProcessing(RUN_PATH)
    image_proc.remove_content_of_folders()
    size = image_proc.split_image()
    remove_content_of_folder_runs()
    start_time = time.time()
    number_of_detected_pores = yolo.detect(False, True)
    end_time = time.time()
    LOG.info("Detection took: " + str(end_time - start_time) + ' seconds')
    image_proc.join_images(size)
    create_pixmap_detected_image(
        '/home/filip/Documents/DP/Git/DP_2021-2022/GUI/PoreDetections/final_fingerprint/pores_predicted_final_image.jpg')
    myWin.spinnerLabel.hide()
    path_to_results = config.get("paths", "path_to_results")
    myWin.number_of_pores_detected_label.setText(str(number_of_detected_pores) + " pores detected!")
    myWin.openDetectedImageButton.setEnabled(True)

def detect_pores_on_block_of_image():
    config = cfg.get_config()
    yolo = yoloDetector.Yolo(myWin.confidenceSlider.value(), myWin.iouSlider.value(), RUN_PATH)
    remove_content_of_folder_runs()
    start_time = time.time()
    number_of_detected_pores = yolo.detect(True, False)
    end_time = time.time()
    LOG.info("Detection took: " + str(end_time - start_time) + ' seconds')
    # input_directory = config.get("paths", "path_to_detected_parts_of_image")
    # file_name = os.listdir(input_directory)
    list_of_images = os.listdir('/home/filip/Documents/DP/Git/DP_2021-2022/GUI/runs/detect/exp/')
    shutil.copyfile('/home/filip/Documents/DP/Git/DP_2021-2022/GUI/runs/detect/exp/'+list_of_images[0],
                    '/home/filip/Documents/DP/Git/DP_2021-2022/GUI/PoreDetections/block_of_image_detected/detected_image.jpg')
    create_pixmap_block_detected('/home/filip/Documents/DP/Git/DP_2021-2022/GUI/PoreDetections/block_of_image_detected/detected_image.jpg')
    myWin.spinnerLabel.hide()
    path_to_results = config.get("paths", "path_to_results")
    myWin.number_of_pores_detected_label.setText(str(number_of_detected_pores) + " pores detected!")
    myWin.openDetectedImageButton.setEnabled(True)


def confidence_slider_event():
    myWin.confidenceLabel.setText('Confidence: ' + str((myWin.confidenceSlider.value() + 1) / 100))


def iou_slider_event():
    myWin.iouLabel.setText('IoU: ' + str((myWin.iouSlider.value() + 1) / 100))


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
    img = Image.open(path_to_detected_image)
    if not myWin.json_is_loaded:
        load_json_button_handle()
    img = img.resize((int(myWin.json['imageWidth']), int(myWin.json['imageHeight'])))
    draw = ImageDraw.Draw(img)
    for shape in myWin.json['shapes']:
        single_shape = [shape['points'][0][0], shape['points'][0][1], shape['points'][1][0], shape['points'][1][1]]
        x = shape['points'][0][0]
        y = shape['points'][0][1]
        x1 = shape['points'][1][0]
        y1 = shape['points'][1][1]
        # all_shapes.append(single_shape)
        draw.ellipse([x-10, y-10, x+10, y+10], fill='yellow', width=2)
        # draw.text([x, y - 10], text)
    img.save('/home/filip/Documents/DP/Git/DP_2021-2022/GUI/PoreDetections/block_of_image_detected/masked_image.jpg')
    create_pixmap_block_detected('/home/filip/Documents/DP/Git/DP_2021-2022/GUI/PoreDetections/block_of_image_detected/masked_image.jpg')
    myWin.mask_turned_on = True

def turn_off_masks_button_clicked():
    create_pixmap_block_detected('/home/filip/Documents/DP/Git/DP_2021-2022/GUI/PoreDetections/block_of_image_detected/detected_image.jpg')
    myWin.mask_turned_on = False

def remove_content_of_folder_runs():
    config = cfg.get_config()
    for filename in os.listdir(config.get("paths", "detections")):
        file_path = os.path.join(config.get("paths", "detections"), filename)
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

if __name__ == '__main__':
    LOG.info('Application started')
    app = QApplication(sys.argv)
    myWin = MyWindow()
    myWin = connect_event_listeners(myWin)
    myWin.configurationGroupbox.setEnabled(True)
    myWin.detectorsTypesGroupBox.setEnabled(True)
    myWin.openDetectedImageButton.setEnabled(False)
    myWin.showMaximized()
    sys.exit(app.exec_())
