from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QFileDialog, QWidget
from PyQt5.QtGui import QPixmap
from MainWindow import *
import os
import shutil
import time
import sys
import logging as LOG

LOG.basicConfig(
    level=LOG.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        LOG.FileHandler("logfile.log"),
        LOG.StreamHandler()
    ]
)



def open_image_button_clicked():
    file_path = None
    fileExplorer = FileExplorer()
    file_path = fileExplorer.openFileNameDialog()
    create_pixmap(file_path)
    LOG.info("Image successfully opened")
    global RUN_PATH
    RUN_PATH = file_path

def detect_pores_button_clicked(self):
    yolo_detector()
    if not myWin.OneStageDetectorCheckBox.isChecked():
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setText("No detector selected ")
        msg.setWindowTitle("Warning")
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        LOG.warning("No detector selected warning MessageBox displayed")
        msg.exec_()

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



def connect_event_listeners(mainWindow):
    mainWindow.OpenImageButton.clicked.connect(open_image_button_clicked)
    mainWindow.OneStageDetectorCheckBox.stateChanged.connect(one_stage_detector_checkbox_state_changed)
    mainWindow.TwoStageDetectorCheckBox.stateChanged.connect(two_stage_detector_checkbox_state_changed)
    mainWindow.detectPoresButton.clicked.connect(detect_pores_button_clicked)
    mainWindow.confidenceSlider.valueChanged.connect(confidence_slider_event)

    return mainWindow


class MyWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyWindow, self).__init__(parent)
        self.setupUi(self)


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
        self.setGeometry(self.left, self.top, self.width, self.height)

    def openFileNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                  "All Files (*);;Python Files (*.py)", options=options)
        return fileName


def create_pixmap(file_path):
    pixmap = QPixmap(file_path)
    myWin.loadedImageLabel.setPixmap(pixmap)
    myWin.loadedImageLabel.resize(460, 620)
    myWin.loadedImageLabel.setScaledContents(True)

def create_pixmap_detected_image(file_path):
    pixmap = QPixmap(file_path)
    myWin.predictedImageLabel.setPixmap(pixmap)
    myWin.predictedImageLabel.resize(460, 620)
    myWin.predictedImageLabel.setScaledContents(True)

def yolo_detector():
    import yolo_detector as yolo
    import image_processing
    imgProc = image_processing.imageProcessing(RUN_PATH)
    imgProc.remove_content_of_folders()
    size = imgProc.splitImage()
    remove_content_of_folder_runs('/home/filip/Documents/DP/Git/DP_2021-2022/GUI/runs/detect/')
    start_time = time.time()
    yolo.model()
    end_time = time.time()
    LOG.info("Detection took: " + str(end_time-start_time) + ' seconds')
    imgProc.joinImages(size)
    create_pixmap_detected_image('/home/filip/Documents/DP/Git/DP_2021-2022/GUI/PoreDetections/final_fingerprint/pores_predicted_final_image.jpg')

def confidence_slider_event():
    myWin.confidenceLabel.setText('Confidence: ' + str(myWin.confidenceSlider.value()+1))

def remove_content_of_folder_runs(folder): # OK
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            LOG.error('Failed to delete %s. Reason: %s' % (file_path, e))

if __name__ == '__main__':
    LOG.info('Application started')
    app = QApplication(sys.argv)
    myWin = MyWindow()
    myWin = connect_event_listeners(myWin)
    myWin.confidenceSlider.setValue(88)
    myWin.showMaximized()
    sys.exit(app.exec_())