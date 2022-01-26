# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QFileDialog, QWidget
from PyQt5.QtGui import QPixmap
from MainWindow import *


def open_image_button_clicked():
    file_path = None
    fileExplorer = FileExplorer()
    file_path = fileExplorer.openFileNameDialog()
    create_pixmap(file_path)

def detect_pores_button_clicked(self):
    if not myWin.OneStageDetectorCheckBox.isChecked():
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        # setting message for Message Box
        msg.setText("No detector selected ")
        # setting Message box window title
        msg.setWindowTitle("Warning")
        # declaring buttons on Message Box
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        # start the app
        msg.exec_()
        # QMessageBox.about("Title", "Message")

def one_stage_detector_checkbox_state_changed(state):
    if (QtCore.Qt.Checked == state):
        print("Check box 1 checked")
    else:
        print("Check box 1 unchecked")

def two_stage_detector_checkbox_state_changed(state):
    if (QtCore.Qt.Checked == state):
        print("Check box 2 checked")
    else:
        print("Check box 2 unchecked")



def connect_event_listeners(mainWindow):
    mainWindow.OpenImageButton.clicked.connect(open_image_button_clicked)
    mainWindow.OneStageDetectorCheckBox.stateChanged.connect(one_stage_detector_checkbox_state_changed)
    mainWindow.TwoStageDetectorCheckBox.stateChanged.connect(two_stage_detector_checkbox_state_changed)
    mainWindow.detectPoresButton.clicked.connect(detect_pores_button_clicked)

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
        if fileName:
            print(fileName)
        return fileName
    # def saveFileDialog(self):
    #     options = QFileDialog.Options()
    #     options |= QFileDialog.DontUseNativeDialog
    #     fileName, _ = QFileDialog.getSaveFileName(self, "QFileDialog.getSaveFileName()", "",
    #                                               "All Files (*);;Text Files (*.txt)", options=options)
    #     if fileName:
    #         print(fileName)


def create_pixmap(file_path):
    pixmap = QPixmap(file_path)
    myWin.loadedImageLabel.setPixmap(pixmap)
    myWin.loadedImageLabel.resize(800, 800)
    myWin.loadedImageLabel.setScaledContents(True)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = MyWindow()
    myWin = connect_event_listeners(myWin)
    # myWin.OneStageDetectorComboBox.addItem("Yolov5")
    # myWin.TwoStageDetectorComboBox.addItem("Yolov52")
    myWin.showMaximized()
    sys.exit(app.exec_())
