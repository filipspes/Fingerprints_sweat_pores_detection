from PyQt5.QtWidgets import QFileDialog, QWidget


class FileExplorer(QWidget):

    def __init__(self, custom_filter=""):
        super().__init__()
        self.title = 'File Explorer'
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 480
        self.custom_filter = custom_filter
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(0, 0, 400, 300)

    def openFileNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "File Explorer", "",
                                                  filter=self.custom_filter, options=options)
        return fileName
