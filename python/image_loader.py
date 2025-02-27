from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import numpy as np
from PIL import Image


class ImageLoader(QWidget):
    imageLoaded = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        self.btn = QPushButton("选择图像")
        self.btn.clicked.connect(self.loadImage)
        layout.addWidget(self.btn)
        self.setLayout(layout)

    def loadImage(self):
        path, _ = QFileDialog.getOpenFileName()
        if path:
            img = np.array(Image.open(path).convert('L')) / 255.0
            self.imageLoaded.emit(img)