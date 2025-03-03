from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import numpy as np
from PIL import Image


class ImageLoader(QWidget):
    imageLoaded = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.initUI()
        # 维度参数
        self._3d_enabled = False
        self._z_depth = 1

    def initUI(self):
        layout = QVBoxLayout()
        self.btn = QPushButton("选择图像")
        self.btn.clicked.connect(self.loadImage)
        layout.addWidget(self.btn)
        self.setLayout(layout)

    def set_3d_params(self, is_3d, z_depth):
        # """接收来自MainWindow的三维参数"""
        self._3d_enabled = is_3d
        self._z_depth = z_depth

    def loadImage(self):
        path, _ = QFileDialog.getOpenFileName()

        if path:
            img = np.array(Image.open(path).convert('L')) / 255.0
            if self._3d_enabled:
                # 创建三维数组
                depth = self._z_depth
                # 沿第三轴复制二维图像
                img = np.stack([img] * depth, axis=2)
            else:
                pass
            self.imageLoaded.emit(img)