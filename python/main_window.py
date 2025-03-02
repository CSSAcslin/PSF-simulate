import sys
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from psf_generator import PSFGenerator
from drawing_widget import DrawingWidget
from image_loader import ImageLoader
from convolution_handler import ConvolutionHandler
from Convolution_Worker import ConvolutionWorker


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.initConnections()

        self.current_psf = None
        self.input_image = None
        self.psf_params = {}
        self.scale_factor = 1.0  # 微米/像素
        self.worker_thread = None

    def initUI(self):
        self.setWindowTitle("光学成像模拟器 v1.2")
        self.setGeometry(100, 100, 1200, 800)

        # 主布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # 左侧控制面板
        control_panel = QTabWidget()
        self.setupPSFTab(control_panel)
        self.setupDrawingTab(control_panel)
        self.setupImageTab(control_panel)

        # 右侧结果显示
        right_container = QWidget()
        right_container.setMinimumWidth(400)
        result_panel = QVBoxLayout(right_container)


        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        result_panel.addWidget(self.canvas)

        # main_layout.addWidget(control_panel, stretch=1)
        # main_layout.addWidget(right_container, stretch=1)

        # 控制按钮
        self.apply_btn = QPushButton("应用PSF运算")
        self.progress = QProgressBar()
        self.progress.setVisible(False)

        result_panel.addWidget(self.progress)
        result_panel.addWidget(self.apply_btn)

        main_layout.addWidget(control_panel, 1)
        main_layout.addWidget(right_container, 1)

    # 选项卡 PSF设置
    def setupPSFTab(self, tab_widget):
        psf_tab = QWidget()
        main_layout = QVBoxLayout(psf_tab)

        # 公共参数初始化
        self.psf_size = QSpinBox()
        self.psf_size.setRange(32, 512)
        self.psf_size.setValue(128)
        # 私有参数初始化
        self.aperture = QLineEdit("0.1")
        self.wavelength = QLineEdit("500e-9")
        self.sigma = QLineEdit("2.0")
        self.motion_length = QLineEdit("20")
        self.motion_angle = QLineEdit("0")

        # 公共参数区域
        public_group = QGroupBox("公共参数")

        public_layout = QFormLayout()
        public_layout.addRow("PSF大小:", self.psf_size)
        public_group.setLayout(public_layout)
        main_layout.addWidget(public_group)


        # 方法选择区域
        private_group = QGroupBox("方法参数")
        private_layout = QVBoxLayout()


        # 区域尺寸设计
        public_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # 垂直方向保持最小必要高度
        public_group.setMaximumHeight(200)
        main_layout.addWidget(public_group, stretch=1)

        private_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(private_group, stretch=3)

        # 方法选择
        self.psf_type = QComboBox()
        self.psf_type.addItems(["艾里斑", "高斯", "运动模糊"])
        private_layout.addWidget(self.psf_type)

        # 创建堆叠窗口存放不同方法的参数
        self.param_stack = QStackedWidget()
        self.param_stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # 艾里斑参数页
        airy_page = QWidget()
        airy_layout = QFormLayout()
        airy_layout.addRow("孔径(m):", self.aperture)
        airy_layout.addRow("波长(m):", self.wavelength)
        airy_page.setLayout(airy_layout)
        self.param_stack.addWidget(airy_page)

        # 高斯参数页
        gauss_page = QWidget()
        gauss_layout = QFormLayout()
        gauss_layout.addRow("Sigma(px):", self.sigma)
        gauss_page.setLayout(gauss_layout)
        self.param_stack.addWidget(gauss_page)

        # 运动模糊参数页
        motion_page = QWidget()
        motion_layout = QFormLayout()
        motion_layout.addRow("运动长度(px):", self.motion_length)
        motion_layout.addRow("运动角度(°):", self.motion_angle)
        motion_page.setLayout(motion_layout)
        self.param_stack.addWidget(motion_page)

        private_layout.addWidget(self.param_stack)
        private_group.setLayout(private_layout)
        main_layout.addWidget(private_group)

        # 生成按钮
        self.generate_psf_btn = QPushButton("生成PSF")
        main_layout.addWidget(self.generate_psf_btn)

        # 连接信号
        self.psf_type.currentIndexChanged.connect(self.param_stack.setCurrentIndex)

        tab_widget.addTab(psf_tab, "PSF设置")

    # 选项卡 绘图
    def setupDrawingTab(self, tab_widget):
        drawing_tab = QWidget()
        layout = QVBoxLayout()

        self.drawing_widget = DrawingWidget(512)
        layout.addWidget(self.drawing_widget)
        # 创建工具栏
        tool_layout = QHBoxLayout()
        self.pen_btn = QPushButton("画笔")
        self.line_btn = QPushButton("直线")
        self.rect_btn = QPushButton("矩形")
        self.ellipse_btn = QPushButton("椭圆")
        self.fill_btn = QPushButton("填充")
        self.clear_btn = QPushButton("清空")

        tool_layout.addWidget(self.pen_btn)
        tool_layout.addWidget(self.line_btn)
        tool_layout.addWidget(self.rect_btn)
        tool_layout.addWidget(self.ellipse_btn)
        tool_layout.addWidget(self.fill_btn)
        tool_layout.addWidget(self.clear_btn)



        # 连接信号
        self.pen_btn.clicked.connect(lambda: self.drawing_widget.setTool("pen"))
        self.line_btn.clicked.connect(lambda: self.drawing_widget.setTool("line"))
        self.rect_btn.clicked.connect(lambda: self.drawing_widget.setTool("rect"))
        self.ellipse_btn.clicked.connect(lambda: self.drawing_widget.setTool("ellipse"))
        self.fill_btn.clicked.connect(lambda: self.drawing_widget.setTool("fill"))
        self.clear_btn.clicked.connect(self.drawing_widget.clear)

        layout.addLayout(tool_layout)
        drawing_tab.setLayout(layout)
        tab_widget.addTab(drawing_tab, "图形绘制")

    # 选项卡 图片上传
    def setupImageTab(self, tab_widget):
        image_tab = QWidget()
        layout = QFormLayout()

        self.image_loader = ImageLoader()
        layout.addWidget(self.image_loader)

        self.scale_input = QLineEdit("1.0")
        layout.addRow("比例(微米/像素):", self.scale_input)

        image_tab.setLayout(layout)
        tab_widget.addTab(image_tab, "图像上传")

    def initConnections(self):
        self.generate_psf_btn.clicked.connect(self.generatePSF)
        self.drawing_widget.imageUpdated.connect(self.handleDrawingUpdate)
        self.image_loader.imageLoaded.connect(self.handleImageLoad)
        self.psf_type.currentIndexChanged.connect(self.updateParamVisibility)
        self.apply_btn.clicked.connect(self.startConvolution)

    def updateParamVisibility(self):
        # 根据PSF类型显示/隐藏参数（需要完善）
        pass

    def generatePSF(self):
        psf_type = self.psf_type.currentText()
        size = self.psf_size.value()

        if psf_type == "艾里斑":
            self.current_psf = PSFGenerator.generate_airy(
                size=size,
                D=float(self.aperture.text()),
                wavelength=float(self.wavelength.text())
            )
        elif psf_type == "高斯":
            self.current_psf = PSFGenerator.generate_gaussian(
                size=size,
                sigma=float(self.sigma.text())
            )
        elif psf_type == "运动模糊":
            self.current_psf = PSFGenerator.generate_motion_blur(
                length=int(self.motion_length.text()),
                angle=float(self.motion_angle.text()),
                size=size
            )

        self.updateResult()

    def startConvolution(self):
        if self.input_image is None:
            QMessageBox.warning(self, "错误", "请先绘制图形或上传图像!")
            return
        if self.current_psf is None:
            QMessageBox.warning(self, "错误", "请先生成PSF核!")
            return

        # 创建进度对话框
        self.progress_dialog = QProgressDialog("正在处理...", "取消", 0, 100, self)
        self.progress_dialog.setWindowTitle("处理进度")
        self.progress_dialog.setWindowModality(Qt.WindowModal)

        # 创建工作线程
        self.worker_thread = QThread()
        self.worker = ConvolutionWorker(
            self.input_image,
            self.current_psf,
            self.scale_factor
        )
        self.worker.moveToThread(self.worker_thread)

        # 连接信号
        self.worker.progress_updated.connect(self.updateProgress)
        self.worker.result_ready.connect(self.handleResult)
        self.worker.error_occurred.connect(self.handleError)
        self.worker_thread.started.connect(self.worker.process)

        # 启动线程
        self.worker_thread.start()
        self.progress_dialog.show()

    def updateProgress(self, value):
        self.progress_dialog.setValue(value)

    def handleResult(self, result):
        self.worker_thread.quit()
        self.worker_thread.wait()
        self.progress_dialog.close()

        # 显示结果
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.imshow(result, cmap='gray')
        self.canvas.draw()

    def handleError(self, message):
        self.worker_thread.quit()
        self.progress_dialog.close()
        QMessageBox.critical(self, "错误", message)

    def handleDrawingUpdate(self, image):
        self.input_image = image
        self.updateResult()

    def handleImageLoad(self, image):
        self.input_image = image
        self.scale_factor = float(self.scale_input.text())
        self.updateResult()

    def updateResult(self):
        if self.input_image is None or self.current_psf is None:
            return

        # 执行卷积
        result = ConvolutionHandler.convolve(
            self.input_image,
            self.current_psf,
            scale_factor=self.scale_factor
        )

        # 更新绘图
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.imshow(result, cmap='hot', vmin=0, vmax=1)
        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())