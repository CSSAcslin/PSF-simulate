import sys

import matplotlib.font_manager
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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


        self.figure = plt.figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        result_panel.addWidget(self.canvas)

        # main_layout.addWidget(control_panel, stretch=1)
        # main_layout.addWidget(right_container, stretch=1)
        # Z轴滑动条
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setTickInterval(5)
        self.slider.setValue(32)
        result_panel.addWidget(self.slider)
        self.slider.setEnabled(False)

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
        self.psf_xy = QSpinBox()
        self.psf_xy.setRange(32, 512)
        self.psf_xy.setValue(128)
        self.psf_z = QLineEdit("64")
        self.psf_dxdy = QLineEdit("0.1e-6")
        self.psf_dz = QLineEdit("0.05e-6")
        self.wavelength = QLineEdit("500e-9")
        self.amplitude = QLineEdit("1")
        # 私有参数初始化
        self.n_bessel = QLineEdit("0")
        self.phase_shift = QLineEdit("90.0")
        self.aperture = QLineEdit("0.1")
        self.sigma = QLineEdit("2.0")
        self.motion_length = QLineEdit("20")
        self.motion_angle = QLineEdit("0")

        # 公共参数区域
        public_group = QGroupBox("公共参数")

        public_layout = QFormLayout()
        public_layout.addRow("PSF生成尺寸xy:", self.psf_xy)
        public_layout.addRow("生成尺寸z:", self.psf_z)
        public_layout.addRow("PSF单位像素长度(m):", self.psf_dxdy)
        public_layout.addRow("PSF单位像素高度(m):", self.psf_dz)
        public_layout.addRow("波长(m):", self.wavelength)
        public_layout.addRow("放大倍数:", self.amplitude)
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
        main_layout.addWidget(private_group, stretch=1)

        # 方法选择
        self.psf_type = QComboBox()
        self.psf_type.addItems(["Bessel 衍射", "Gaussian 衍射", "艾里斑", "高斯", "运动模糊"])
        private_layout.addWidget(self.psf_type)

        # 创建堆叠窗口存放不同方法的参数
        self.param_stack = QStackedWidget()
        self.param_stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Bessel
        bessel_page = QWidget()
        bessel_layout = QFormLayout()
        bessel_layout.addRow("贝塞尔阶数:", self.n_bessel)
        bessel_layout.addRow("相位移(°):", self.phase_shift)
        bessel_page.setLayout(bessel_layout)
        self.param_stack.addWidget(bessel_page)

        # Gaussian
        gaussian_page = QWidget()
        gaussian_layout = QFormLayout()
        # gaussian_layout.addRow(":", self.n_bessel)
        gaussian_page.setLayout(gaussian_layout)
        self.param_stack.addWidget(gaussian_page)

        # 艾里斑参数页
        airy_page = QWidget()
        airy_layout = QFormLayout()
        airy_layout.addRow("孔径(m):", self.aperture)
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

        # +++ 新增三维绘图选项(v0.2.0) +++
        dimension_layout = QHBoxLayout()
        self.dim_checkbox = QCheckBox("启用三维绘图")
        self.z_depth_label = QLabel("Z轴厚度:")
        self.z_depth_input = QSpinBox()
        self.z_depth_input.setMinimum(1)
        self.z_depth_input.setValue(1)
        self.z_depth_input.setEnabled(False)
        self.z_depth_label.setVisible(False)
        self.z_depth_input.setVisible(False)

        # 连接复选框状态变化信号
        self.dim_checkbox.stateChanged.connect(self.toggle_3d_options)
        self.dim_checkbox.stateChanged.connect(self.update_drawing_3d_params) #3d信号
        self.z_depth_input.valueChanged.connect(self.update_drawing_3d_params)

        dimension_layout.addWidget(self.dim_checkbox)
        dimension_layout.addWidget(self.z_depth_label)
        dimension_layout.addWidget(self.z_depth_input)
        dimension_layout.addStretch()
        layout.addLayout(dimension_layout)

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

    def toggle_3d_options(self, state):
        """切换三维选项可见性"""
        is_3d = state == Qt.Checked
        self.z_depth_label.setVisible(is_3d)
        self.z_depth_input.setVisible(is_3d)
        self.z_depth_input.setEnabled(is_3d)

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
        # 根据PSF类型显示/隐藏参数（目前不需要完善）
        pass

    def generatePSF(self):
        psf_type = self.psf_type.currentText()
        size_xy = self.psf_xy.value()
        size_z = float(self.psf_z.text())
        size_dxdy = float(self.psf_dxdy.text())
        size_dz = float(self.psf_dz.text())
        amplitude = float(self.amplitude.text())


        if psf_type == "Bessel 衍射":
            self.current_psf = PSFGenerator.generate_bessel(
                size = size_xy,
                size_z = size_z,
                size_dxdy =size_dxdy,
                size_dz = size_dz,
                amplitude = amplitude,
                wavelength = float(self.wavelength.text()),
                n_bessel = float(self.n_bessel.text()),
                phase_shift = float(self.phase_shift.text())
            )
        elif psf_type == "Gaussian 衍射":
            self.current_psf = PSFGenerator.generate_gaussian(
                size = size_xy,
                size_z = size_z,
                size_dxdy = size_dxdy,
                size_dz = size_dz,
                amplitude = amplitude,
                wavelength = float(self.wavelength.text()),
            )
        elif psf_type == "艾里斑":
            self.current_psf = PSFGenerator.generate_airy(
                size=size_xy,
                size_z=size_z,
                size_dxdy=size_dxdy,
                size_dz=size_dz,
                amplitude=amplitude,
                wavelength=float(self.wavelength.text()),
                D=float(self.aperture.text()),
            )
        elif psf_type == "高斯":
            self.current_psf = PSFGenerator.generate_gaussian_old(
                size=size_xy,
                # size_z=size_z,
                # size_dxdy=size_dxdy,
                # size_dz=size_dz,
                # amplitude=amplitude,
                # wavelength=float(self.wavelength.text()),
                sigma=float(self.sigma.text())
            )
        elif psf_type == "运动模糊":
            self.current_psf = PSFGenerator.generate_motion_blur(
                size=size_xy,
                # size_z=size_z,
                # size_dxdy=size_dxdy,
                # size_dz=size_dz,
                # amplitude=amplitude,
                # wavelength=float(self.wavelength.text()),
                length=int(self.motion_length.text()),
                angle=float(self.motion_angle.text()),
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

    def get_drawing_3d_params(self):
        # """返回绘图相关的三维参数"""
        return {
            'is_3d': self.dim_checkbox.isChecked(),
            'z_depth': self.z_depth_input.value()
        }

    def update_drawing_3d_params(self):
        # """当参数变化时更新绘图控件"""
        params = self.get_drawing_3d_params()
        self.drawing_widget.set_3d_params(params['is_3d'], params['z_depth'])
        self.image_loader.set_3d_params(params['is_3d'], params['z_depth'])

    def enable_3d_visualization(self, depth):
        # """激活三维可视化组件"""
        self.slider.setRange(0, depth - 1)
        self.slider.setEnabled(True)
        self.current_z_layer = depth / 2
        self.slider.setValue(self.current_z_layer)

    def disable_3d_visualization(self):
        # """禁用三维可视化组件"""
        self.slider.setEnabled(False)
        self.current_z_layer = None

    def updateResult(self):
        if self.input_image is None or self.current_psf is None:
            return

        # 执行卷积
        result = ConvolutionHandler.convolve(
            self.input_image,
            self.current_psf,
            scale_factor=self.scale_factor
        )
        # 根据输入维度调整处理逻辑
        if self.input_image.ndim == 3:
            # 处理三维输入的逻辑


            self.enable_3d_visualization(self.input_image.shape[2])
        else:
            # 原有二维处理逻辑


            self.disable_3d_visualization()

        # 更新绘图
        font1 = matplotlib.font_manager.FontProperties(fname= r"C:\Windows\Fonts\msyh.ttc")

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        image = ax.imshow(result,
                  cmap='viridis',
                  vmin=0,
                  vmax=1,
                  )
        ax_cb = inset_axes(ax, width="3%", height="100%", loc='lower left', bbox_to_anchor=(1.02, 0., 1, 1),bbox_transform=ax.transAxes, borderpad=0)
        self.figure.colorbar(image, ax=ax, cax=ax_cb, label = 'Intensity')
        ax.set_title(f"{self.psf_type.currentText()} PSF (z={self.current_z_layer})",fontproperties=font1)
        ax.set_xlabel("X position (μm)")
        ax.set_ylabel("Y position (μm)")
        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())