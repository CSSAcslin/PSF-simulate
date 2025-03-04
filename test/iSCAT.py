import numpy as np
from scipy.special import jn
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout,
                             QSlider, QWidget, QComboBox)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


def generate_psf(mode, ** params):
    """生成3D PSF数据（支持贝塞尔和高斯两种模式）"""
    # 通用参数
    nx, ny, nz = params['nx'], params['ny'], params['nz']
    dx, dy, dz = params['dx'], params['dy'], params['dz']
    wavelength = params['wavelength']

    # 坐标网格（三维中心对称）
    x = (np.arange(nx) - nx // 2) * dx
    y = (np.arange(ny) - ny // 2) * dy
    z = (np.arange(nz) - nz // 2) * dz

    if mode == 'bessel':
        # 贝塞尔干涉模式
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        r_xy = np.sqrt(X ** 2 + Y ** 2)  # 横向径向距离
        k = 2 * np.pi / wavelength

        # 贝塞尔函数（二维）与轴向干涉的组合
        bessel = jn(params['n_bessel'], k * r_xy)
        interference = np.cos(k * Z + params['phase_shift'])
        psf = params['amplitude'] * bessel * interference

    elif mode == 'gaussian':
        # 高斯衍射模式
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)  # 三维径向距离

        # 根据阿贝衍射极限计算标准差
        fwhm = wavelength / 2
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))  # FWHM = 2.355σ
        psf = np.exp(-r ** 2 / (2 * sigma ** 2))

    # 统一后处理
    psf = np.abs(psf)  # 取绝对值保证非负
    psf /= psf.max()  # 归一化到[0,1]
    return psf.astype(np.float32)


class PSFViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        # 初始化参数
        self.params = {
            'nx': 128, 'ny': 128, 'nz': 64,
            'dx': 0.1e-6, 'dy': 0.1e-6, 'dz': 0.05e-6,
            'n_bessel': 1, 'amplitude': 1.0,
            'wavelength': 633e-9, 'phase_shift': np.pi / 4
        }
        self.mode = 'bessel'
        self.initUI()
        self.update_psf()

    def initUI(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()

        # 模式选择
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(['bessel', 'gaussian'])
        self.mode_selector.currentTextChanged.connect(self.change_mode)
        layout.addWidget(self.mode_selector)

        # 可视化组件
        self.fig = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        # Z轴滑动条
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setTickInterval(5)
        layout.addWidget(self.slider)

        main_widget.setLayout(layout)
        self.setWindowTitle("3D PSF Viewer")
        self.resize(800, 600)

    def change_mode(self, mode):
        """切换PSF生成模式"""
        self.mode = mode
        self.update_psf()

    def update_psf(self):
        """生成新PSF数据并更新显示"""
        self.psf = generate_psf(self.mode, ** self.params)
        self.vmin, self.vmax = np.min(self.psf), np.max(self.psf)
        self.slider.setMaximum(self.params['nz'] - 1)
        self.slider.valueChanged.connect(self.update_display)
        self.update_display(0)

    def update_display(self, z_index):
        """更新可视化显示"""
        # 获取当前层数据
        layer = self.psf[:, :, z_index]

        # 清除旧图
        self.fig.clear()

        # 绘制新图
        ax = self.fig.add_subplot(111)
        im = ax.imshow(layer,
                       cmap='viridis',
                       vmin=self.vmin,
                       vmax=self.vmax,
                       # extent=[-self.params['dx'] * 64, self.params['dx'] * 64,
                       #         -self.params['dy'] * 64, self.params['dy'] * 64],
                       # origin='lower'
                       )

        # 添加标注
        ax.set_title(f"{self.mode.capitalize()} PSF (z={z_index})")
        ax.set_xlabel("X position (μm)")
        ax.set_ylabel("Y position (μm)")
        self.fig.colorbar(im, ax=ax, label='Intensity')
        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication([])
    viewer = PSFViewer()
    viewer.show()
    app.exec_()