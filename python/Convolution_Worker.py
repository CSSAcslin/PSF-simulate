from PyQt5.QtCore import QObject, pyqtSignal
import numpy as np
from convolution_handler import ConvolutionHandler
class ConvolutionWorker(QObject):
    progress_updated = pyqtSignal(int)
    result_ready = pyqtSignal(np.ndarray)
    error_occurred = pyqtSignal(str)

    def __init__(self, image, psf, scale_factor):
        super().__init__()
        self.image = image
        self.psf = psf
        self.scale_factor = scale_factor
        self._is_running = True

    def process(self):
        try:
            # 模拟进度更新
            self.progress_updated.emit(10)

            # 执行卷积
            result = ConvolutionHandler.convolve(
                self.image,
                self.psf,
                self.scale_factor,
                progress_callback=self.update_progress
            )

            self.progress_updated.emit(100)
            self.result_ready.emit(result)

        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            self._is_running = False

    def update_progress(self, value):
        if self._is_running:
            self.progress_updated.emit(value)

    def stop(self):
        self._is_running = False