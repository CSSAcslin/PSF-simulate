import numpy as np
from scipy.signal import fftconvolve


class ConvolutionHandler:
    @staticmethod
    def convolve(image, psf, scale_factor=1.0, progress_callback=None):
        # 验证输入
        if image.ndim != 2:
            raise ValueError("输入图像必须是二维灰度图")
        if psf.shape[0] != psf.shape[1]:
            raise ValueError("PSF核必须是正方形")

        # 调整PSF尺寸
        scaled_psf = psf  # 实际应实现物理缩放逻辑

        # 使用FFT加速卷积
        result = fftconvolve(image, scaled_psf, mode='same')

        if progress_callback:
            progress_callback(50)  # 模拟进度

        # 后处理
        result = np.clip(result, 0, 1)
        return result