import numpy as np
from scipy.signal import fftconvolve


class ConvolutionHandler:
    @staticmethod
    def convolve(image, psf, scale_factor=1.0, z_index=None, progress_callback=None):
        # 验证输入
        # if image.ndim != 2:
        #     raise ValueError("输入图像必须是二维灰度图")
        if psf.ndim not in [2, 3]:
            raise ValueError("PSF必须是二维或三维数组")
        if psf.ndim == 3 and z_index is None:
            raise ValueError("三维PSF需要指定z_index参数")

        # 调整PSF尺寸
        scaled_psf = psf  # 实际应实现物理缩放逻辑

        # 处理不同维度的PSF
        if psf.ndim == 3:
            if z_index < 0 or z_index >= psf.shape[2]:
                raise ValueError("z_index超出有效范围")
            current_psf = scaled_psf[:, :, z_index]
        else:
            current_psf = scaled_psf

        # 使用FFT加速卷积
        result = fftconvolve(image, scaled_psf, mode='same')

        if progress_callback:
            progress_callback(50)  # 模拟进度

        # 后处理
        result = np.clip(result, 0, 1)
        return result