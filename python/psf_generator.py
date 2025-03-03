import numpy as np
from scipy.special import j1, jn
from scipy.ndimage import rotate

class PSFGenerator:

    def __init__(self):
        self.params = {
            'nx': 128, 'ny': 128, 'nz': 64,
            'dx': 0.1e-6, 'dy': 0.1e-6, 'dz': 0.05e-6,
            'n_bessel': 0, 'amplitude': 1.0,
            'wavelength': 633e-9, 'phase_shift': np.pi / 4
        }
    @staticmethod
    def generate_bessel(size=128,size_z=64,size_dxdy = 0.1e-6,size_dz=0.05e-6,amplitude = 1,wavelength=500e-9,
                        n_bessel = 0,phase_shift = 90.0):
        # 贝塞尔干涉模式
        x = (np.arange(size) - size // 2) * size_dxdy
        y = (np.arange(size) - size // 2) * size_dxdy
        z = (np.arange(size_z) - size_z // 2) * size_dz
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        r_xy = np.sqrt(X ** 2 + Y ** 2)  # 横向径向距离
        k = 2 * np.pi / wavelength

        # 贝塞尔函数（二维）与轴向干涉的组合
        bessel = jn(n_bessel, k * r_xy)
        interference = np.cos(k * Z + phase_shift/180.0 * np.pi)
        psf = amplitude * bessel * interference
        psf = np.abs(psf)  # 取绝对值保证非负
        psf /= psf.max()  # 归一化到[0,1]
        return psf.astype(np.float32)

    @staticmethod
    def generate_gaussian(size=128,size_z=64,size_dxdy = 0.1e-6,size_dz=0.05e-6,amplitude = 1,wavelength=500e-9,
                          f=1.0):
        # 高斯衍射模式
        x,y = (np.arange(size) - size // 2) * size_dxdy
        z = (np.arange(size_z) - size_z // 2) * size_dz
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)  # 三维径向距离

        # 根据阿贝衍射极限计算标准差
        fwhm = wavelength / 2
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))  # FWHM = 2.355σ
        psf = np.exp(-r ** 2 / (2 * sigma ** 2))
        psf = np.abs(psf)  # 取绝对值保证非负
        psf /= psf.max()  # 归一化到[0,1]
        return psf.astype(np.float32)

    @staticmethod
    def generate_airy(size=128,size_z=64,size_dxdy = 0.1e-6,size_dz=0.05e-6,amplitude = 1,wavelength=500e-9,
                      D=0.1, f=1.0):
        k = np.pi * D / (wavelength * f)
        x = np.linspace(-1e-5, 1e-5, size)
        y = np.linspace(-1e-5, 1e-5, size)
        xx, yy = np.meshgrid(x, y)
        r = np.sqrt(xx**2 + yy**2)
        kr = k * r
        psf = (2 * j1(kr) / kr)**2
        psf[np.isnan(psf)] = 1.0
        return psf / psf.sum()

    @staticmethod
    def generate_gaussian_old(size=128, sigma=2.0):
        x = np.linspace(-size//2, size//2, size)
        xx, yy = np.meshgrid(x, x)
        psf = np.exp(-(xx**2 + yy**2)/(2*sigma**2))
        return psf / psf.sum()

    @staticmethod
    def generate_motion_blur(length=20, angle=0, size=128):
        kernel = np.zeros((size, size))
        center = size//2
        kernel[center-length//2:center+length//2, center] = 1
        kernel = rotate(kernel, angle, reshape=False)
        return kernel / kernel.sum()