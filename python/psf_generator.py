import numpy as np
from scipy.special import j1
from scipy.ndimage import rotate

class PSFGenerator:
    @staticmethod
    def generate_airy(size=128, D=0.1, wavelength=500e-9, f=1.0):
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
    def generate_gaussian(size=128, sigma=2.0):
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