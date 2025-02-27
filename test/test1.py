import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j1
from scipy.signal import convolve2d
from PIL import Image  # 用于加载外部图像
from matplotlib.font_manager import FontProperties #处理字符显示问题

# ====================== 1. 生成艾里斑PSF核 ======================
def generate_airy_psf(size=256, wavelength=500e-9, D=0.1, f=1.0):
    """生成艾里斑PSF核（归一化）"""
    k = np.pi * D / (wavelength * f)
    x = np.linspace(-1e-5, 1e-5, size)
    y = np.linspace(-1e-5, 1e-5, size)
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt(xx**2 + yy**2)
    kr = k * r
    psf = (2 * j1(kr) / kr)**2
    psf[np.isnan(psf)] = 1.0  # 中心点
    psf /= psf.sum()  # 归一化以确保总能量不变
    return psf

# 生成PSF核（可调整size控制核大小）
psf_kernel = generate_airy_psf(size=128)
psf_kernel = psf_kernel / psf_kernel.max()  # 可选：将PSF归一化到0-1范围

# ====================== 2. 定义或加载输入图像 ======================
# 选项1：自定义绘制图形（示例：白色矩形）
image = np.zeros((512, 512))
image[200:300, 200:300] = 1.0  # 在中心绘制矩形

# 选项2：加载外部图像（取消注释以下代码）
# image_path = "your_image.jpg"  # 替换为你的图像路径
# image = np.array(Image.open(image_path).convert('L'))  # 转为灰度
# image = image / 255.0  # 归一化到0-1范围

# ====================== 3. 应用PSF卷积 ======================
# 对输入图像进行卷积（使用FFT加速的卷积，边界处理为'same'）
blurred_image = convolve2d(image, psf_kernel, mode='same', boundary='symm')

# ====================== 4. 可视化结果 ======================
plt.figure(figsize=(12, 6))
my_font = FontProperties(fname=r"c:\windows\fonts\msyh.ttc", size=12)
# 原始图像
plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('原始图像',fontproperties=my_font)
plt.axis('off')

# PSF核
plt.subplot(1, 3, 2)
plt.imshow(psf_kernel, cmap='hot')
plt.title('艾里斑PSF核',fontproperties=my_font)
plt.axis('off')

# 模糊后图像
plt.subplot(1, 3, 3)
plt.imshow(blurred_image, cmap='gray')
plt.title('PSF卷积后图像',fontproperties=my_font)
plt.axis('off')

plt.tight_layout()
plt.show()