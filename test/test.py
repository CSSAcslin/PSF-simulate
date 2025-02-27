import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j1  # 第一类贝塞尔函数
from matplotlib.font_manager import FontProperties #处理字符显示问题


# 参数设置
lambda_wavelength = 500e-9  # 光波长 (500 nm)
D = 0.01                     # 孔径直径 (0.1 m)
f = 0.1                     # 焦距 (1 m)
k = np.pi * D / (lambda_wavelength * f)  # 波数相关参数

# 生成二维坐标网格
N = 512                     # 图像分辨率
x = np.linspace(-1e-5, 1e-5, N)  # x范围（单位：米）
y = np.linspace(-1e-5, 1e-5, N)  # y范围（单位：米）
xx, yy = np.meshgrid(x, y)
r = np.sqrt(xx**2 + yy**2)  # 计算径向距离

# 计算艾里斑强度分布
kr = k * r
airy_disk = (2 * j1(kr) / kr)**2  # 艾里斑公式
airy_disk[np.isnan(airy_disk)] = 1.0  # 处理r=0处的NaN（中心点）

# 可视化
plt.figure(figsize=(10, 8))

my_font = FontProperties(fname=r"c:\windows\fonts\msyh.ttc", size=12)
# 绘制二维艾里斑
plt.subplot(2, 1, 1)
plt.imshow(airy_disk, cmap='hot', extent=[-1e5*x[-1], 1e5*x[-1], -1e5*y[-1], 1e5*y[-1]])
plt.colorbar(label='intensity')
plt.xlabel('x (微米)', fontproperties=my_font)
plt.ylabel('y (微米)', fontproperties=my_font)
plt.title('艾里斑的二维强度分布', fontproperties=my_font)

# 绘制横向截面（一维强度分布）
plt.subplot(2, 1, 2)
profile = airy_disk[N//2, :]  # 中心水平线
plt.plot(1e6 * x, profile / np.max(profile), 'r-', lw=2)  # 归一化并转换为微米
plt.xlabel('x (μm)')
plt.ylabel('归一化强度', fontproperties=my_font)
plt.title('横向一维强度分布', fontproperties=my_font)
plt.tight_layout()
plt.show()