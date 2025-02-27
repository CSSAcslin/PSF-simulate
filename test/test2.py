import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scipy.signal import convolve2d
from scipy.special import j1

# ====================== 1. 全局参数和PSF生成 ======================
# 图像参数
canvas_size = 512  # 画布大小
psf_size = 128     # PSF核大小

# 生成艾里斑PSF核
def generate_airy_psf(size=psf_size, wavelength=500e-9, D=0.1, f=1.0):
    k = np.pi * D / (wavelength * f)
    x = np.linspace(-1e-5, 1e-5, size)
    y = np.linspace(-1e-5, 1e-5, size)
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt(xx**2 + yy**2)
    kr = k * r
    psf = (2 * j1(kr) / kr)**2
    psf[np.isnan(psf)] = 1.0
    psf /= psf.sum()  # 归一化
    return psf

psf_kernel = generate_airy_psf()

# 初始化画布和图像
image = np.zeros((canvas_size, canvas_size))  # 初始全黑画布
drawing = False  # 标记是否正在绘制
last_point = None  # 记录上一个鼠标位置

# ====================== 2. 交互事件处理 ======================
def on_press(event):
    global drawing, last_point
    if event.inaxes != ax_original:  # 仅在左侧画布触发
        return
    drawing = True
    last_point = (int(event.xdata), int(event.ydata))

def on_motion(event):
    global last_point
    if not drawing or event.inaxes != ax_original:
        return
    x, y = int(event.xdata), int(event.ydata)
    if last_point:
        # 在两点之间画线（模拟画笔）
        dx, dy = x - last_point[0], y - last_point[1]
        steps = max(abs(dx), abs(dy))
        for i in range(steps + 1):
            xi = last_point[0] + int(i * dx / steps)
            yi = last_point[1] + int(i * dy / steps)
            if 0 <= xi < canvas_size and 0 <= yi < canvas_size:
                image[yi, xi] = 1.0  # 白色绘制
    last_point = (x, y)
    update_display()

def on_release(event):
    global drawing, last_point
    drawing = False
    last_point = None

def update_display():
    # 计算卷积并更新显示
    blurred = convolve2d(image, psf_kernel, mode='same', boundary='symm')
    ax_original.imshow(image, cmap='gray', vmin=0, vmax=1)
    ax_blurred.imshow(blurred, cmap='gray', vmin=0, vmax=1)
    plt.draw()

# ====================== 3. 创建交互界面 ======================
fig, (ax_original, ax_blurred) = plt.subplots(1, 2, figsize=(12, 6))
ax_original.set_title("drawing area(click and drag)")
ax_blurred.set_title("Airy Disk' PSF")
for ax in [ax_original, ax_blurred]:
    ax.axis('off')

# 绑定事件
fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('motion_notify_event', on_motion)
fig.canvas.mpl_connect('button_release_event', on_release)

# 添加清空按钮
def clear_canvas(event):
    global image
    image = np.zeros((canvas_size, canvas_size))
    update_display()

ax_clear = plt.axes([0.4, 0.05, 0.2, 0.05])  # 按钮位置
btn_clear = Button(ax_clear, 'clear all')
btn_clear.on_clicked(clear_canvas)

# 初始化显示
update_display()
plt.show()