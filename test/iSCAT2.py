import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv


def born_wolf_iPSF(NA=1.4, wavelength=500e-9, n=1.518, z=0,
                   size=5e-6, pixels=512, theta_steps=200):
    """
    Born & Wolf PSF模型生成干涉点扩散函数

    参数:
        NA: 数值孔径
        wavelength: 波长（米）
        n: 物镜浸没介质折射率
        z: 离焦量（米）
        size: 成像区域尺寸（米）
        pixels: 图像分辨率
        theta_steps: 角向积分步数

    返回:
        iPSF: 干涉强度分布
    """
    k = 2 * np.pi * n / wavelength
    alpha = np.arcsin(NA / n)

    # 生成坐标网格
    x = np.linspace(-size / 2, size / 2, pixels)
    y = x.copy()
    X, Y = np.meshgrid(x, y)
    r = np.sqrt(X ** 2 + Y ** 2)

    # 初始化电场
    E_scat = np.zeros_like(r, dtype=np.complex128)

    # 角向积分
    theta = np.linspace(0, alpha, theta_steps)
    d_theta = alpha / theta_steps

    for theta_i in theta:
        # Born & Wolf核心积分项
        kr = k * r * np.sin(theta_i)
        phase = np.exp(1j * k * z * np.cos(theta_i))

        # 角向积分权重
        integrand = np.sqrt(np.cos(theta_i)) * jv(0, kr) * phase * np.sin(theta_i)

        E_scat += integrand * d_theta

    # 标定参考光（假设单位振幅）
    E_ref = 1.0

    # 干涉强度计算
    I_interference = np.abs(E_ref + E_scat) ** 2

    return I_interference


# 参数设置示例
psf_params = {
    "NA": 1.4,
    "wavelength": 550e-9,
    "n": 1.518,
    "z": 0.5e-6,
    "size": 4e-6,
    "pixels": 128,
    "theta_steps": 3000
}

# 生成iPSF
iPSF = born_wolf_iPSF(**psf_params)

# 可视化
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(iPSF, cmap='inferno',
           extent=[-2e-6, 2e-6, -2e-6, 2e-6])
plt.title('Interferometric PSF')
plt.colorbar()

# 横向剖面
plt.subplot(122)
profile = iPSF[psf_params['pixels'] // 2, :]
plt.plot(np.linspace(-2e-6, 2e-6, psf_params['pixels']), profile)
plt.title('横向强度剖面')
plt.xlabel('Position (m)')
plt.tight_layout()
plt.show()