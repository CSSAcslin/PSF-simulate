import numpy as np
from scipy.special import jv, jn_zeros
from scipy.fft import fftn, ifftn, fftfreq

# 通用参数初始化示例
params = {
    'lambda_': 500e-9,  # 波长 (m)
    'NA': 1.4,  # 数值孔径
    'n_immersion': 1.515,  # 浸没介质折射率
    'pixel_size': 100e-9,  # 像素尺寸 (m)
    'volume_shape': (256, 256, 64),  # (x_size, y_size, z_size)
    'z_step': 0.1e-6  # 轴向步长 (m)
}


# ===================== Richards & Wolf 矢量衍射模型 =====================
def richards_wolf_psf(lambda_, NA, n_immersion, pixel_size, volume_shape):
    """矢量衍射模型 (需GPU加速)"""
    k = 2 * np.pi / lambda_
    alpha = np.arcsin(NA / n_immersion)

    # 生成空间网格
    x = np.arange(-volume_shape[0] // 2, volume_shape[0] // 2) * pixel_size
    y = np.arange(-volume_shape[1] // 2, volume_shape[1] // 2) * pixel_size
    z = np.arange(-volume_shape[2] // 2, volume_shape[2] // 2) * pixel_size
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # 极坐标离散化 (示例简化)
    theta = np.linspace(0, alpha, 100)
    phi = np.linspace(0, 2 * np.pi, 100)
    Theta, Phi = np.meshgrid(theta, phi)

    # 矢量衍射积分 (伪代码示意)
    I0 = np.zeros_like(X)
    I1 = np.zeros_like(X)
    for t, p in zip(Theta.ravel(), Phi.ravel()):
        # 偏振项计算
        phase = k * (X * np.sin(t) * np.cos(p) + Y * np.sin(t) * np.sin(p) + Z * np.cos(t))
        I0 += np.sqrt(np.cos(t)) * jv(0, k * np.sin(t) * np.sqrt(X ** 2 + Y ** 2)) * np.exp(1j * phase)
        I1 += np.sqrt(np.cos(t)) * jv(1, k * np.sin(t) * np.sqrt(X ** 2 + Y ** 2)) * np.exp(1j * phase)

    PSF = np.abs(I0) ** 2 + 2 * np.abs(I1) ** 2
    return PSF / np.max(PSF)


# ===================== Gibson & Lanni 多层模型 =====================
def gibson_lanni_psf(lambda_, NA, n_layers, t_layers, pixel_size, volume_shape):
    """多层介质模型"""
    k = 2 * np.pi / lambda_
    x = np.fft.fftshift(np.fft.fftfreq(volume_shape[0], pixel_size / (2 * np.pi)))
    y = np.fft.fftshift(np.fft.fftfreq(volume_shape[1], pixel_size / (2 * np.pi)))
    X, Y = np.meshgrid(x, y, indexing='ij')
    rho = np.sqrt(X ** 2 + Y ** 2) / (NA * k)

    # 相位延迟计算
    W = 0
    for n, t in zip(n_layers, t_layers):
        W += k * t * np.sqrt(n ** 2 - (NA * rho) ** 2)

    # 标量衍射积分
    PSF_2D = np.abs(np.fft.ifft2(np.exp(1j * W))) ** 2
    return np.tile(PSF_2D, (volume_shape[2], 1, 1)).T  # 简化的轴向扩展


# ===================== 可变折射率扩展模型 =====================
def variable_refractive_psf(lambda_, NA, n_func, t_sample, volume_shape):
    """折射率轴向变化的模型"""
    z_positions = np.linspace(0, t_sample, volume_shape[2])
    PSF_vol = np.empty(volume_shape)

    for i, z in enumerate(z_positions):
        n_current = n_func(z)  # 用户自定义折射率函数
        # 调用Gibson & Lanni模型并修改折射率
        PSF_vol[:, :, i] = gibson_lanni_psf(lambda_, NA, [n_current], [z], ...)

    return PSF_vol


# ===================== Born & Wolf 标量模型 =====================
def born_wolf_psf(lambda_, NA, pixel_size, volume_shape):
    """经典标量模型"""
    k = 2 * np.pi / lambda_
    x = np.arange(-volume_shape[0] // 2, volume_shape[0] // 2) * pixel_size
    y = np.arange(-volume_shape[1] // 2, volume_shape[1] // 2) * pixel_size
    X, Y = np.meshgrid(x, y, indexing='ij')
    r = np.sqrt(X ** 2 + Y ** 2)

    # 解析积分近似
    PSF_2D = (2 * jv(1, k * NA * r) / (k * NA * r)) ** 2
    return np.tile(PSF_2D, (volume_shape[2], 1, 1)).T


# ===================== 傅里叶离焦模型 =====================
def fourier_defocus_psf(lambda_, NA, z_range, volume_shape):
    """频域离焦建模"""
    omega_x = fftfreq(volume_shape[0])
    omega_y = fftfreq(volume_shape[1])
    Wx, Wy = np.meshgrid(omega_x, omega_y)
    omega_r = np.sqrt(Wx ** 2 + Wy ** 2)

    PSF_vol = np.empty(volume_shape)
    for i, z in enumerate(z_range):
        sigma_z = 0.1 * abs(z)  # 示例参数
        h_z = np.exp(-sigma_z ** 2 * omega_r ** 2) * np.sinc(z * omega_r)
        PSF_vol[:, :, i] = np.abs(ifftn(h_z))

    return PSF_vol


# ===================== 高斯模型 =====================
def gaussian_psf(sigma_0, k_z, volume_shape):
    """轴向变化高斯PSF"""
    z = np.arange(volume_shape[2])
    sigma = sigma_0 + k_z * z
    PSF_vol = np.empty(volume_shape)

    for i in range(volume_shape[2]):
        x = np.arange(-volume_shape[0] // 2, volume_shape[0] // 2)
        y = np.arange(-volume_shape[1] // 2, volume_shape[1] // 2)
        X, Y = np.meshgrid(x, y)
        PSF_vol[:, :, i] = np.exp(-(X ** 2 + Y ** 2) / (2 * sigma[i] ** 2))

    return PSF_vol


# ===================== 参数初始化示例 =====================
if __name__ == "__main__":
    # Richards & Wolf 参数
    psf_rw = richards_wolf_psf(
        lambda_=500e-9,
        NA=1.4,
        n_immersion=1.515,
        pixel_size=100e-9,
        volume_shape=(256, 256, 64)
    )

    # Gibson & Lanni 参数
    psf_gl = gibson_lanni_psf(
        lambda_=488e-9,
        NA=1.2,
        n_layers=[1.33, 1.52, 1.515],  # 样品/玻片/浸没层
        t_layers=[10e-6, 170e-6, 100e-6],  # 厚度
        pixel_size=65e-9,
        volume_shape=(512, 512, 32)
    )

    # 高斯模型参数
    psf_gauss = gaussian_psf(
        sigma_0=1.0,
        k_z=0.05,
        volume_shape=(128, 128, 16)
    )