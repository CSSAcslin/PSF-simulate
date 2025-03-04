import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv  # 贝塞尔函数用于轴对称积分


def generate_iPSF_3D(NA=0.8, wavelength=0.5e-6, n=1.33, z_range=2e-6, pixel_size=0.1e-6, grid_size=64):
    """
    生成iSCAT的三维iPSF分布（简化模型）

    参数:
        NA (float): 数值孔径
        wavelength (float): 光波长（米）
        n (float): 介质折射率
        z_range (float): 轴向扫描范围（米）
        pixel_size (float): 像素尺寸（米）
        grid_size (int): 网格尺寸（N x N x N_z）

    返回:
        iPSF_3D (ndarray): 三维iPSF强度分布 [Z, Y, X]
    """
    k = 2 * np.pi * n / wavelength  # 波数
    alpha = np.arcsin(NA / n)  # 最大接收角

    # 生成空间坐标网格
    x = y = np.linspace(-grid_size // 2 * pixel_size, grid_size // 2 * pixel_size, grid_size)
    z = np.linspace(-z_range / 2, z_range / 2, grid_size)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # 径向坐标r和方位角phi
    r = np.sqrt(X ** 2 + Y ** 2)
    phi = np.arctan2(Y, X)

    # 初始化场分布
    E_ref = 1.0  # 假设参考场为均匀场（简化）
    E_scat = np.zeros_like(X, dtype=np.complex128)

    # 数值积分参数（θ离散化）
    theta_steps = 50
    theta = np.linspace(0, alpha, theta_steps)
    d_theta = alpha / theta_steps

    # 矢量衍射积分（轴对称简化）
    for theta_i in theta:
        # 离焦相位项: exp(ikz cosθ)
        phase_z = np.exp(1j * k * Z * np.cos(theta_i))

        # 透射系数简化模型（假设t_p=1, t_s=1）
        t_coeff = 1.0  # 更复杂模型需替换为θ的函数

        # 径向积分项（贝塞尔函数展开）
        kr = k * r * np.sin(theta_i)
        J0 = jv(0, kr)  # 0阶贝塞尔函数

        # 积分累加
        E_scat += (np.sin(theta_i) * np.sqrt(np.cos(theta_i)) * t_coeff *
                   J0 * phase_z) * d_theta

    # 干涉强度计算
    I_total = np.abs(E_ref + E_scat) ** 2

    return I_total


# 参数设置
params = {
    'NA': 0.9,
    'wavelength': 500e-9,
    'n': 1.33,
    'z_range': 2e-6,
    'pixel_size': 100e-9,
    'grid_size': 64
}

# 生成iPSF
iPSF_3D = generate_iPSF_3D(**params)

# 可视化中间切片
z_slice = iPSF_3D[params['grid_size'] // 2, :, :]
plt.imshow(z_slice, cmap='viridis',
           extent=[-params['grid_size'] // 2 * params['pixel_size'] * 1e6,
                   params['grid_size'] // 2 * params['pixel_size'] * 1e6,
                   -params['grid_size'] // 2 * params['pixel_size'] * 1e6,
                   params['grid_size'] // 2 * params['pixel_size'] * 1e6])
plt.xlabel('X (μm)')
plt.ylabel('Y (μm)')
plt.colorbar(label='Intensity (a.u.)')
plt.title('iPSF XY Slice at z=0')
plt.show()