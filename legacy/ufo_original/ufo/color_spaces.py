# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class RGB2Lab(nn.Module):
#     """
#     将 [B, 3, H, W] 的 RGB 图像转换为 Lab 颜色空间
#     输入范围: [0,1]
#     输出: L in [0,1], a in [-1,1], b in [-1,1] (归一化版本)
#     """

#     def __init__(self):
#         super(RGB2Lab, self).__init__()
#         # sRGB to XYZ 变换矩阵 (D65)
#         rgb_to_xyz = torch.tensor([
#             [0.412453, 0.357580, 0.180423],
#             [0.212671, 0.715160, 0.072169],
#             [0.019334, 0.119193, 0.950227]
#         ], dtype=torch.float32)

#         # 参考白点 D65
#         white_point = torch.tensor([0.95047, 1.0, 1.08883], dtype=torch.float32)
#         # 注册为缓冲区，这样它们会自动跟随模块移动到正确的设备
#         self.register_buffer('rgb_to_xyz', rgb_to_xyz)
#         self.register_buffer('white_point', white_point)

#     def _f(self, t):
#         delta = 6 / 29
#         return torch.where(t > delta ** 3, t.pow(1 / 3), (t / (3 * delta ** 2)) + (4 / 29))

#     def forward(self, img):
#         # img: [B, 3, H, W], 范围 [0,1]
#         B, C, H, W = img.shape
#         assert C == 3, "输入必须是RGB图像"

#         # 确保缓冲区在正确的设备上
#         device = img.device
#         rgb_to_xyz = self.rgb_to_xyz.to(device)
#         white_point = self.white_point.to(device)

#         # 1. sRGB -> Linear RGB
#         mask = (img <= 0.04045).float()
#         img_linear = mask * (img / 12.92) + (1 - mask) * (((img + 0.055) / 1.055) ** 2.4)

#         # 2. Linear RGB -> XYZ
#         img_flat = img_linear.permute(0, 2, 3, 1).reshape(-1, 3)  # [B*H*W, 3]
#         xyz = torch.matmul(img_flat, self.rgb_to_xyz.to(img.device).T)
#         # xyz = torch.matmul(img_flat, self.rgb_to_xyz.T)  # [B*H*W, 3]
#         xyz = xyz / self.white_point.to(img.device)  # 归一化白点
#         xyz = xyz.reshape(B, H, W, 3).permute(0, 3, 1, 2)  # [B, 3, H, W]

#         # 3. XYZ -> Lab
#         fx, fy, fz = self._f(xyz[:, 0]), self._f(xyz[:, 1]), self._f(xyz[:, 2])
#         L = 116 * fy - 16
#         a = 500 * (fx - fy)
#         b = 200 * (fy - fz)

#         lab = torch.stack([L, a, b], dim=1)

#         # 归一化处理
#         L_norm = lab[:, 0:1, :, :] / 100.0  # L→[0,1]
#         a_norm = lab[:, 1:2, :, :] / 128.0  # a→[-1,1]
#         b_norm = lab[:, 2:3, :, :] / 128.0  # b→[-1,1]
#         lab_norm = torch.cat([L_norm, a_norm, b_norm], dim=1)

#         return lab_norm


# class Lab2RGB(nn.Module):
#     """
#     将 [B, 3, H, W] 的 Lab 图像转换为 RGB 颜色空间
#     输入: L in [0,1], a in [-1,1], b in [-1,1] (归一化版本)
#     输出范围: [0,1]
#     """

#     def __init__(self):
#         super(Lab2RGB, self).__init__()
#         # XYZ to sRGB 变换矩阵 (D65)
#         xyz_to_rgb = torch.tensor([
#             [3.240479, -1.537150, -0.498535],
#             [-0.969256, 1.875992, 0.041556],
#             [0.055648, -0.204043, 1.057311]
#         ], dtype=torch.float32)

#         # 参考白点 D65
#         white_point = torch.tensor([0.95047, 1.0, 1.08883], dtype=torch.float32)
#         # 注册为缓冲区，这样它们会自动跟随模块移动到正确的设备
#         self.register_buffer('xyz_to_rgb', xyz_to_rgb)
#         self.register_buffer('white_point', white_point)

#     def _finv(self, t):
#         delta = 6 / 29
#         return torch.where(t > delta, t ** 3, 3 * delta ** 2 * (t - 4 / 29))

#     def forward(self, lab_norm):
#         # lab_norm: [B, 3, H, W], L in [0,1], a/b in [-1,1]
#         B, C, H, W = lab_norm.shape
#         assert C == 3, "输入必须是Lab图像"

#         # 确保缓冲区在正确的设备上
#         device = lab_norm.device
#         xyz_to_rgb = self.xyz_to_rgb.to(device)
#         white_point = self.white_point.to(device)

#         # 1. 反归一化处理
#         L = lab_norm[:, 0:1, :, :] * 100.0
#         a = lab_norm[:, 1:2, :, :] * 128.0
#         b = lab_norm[:, 2:3, :, :] * 128.0
#         lab = torch.cat([L, a, b], dim=1)

#         # 2. 确保范围约束（防止模型输出异常）
#         lab = torch.cat([
#             torch.clamp(lab[:, 0:1, :, :], 0.0, 100.0),
#             torch.clamp(lab[:, 1:, :, :], -128.0, 127.0)
#         ], dim=1)

#         L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]

#         # 3. Lab -> XYZ
#         fy = (L + 16) / 116
#         fx = fy + (a / 500)
#         fz = fy - (b / 200)

#         xr = self._finv(fx)
#         yr = self._finv(fy)
#         zr = self._finv(fz)

#         X = xr * self.white_point[0]
#         Y = yr * self.white_point[1]
#         Z = zr * self.white_point[2]

#         xyz = torch.stack([X, Y, Z], dim=1)  # [B, 3, H, W]

#         # 4. XYZ -> Linear RGB
#         xyz_flat = xyz.permute(0, 2, 3, 1).reshape(-1, 3)
#         rgb_linear = torch.matmul(xyz_flat, self.xyz_to_rgb.T)
#         rgb_linear = rgb_linear.reshape(B, H, W, 3).permute(0, 3, 1, 2)

#         # 5. Linear RGB -> sRGB
#         mask = (rgb_linear <= 0.0031308).float()
#         rgb = mask * (12.92 * rgb_linear) + (1 - mask) * (1.055 * torch.clamp(rgb_linear, min=0) ** (1 / 2.4) - 0.055)

#         # 限制到 [0,1]
#         rgb = torch.clamp(rgb, 0.0, 1.0)
#         return rgb




# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class RGB2Lab(nn.Module):
#     """
#     将 [B, 3, H, W] 的 RGB 图像转换为 Lab 颜色空间
#     输入范围: [0,1]
#     输出: L in [0,1], a in [-1,1], b in [-1,1] (归一化版本)
#     """

#     def __init__(self):
#         super(RGB2Lab, self).__init__()
#         # sRGB to XYZ 变换矩阵 (D65)
#         rgb_to_xyz = torch.tensor([
#             [0.412453, 0.357580, 0.180423],
#             [0.212671, 0.715160, 0.072169],
#             [0.019334, 0.119193, 0.950227]
#         ], dtype=torch.float32)

#         # 参考白点 D65
#         white_point = torch.tensor([0.95047, 1.0, 1.08883], dtype=torch.float32)
#         # 注册为缓冲区，这样它们会自动跟随模块移动到正确的设备
#         self.register_buffer('rgb_to_xyz', rgb_to_xyz)
#         self.register_buffer('white_point', white_point)

#     # def _f(self, t):
#     #     delta = 6 / 29
#     #     return torch.where(t > delta ** 3, t.pow(1 / 3), (t / (3 * delta ** 2)) + (4 / 29))
    
#     def _f(self, t):
#         delta = 6 / 29
#         # 安全性优化：防止对接近0的数求 1/3 次幂导致梯度爆炸
#         return torch.where(t > delta ** 3, 
#                            torch.clamp(t, min=1e-8).pow(1 / 3), 
#                            (t / (3 * delta ** 2)) + (4 / 29))
    

#     def forward(self, img):
#         # img: [B, 3, H, W], 范围 [0,1]
#         B, C, H, W = img.shape
#         assert C == 3, "输入必须是RGB图像"

#         # 确保缓冲区在正确的设备上
#         device = img.device
#         rgb_to_xyz = self.rgb_to_xyz.to(device)
#         white_point = self.white_point.to(device)

#         # 1. sRGB -> Linear RGB
#         mask = (img <= 0.04045).float()
#         img_linear = mask * (img / 12.92) + (1 - mask) * (((img + 0.055) / 1.055) ** 2.4)

#         # 2. Linear RGB -> XYZ
#         img_flat = img_linear.permute(0, 2, 3, 1).reshape(-1, 3)  # [B*H*W, 3]
#         xyz = torch.matmul(img_flat, rgb_to_xyz.T)
#         # xyz = torch.matmul(img_flat, self.rgb_to_xyz.T)  # [B*H*W, 3]
#         xyz = xyz / white_point  # 归一化白点
#         xyz = xyz.reshape(B, H, W, 3).permute(0, 3, 1, 2)  # [B, 3, H, W]

#         # 3. XYZ -> Lab
#         fx, fy, fz = self._f(xyz[:, 0]), self._f(xyz[:, 1]), self._f(xyz[:, 2])
#         L = 116 * fy - 16
#         a = 500 * (fx - fy)
#         b = 200 * (fy - fz)

#         lab = torch.stack([L, a, b], dim=1)

#         # 归一化处理
#         L_norm = lab[:, 0:1, :, :] / 100.0  # L→[0,1]
#         a_norm = lab[:, 1:2, :, :] / 128.0  # a→[-1,1]
#         b_norm = lab[:, 2:3, :, :] / 128.0  # b→[-1,1]
#         lab_norm = torch.cat([L_norm, a_norm, b_norm], dim=1)

#         return lab_norm


# class Lab2RGB(nn.Module):
#     """
#     将 [B, 3, H, W] 的 Lab 图像转换为 RGB 颜色空间
#     输入: L in [0,1], a in [-1,1], b in [-1,1] (归一化版本)
#     输出范围: [0,1]
#     """

#     def __init__(self):
#         super(Lab2RGB, self).__init__()
#         # XYZ to sRGB 变换矩阵 (D65)
#         xyz_to_rgb = torch.tensor([
#             [3.240479, -1.537150, -0.498535],
#             [-0.969256, 1.875992, 0.041556],
#             [0.055648, -0.204043, 1.057311]
#         ], dtype=torch.float32)

#         # 参考白点 D65
#         white_point = torch.tensor([0.95047, 1.0, 1.08883], dtype=torch.float32)
#         # 注册为缓冲区，这样它们会自动跟随模块移动到正确的设备
#         self.register_buffer('xyz_to_rgb', xyz_to_rgb)
#         self.register_buffer('white_point', white_point)

#     def _finv(self, t):
#         delta = 6 / 29
#         return torch.where(t > delta, t ** 3, 3 * delta ** 2 * (t - 4 / 29))

#     def forward(self, lab_norm):
#         # lab_norm: [B, 3, H, W], L in [0,1], a/b in [-1,1]
#         B, C, H, W = lab_norm.shape
#         assert C == 3, "输入必须是Lab图像"

#         # 确保缓冲区在正确的设备上
#         device = lab_norm.device
#         xyz_to_rgb = self.xyz_to_rgb.to(device)
#         white_point = self.white_point.to(device)

#         # 1. 反归一化处理
#         L = lab_norm[:, 0:1, :, :] * 100.0
#         a = lab_norm[:, 1:2, :, :] * 128.0
#         b = lab_norm[:, 2:3, :, :] * 128.0
#         lab = torch.cat([L, a, b], dim=1)

#         # 2. 确保范围约束（防止模型输出异常）
#         lab = torch.cat([
#             torch.clamp(lab[:, 0:1, :, :], 0.0, 100.0),
#             torch.clamp(lab[:, 1:, :, :], -128.0, 127.0)
#         ], dim=1)

#         L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]

#         # 3. Lab -> XYZ
#         fy = (L + 16) / 116
#         fx = fy + (a / 500)
#         fz = fy - (b / 200)

#         xr = self._finv(fx)
#         yr = self._finv(fy)
#         zr = self._finv(fz)

#         X = xr * white_point[0]
#         Y = yr * white_point[1]
#         Z = zr * white_point[2]

#         xyz = torch.stack([X, Y, Z], dim=1)  # [B, 3, H, W]

#         # 4. XYZ -> Linear RGB
#         xyz_flat = xyz.permute(0, 2, 3, 1).reshape(-1, 3)
#         rgb_linear = torch.matmul(xyz_flat, xyz_to_rgb.T)
#         rgb_linear = rgb_linear.reshape(B, H, W, 3).permute(0, 3, 1, 2)

#         # 5. Linear RGB -> sRGB
#         mask = (rgb_linear <= 0.0031308).float()
#         rgb = mask * (12.92 * rgb_linear) + (1 - mask) * (1.055 * torch.clamp(rgb_linear, min=0) ** (1 / 2.4) - 0.055)

#         # 限制到 [0,1]
#         rgb = torch.clamp(rgb, 0.0, 1.0)
#         return rgb



import torch
import torch.nn as nn

class RGB2Lab(nn.Module):
    """
    将 [B, 3, H, W] 的 RGB 图像转换为 Lab 颜色空间
    """
    def __init__(self):
        super(RGB2Lab, self).__init__()
        rgb_to_xyz = torch.tensor([
            [0.412453, 0.357580, 0.180423],
            [0.212671, 0.715160, 0.072169],
            [0.019334, 0.119193, 0.950227]
        ], dtype=torch.float32)
        white_point = torch.tensor([0.95047, 1.0, 1.08883], dtype=torch.float32)
        self.register_buffer('rgb_to_xyz', rgb_to_xyz)
        self.register_buffer('white_point', white_point)

    def _f(self, t):
        delta = 6 / 29
        # 安全性优化：防止对接近0的数求 1/3 次幂导致梯度爆炸
        return torch.where(t > delta ** 3, 
                           torch.clamp(t, min=1e-8).pow(1 / 3), 
                           (t / (3 * delta ** 2)) + (4 / 29))

    def forward(self, img):
        B, C, H, W = img.shape
        rgb_to_xyz = self.rgb_to_xyz.to(img.device)
        white_point = self.white_point.to(img.device)

        mask = (img <= 0.04045).float()
        img_linear = mask * (img / 12.92) + (1 - mask) * (((img + 0.055) / 1.055) ** 2.4)

        img_flat = img_linear.permute(0, 2, 3, 1).reshape(-1, 3)
        xyz = torch.matmul(img_flat, rgb_to_xyz.T)
        xyz = xyz / white_point
        xyz = xyz.reshape(B, H, W, 3).permute(0, 3, 1, 2)

        fx, fy, fz = self._f(xyz[:, 0]), self._f(xyz[:, 1]), self._f(xyz[:, 2])
        L = 116 * fy - 16
        a = 500 * (fx - fy)
        b = 200 * (fy - fz)

        lab = torch.stack([L, a, b], dim=1)
        L_norm = lab[:, 0, :, :] / 100.0 # 修改：使用索引 0 而不是切片 0:1
        a_norm = lab[:, 1, :, :] / 128.0
        b_norm = lab[:, 2, :, :] / 128.0
        
        # 重新堆叠回 [B, 3, H, W]
        lab_norm = torch.stack([L_norm, a_norm, b_norm], dim=1)
        return lab_norm


class Lab2RGB(nn.Module):
    """
    将 [B, 3, H, W] 的 Lab 图像转换为 RGB 颜色空间
    """
    def __init__(self):
        super(Lab2RGB, self).__init__()
        xyz_to_rgb = torch.tensor([
            [3.240479, -1.537150, -0.498535],
            [-0.969256, 1.875992, 0.041556],
            [0.055648, -0.204043, 1.057311]
        ], dtype=torch.float32)
        white_point = torch.tensor([0.95047, 1.0, 1.08883], dtype=torch.float32)
        self.register_buffer('xyz_to_rgb', xyz_to_rgb)
        self.register_buffer('white_point', white_point)

    def _finv(self, t):
        delta = 6 / 29
        return torch.where(t > delta, t ** 3, 3 * delta ** 2 * (t - 4 / 29))

    def forward(self, lab_norm):
        B, C, H, W = lab_norm.shape
        xyz_to_rgb = self.xyz_to_rgb.to(lab_norm.device)
        white_point = self.white_point.to(lab_norm.device)

        # 1. 反归一化处理
        # [关键修复] 使用索引 [:, 0] 确保取出的是 [B, H, W] 而不是 [B, 1, H, W]
        L = lab_norm[:, 0, :, :] * 100.0
        a = lab_norm[:, 1, :, :] * 128.0
        b = lab_norm[:, 2, :, :] * 128.0
        
        # 2. Lab -> XYZ
        fy = (L + 16) / 116
        fx = fy + (a / 500)
        fz = fy - (b / 200)

        xr = self._finv(fx)
        yr = self._finv(fy)
        zr = self._finv(fz)

        X = xr * white_point[0]
        Y = yr * white_point[1]
        Z = zr * white_point[2]

        # 此时 X,Y,Z 都是 [B, H, W]，stack 后变成 [B, 3, H, W]
        xyz = torch.stack([X, Y, Z], dim=1)

        # 4. XYZ -> Linear RGB
        xyz_flat = xyz.permute(0, 2, 3, 1).reshape(-1, 3)
        rgb_linear = torch.matmul(xyz_flat, xyz_to_rgb.T)
        rgb_linear = rgb_linear.reshape(B, H, W, 3).permute(0, 3, 1, 2)

        # 5. Linear RGB -> sRGB (核心修复：防止 NaN)
        mask = (rgb_linear <= 0.0031308).float()
        
        # [关键修复]：限制最小值为 1e-8，防止 0 的幂运算导致梯度爆炸 (NaN)
        safe_linear = torch.clamp(rgb_linear, min=1e-8)
        
        rgb = mask * (12.92 * rgb_linear) + (1 - mask) * (1.055 * safe_linear ** (1 / 2.4) - 0.055)

        # 限制到 [0,1]
        rgb = torch.clamp(rgb, 0.0, 1.0)
        return rgb