# KL.py - KernelGAN实现 (功能封装) - SNN版本 (使用整合.py中的脉冲神经元)
import os
import math
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils as vutils


# ========== 从整合.py导入脉冲神经元定义 ==========
class BPTTSurrogateFn(torch.autograd.Function):
    """
    阶跃函数的替代梯度，用于 BPTT
    前向: Heaviside step
    反向: 使用 sigmoid 近似 (可选换成其他函数)
    """

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input > 0).float()  # 前向仍是硬阈值发放

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        # 使用 sigmoid 近似导数作为替代梯度
        slope = 10.0
        grad_input = grad_output * slope * torch.sigmoid(slope * input) * (1 - torch.sigmoid(slope * input))
        return grad_input


class TemporalEncoder(nn.Module):
    """
    将静态输入扩展为时序输入，或者对时序输入加噪声/扰动以创造时序性。
    """

    def __init__(self, time_steps=5, mode="repeat", noise_scale=0.1):
        super(TemporalEncoder, self).__init__()
        self.time_steps = time_steps
        self.mode = mode  # "repeat" 或 "noisy"
        self.noise_scale = noise_scale

    def forward(self, x):
        """
        输入: x [batch, ...]  (静态输入)
        输出: seq [batch, time_steps, ...] (时序展开后的输入)
        """
        if self.mode == "repeat":
            # 直接复制成多个时间步
            seq = x.unsqueeze(1).repeat(1, self.time_steps, *([1] * (x.ndim - 1)))
        elif self.mode == "noisy":
            # 每个时间步加不同噪声，创造动态性
            seqs = []
            for t in range(self.time_steps):
                noise = torch.randn_like(x) * self.noise_scale
                seqs.append(x + noise)
            seq = torch.stack(seqs, dim=1)
        else:
            raise ValueError(f"未知模式: {self.mode}")
        return seq.permute(1, 0, 2, 3, 4)  # [T, B, C, H, W]


class MultiStepPmLIFNode(nn.Module):
    def __init__(self, v_th=1.0, v_reset=0.0, tau=2.0, surrogate_fn=None):
        """
        多步版 PmLIF 节点 (支持 [T, B, ...] 输入)

        参数:
            v_th: 阈值 (可学习)
            v_reset: 重置电位
            tau: 时间常数 (可学习)
            surrogate_fn: 替代梯度函数 (默认 BPTT)
        """
        super().__init__()
        self.v_th = nn.Parameter(torch.tensor(v_th, dtype=torch.float32))
        self.v_reset = v_reset
        self.tau = nn.Parameter(torch.tensor(tau, dtype=torch.float32))
        self.surrogate_fn = surrogate_fn or (lambda x: BPTTSurrogateFn.apply(x))

        self.v = None  # 膜电位状态

    def forward(self, x_seq: torch.Tensor):
        """
        输入:  [T, B, ...]
        输出:  [T, B, ...] (spikes)
        """
        T, B = x_seq.shape[0], x_seq.shape[1]
        if self.v is None:
            self.v = torch.zeros_like(x_seq[0])

        spikes = []
        for t in range(T):
            x = x_seq[t]

            # 电压更新公式
            h = self.v + (1.0 / self.tau) * (-(self.v - self.v_reset) + x)

            # 产生脉冲
            s = self.surrogate_fn(h - self.v_th)

            # 更新电位
            self.v = h * (1 - s) + self.v_reset * s

            spikes.append(s)

        return torch.stack(spikes, dim=0)

    def reset(self):
        """重置膜电位"""
        self.v = None


class MultiStepTernaryPmLIFNode(nn.Module):
    """
    三元脉冲神经元 (Ternary Spiking Neuron)
    输出脉冲 {-1, 0, +1}
    """

    def __init__(self, v_th=1.0, v_reset=0.0, tau=2.0, surrogate_fn=None):
        super().__init__()
        self.v_th = nn.Parameter(torch.tensor(v_th, dtype=torch.float32))
        self.v_reset = v_reset
        self.tau = nn.Parameter(torch.tensor(tau, dtype=torch.float32))
        self.surrogate_fn = surrogate_fn or (lambda x: BPTTSurrogateFn.apply(x))
        self.v = None  # 膜电位

    def forward(self, x_seq: torch.Tensor):
        T, B = x_seq.shape[0], x_seq.shape[1]
        if self.v is None:
            self.v = torch.zeros_like(x_seq[0])

        spikes = []
        for t in range(T):
            x = x_seq[t]

            # 膜电位更新
            h = self.v + (1.0 / self.tau) * (-(self.v - self.v_reset) + x)

            # 三元发放规则
            s_pos = self.surrogate_fn(h - self.v_th)  # 正阈值 → +1
            s_neg = self.surrogate_fn(-h - self.v_th)  # 负阈值 → -1
            s = s_pos - s_neg  # {-1,0,1}

            # 膜电位更新（发放则复位）
            self.v = h * (1 - (s_pos + s_neg)) + self.v_reset * (s_pos + s_neg)

            spikes.append(s)

        return torch.stack(spikes, dim=0)

    def reset(self):
        self.v = None


# ========== 时序输入生成函数 ==========
def create_temporal_input(image_tensor, time_steps=5):
    batch_size, channels, height, width = image_tensor.shape
    temporal_input = torch.zeros((batch_size, time_steps, channels, height, width), device=image_tensor.device)
    for b in range(batch_size):
        image_pil = transforms.ToPILImage()(image_tensor[b].cpu())
        temporal_input[b, 0] = image_tensor[b]
        for t in range(1, time_steps):
            transform = transforms.Compose([
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
                transforms.ToTensor(),
            ])
            transformed_img = transform(image_pil).to(image_tensor.device)
            temporal_input[b, t] = transformed_img
    return temporal_input


# ========== SNN版本的判别器 (使用整合.py中的脉冲神经元) ==========
class SNNPatchDiscriminator(nn.Module):
    def __init__(self, in_ch=3, ndf=64, time_steps=5):
        super().__init__()
        self.time_steps = time_steps

        self.conv1 = nn.Conv2d(in_ch, ndf, 4, 2, 1)
        # 使用整合.py中的三元脉冲神经元
        self.lif1 = MultiStepTernaryPmLIFNode(v_th=0.05, v_reset=0.0, tau=0.5)

        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1)
        self.lif2 = MultiStepTernaryPmLIFNode(v_th=0.05, v_reset=0.0, tau=0.5)

        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1)
        self.lif3 = MultiStepTernaryPmLIFNode(v_th=0.05, v_reset=0.0, tau=0.5)

        self.conv4 = nn.Conv2d(ndf * 4, 1, 3, 1, 1)  # 输出 patch-wise 的真伪评分

    def forward(self, x):
        self.lif1.reset()
        self.lif2.reset()
        self.lif3.reset()
        temporal_input = create_temporal_input(x, self.time_steps)
        # 转换维度为 [T, B, C, H, W]
        x_seq = temporal_input.permute(1, 0, 2, 3, 4)

        # 处理每个卷积层
        T, B, C, H, W = x_seq.shape

        # 第一层
        x_flat = x_seq.reshape(T * B, C, H, W)
        out = self.conv1(x_flat)
        _, C1, H1, W1 = out.shape
        out_seq = out.reshape(T, B, C1, H1, W1)
        out_seq = self.lif1(out_seq)

        # 第二层
        out_flat = out_seq.reshape(T * B, C1, H1, W1)
        out = self.conv2(out_flat)
        _, C2, H2, W2 = out.shape
        out_seq = out.reshape(T, B, C2, H2, W2)
        out_seq = self.lif2(out_seq)

        # 第三层
        out_flat = out_seq.reshape(T * B, C2, H2, W2)
        out = self.conv3(out_flat)
        _, C3, H3, W3 = out.shape
        out_seq = out.reshape(T, B, C3, H3, W3)
        out_seq = self.lif3(out_seq)

        # 输出层
        out_flat = out_seq.reshape(T * B, C3, H3, W3)
        out = self.conv4(out_flat)
        _, C4, H4, W4 = out.shape
        out_seq = out.reshape(T, B, C4, H4, W4)

        # 时间维度上平均
        out = out_seq.mean(dim=0)  # [B, 1, H4, W4]

        return out


# --------------------------
# 工具 & 复现性
# --------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


# --------------------------
# 仅使用 LR（珊瑚）图像的数据集
# --------------------------
class LRPatchDataset(Dataset):
    def __init__(self, image_dir, patch_size=64, num_patches_per_image=16):
        super().__init__()
        self.image_paths = [
            os.path.join(image_dir, f) for f in os.listdir(image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
        ]
        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No images found in {image_dir}")
        self.patch_size = patch_size
        self.num_patches_per_image = num_patches_per_image
        self.tf = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths) * self.num_patches_per_image

    def __getitem__(self, idx):
        path = self.image_paths[idx // self.num_patches_per_image]
        img = Image.open(path).convert("RGB")
        W, H = img.size
        ps = self.patch_size
        if W < ps or H < ps:
            img = img.resize((max(W, ps), max(H, ps)), Image.BICUBIC)
            W, H = img.size
        x = random.randint(0, W - ps)
        y = random.randint(0, H - ps)
        patch = img.crop((x, y, x + ps, y + ps))
        patch = self.tf(patch)
        return patch


# --------------------------
# 可学习的退化核 + 噪声参数
# --------------------------
class KernelAndNoise(nn.Module):
    def __init__(self, kernel_size=15, channels=3):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.k = kernel_size
        self.kernel_logits = nn.Parameter(torch.zeros(1, 1, self.k, self.k))
        nn.init.normal_(self.kernel_logits, mean=0.0, std=0.01)

        self.noise_mu = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.noise_rho = nn.Parameter(torch.full((1, channels, 1, 1), -2.0))

    def normalized_kernel(self):
        k = self.kernel_logits.view(1, 1, -1)
        k = torch.softmax(k, dim=-1).view(1, 1, self.k, self.k)
        return k

    def sigma(self):
        return F.softplus(self.noise_rho) + 1e-6


# --------------------------
# 退化算子
# --------------------------
class DegradationOperator(nn.Module):
    def __init__(self, kernel_module: KernelAndNoise, scale=2):
        super().__init__()
        self.params = kernel_module
        self.scale = scale

    def forward(self, lr_patch):
        B, C, H, W = lr_patch.shape
        up = F.interpolate(lr_patch, scale_factor=self.scale, mode='bicubic', align_corners=False)

        k = self.params.normalized_kernel()
        k_rep = k.expand(C, 1, k.size(-2), k.size(-1))
        blurred = F.conv2d(up, k_rep, bias=None, stride=1, padding=self.params.k // 2, groups=C)

        down = F.interpolate(blurred, size=(H, W), mode='bicubic', align_corners=False)

        mu = self.params.noise_mu
        sigma = self.params.sigma()
        noise = torch.randn_like(down) * sigma + mu
        out = torch.clamp(down + noise, 0.0, 1.0)
        return out


# --------------------------
# 高斯先验
# --------------------------
def gaussian_kernel_like(kernel_tensor):
    _, _, kH, kW = kernel_tensor.shape
    cy, cx = (kH - 1) / 2.0, (kW - 1) / 2.0
    y, x = torch.meshgrid(torch.arange(kH), torch.arange(kW), indexing='ij')
    y, x = y.to(kernel_tensor.device).float(), x.to(kernel_tensor.device).float()
    dist2 = (y - cy) ** 2 + (x - cx) ** 2
    sigma_gauss = max(kH, kW) / 6.0
    g = torch.exp(-dist2 / (2 * sigma_gauss ** 2))
    g = g / (g.sum() + 1e-8)
    return g.view(1, 1, kH, kW)


# --------------------------
# 训练器
# --------------------------
class KernelGAN_UnpairedLR:
    def __init__(
            self,
            kernel_size=15,
            scale=2,
            patch_size=64,
            batch_size=16,
            lr_g=1e-3,
            lr_d=2e-4,
            device='cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self.device = device
        self.scale = scale

        self.kn = KernelAndNoise(kernel_size=kernel_size, channels=3).to(device)
        self.deg = DegradationOperator(self.kn, scale=scale).to(device)
        self.D = SNNPatchDiscriminator(time_steps=5).to(device)

        self.opt_g = torch.optim.Adam(list(self.kn.parameters()), lr=lr_g, betas=(0.5, 0.999))
        self.opt_d = torch.optim.Adam(self.D.parameters(), lr=lr_d, betas=(0.5, 0.999))

        self.lambda_rec = 1.0
        self.lambda_adv = 0.05
        self.lambda_kprior = 0.01

    def train(self, dataset, epochs=50, log_interval=100, save_dir='./kernelgan_lr_only'):
        os.makedirs(save_dir, exist_ok=True)
        loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
        step = 0
        bce = nn.BCEWithLogitsLoss()

        for epoch in range(1, epochs + 1):
            for i, lr_patch in enumerate(loader):
                lr_patch = lr_patch.to(self.device)

                self.D.train()
                self.opt_d.zero_grad()

                with torch.no_grad():
                    fake_lr = self.deg(lr_patch)

                pred_real = self.D(lr_patch)
                pred_fake = self.D(fake_lr)

                real_label = torch.ones_like(pred_real) * 0.9
                fake_label = torch.zeros_like(pred_fake) * 0.1

                d_loss_real = bce(pred_real, real_label)
                d_loss_fake = bce(pred_fake, fake_label)
                d_loss = (d_loss_real + d_loss_fake) * 0.5
                d_loss.backward()
                self.opt_d.step()

                self.opt_g.zero_grad()
                fake_lr = self.deg(lr_patch)

                rec_loss = F.l1_loss(fake_lr, lr_patch)

                pred_fake = self.D(fake_lr)
                g_adv = bce(pred_fake, torch.ones_like(pred_fake) * 0.9)

                k = self.kn.normalized_kernel()
                gtmpl = gaussian_kernel_like(k)
                k_prior = F.mse_loss(k, gtmpl)

                g_loss = self.lambda_rec * rec_loss + self.lambda_adv * g_adv + self.lambda_kprior * k_prior
                g_loss.backward()
                self.opt_g.step()

                if (i + 1) % log_interval == 0:
                    print(f"Epoch[{epoch}/{epochs}] Iter[{i + 1}/{len(loader)}] "
                          f"D: {d_loss.item():.4f} | G: {g_loss.item():.4f} "
                          f"(rec {rec_loss.item():.4f}, adv {g_adv.item():.4f}, kprior {k_prior.item():.4f})")
                step += 1

            self._save_debug_images(epoch, save_dir, loader)
            self._save_checkpoint(epoch, save_dir)

        print("Training finished. Learned kernel & noise ready to use.")

    @torch.no_grad()
    def _save_debug_images(self, epoch, save_dir, loader):
        self.D.eval()
        self.kn.eval()
        batch = next(iter(loader))
        batch = batch[:8].to(self.device)
        fake = self.deg(batch)

        grid_real = vutils.make_grid(batch, nrow=4, normalize=True)
        grid_fake = vutils.make_grid(fake, nrow=4, normalize=True)
        vutils.save_image(grid_real, os.path.join(save_dir, f'epoch_{epoch:03d}_real.png'))
        vutils.save_image(grid_fake, os.path.join(save_dir, f'epoch_{epoch:03d}_fake.png'))

        k = self.kn.normalized_kernel().squeeze().cpu().numpy()
        k = (k - k.min()) / (k.max() - k.min() + 1e-8)
        k_img = Image.fromarray((k * 255).astype(np.uint8))
        k_img = k_img.resize((256, 256), Image.NEAREST)
        k_img.save(os.path.join(save_dir, f'epoch_{epoch:03d}_kernel.png'))

    def _save_checkpoint(self, epoch, save_dir):
        ckpt = {
            'epoch': epoch,
            'kernel_logits': self.kn.kernel_logits.detach().cpu(),
            'noise_mu': self.kn.noise_mu.detach().cpu(),
            'noise_rho': self.kn.noise_rho.detach().cpu(),
            'state_dict_kn': self.kn.state_dict(),
            'state_dict_D': self.D.state_dict(),
        }
        torch.save(ckpt, os.path.join(save_dir, f'epoch_{epoch:03d}.pth'))

    # --------------------------
    # 推理：先模糊再下采样再加噪声
    # --------------------------
    @torch.no_grad()
    def degrade_hr_image(self, hr_image_path, out_lr_path, out_hr_like_path=None):
        """
        hr_image_path: HR（乌龟）图像路径
        out_lr_path: 输出的降采样后退化 LR 图像（更贴近 LR 珊瑚风格）
        out_hr_like_path: （可选）输出与 HR 同分辨率的"模糊+噪声但不降采样"图像
        """
        self.kn.eval()
        img = Image.open(hr_image_path).convert("RGB")
        tf = transforms.ToTensor()
        x = tf(img).unsqueeze(0).to(self.device)  # [0,1]

        # 1) 模糊
        k = self.kn.normalized_kernel()
        k_rep = k.expand(3, 1, k.size(-2), k.size(-1))
        blurred = F.conv2d(x, k_rep, stride=1, padding=k.size(-1) // 2, groups=3)

        # 2) 下采样
        H, W = x.shape[-2:]
        lrH, lrW = H // self.scale, W // self.scale
        down = F.interpolate(blurred, size=(lrH, lrW), mode='bicubic', align_corners=False)

        # 3) 加噪声
        mu = self.kn.noise_mu
        sigma = self.kn.sigma()
        noise = torch.randn_like(down) * sigma + mu
        deg_lr = torch.clamp(down + noise, 0.0, 1.0)

        vutils.save_image(deg_lr, out_lr_path, normalize=True)

        # 可选：输出 HR 尺度的"模糊+噪声但不缩小"
        if out_hr_like_path is not None:
            noise_hr = torch.randn_like(blurred) * sigma + mu
            hr_like = torch.clamp(blurred + noise_hr, 0.0, 1.0)
            vutils.save_image(hr_like, out_hr_like_path, normalize=True)

        print(f"Saved LR-degraded image to: {out_lr_path}")
        if out_hr_like_path:
            print(f"Saved HR-like blurred+noisy image to: {out_hr_like_path}")

    def estimate_kernel(self, image_path):
        return self.kn.normalized_kernel().squeeze().cpu().numpy()


