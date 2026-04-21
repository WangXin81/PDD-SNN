import os

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from ..evaluation.metrics import calculate_uiqm
from ..models.color_spaces import RGB2Lab
from ..models.networks import ReconstructionModule
from ..utils.common import EnergyMeter

try:
    import lpips
except ImportError:
    lpips = None


def ssim_tensor_function(img1, img2):
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu1 = F.avg_pool2d(img1, 3, 1, 1)
    mu2 = F.avg_pool2d(img2, 3, 1, 1)
    sigma1_sq = F.avg_pool2d(img1 ** 2, 3, 1, 1) - mu1 ** 2
    sigma2_sq = F.avg_pool2d(img2 ** 2, 3, 1, 1) - mu2 ** 2
    sigma12 = F.avg_pool2d(img1 * img2, 3, 1, 1) - mu1 * mu2
    ssim_map = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean().item()


class SNNInferenceEngine:
    def __init__(self, config, model_path=None, device=None, skip_weights=False):
        self.config = config
        self.device = device or config.DEVICE
        self.model = ReconstructionModule(**config.RECONSTRUCTION_CONFIG, device=self.device).to(self.device)
        self.rgb2lab = RGB2Lab().to(self.device)
        self.transform = transforms.ToTensor()
        self.patch_size = getattr(config, "INFER_PATCH_SIZE", None) or 256
        self.scale = config.RECONSTRUCTION_CONFIG.get("upscale_factor", config.UPSCALE_FACTOR)
        try:
            if lpips is None:
                raise ImportError("lpips is not installed")
            self.lpips_loss = lpips.LPIPS(net="alex").to(self.device)
            self.lpips_loss.eval()
        except Exception:
            self.lpips_loss = None
        if model_path and not skip_weights:
            self._load_weights(model_path)
        self.model.eval()

    def _load_weights(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        state_dict = checkpoint.get("reconstruction_module_state_dict", checkpoint.get("state_dict", checkpoint))
        new_state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
        self.model.load_state_dict(new_state_dict, strict=False)

    def infer(self, lr_path, hr_path=None, save_path=None):
        lr_img = Image.open(lr_path).convert("RGB")
        lr_tensor = self.transform(lr_img).unsqueeze(0).to(self.device)
        b, c, h_old, w_old = lr_tensor.size()
        pad_h = (self.patch_size - h_old % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w_old % self.patch_size) % self.patch_size
        lr_padded = F.pad(lr_tensor, (0, pad_w, 0, pad_h), mode="replicate")
        _, _, h_pad, w_pad = lr_padded.size()
        sr_padded = torch.zeros((b, 3, h_pad * self.scale, w_pad * self.scale), device=self.device)

        meter = None
        try:
            meter = EnergyMeter(
                self.model,
                input_size=(1, 3, self.patch_size, self.patch_size),
                device=self.device,
                time_steps=self.config.RECONSTRUCTION_CONFIG.get("time_steps", self.config.TIME_STEPS),
            )
            meter.register_hooks()
        except Exception:
            meter = None

        with torch.no_grad():
            for y in range(0, h_pad, self.patch_size):
                for x in range(0, w_pad, self.patch_size):
                    lr_patch = lr_padded[..., y:y + self.patch_size, x:x + self.patch_size]
                    for module in self.model.modules():
                        if hasattr(module, "reset"):
                            module.reset()
                    sr_patch = self.model(self.rgb2lab(lr_patch))
                    sr_padded[..., y * self.scale:(y + self.patch_size) * self.scale, x * self.scale:(x + self.patch_size) * self.scale] = sr_patch

        energy_results = meter.calculate_metrics() if meter else {}
        if meter:
            meter.remove_hooks()

        sr_tensor = torch.clamp(sr_padded[..., :h_old * self.scale, :w_old * self.scale], 0, 1)
        results = {
            "psnr": 0.0,
            "ssim": 0.0,
            "lpips": 0.0,
            "uiqm": calculate_uiqm(sr_tensor),
            "gsops": energy_results.get("GSOPs", 0.0),
            "energy": energy_results.get("Energy_SNN (J)", 0.0),
        }

        if hr_path and os.path.exists(hr_path):
            hr_tensor = self.transform(Image.open(hr_path).convert("RGB")).unsqueeze(0).to(self.device)
            hr_tensor = hr_tensor[..., : h_old * self.scale, : w_old * self.scale]
            shave = max(1, self.scale)
            sr_c = sr_tensor[..., shave:-shave, shave:-shave]
            hr_c = hr_tensor[..., shave:-shave, shave:-shave]

            def to_y(img):
                return 0.257 * img[:, 0] + 0.504 * img[:, 1] + 0.098 * img[:, 2] + (16 / 255)

            sr_y = to_y(sr_c).unsqueeze(1)
            hr_y = to_y(hr_c).unsqueeze(1)
            mse = torch.mean((sr_y - hr_y) ** 2)
            results["psnr"] = 10.0 * torch.log10(1.0 / mse).item() if mse > 0 else 100.0
            results["ssim"] = ssim_tensor_function(sr_y, hr_y)
            if self.lpips_loss is not None:
                results["lpips"] = self.lpips_loss(sr_c * 2 - 1, hr_c * 2 - 1).mean().item()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            save_image(sr_tensor, save_path)
        return results
