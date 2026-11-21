"""
图像特征提取模块 (Week 1 优化版)
升级点（对齐架构报告第六章“编辑变换片段检索”研究主线）：
1. 支持原图 + 水平镜像（hflip）+ 中心裁剪（center_crop）多视角特征提取，
   使模型在面对常见的镜像翻转、轻度裁剪搬运时仍能稳定召回底库原图。
2. 保留 CLIP / ResNet / OpenCV 512 维三层自动降级机制，无 GPU 环境也可直接联调。
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from .schemas import ImageFeatureRecord


def compute_sha256(file_path: str | Path) -> str:
    """计算文件 SHA-256 哈希，用于文件一致性校验与记录追溯。"""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_dhash(image_bgr: np.ndarray, hash_size: int = 8) -> str:
    """计算 64 位差值感知哈希（dHash）。"""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    bits = "".join("1" if v else "0" for v in diff.flatten())
    return f"{int(bits, 2):0{hash_size * hash_size // 4}x}"


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    vec = vec.astype(np.float32).flatten()
    norm = float(np.linalg.norm(vec))
    if norm < 1e-12:
        return vec
    return vec / norm


class ImageFeatureExtractor:
    """统一图像特征提取器（支持多视角编辑变换鲁棒提取）。"""

    def __init__(self, backend: str = "auto", device: Optional[str] = None) -> None:
        self.requested_backend = backend
        self.device = device
        self.backend_name = "fallback_cv"
        self.dim = 512
        self._model = None
        self._preprocess = None

        if backend in ("auto", "clip"):
            if self._try_init_clip():
                return
            if backend == "clip":
                raise RuntimeError("已指定 backend='clip'，但当前环境缺少 torch 或 open_clip_torch。")

        if backend == "resnet":
            if self._try_init_resnet():
                return
            raise RuntimeError("已指定 backend='resnet'，但当前环境缺少 torch 或 torchvision。")

    def _try_init_clip(self) -> bool:
        try:
            import torch
            import open_clip  # type: ignore

            dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
            model.eval().to(dev)
            self._model = model
            self._preprocess = preprocess
            self.device = dev
            self.backend_name = "clip-vit-b32"
            self.dim = 512
            return True
        except Exception:
            return False

    def _try_init_resnet(self) -> bool:
        try:
            import torch
            import torchvision.models as models  # type: ignore
            import torchvision.transforms as T  # type: ignore

            dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            weights = models.ResNet18_Weights.DEFAULT
            net = models.resnet18(weights=weights)
            net.fc = torch.nn.Identity()
            net.eval().to(dev)
            self._model = net
            self._preprocess = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self.device = dev
            self.backend_name = "resnet18"
            self.dim = 512
            return True
        except Exception:
            return False

    def _extract_fallback_cv(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        512 维多线索视觉描述子（空间分块 HSV + 梯度方向 + 低频 DCT）。
        """
        img = cv2.resize(image_bgr, (256, 256), interpolation=cv2.INTER_AREA)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv_feats = []
        for r in range(2):
            for c in range(2):
                patch = hsv[r * 128 : (r + 1) * 128, c * 128 : (c + 1) * 128]
                h_hist = cv2.calcHist([patch], [0], None, [16], [0, 180]).flatten()
                s_hist = cv2.calcHist([patch], [1], None, [8], [0, 256]).flatten()
                v_hist = cv2.calcHist([patch], [2], None, [8], [0, 256]).flatten()
                hsv_feats.append(_l2_normalize(np.concatenate([h_hist, s_hist, v_hist])))
        color_vec = np.concatenate(hsv_feats)  # 128

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        grad_feats = []
        for r in range(4):
            for c in range(4):
                m_patch = mag[r * 64 : (r + 1) * 64, c * 64 : (c + 1) * 64]
                a_patch = ang[r * 64 : (r + 1) * 64, c * 64 : (c + 1) * 64]
                hist, _ = np.histogram(a_patch, bins=16, range=(0.0, 360.0), weights=m_patch)
                grad_feats.append(_l2_normalize(hist))
        grad_vec = np.concatenate(grad_feats)  # 256

        small_gray = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA) / 255.0
        dct = cv2.dct(small_gray)
        dct_vec = _l2_normalize(dct[:12, :12].flatten()[:128])  # 128

        return _l2_normalize(np.concatenate([color_vec, grad_vec, dct_vec]))

    def extract_from_bgr(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, str]:
        """从 OpenCV BGR 图像矩阵提取 (512维归一化向量, dHash)。"""
        phash = compute_dhash(image_bgr)

        if self.backend_name == "clip-vit-b32" and self._model is not None:
            import torch

            pil_img = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
            tensor = self._preprocess(pil_img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feats = self._model.encode_image(tensor)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats.cpu().numpy().astype(np.float32).flatten(), phash

        if self.backend_name == "resnet18" and self._model is not None:
            import torch

            pil_img = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
            tensor = self._preprocess(pil_img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feats = self._model(tensor)
            return _l2_normalize(feats.cpu().numpy()), phash

        return self._extract_fallback_cv(image_bgr), phash

    def extract_multi_view_from_bgr(self, image_bgr: np.ndarray) -> Dict[str, np.ndarray]:
        """
        提取多视角查询特征（原图 / 水平镜像 / 中心85%裁剪），
        用于提升对镜像、裁边等常见编辑变换的召回鲁棒性。
        """
        vec_orig, _ = self.extract_from_bgr(image_bgr)
        flipped = cv2.flip(image_bgr, 1)
        vec_hflip, _ = self.extract_from_bgr(flipped)

        h, w = image_bgr.shape[:2]
        dh, dw = int(h * 0.075), int(w * 0.075)
        cropped = image_bgr[dh : h - dh, dw : w - dw] if (h > 40 and w > 40) else image_bgr
        vec_crop, _ = self.extract_from_bgr(cropped)

        return {
            "original": vec_orig,
            "hflip": vec_hflip,
            "center_crop": vec_crop,
        }

    def extract_file(
        self, file_path: str | Path, item_id: Optional[str] = None, with_hflip: bool = False
    ) -> ImageFeatureRecord:
        """读取本地图片文件并返回结构化特征记录。"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"图片文件不存在: {path}")

        raw_bytes = np.fromfile(str(path), dtype=np.uint8)
        image_bgr = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"无法解码图片文件: {path}")

        h, w = image_bgr.shape[:2]
        vec, phash = self.extract_from_bgr(image_bgr)
        hflip_vec = None
        if with_hflip:
            hflip_vec, _ = self.extract_from_bgr(cv2.flip(image_bgr, 1))

        sha256 = compute_sha256(path)
        return ImageFeatureRecord(
            item_id=item_id or path.stem,
            file_path=str(path),
            sha256=sha256,
            width=int(w),
            height=int(h),
            phash=phash,
            model_name=self.backend_name,
            embedding_dim=int(vec.shape[0]),
            embedding=vec.tolist(),
            hflip_embedding=hflip_vec.tolist() if hflip_vec is not None else None,
        )
