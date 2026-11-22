"""
图像特征提取模块 (v0.3 - 修复自动降级链并绑定模型签名)
改进点（对应评审意见 #2）：
1. auto 模式按顺序依次尝试 clip -> resnet -> fallback_cv，不再跳过 ResNet。
2. 不再静默吞掉异常：记录每一步降级原因到 fallback_history 并输出 warning 日志。
3. 暴露 model_name / model_weights / preprocess_version 与 signature()，供索引入库与查询时强制校验一致性。
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from .schemas import ImageFeatureRecord

logger = logging.getLogger(__name__)


def compute_sha256(file_path: str | Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_dhash(image_bgr: np.ndarray, hash_size: int = 8) -> str:
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
    """统一图像特征提取器（带完整降级日志与模型版本签名）。"""

    def __init__(self, backend: str = "auto", device: Optional[str] = None) -> None:
        self.requested_backend = backend
        self.device = device
        self.backend_name = "fallback_cv"
        self.model_weights = "hsv128+grad256+dct128"
        self.preprocess_version = "cv2-256x256-v1"
        self.dim = 512
        self._model = None
        self._preprocess = None
        self.fallback_history: List[str] = []

        if backend == "auto":
            if self._try_init_clip():
                return
            if self._try_init_resnet():
                return
            logger.warning(
                "视觉特征提取器已自动降级为 fallback_cv，降级链路: %s",
                " | ".join(self.fallback_history),
            )
        elif backend == "clip":
            if not self._try_init_clip():
                raise RuntimeError(f"初始化 CLIP 失败: {'; '.join(self.fallback_history)}")
        elif backend == "resnet":
            if not self._try_init_resnet():
                raise RuntimeError(f"初始化 ResNet 失败: {'; '.join(self.fallback_history)}")
        elif backend == "fallback_cv":
            pass
        else:
            raise ValueError(f"不支持的 backend 参数: {backend}")

    @property
    def signature(self) -> str:
        return f"{self.backend_name}:{self.model_weights}:{self.preprocess_version}:dim{self.dim}"

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
            self.model_weights = "openai-vit-b-32"
            self.preprocess_version = "openclip-224-v1"
            self.dim = 512
            return True
        except Exception as exc:
            msg = f"clip 不可用 ({type(exc).__name__}: {exc})"
            self.fallback_history.append(msg)
            logger.info(msg)
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
            self.model_weights = "torchvision-resnet18-default"
            self.preprocess_version = "imagenet-224-v1"
            self.dim = 512
            return True
        except Exception as exc:
            msg = f"resnet 不可用 ({type(exc).__name__}: {exc})"
            self.fallback_history.append(msg)
            logger.info(msg)
            return False

    def _extract_fallback_cv(self, image_bgr: np.ndarray) -> np.ndarray:
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
        vec_orig, _ = self.extract_from_bgr(image_bgr)
        vec_hflip, _ = self.extract_from_bgr(cv2.flip(image_bgr, 1))

        h, w = image_bgr.shape[:2]
        dh, dw = int(h * 0.075), int(w * 0.075)
        cropped = image_bgr[dh : h - dh, dw : w - dw] if (h > 40 and w > 40) else image_bgr
        vec_crop, _ = self.extract_from_bgr(cropped)

        return {"original": vec_orig, "hflip": vec_hflip, "center_crop": vec_crop}

    def extract_file(
        self, file_path: str | Path, item_id: Optional[str] = None, with_hflip: bool = False
    ) -> ImageFeatureRecord:
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

        return ImageFeatureRecord(
            item_id=item_id or path.stem,
            file_path=str(path),
            sha256=compute_sha256(path),
            width=int(w),
            height=int(h),
            phash=phash,
            model_name=self.backend_name,
            model_weights=self.model_weights,
            preprocess_version=self.preprocess_version,
            embedding_dim=int(vec.shape[0]),
            embedding=vec.tolist(),
            hflip_embedding=hflip_vec.tolist() if hflip_vec is not None else None,
        )
