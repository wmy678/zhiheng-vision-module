"""
视频抽帧模块 (v0.3 - 全片均匀覆盖 + 结构/亮度/颜色多线索变化检测)
改进点（对应评审意见 #4）：
1. 解决仅用 HSV H/S 直方图无法感知黑白运动物体和空间结构变化的问题：
   结合 灰度像素空间差异 (MAD) + 边缘结构变化 + HSV 颜色直方图巴氏距离 综合评分。
2. 解决长视频达到 max_frames 后截断丢失后半段的问题：
   采用“全片基础均匀采样 + 场景/运动变化补帧”策略，长视频自动按总时长自适应调整基础步长，
   保证覆盖全片并返回 coverage_start_sec / coverage_end_sec / coverage_ratio。
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .feature_extractor import compute_dhash, compute_sha256
from .schemas import VideoExtractionResult, VideoFrameItem


def _frame_signature(image_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    small = cv2.resize(image_bgr, (160, 90), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
    edges = cv2.Canny(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), 60, 150).astype(np.float32) / 255.0
    return gray, edges, hist


def _combined_frame_diff(
    prev_sig: Tuple[np.ndarray, np.ndarray, np.ndarray],
    curr_sig: Tuple[np.ndarray, np.ndarray, np.ndarray],
) -> float:
    """
    综合差异度（同时敏感于空间位移、亮度变化、黑白运动物体和色彩切换）：
    - 灰度空间平均绝对差 (MAD)
    - 边缘空间绝对差 (对黑白几何/文字/物体移动高度敏感)
    - HSV 颜色直方图巴氏距离
    取三者最大值并归一化到 [0, 1]。
    """
    g1, e1, h1 = prev_sig
    g2, e2, h2 = curr_sig
    gray_mad = float(np.mean(np.abs(g1 - g2)))
    edge_diff = float(np.mean(np.abs(e1 - e2)))
    color_diff = float(cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA))
    # 放大微小局部运动在全局均值下的响应，便于 0.05 阈值稳定捕获运动物体
    return float(min(1.0, max(gray_mad * 3.0, edge_diff * 4.0, color_diff)))


def extract_video_keyframes(
    video_path: str | Path,
    output_dir: str | Path,
    sample_fps: float = 1.0,
    min_scene_diff: float = 0.06,
    base_interval_sec: float = 3.0,
    max_frames: int = 300,
    video_id: Optional[str] = None,
) -> VideoExtractionResult:
    """
    全片均匀覆盖 + 结构变化补帧。
    - 即使画面变化极小，每隔 base_interval_sec（长视频自动拉伸以覆盖全片）也会保底保留一帧；
    - 只要画面发生明显运动或场景切换（综合差异 >= min_scene_diff），立即补帧。
    """
    vpath = Path(video_path)
    if not vpath.exists():
        raise FileNotFoundError(f"视频文件不存在: {vpath}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {vpath}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    if fps <= 0:
        fps = 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration_sec = round(total_frames / fps, 3) if total_frames > 0 else 0.0

    # 为长视频自适应调整采样频率，确保在 max_frames 预算内均匀覆盖全片，而不是截断后半段
    effective_fps = max(sample_fps, 0.1)
    if duration_sec > 0 and max_frames > 0:
        max_sustainable_fps = (max_frames * 0.8) / duration_sec
        if effective_fps > max_sustainable_fps:
            effective_fps = max(0.05, max_sustainable_fps)
        min_base_interval = duration_sec / max(1, int(max_frames * 0.5))
        base_interval_sec = max(base_interval_sec, min_base_interval)

    step = max(1, int(round(fps / effective_fps)))
    vid = video_id or vpath.stem

    sampled: List[VideoFrameItem] = []
    prev_sig: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
    last_saved_ts: float = -1e9
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % step == 0:
                ts = round(frame_idx / fps, 2)
                sig = _frame_signature(frame)
                diff = 1.0 if prev_sig is None else _combined_frame_diff(prev_sig, sig)

                is_base = (prev_sig is None) or ((ts - last_saved_ts) >= base_interval_sec - 1e-6)
                is_scene = diff >= min_scene_diff

                if is_base or is_scene:
                    reason = "scene_change" if (prev_sig is not None and is_scene) else "base_sample"
                    frame_name = f"{vid}_f{frame_idx:06d}_{ts:07.2f}s.jpg"
                    frame_file = out_dir / frame_name
                    ok_enc, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                    if ok_enc:
                        buf.tofile(str(frame_file))

                    sampled.append(
                        VideoFrameItem(
                            frame_index=frame_idx,
                            timestamp_sec=ts,
                            frame_path=str(frame_file),
                            phash=compute_dhash(frame),
                            scene_diff=round(diff, 4),
                            reason=reason,
                        )
                    )
                    prev_sig = sig
                    last_saved_ts = ts
                    if len(sampled) >= max_frames:
                        break

            frame_idx += 1
    finally:
        cap.release()

    if duration_sec == 0.0 and frame_idx > 0:
        duration_sec = round(frame_idx / fps, 3)
        total_frames = frame_idx

    cov_start = sampled[0].timestamp_sec if sampled else 0.0
    cov_end = sampled[-1].timestamp_sec if sampled else 0.0
    cov_ratio = round(min(1.0, (cov_end - cov_start + (1.0 / effective_fps)) / duration_sec), 4) if (sampled and duration_sec > 0) else 0.0

    return VideoExtractionResult(
        video_id=vid,
        file_path=str(vpath),
        sha256=compute_sha256(vpath),
        fps=round(fps, 3),
        total_frames=total_frames,
        duration_sec=duration_sec,
        width=width,
        height=height,
        coverage_start_sec=cov_start,
        coverage_end_sec=cov_end,
        coverage_ratio=cov_ratio,
        sampled_frames=sampled,
    )
