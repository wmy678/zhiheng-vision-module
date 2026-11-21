"""
视频抽帧模块 (Week 1)
支持按固定采样率（如 1 fps）抽帧，并结合 HSV 直方图差异进行镜头去重，
输出带精确时间戳（timestamp_sec）的关键帧集合，便于后续视频片段定位与入库。
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from .feature_extractor import compute_dhash, compute_sha256
from .schemas import VideoExtractionResult, VideoFrameItem


def _frame_hist(image_bgr: np.ndarray) -> np.ndarray:
    small = cv2.resize(image_bgr, (160, 90), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
    return hist


def extract_video_keyframes(
    video_path: str | Path,
    output_dir: str | Path,
    sample_fps: float = 1.0,
    min_scene_diff: float = 0.08,
    max_frames: int = 300,
    video_id: Optional[str] = None,
) -> VideoExtractionResult:
    """
    从视频中抽取关键帧并落盘保存为 JPEG。

    参数:
        video_path: 视频文件路径
        output_dir: 抽帧图片输出目录
        sample_fps: 每秒基础采样帧数（默认 1.0，即每秒检查 1 帧）
        min_scene_diff: 相邻保留帧的最小直方图差异度（0 表示不做镜头去重，保留所有采样帧）
        max_frames: 单视频最大抽帧数上限
        video_id: 自定义视频业务 ID
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

    step = max(1, int(round(fps / max(sample_fps, 0.1))))
    vid = video_id or vpath.stem

    sampled: List[VideoFrameItem] = []
    prev_hist: Optional[np.ndarray] = None
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % step == 0:
                hist = _frame_hist(frame)
                if prev_hist is None:
                    diff = 1.0
                else:
                    # 巴氏距离：0 表示完全一致，越大表示画面变化越明显
                    diff = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA))

                if prev_hist is None or diff >= min_scene_diff:
                    ts = round(frame_idx / fps, 2)
                    frame_name = f"{vid}_f{frame_idx:06d}_{ts:07.2f}s.jpg"
                    frame_file = out_dir / frame_name
                    ok_enc, buf = cv2.imencode(
                        ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90]
                    )
                    if ok_enc:
                        buf.tofile(str(frame_file))

                    sampled.append(
                        VideoFrameItem(
                            frame_index=frame_idx,
                            timestamp_sec=ts,
                            frame_path=str(frame_file),
                            phash=compute_dhash(frame),
                            scene_diff=round(diff, 4),
                        )
                    )
                    prev_hist = hist
                    if len(sampled) >= max_frames:
                        break

            frame_idx += 1
    finally:
        cap.release()

    if duration_sec == 0.0 and frame_idx > 0:
        duration_sec = round(frame_idx / fps, 3)
        total_frames = frame_idx

    return VideoExtractionResult(
        video_id=vid,
        file_path=str(vpath),
        sha256=compute_sha256(vpath),
        fps=round(fps, 3),
        total_frames=total_frames,
        duration_sec=duration_sec,
        width=width,
        height=height,
        sampled_frames=sampled,
    )
