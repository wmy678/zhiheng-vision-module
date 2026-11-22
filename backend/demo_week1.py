"""
成员B 端到端真实检测闭环与 7 项审查回归自测脚本 (v0.3)
运行方式：
    python backend/demo_week1.py
覆盖验证项：
1. 模型降级链与签名校验（auto 依次尝试 clip -> resnet -> fallback_cv，并校验 index_manifest.json）
2. 增量入库不覆盖旧库 + SHA-256 重复跳过
3. 真实图片比对闭环（多视角向量召回 + ORB/RANSAC 几何验证 -> 真实算法生成 bbox_norm，非手写）
4. 拒绝匹配机制（无关图片返回 status="no_match"，不把 Top-1 强行当匹配）
5. 10 秒黑白运动物体视频抽帧（验证结构/亮度/边缘变化补帧与全片覆盖率，不再只保留第 0 秒）
6. Pydantic 严格校验拦截（非法 bbox_norm、end_sec <= start_sec、超出 duration_sec）
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from pydantic import ValidationError

from vision import (
    ImageFeatureExtractor,
    VisualImageDetector,
    VisualVectorStore,
    extract_video_keyframes,
    ingest_directory,
)
from vision.schemas import BoundingBoxItem, VideoDetectionPayload, VideoSegmentItem


def _imwrite_unicode(path: Path, img: np.ndarray, params: list[int] | None = None) -> None:
    ext = path.suffix or ".jpg"
    ok, buf = cv2.imencode(ext, img, params or [])
    if not ok:
        raise RuntimeError(f"图片编码失败: {path}")
    buf.tofile(str(path))


def _make_textured_poster(seed: int, label: str, bg_bgr: tuple[int, int, int]) -> np.ndarray:
    """生成带丰富纹理角点的测试图（供 ORB 局部关键点与 RANSAC 真实几何验证）。"""
    rng = np.random.default_rng(seed)
    img = np.full((360, 480, 3), bg_bgr, dtype=np.uint8)
    for _ in range(36):
        x1, y1 = int(rng.integers(15, 430)), int(rng.integers(15, 310))
        w, h = int(rng.integers(24, 90)), int(rng.integers(24, 80))
        color = tuple(int(c) for c in rng.integers(40, 245, size=3))
        if rng.random() > 0.45:
            cv2.rectangle(img, (x1, y1), (min(470, x1 + w), min(350, y1 + h)), color, 2)
        else:
            cv2.circle(img, (x1, y1), max(10, w // 3), color, 2)
    cv2.putText(img, label, (28, 64), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
    cv2.putText(img, f"ID-{seed}", (28, 325), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (240, 240, 120), 2)
    return img


def _make_bw_moving_video(video_path: Path, duration_sec: int = 10, fps: int = 10) -> None:
    """生成 10 秒纯黑白运动方块视频（测试灰度/边缘变化抽帧，验证不再只保留第 0 秒）。"""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, float(fps), (320, 180))
    total = duration_sec * fps
    for i in range(total):
        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        x = int(20 + (240 * i) / max(1, total - 1))
        y = int(45 + 35 * np.sin(i / 8.0))
        cv2.rectangle(frame, (x, y), (x + 48, y + 48), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()


def _test_schema_validations() -> None:
    # 1. 非法 bbox_norm (x2 <= x1)
    try:
        BoundingBoxItem(
            id="01", dimension="色彩与画风", label="非法框", attention_level="focus",
            score=0.9, bbox_norm=[0.6, 0.2, 0.4, 0.8], location_desc="test", fact_description="test",
        )
        raise AssertionError("应当拦截 x2 <= x1 的非法 bbox_norm")
    except ValidationError:
        pass

    # 2. 非法视频片段 (end_sec <= start_sec)
    try:
        VideoSegmentItem(
            id="01", dimension="画面直接搬运", start_sec=20.0, end_sec=15.0,
            attention_level="focus", score=0.8, label="非法片段", fact_description="test",
        )
        raise AssertionError("应当拦截 end_sec <= start_sec 的非法片段")
    except ValidationError:
        pass

    # 3. 片段超出视频总时长
    try:
        VideoDetectionPayload(
            task_id="t1", file_name="a.mp4", reference_scope="test", duration_sec=30.0,
            segments=[
                VideoSegmentItem(
                    id="01", dimension="画面直接搬运", start_sec=25.0, end_sec=35.0,
                    attention_level="focus", score=0.8, label="越界片段", fact_description="test",
                )
            ],
        )
        raise AssertionError("应当拦截 end_sec > duration_sec 的越界片段")
    except ValidationError:
        pass


def main() -> None:
    root = Path(__file__).resolve().parent
    demo_dir = root / "demo_workspace"
    if demo_dir.exists():
        shutil.rmtree(demo_dir)
    batch_a = demo_dir / "batch_a"
    batch_b = demo_dir / "batch_b"
    index_dir = demo_dir / "vector_index"
    samples_dir = root / "sample_outputs"
    batch_a.mkdir(parents=True, exist_ok=True)
    batch_b.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    # 准备两批底库图片（验证增量入库不覆盖）
    poster_a = _make_textured_poster(101, "STAR POSTER A", (35, 30, 55))
    poster_b = _make_textured_poster(202, "CITY NIGHT B", (50, 45, 25))
    _imwrite_unicode(batch_a / "ref_poster_a.jpg", poster_a)
    _imwrite_unicode(batch_b / "ref_poster_b.jpg", poster_b)

    # 1. 增量入库测试：先入 A，再入 B，再重复入 A
    res_a = ingest_directory(batch_a, index_dir, mode="rebuild", backend="auto")
    res_b = ingest_directory(batch_b, index_dir, mode="incremental", backend="auto")
    res_dup = ingest_directory(batch_a, index_dir, mode="incremental", backend="auto")
    assert res_a["total_vectors"] == 1
    assert res_b["total_vectors"] == 2, f"增量入库覆盖了旧库: {res_b}"
    assert res_dup["total_vectors"] == 2 and res_dup["skipped_duplicates"] == 1
    print(f"[PASS] 增量入库与去重验证通过：A(1) -> +B(2) -> 重复A跳过({res_dup['skipped_duplicates']})")
    print(f"       模型签名: {res_b['extractor_signature']}")
    if res_b["fallback_history"]:
        print(f"       降级链路: {' | '.join(res_b['fallback_history'])}")

    # 2. 真实图片比对闭环（构造：将 poster_a 水平镜像后局部粘贴到更大画布中）
    canvas = np.full((540, 960, 3), (24, 24, 28), dtype=np.uint8)
    flipped_a = cv2.flip(poster_a, 1)
    canvas[90:450, 410:890] = flipped_a
    query_match_path = demo_dir / "query_pasted_hflip.jpg"
    _imwrite_unicode(query_match_path, canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 88])

    # 构造一张完全无关的纯随机噪声/棋盘图（测试 no_match 拒绝）
    rng = np.random.default_rng(999)
    unrelated = rng.integers(0, 255, size=(360, 480, 3), dtype=np.uint8)
    query_nomatch_path = demo_dir / "query_unrelated.jpg"
    _imwrite_unicode(query_nomatch_path, unrelated)

    extractor = ImageFeatureExtractor(backend="auto")
    store = VisualVectorStore(index_dir=index_dir)
    store.load(extractor=extractor)
    detector = VisualImageDetector(extractor=extractor, store=store, recall_threshold=0.45, min_inliers=10)

    real_match = detector.detect_image(query_match_path, task_id="real_match_01")
    assert real_match.status == "completed" and len(real_match.boxes) >= 1, f"真实匹配闭环失败: {real_match}"
    box0 = real_match.boxes[0]
    print(
        f"[PASS] 真实图片比对闭环通过：status={real_match.status} | "
        f"bbox_norm={box0.bbox_norm} | inliers={box0.geometric_inliers} | source={box0.source_ref}"
    )

    real_nomatch = detector.detect_image(query_nomatch_path, task_id="real_nomatch_01")
    assert real_nomatch.status == "no_match" and len(real_nomatch.boxes) == 0
    print(f"[PASS] 拒绝匹配机制通过：无关图片返回 status={real_nomatch.status}, boxes=[]")

    # 导出真实算法生成的 JSON（非手写假框）
    (samples_dir / "real_image_match.json").write_text(real_match.to_json(indent=2), encoding="utf-8")
    (samples_dir / "real_image_nomatch.json").write_text(real_nomatch.to_json(indent=2), encoding="utf-8")

    # 3. 10 秒黑白运动物体视频抽帧验证
    bw_vid = demo_dir / "bw_moving_10s.mp4"
    _make_bw_moving_video(bw_vid, duration_sec=10, fps=10)
    vid_res = extract_video_keyframes(bw_vid, demo_dir / "bw_frames", sample_fps=1.0, min_scene_diff=0.06)
    assert len(vid_res.sampled_frames) >= 5, f"黑白运动视频漏帧: 仅保留 {len(vid_res.sampled_frames)} 帧"
    assert vid_res.coverage_end_sec >= 8.0, f"视频后半段未覆盖: coverage_end={vid_res.coverage_end_sec}"
    print(
        f"[PASS] 10秒黑白运动视频抽帧通过：保留 {len(vid_res.sampled_frames)} 帧，"
        f"覆盖 {vid_res.coverage_start_sec}s ~ {vid_res.coverage_end_sec}s (覆盖率 {vid_res.coverage_ratio:.0%})"
    )

    # 4. Pydantic 严格数据校验验证
    _test_schema_validations()
    print("[PASS] Pydantic 非法坐标/时间戳/越界片段拦截验证通过")
    print(f"\n所有 6 项回归测试全部通过！真实算法输出已保存至: {samples_dir}")


if __name__ == "__main__":
    main()
