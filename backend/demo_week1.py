"""
成员B 第1周端到端自测与演示脚本 (v0.2 优化版)
升级内容：
1. 在原有检索自测基础上，新增【镜像翻转 + 轻度裁剪】编辑变换查询图，
   验证多视角检索仍能稳定 Top-1 命中底库原图并标记变换线索。
2. 导出的 sample_image_result.json / sample_video_result.json 完全对齐：
   - 《知衡 v2.4》图片四轴 / 视频五轴指标
   - 功能二“01/02/03”固定编号、关注程度（focus/moderate/auxiliary）、检测事实与比对依据
   - 视频三轨时间轴（visual_reuse / ai_anomaly / plot_av）
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from vision import ImageFeatureExtractor, VisualVectorStore, ingest_directory
from vision.schemas import (
    BoundingBoxItem,
    ImageAxisScores,
    ImageDetectionPayload,
    VideoAxisScores,
    VideoDetectionPayload,
    VideoSegmentItem,
)

def _imwrite_unicode(path: Path, img: np.ndarray, params: list[int] | None = None) -> None:
    ext = path.suffix or ".jpg"
    ok, buf = cv2.imencode(ext, img, params or [])
    if not ok:
        raise RuntimeError(f"图片编码失败: {path}")
    buf.tofile(str(path))

def _generate_synthetic_assets(media_dir: Path) -> tuple[Path, Path]:
    img_dir = media_dir / "images"
    vid_dir = media_dir / "videos"
    img_dir.mkdir(parents=True, exist_ok=True)
    vid_dir.mkdir(parents=True, exist_ok=True)

    img1 = np.full((360, 640, 3), (45, 32, 24), dtype=np.uint8)
    cv2.rectangle(img1, (40, 40), (300, 310), (210, 130, 50), -1)
    cv2.circle(img1, (470, 180), 95, (70, 190, 240), -1)
    cv2.putText(img1, "REF-01 POSTER", (55, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    _imwrite_unicode(img_dir / "ref_poster_01.jpg", img1)

    img2 = np.full((360, 640, 3), (38, 62, 40), dtype=np.uint8)
    cv2.circle(img2, (320, 125), 55, (195, 215, 245), -1)
    cv2.rectangle(img2, (235, 185), (405, 345), (90, 165, 220), -1)
    cv2.putText(img2, "REF-02 PORTRAIT", (40, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (230, 245, 230), 2)
    _imwrite_unicode(img_dir / "ref_portrait_02.jpg", img2)

    img3 = np.full((360, 640, 3), (68, 28, 52), dtype=np.uint8)
    for x in range(0, 640, 40):
        cv2.line(img3, (x, 0), (x, 360), (140, 90, 180), 1)
    for y in range(0, 360, 40):
        cv2.line(img3, (0, y), (640, y), (140, 90, 180), 1)
    cv2.rectangle(img3, (180, 90), (460, 270), (220, 180, 90), 3)
    _imwrite_unicode(img_dir / "ref_grid_03.jpg", img3)

    video_path = vid_dir / "ref_clip_01.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 10.0, (320, 180))
    for s_idx, (base_img, tag) in enumerate([(img1, "SCENE 1"), (img2, "SCENE 2"), (img3, "SCENE 3")]):
        frame_base = cv2.resize(base_img, (320, 180))
        for f in range(20):
            frame = frame_base.copy()
            cv2.putText(frame, f"{tag} t={s_idx * 2 + f / 10.0:.1f}s", (12, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            writer.write(frame)
    writer.release()

    # 查询图1：轻度压缩 + 亮度微调
    query_edited = media_dir.parent / "query_poster_edited.jpg"
    _imwrite_unicode(query_edited, cv2.convertScaleAbs(img1, alpha=1.06, beta=8), [int(cv2.IMWRITE_JPEG_QUALITY), 75])
    # 查询图2：水平镜像翻转 + 裁边（模拟对抗搬运编辑）
    query_flipped = media_dir.parent / "query_poster_flipped_crop.jpg"
    flipped = cv2.flip(img1, 1)
    cropped = flipped[12:348, 20:620]
    _imwrite_unicode(query_flipped, cropped, [int(cv2.IMWRITE_JPEG_QUALITY), 75])

    return query_edited, query_flipped


def _export_sample_payloads(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    img_payload = ImageDetectionPayload(
        task_id="zh_img_2026_w1",
        status="completed",
        file_name="太空邮差_宣传海报_待检稿.png",
        reference_scope="知识产权底库 v2.4（12,846 部作品）· 智能自查模式",
        image_width=1280,
        image_height=720,
        mode="smart_check",
        axis_scores=ImageAxisScores(
            character_design=86.0,
            pose_composition=78.0,
            color_style=72.0,
            text_logo=81.0,
            aigc_artifact_score=69.0,
            watermark_logo_detected=True,
        ),
        boxes=[
            BoundingBoxItem(
                id="01",
                dimension="角色形象",
                label="IP 形象重合",
                attention_level="focus",
                score=0.91,
                bbox_norm=[0.14, 0.16, 0.42, 0.78],
                location_desc="画面左侧主体人物区域",
                fact_description="主体角色头盔轮廓、肩部配色及站位比例与参考库条目存在较高结构重合。",
                source_ref="底库作品《星际投递员_角色设定图》(ID: IP-2048)",
                evidence_note="特征余弦相似度 0.91；系统未检出对应角色授权凭证，建议核对授权范围（演示数据）。",
            ),
            BoundingBoxItem(
                id="02",
                dimension="画面文字/Logo",
                label="品牌标志雷同",
                attention_level="focus",
                score=0.84,
                bbox_norm=[0.63, 0.18, 0.87, 0.43],
                location_desc="右上角徽标与文字区",
                fact_description="徽标外框几何结构与底库注册图形标志局部特征接近，文字排版位置对应。",
                source_ref="底库作品《星邮局品牌视觉规范》(ID: TM-0892)",
                evidence_note="局部轮廓匹配度 0.84，请人工核查是否属于授权合作物料（演示数据）。",
            ),
            BoundingBoxItem(
                id="03",
                dimension="生成痕迹",
                label="局部频域伪影",
                attention_level="moderate",
                score=0.69,
                bbox_norm=[0.43, 0.56, 0.58, 0.74],
                location_desc="中部手持信封与指节局部",
                fact_description="手指关节边缘存在轻度频域纹理不一致与笔触过渡断层。",
                source_ref="内部生成痕迹基线（非底库版权比对）",
                evidence_note="属于生成式辅助绘制常见局部特征，仅作复核参考，不代表权属结论（演示数据）。",
            ),
        ],
        summary_note="当前参考范围内共定位 3 处待复核区域（重点关注 2 处，一般关注 1 处），覆盖角色形象、画面文字/Logo 与生成痕迹维度；结论仅供授权核查与人工复核。",
    )

    vid_payload = VideoDetectionPayload(
        task_id="zh_vid_2026_w1",
        status="completed",
        file_name="城市夜游短片_成片V1.mp4",
        reference_scope="知识产权底库 v2.4（12,846 部作品）· 智能自查模式",
        duration_sec=60.0,
        mode="smart_check",
        axis_scores=VideoAxisScores(
            direct_footage_reuse=84.0,
            character_visual=73.0,
            dialogue_transcript=46.0,
            voiceprint_match=38.0,
            story_plot=61.0,
            ai_face_swap_confidence=76.0,
            av_sync_consistency=89.0,
        ),
        segments=[
            VideoSegmentItem(
                id="01",
                track="visual_reuse",
                dimension="画面直接搬运",
                start_sec=12.0,
                end_sec=24.0,
                attention_level="focus",
                score=0.93,
                label="镜像裁剪搬运片段",
                transform_hint="水平镜像 + 轻度裁边",
                fact_description="12s–24s 连续关键帧在水平镜像视角下与底库参考视频对应片段高度一致。",
                source_ref="底库短片《夜航城市纪实》(00:08 - 00:20)",
                keyframe_time_sec=16.0,
                evidence_note="多视角检索命中水平镜像视角（相似度 0.93），当前未关联该片段授权记录（演示数据）。",
            ),
            VideoSegmentItem(
                id="02",
                track="ai_anomaly",
                dimension="人物形象",
                start_sec=31.0,
                end_sec=35.0,
                attention_level="focus",
                score=0.76,
                label="疑似 AI 换脸片段",
                transform_hint="帧级面部融合边界异常",
                fact_description="31s–35s 主播面部边缘存在帧间闪烁与肤色纹理过渡异常。",
                source_ref="底库人物设定《签约出镜人_A04》",
                keyframe_time_sec=33.0,
                evidence_note="帧级异常置信度 0.76，需结合原始拍摄工程文件或肖像授权确认（演示数据）。",
            ),
            VideoSegmentItem(
                id="03",
                track="plot_av",
                dimension="故事剧情",
                start_sec=42.0,
                end_sec=55.0,
                attention_level="auxiliary",
                score=0.58,
                label="转场空镜与节奏接近",
                transform_hint=None,
                fact_description="42s–55s 连续三个城市夜景空镜的景别顺序与参考片相近。",
                source_ref="底库短片《夜航城市纪实》(00:38 - 00:50)",
                keyframe_time_sec=48.0,
                evidence_note="属于辅助观察线索，公共夜景题材存在天然巧合可能，建议人工复核（演示数据）。",
            ),
        ],
        summary_note="全片 60 秒共定位 3 段待复核区间（重点关注 2 段，辅助观察 1 段），分布于画面搬运轨、异常线索轨与剧情关联轨。",
    )

    (out_dir / "sample_image_result.json").write_text(img_payload.model_dump_json(indent=2), encoding="utf-8")
    (out_dir / "sample_video_result.json").write_text(vid_payload.model_dump_json(indent=2), encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parent
    demo_dir = root / "demo_workspace"
    media_dir = demo_dir / "reference_media"
    index_dir = demo_dir / "vector_index"
    samples_dir = root / "sample_outputs"

    query_edited, query_flipped = _generate_synthetic_assets(media_dir)
    summary = ingest_directory(media_dir=media_dir, index_dir=index_dir, backend="auto", sample_fps=1.0, min_scene_diff=0.08)
    print("=== 首批底库入库完成 ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    store = VisualVectorStore(index_dir=index_dir, dim=512)
    loaded = store.load()
    extractor = ImageFeatureExtractor(backend="auto")

    # 自测1：压缩+调色图检索
    q_rec = extractor.extract_file(query_edited, item_id="query_poster_edited")
    hits = store.search(np.array(q_rec.embedding, dtype=np.float32), top_k=3)
    print(f"\n=== 检索自测 1：压缩+调色图 (索引条目: {loaded}, 引擎: {store.engine_name}) ===")
    for rank, (meta, sim) in enumerate(hits, start=1):
        ts_str = f" @ {meta.frame_timestamp_sec}s" if meta.frame_timestamp_sec is not None else ""
        print(f"  Top-{rank}: {meta.title}{ts_str} | cosine_sim={sim:.4f}")
    assert hits and "ref_poster_01" in hits[0][0].title

    # 自测2：水平镜像+裁剪编辑变换检索（多视角取最大相似度）
    raw_bytes = np.fromfile(str(query_flipped), dtype=np.uint8)
    img_bgr = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
    views = extractor.extract_multi_view_from_bgr(img_bgr)
    best_view = ""
    best_meta = None
    best_sim = -1.0
    for view_name, vec in views.items():
        v_hits = store.search(vec, top_k=1)
        if v_hits and v_hits[0][1] > best_sim:
            best_sim = v_hits[0][1]
            best_meta = v_hits[0][0]
            best_view = view_name

    print(f"\n=== 检索自测 2：水平镜像+裁剪图（多视角鲁棒检索）===")
    print(f"  最佳命中视角: {best_view} | Top-1: {best_meta.title if best_meta else 'N/A'} | cosine_sim={best_sim:.4f}")
    assert best_meta is not None and "ref_poster_01" in best_meta.title

    _export_sample_payloads(samples_dir)
    print(f"\n=== 对齐《知衡 v2.4》示例 JSON 已导出至: {samples_dir} ===")


if __name__ == "__main__":
    main()
