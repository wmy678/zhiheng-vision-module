"""
真实图片比对闭环与几何验证模块 (v0.3 新增)
改进点（对应评审意见 #1、#5）：
1. 完整闭环：待检图 -> 多视角向量 Top-K 召回 -> ORB 局部特征匹配 + Lowe 比率测试 + RANSAC 单应性几何验证
   -> 计算真实归一化匹配区域 bbox_norm -> 输出 ImageDetectionPayload。
2. 拒绝匹配机制：若 Top-K 候选相似度低于阈值且几何验证未通过，明确返回 status="no_match" 与 boxes=[]，
   绝不强行把检索第一名当作侵权或复用匹配。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .faiss_ingest import VisualVectorStore
from .feature_extractor import ImageFeatureExtractor
from .schemas import (
    BoundingBoxItem,
    ImageAxisScores,
    ImageDetectionPayload,
    ReferenceMetaRecord,
)


def _imread_unicode(path: str | Path) -> Optional[np.ndarray]:
    p = Path(path)
    if not p.exists():
        return None
    raw = np.fromfile(str(p), dtype=np.uint8)
    return cv2.imdecode(raw, cv2.IMREAD_COLOR)


def _describe_location(bbox_norm: List[float]) -> str:
    x1, y1, x2, y2 = bbox_norm
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    horiz = "左侧" if cx < 0.38 else ("右侧" if cx > 0.62 else "中部")
    vert = "上方" if cy < 0.38 else ("下方" if cy > 0.62 else "中段")
    area_pct = int(round((x2 - x1) * (y2 - y1) * 100))
    return f"画面{horiz}{vert}区域（约占画幅 {area_pct}%）"


def verify_geometric_match(
    query_bgr: np.ndarray,
    ref_bgr: np.ndarray,
    min_inliers: int = 10,
    ratio_thresh: float = 0.78,
) -> Tuple[bool, int, float, Optional[List[float]], str]:
    """
    使用 ORB 局部关键点 + Lowe Ratio Test + RANSAC Homography 进行几何验证。
    同时测试原图与水平镜像两个方向，返回：
        (是否通过几何验证, 内点数, 内点比例, 待检图上的真实归一化框 [x1,y1,x2,y2], 命中方向)
    """
    orb = cv2.ORB_create(nfeatures=1200)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    q_gray = cv2.cvtColor(query_bgr, cv2.COLOR_BGR2GRAY)
    qh, qw = q_gray.shape[:2]
    kp_q, des_q = orb.detectAndCompute(q_gray, None)
    if des_q is None or len(kp_q) < min_inliers:
        return False, 0, 0.0, None, "none"

    best_result: Tuple[bool, int, float, Optional[List[float]], str] = (False, 0, 0.0, None, "none")

    for orient, ref_img in [("original", ref_bgr), ("hflip", cv2.flip(ref_bgr, 1))]:
        r_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
        kp_r, des_r = orb.detectAndCompute(r_gray, None)
        if des_r is None or len(kp_r) < min_inliers:
            continue

        raw_matches = bf.knnMatch(des_r, des_q, k=2)
        good = []
        for pair in raw_matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < ratio_thresh * n.distance:
                    good.append(m)

        if len(good) < min_inliers:
            continue

        src_pts = np.float32([kp_r[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_q[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None or mask is None:
            continue

        inlier_mask = mask.ravel().astype(bool)
        inliers = int(np.sum(inlier_mask))
        inlier_ratio = float(inliers / max(1, len(good)))

        if inliers >= min_inliers and inliers > best_result[1]:
            inlier_q_pts = dst_pts[inlier_mask].reshape(-1, 2)
            # 基于 RANSAC 内点外接矩形计算待检图上的真实匹配区域，适当外扩 3% 留出边界余量
            x1 = max(0.0, float(np.min(inlier_q_pts[:, 0]) / qw) - 0.03)
            y1 = max(0.0, float(np.min(inlier_q_pts[:, 1]) / qh) - 0.03)
            x2 = min(1.0, float(np.max(inlier_q_pts[:, 0]) / qw) + 0.03)
            y2 = min(1.0, float(np.max(inlier_q_pts[:, 1]) / qh) + 0.03)
            if x2 - x1 > 0.03 and y2 - y1 > 0.03:
                best_result = (True, inliers, round(inlier_ratio, 4), [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)], orient)

    return best_result


class VisualImageDetector:
    """
    真实图片比对检测器：
    向量多视角召回 Top-K -> 逐候选执行局部特征几何验证 -> 生成真实坐标框或拒绝匹配 (no_match)。
    """

    def __init__(
        self,
        extractor: ImageFeatureExtractor,
        store: VisualVectorStore,
        recall_threshold: float = 0.55,
        min_inliers: int = 10,
    ) -> None:
        self.extractor = extractor
        self.store = store
        self.recall_threshold = recall_threshold
        self.min_inliers = min_inliers

    def detect_image(
        self,
        query_path: str | Path,
        task_id: str = "real_img_task",
        top_k: int = 5,
        mode: str = "smart_check",
    ) -> ImageDetectionPayload:
        qpath = Path(query_path)
        query_bgr = _imread_unicode(qpath)
        if query_bgr is None:
            return ImageDetectionPayload(
                task_id=task_id,
                status="failed",
                error_message=f"无法读取待检图片: {qpath.name}",
                file_name=qpath.name,
                reference_scope=f"本地视觉底库 ({len(self.store.records)} 条向量)",
                model_signature=self.extractor.signature,
                image_width=1,
                image_height=1,
                mode=mode,  # type: ignore[arg-type]
                boxes=[],
                summary_note="图片解码失败，未执行检索。",
            )

        qh, qw = query_bgr.shape[:2]
        self.store.verify_extractor_compatibility(self.extractor)

        # 1. 多视角向量召回（按 source_id 合并最高分候选）
        views = self.extractor.extract_multi_view_from_bgr(query_bgr)
        cand_map: Dict[str, Tuple[ReferenceMetaRecord, float, str]] = {}
        for v_name, vec in views.items():
            for meta, sim in self.store.search(vec, top_k=top_k, extractor=self.extractor):
                if meta.media_type != "image":
                    continue
                prev = cand_map.get(meta.source_id)
                if prev is None or sim > prev[1]:
                    cand_map[meta.source_id] = (meta, sim, v_name)

        ranked_cands = sorted(cand_map.values(), key=lambda x: -x[1])[:top_k]

        # 2. 对 Top-K 候选逐个做 ORB + RANSAC 几何验证（不盲信 Top-1 向量余弦分数）
        verified_boxes: List[BoundingBoxItem] = []
        best_style_sim: Optional[float] = None

        for meta, sim, view_name in ranked_cands:
            if sim < self.recall_threshold:
                continue
            ref_bgr = _imread_unicode(meta.file_path)
            if ref_bgr is None:
                continue

            ok_geo, inliers, inlier_ratio, bbox_norm, geo_orient = verify_geometric_match(
                query_bgr=query_bgr,
                ref_bgr=ref_bgr,
                min_inliers=self.min_inliers,
            )
            if not ok_geo or bbox_norm is None:
                continue

            idx_str = f"{len(verified_boxes) + 1:02d}"
            combined_score = round(min(1.0, 0.5 * sim + 0.5 * min(1.0, inliers / 40.0)), 4)
            best_style_sim = max(best_style_sim or 0.0, round(sim * 100.0, 1))
            att_level = "focus" if combined_score >= 0.78 else ("moderate" if combined_score >= 0.6 else "auxiliary")
            orient_desc = "水平镜像视角" if (geo_orient == "hflip" or view_name == "hflip") else "正向视角"

            verified_boxes.append(
                BoundingBoxItem(
                    id=idx_str,
                    dimension="色彩与画风",
                    label=f"局部画面结构匹配（{orient_desc}）",
                    attention_level=att_level,  # type: ignore[arg-type]
                    score=combined_score,
                    bbox_norm=bbox_norm,
                    location_desc=_describe_location(bbox_norm),
                    fact_description=(
                        f"在{orient_desc}下与底库参考图《{meta.title}》完成局部特征对齐，"
                        f"RANSAC 几何验证通过（{inliers} 个一致内点，内点率 {inlier_ratio:.0%}）。"
                    ),
                    source_ref=f"{meta.title} (ID: {meta.source_id})",
                    evidence_note=(
                        f"向量余弦相似度 {sim:.3f}（召回视角: {view_name}），"
                        f"局部关键点几何验证通过；结果仅反映画面结构重合，需人工核验授权状态。"
                    ),
                    geometric_inliers=inliers,
                    reviewed=False,
                )
            )

        # 未实现维度明确为 None + not_implemented，不填假 0 分
        axis_scores = ImageAxisScores(
            color_style=best_style_sim,
            dimension_status={
                "character_design": "not_implemented",
                "pose_composition": "not_implemented",
                "color_style": "detected" if best_style_sim is not None else "detected",
                "text_logo": "not_implemented",
                "aigc_artifact_score": "not_implemented",
            },
        )

        if not verified_boxes:
            return ImageDetectionPayload(
                task_id=task_id,
                status="no_match",
                file_name=qpath.name,
                reference_scope=f"本地视觉底库（{len(self.store.records)} 条向量）",
                model_signature=self.extractor.signature,
                image_width=qw,
                image_height=qh,
                mode=mode,  # type: ignore[arg-type]
                axis_scores=axis_scores,
                boxes=[],
                summary_note="当前参考范围内未发现通过几何验证的匹配区域（未检出不代表不存在库外来源风险）。",
            )

        return ImageDetectionPayload(
            task_id=task_id,
            status="completed",
            file_name=qpath.name,
            reference_scope=f"本地视觉底库（{len(self.store.records)} 条向量）",
            model_signature=self.extractor.signature,
            image_width=qw,
            image_height=qh,
            mode=mode,  # type: ignore[arg-type]
            axis_scores=axis_scores,
            boxes=verified_boxes,
            summary_note=(
                f"在当前参考范围内共定位 {len(verified_boxes)} 处通过几何验证的相似区域；"
                "其余未接入算法维度标注为“未检测”，所有结果仅供人工授权复核。"
            ),
        )
