"""
成员B（视觉/视频模块）数据结构定义 (v0.3 - 兼容 Pydantic v1 & v2，补齐严格校验与“未检测”状态区分)
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
import pydantic
from pydantic import BaseModel, Field

PYDANTIC_V2 = int(pydantic.__version__.split(".")[0]) >= 2
if PYDANTIC_V2:
    from pydantic import field_validator, model_validator
else:
    from pydantic import root_validator, validator


class CompatModel(BaseModel):
    """兼容 Pydantic v1 与 v2 的序列化基类"""

    def to_json(self, indent: Optional[int] = None) -> str:
        if PYDANTIC_V2:
            return self.model_dump_json(indent=indent)
        return self.json(ensure_ascii=False, indent=indent)

    @classmethod
    def from_json(cls, raw: str) -> Any:
        if PYDANTIC_V2:
            return cls.model_validate_json(raw)
        return cls.parse_raw(raw)


AttentionLevel = Literal["focus", "moderate", "auxiliary"]
CompareMode = Literal["smart_check", "targeted_compare"]
MediaType = Literal["image", "video"]
TaskStatus = Literal["loading", "completed", "no_match", "partial", "failed"]
DimensionState = Literal["detected", "not_implemented", "failed"]


class ImageFeatureRecord(CompatModel):
    item_id: str = Field(..., description="素材唯一 ID")
    file_path: str = Field(..., description="原始文件路径")
    sha256: str = Field(..., description="文件 SHA-256 校验值")
    width: int = Field(..., gt=0, description="图像宽度（像素）")
    height: int = Field(..., gt=0, description="图像高度（像素）")
    phash: str = Field(..., description="64 位感知哈希（十六进制）")
    model_name: str = Field(..., description="特征提取后端名称")
    model_weights: str = Field(..., description="权重/参数版本标识")
    preprocess_version: str = Field(..., description="预处理管线版本")
    embedding_dim: int = Field(..., gt=0, description="特征向量维度")
    embedding: List[float] = Field(..., description="原图 L2 归一化特征向量")
    hflip_embedding: Optional[List[float]] = Field(None, description="水平镜像特征向量")


class VideoFrameItem(CompatModel):
    frame_index: int = Field(..., ge=0)
    timestamp_sec: float = Field(..., ge=0.0)
    frame_path: str
    phash: str
    scene_diff: float = Field(0.0, ge=0.0)
    reason: Literal["base_sample", "scene_change"] = "base_sample"


class VideoExtractionResult(CompatModel):
    video_id: str
    file_path: str
    sha256: str
    fps: float
    total_frames: int
    duration_sec: float
    width: int
    height: int
    coverage_start_sec: float = 0.0
    coverage_end_sec: float = 0.0
    coverage_ratio: float = Field(0.0, ge=0.0, le=1.0)
    sampled_frames: List[VideoFrameItem]


class ImageAxisScores(CompatModel):
    character_design: Optional[float] = Field(None, ge=0.0, le=100.0)
    pose_composition: Optional[float] = Field(None, ge=0.0, le=100.0)
    color_style: Optional[float] = Field(None, ge=0.0, le=100.0)
    text_logo: Optional[float] = Field(None, ge=0.0, le=100.0)
    aigc_artifact_score: Optional[float] = Field(None, ge=0.0, le=100.0)
    watermark_logo_detected: Optional[bool] = None
    dimension_status: Dict[str, DimensionState] = Field(
        default_factory=lambda: {
            "character_design": "not_implemented",
            "pose_composition": "not_implemented",
            "color_style": "not_implemented",
            "text_logo": "not_implemented",
            "aigc_artifact_score": "not_implemented",
        }
    )


class VideoAxisScores(CompatModel):
    direct_footage_reuse: Optional[float] = Field(None, ge=0.0, le=100.0)
    character_visual: Optional[float] = Field(None, ge=0.0, le=100.0)
    dialogue_transcript: Optional[float] = Field(None, ge=0.0, le=100.0)
    voiceprint_match: Optional[float] = Field(None, ge=0.0, le=100.0)
    story_plot: Optional[float] = Field(None, ge=0.0, le=100.0)
    ai_face_swap_confidence: Optional[float] = Field(None, ge=0.0, le=100.0)
    av_sync_consistency: Optional[float] = Field(None, ge=0.0, le=100.0)
    dimension_status: Dict[str, DimensionState] = Field(
        default_factory=lambda: {
            "direct_footage_reuse": "not_implemented",
            "character_visual": "not_implemented",
            "dialogue_transcript": "not_implemented",
            "voiceprint_match": "not_implemented",
            "story_plot": "not_implemented",
            "ai_face_swap_confidence": "not_implemented",
            "av_sync_consistency": "not_implemented",
        }
    )


def _check_bbox_norm(v: List[float]) -> List[float]:
    if len(v) != 4:
        raise ValueError(f"bbox_norm 必须包含 4 个坐标值，当前: {v}")
    x1, y1, x2, y2 = v
    if not all(0.0 <= c <= 1.0 for c in v):
        raise ValueError(f"bbox_norm 坐标必须在 [0, 1] 范围内，当前值: {v}")
    if not (x2 > x1 and y2 > y1):
        raise ValueError(f"bbox_norm 必须满足 x2 > x1 且 y2 > y1，当前值: {v}")
    return [round(float(c), 4) for c in v]


class BoundingBoxItem(CompatModel):
    id: str
    dimension: str
    label: str
    attention_level: AttentionLevel
    score: float = Field(..., ge=0.0, le=1.0)
    bbox_norm: List[float]
    location_desc: str
    fact_description: str
    source_ref: Optional[str] = None
    evidence_note: str = ""
    geometric_inliers: Optional[int] = Field(None, ge=0)
    reviewed: bool = False

    if PYDANTIC_V2:
        @field_validator("bbox_norm")
        @classmethod
        def validate_bbox(cls, v: List[float]) -> List[float]:
            return _check_bbox_norm(v)
    else:
        @validator("bbox_norm")
        def validate_bbox(cls, v: List[float]) -> List[float]:
            return _check_bbox_norm(v)


class ImageDetectionPayload(CompatModel):
    task_id: str
    status: TaskStatus = "completed"
    error_message: Optional[str] = None
    file_name: str
    reference_scope: str
    model_signature: str = ""
    image_width: int = Field(..., gt=0)
    image_height: int = Field(..., gt=0)
    mode: CompareMode = "smart_check"
    axis_scores: Optional[ImageAxisScores] = None
    boxes: List[BoundingBoxItem] = Field(default_factory=list)
    summary_note: str = ""


class VideoSegmentItem(CompatModel):
    id: str
    track: Literal["visual_reuse", "ai_anomaly", "plot_av"] = "visual_reuse"
    dimension: str
    start_sec: float = Field(..., ge=0.0)
    end_sec: float = Field(..., gt=0.0)
    attention_level: AttentionLevel
    score: float = Field(..., ge=0.0, le=1.0)
    label: str
    transform_hint: Optional[str] = None
    fact_description: str
    source_ref: Optional[str] = None
    keyframe_time_sec: Optional[float] = Field(None, ge=0.0)
    evidence_note: str = ""
    reviewed: bool = False

    if PYDANTIC_V2:
        @model_validator(mode="after")
        def validate_time_range(self) -> "VideoSegmentItem":
            if self.end_sec <= self.start_sec:
                raise ValueError(f"视频片段必须满足 end_sec > start_sec，当前: [{self.start_sec}, {self.end_sec}]")
            return self
    else:
        @root_validator
        def validate_time_range(cls, values: Dict[str, Any]) -> Dict[str, Any]:
            s, e = values.get("start_sec"), values.get("end_sec")
            if s is not None and e is not None and e <= s:
                raise ValueError(f"视频片段必须满足 end_sec > start_sec，当前: [{s}, {e}]")
            return values


class VideoDetectionPayload(CompatModel):
    task_id: str
    status: TaskStatus = "completed"
    error_message: Optional[str] = None
    file_name: str
    reference_scope: str
    model_signature: str = ""
    duration_sec: float = Field(..., gt=0.0)
    mode: CompareMode = "smart_check"
    axis_scores: Optional[VideoAxisScores] = None
    segments: List[VideoSegmentItem] = Field(default_factory=list)
    summary_note: str = ""

    if PYDANTIC_V2:
        @model_validator(mode="after")
        def validate_segments_within_duration(self) -> "VideoDetectionPayload":
            for seg in self.segments:
                if seg.end_sec > self.duration_sec + 1e-6:
                    raise ValueError(f"片段 {seg.id} 结束时间 {seg.end_sec}s 超出视频总时长 {self.duration_sec}s")
            return self
    else:
        @root_validator
        def validate_segments_within_duration(cls, values: Dict[str, Any]) -> Dict[str, Any]:
            dur = values.get("duration_sec")
            segs = values.get("segments") or []
            if dur is not None:
                for seg in segs:
                    if seg.end_sec > dur + 1e-6:
                        raise ValueError(f"片段 {seg.id} 结束时间 {seg.end_sec}s 超出视频总时长 {dur}s")
            return values


class ReferenceMetaRecord(CompatModel):
    vector_id: int
    source_id: str
    media_type: MediaType
    title: str
    file_path: str
    sha256: str
    frame_timestamp_sec: Optional[float] = None
    phash: str
    author: str = "未填写"
    license_status: str = "待核验"
    model_name: str
    model_weights: str = "default"
    preprocess_version: str = "v1"


class IndexManifest(CompatModel):
    model_name: str
    model_weights: str
    preprocess_version: str
    embedding_dim: int
    engine: str
    total_vectors: int
    unique_sha256_count: int
