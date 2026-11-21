"""
成员B（视觉/视频模块）数据结构定义 (v0.2 - 对齐《知衡 v2.4》与功能二研判规范)
关键对齐点：
1. 维度口径对齐任务一 PPT 第 2 页：
   - 图片四轴：角色形象(40%) / 动作与站位(25%) / 色彩与画风(20%) / 画面文字与Logo(15%)
   - 视频五轴：画面直接搬运(30%) / 人物形象(20%) / 原声台词(20%) / 声纹特征(15%) / 故事剧情(15%)
2. 研判口径对齐功能二与架构报告：
   - 用 attention_level（重点关注 / 一般关注 / 辅助观察）替代法律定性词
   - 区分“检测事实（fact_description）”与“比对依据（evidence_note）”
   - 支持“智能自查（smart_check）”与“定向对比（targeted_compare）”双模式
"""
from __future__ import annotations

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field


AttentionLevel = Literal["focus", "moderate", "auxiliary"]  # 重点关注 / 一般关注 / 辅助观察
CompareMode = Literal["smart_check", "targeted_compare"]
MediaType = Literal["image", "video"]
TaskStatus = Literal["completed", "no_match", "partial", "failed"]


class ImageFeatureRecord(BaseModel):
    """单张图片特征提取结果（含多视角鲁棒特征支持）"""
    item_id: str = Field(..., description="素材唯一 ID")
    file_path: str = Field(..., description="原始文件路径")
    sha256: str = Field(..., description="文件 SHA-256 校验值")
    width: int = Field(..., description="图像宽度（像素）")
    height: int = Field(..., description="图像高度（像素）")
    phash: str = Field(..., description="64 位感知哈希（十六进制）")
    model_name: str = Field(..., description="特征提取后端名称")
    embedding_dim: int = Field(..., description="特征向量维度")
    embedding: List[float] = Field(..., description="原图 L2 归一化特征向量")
    hflip_embedding: Optional[List[float]] = Field(
        None, description="水平镜像视角特征向量（用于抗镜像搬运检索）"
    )


class VideoFrameItem(BaseModel):
    """视频单帧抽帧记录"""
    frame_index: int = Field(..., description="原始视频帧序号")
    timestamp_sec: float = Field(..., description="时间戳（秒）")
    frame_path: str = Field(..., description="保存的关键帧图片路径")
    phash: str = Field(..., description="该帧感知哈希")
    scene_diff: float = Field(0.0, description="与上一保留帧的直方图差异度 (0~1)")


class VideoExtractionResult(BaseModel):
    """视频抽帧汇总结果"""
    video_id: str
    file_path: str
    sha256: str
    fps: float
    total_frames: int
    duration_sec: float
    width: int
    height: int
    sampled_frames: List[VideoFrameItem]


class ImageAxisScores(BaseModel):
    """图片四轴相似度指标（供功能一雷达图与维度明细消费）"""
    character_design: float = Field(..., ge=0.0, le=100.0, description="角色形象 (权重 40%)")
    pose_composition: float = Field(..., ge=0.0, le=100.0, description="动作与站位 (权重 25%)")
    color_style: float = Field(..., ge=0.0, le=100.0, description="色彩与画风 (权重 20%)")
    text_logo: float = Field(..., ge=0.0, le=100.0, description="画面文字 / Logo (权重 15%)")
    aigc_artifact_score: float = Field(0.0, ge=0.0, le=100.0, description="生成痕迹检测（频域伪影/笔触不一致）")
    watermark_logo_detected: bool = Field(False, description="是否检出疑似水印/Logo")


class VideoAxisScores(BaseModel):
    """视频五轴相似度指标（供功能一雷达图与维度明细消费）"""
    direct_footage_reuse: float = Field(..., ge=0.0, le=100.0, description="画面直接搬运 (权重 30%)")
    character_visual: float = Field(..., ge=0.0, le=100.0, description="人物形象 (权重 20%)")
    dialogue_transcript: float = Field(0.0, ge=0.0, le=100.0, description="原声台词 (权重 20%，由成员A补充)")
    voiceprint_match: float = Field(0.0, ge=0.0, le=100.0, description="声纹特征 (权重 15%，由成员A补充)")
    story_plot: float = Field(0.0, ge=0.0, le=100.0, description="故事剧情 (权重 15%)")
    ai_face_swap_confidence: float = Field(0.0, ge=0.0, le=100.0, description="AI 换脸置信度（帧级）")
    av_sync_consistency: float = Field(100.0, ge=0.0, le=100.0, description="音画一致性评分")


class BoundingBoxItem(BaseModel):
    """图片边界框标记条目（供前端 ImageBBoxAnnotator 消费）"""
    id: str = Field(..., description="固定标记编号，如 '01'、'02'")
    dimension: str = Field(..., description="所属维度：角色形象 | 动作与站位 | 色彩与画风 | 画面文字/Logo | 生成痕迹")
    label: str = Field(..., description="分析标签，如 'IP 形象重合' / '品牌标志雷同' / '手部生成异常'")
    attention_level: AttentionLevel = Field(..., description="关注程度：focus(重点关注) | moderate(一般关注) | auxiliary(辅助观察)")
    score: float = Field(..., ge=0.0, le=1.0, description="局部特征相似度/置信度 (0~1)")
    bbox_norm: List[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="归一化边界框 [x1, y1, x2, y2]，范围 0~1",
    )
    location_desc: str = Field(..., description="区域文字位置，如 '画面左侧主体人物头部区域'")
    fact_description: str = Field(..., description="客观检测事实（先描述事实）")
    source_ref: Optional[str] = Field(None, description="命中的底库参考素材或定向对比文件")
    evidence_note: str = Field("", description="比对依据与人工复核提示（不直接作侵权裁决）")
    reviewed: bool = Field(False, description="人工是否已核对")


class ImageDetectionPayload(BaseModel):
    """图片检测返回 JSON 结构"""
    task_id: str
    status: TaskStatus = "completed"
    file_name: str
    reference_scope: str = Field(..., description="如 '知识产权底库 v2.4 (12,846 部作品)' 或 '定向文件：xxx.png'")
    image_width: int
    image_height: int
    mode: CompareMode = "smart_check"
    axis_scores: Optional[ImageAxisScores] = None
    boxes: List[BoundingBoxItem]
    summary_note: str = Field("", description="全局归纳（仅陈述事实与分布，不做法律侵权定性）")


class VideoSegmentItem(BaseModel):
    """视频多轨时间轴风险片段（供前端 VideoTimelineMarkers 消费）"""
    id: str = Field(..., description="固定标记编号，如 '01'、'02'")
    track: Literal["visual_reuse", "ai_anomaly", "plot_av"] = Field(
        "visual_reuse",
        description="时间轨分类：visual_reuse(画面与搬运) | ai_anomaly(AI换脸/异常线索) | plot_av(剧情/音画关联)",
    )
    dimension: str = Field(..., description="所属五轴维度，如 '画面直接搬运' / '人物形象'")
    start_sec: float = Field(..., ge=0.0, description="片段起始时间（秒）")
    end_sec: float = Field(..., ge=0.0, description="片段结束时间（秒）")
    attention_level: AttentionLevel = Field(..., description="关注程度：focus | moderate | auxiliary")
    score: float = Field(..., ge=0.0, le=1.0, description="片段最高相似度/置信度")
    label: str = Field(..., description="分析标签，如 '画面搬运' / 'AI 换脸' / '镜像裁剪复用'")
    transform_hint: Optional[str] = Field(None, description="检测到的编辑变换线索，如 '水平镜像 + 轻度裁剪'")
    fact_description: str = Field(..., description="客观检测事实")
    source_ref: Optional[str] = Field(None, description="命中的底库参考视频与对应时间段")
    keyframe_time_sec: Optional[float] = Field(None, description="代表关键帧时间戳")
    evidence_note: str = Field("", description="比对依据与复核提示")
    reviewed: bool = Field(False, description="人工是否已核对")


class VideoDetectionPayload(BaseModel):
    """视频检测返回 JSON 结构"""
    task_id: str
    status: TaskStatus = "completed"
    file_name: str
    reference_scope: str
    duration_sec: float
    mode: CompareMode = "smart_check"
    axis_scores: Optional[VideoAxisScores] = None
    segments: List[VideoSegmentItem]
    summary_note: str = ""


class ReferenceMetaRecord(BaseModel):
    """底库入库元数据记录"""
    vector_id: int = Field(..., description="向量索引中的顺序行号")
    source_id: str = Field(..., description="素材业务 ID")
    media_type: MediaType
    title: str
    file_path: str
    sha256: str
    frame_timestamp_sec: Optional[float] = Field(None, description="视频关键帧时间戳（秒）")
    phash: str
    author: str = "未填写"
    license_status: str = "待核验"
    model_name: str
