# 成员B（视觉/视频算法 + 视觉前端组件）v0.3 升级说明

本版本已逐项落实代码审查提出的 7 项核心改进：

1. **真实图片比对闭环（`backend/vision/matcher.py`）**：新增 `VisualImageDetector`，打通“待检图 → 多视角向量 Top-K 召回 → ORB 局部关键点 + Lowe 比率测试 + RANSAC 单应性几何验证 → 计算真实归一化边界框 `bbox_norm` → 输出结构化 JSON”，未接入维度明确返回 `null` + `not_implemented`。
2. **模型自动降级链与索引签名绑定（`feature_extractor.py` + `faiss_ingest.py`）**：`auto` 模式按 `clip -> resnet -> fallback_cv` 顺序尝试并记录 `fallback_history` 日志；入库时写入 `index_manifest.json`，加载与检索前强制校验模型、权重与预处理版本一致。
3. **增量入库与去重校验（`faiss_ingest.py`）**：区分 `mode="incremental"`（默认，保留旧库并按 SHA-256 去重）与 `mode="rebuild"`，保存和加载时同步校验向量数与元数据条数一致。
4. **结构感知与全片均匀抽帧（`video_frame_extractor.py`）**：融合灰度空间 MAD + Canny 边缘差异 + HSV 直方图差异，可捕获纯黑白运动物体变化；采用“全片自适应基础采样 + 场景变化补帧”避免长视频后半段截断，并返回实际覆盖区间与覆盖率。
5. **拒绝匹配机制（`matcher.py`）**：Top-K 召回后必须通过相似度阈值与 RANSAC 几何内点验证，无关图片返回 `status="no_match"` 和空 `boxes`，不盲信检索第一名。
6. **Pydantic 严格校验与“未检测”口径（`schemas.py`）**：强制校验 `bbox_norm` 在 `[0, 1]` 且 `x2 > x1, y2 > y1`，校验视频 `end_sec > start_sec` 且不超过 `duration_sec`；未实现指标默认为 `None`。
7. **前端多状态区分与真实 `<video>` 联动（`frontend/components/`）**：`ImageBBoxAnnotator.jsx` 与 `VideoTimelineMarkers.jsx` 支持区分 `loading / failed / no_match / partial / completed` 五种状态；时间轴组件支持绑定真实 `<video>` 的 `timeupdate` 事件与点击跳转 `currentTime`。

## 运行全套 6 项回归自测

```bash
pip install -r backend/requirements.txt
python backend/demo_week1.py
```
