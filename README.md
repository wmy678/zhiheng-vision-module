# 成员B（视觉/视频算法 + 视觉前端组件）第1周交付说明（v0.2 优化版）

本交付包已结合团队《知衡 · AIGC内容版权比对与合规预检系统 v2.4》设计稿、功能二研判工作台规范及架构建议报告完成全面对齐：

1. **指标与维度口径对齐（任务一 PPT 第 2 页）**：
   - `ImageAxisScores`：角色形象(40%) / 动作与站位(25%) / 色彩与画风(20%) / 画面文字与Logo(15%) + 生成痕迹检测 + 水印/Logo 检出。
   - `VideoAxisScores`：画面直接搬运(30%) / 人物形象(20%) / 原声台词(20%) / 声纹特征(15%) / 故事剧情(15%) + AI换脸置信度 + 音画一致性。
2. **研判边界与字段规范对齐（功能二 + 架构报告）**：
   - 采用固定编号（`01`/`02`/`03`）与关注程度分级（`focus` 重点关注 / `moderate` 一般关注 / `auxiliary` 辅助观察），**不把相似度直接等同于法律侵权概率**。
   - 区分 `fact_description`（客观检测事实）与 `evidence_note`（比对依据），支持 `smart_check`（智能自查）与 `targeted_compare`（定向对比）双模式，以及“当前参考范围内未发现匹配”空状态。
3. **编辑变换鲁棒性基线（架构报告第六章）**：
   - `ImageFeatureExtractor` 新增 `extract_multi_view_from_bgr`（原图 + 水平镜像 + 中心裁剪），并在 `demo_week1.py` 中通过了**水平镜像 + 裁边查询图 Top-1 召回原图**的自测验证。
4. **前端组件对齐（《知衡 v2.4》浅色工作台）**：
   - `ImageBBoxAnnotator.jsx`：支持归一化坐标标注框、四轴维度标签、关注程度筛选、缩放与空状态提示。
   - `VideoTimelineMarkers.jsx`：升级为**三轨时间轴**（画面与搬运轨、AI异常线索轨、剧情/音画关联轨），支持点击跳转、播放头同步与双向联动。
   - `frontend/demo/week1_preview.html`：可直接用浏览器打开截图/录屏提交群内周报。

## 运行自测

```bash
pip install -r backend/requirements.txt
python backend/demo_week1.py
```
