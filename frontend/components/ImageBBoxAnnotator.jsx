import React, { useState } from "react";
import "./visual-components.css";

const ATTENTION_META = {
  focus: { text: "重点关注", color: "#DC2626", bg: "rgba(220, 38, 38, 0.16)" },
  moderate: { text: "一般关注", color: "#D97706", bg: "rgba(217, 119, 6, 0.15)" },
  auxiliary: { text: "辅助观察", color: "#2563EB", bg: "rgba(37, 99, 235, 0.12)" },
};

/**
 * 图片边界框标记组件（成员B v0.3 - 支持完整任务状态区分）
 *
 * Props:
 * - status: "loading" | "completed" | "no_match" | "partial" | "failed"
 * - errorMessage?: string
 * - imageUrl: string
 * - fileName?: string
 * - referenceScope?: string
 * - boxes: Array<BoundingBoxItem>
 * - selectedId?: string | null
 * - onSelect?: (box) => void
 */
export default function ImageBBoxAnnotator({
  status = "completed",
  errorMessage = "",
  imageUrl,
  fileName = "待检图片",
  referenceScope = "本地视觉底库",
  boxes = [],
  selectedId = null,
  onSelect,
}) {
  const [showBoxes, setShowBoxes] = useState(true);
  const [filterLevel, setFilterLevel] = useState("all");
  const [zoom, setZoom] = useState(1);
  const [hoveredId, setHoveredId] = useState(null);

  const visibleBoxes = boxes.filter(
    (b) => filterLevel === "all" || b.attention_level === filterLevel
  );

  const renderStateBanner = () => {
    if (status === "loading") {
      return <div className="zh-state-banner zh-state-loading">正在执行图像特征提取与局部几何验证，请稍候…</div>;
    }
    if (status === "failed") {
      return (
        <div className="zh-state-banner zh-state-failed">
          检测失败：{errorMessage || "图片解析或检索异常（未完成检测，不代表无风险）"}
        </div>
      );
    }
    if (status === "partial") {
      return (
        <div className="zh-state-banner zh-state-partial">
          部分维度已完成检测，其余维度标注为“未检测”，请结合已定位结果人工复核。
        </div>
      );
    }
    if (status === "no_match" || (status === "completed" && boxes.length === 0)) {
      return (
        <div className="zh-state-banner zh-state-nomatch">
          当前参考范围内未发现通过几何验证的匹配区域（未检出不代表不存在库外来源风险）。
        </div>
      );
    }
    return null;
  };

  return (
    <div className="zh-card">
      <div className="zh-toolbar">
        <div className="zh-toolbar-left">
          <span className="zh-title">{fileName}</span>
          <span className="zh-pill">{referenceScope}</span>
          <span className="zh-pill">
            {status === "completed" || status === "partial" ? `定位 ${boxes.length} 处` : `状态：${status}`}
          </span>
        </div>
        <div className="zh-toolbar-right">
          <div className="zh-filter-group">
            {[
              ["all", "全部"],
              ["focus", "重点关注"],
              ["moderate", "一般关注"],
              ["auxiliary", "辅助观察"],
            ].map(([key, label]) => (
              <button
                key={key}
                type="button"
                className={`zh-btn ${filterLevel === key ? "zh-btn-active" : ""}`}
                onClick={() => setFilterLevel(key)}
              >
                {label}
              </button>
            ))}
          </div>
          <button type="button" className="zh-btn" onClick={() => setShowBoxes((v) => !v)}>
            {showBoxes ? "隐藏标注框" : "显示标注框"}
          </button>
          <button type="button" className="zh-btn" onClick={() => setZoom((z) => Math.max(0.75, +(z - 0.25).toFixed(2)))}>
            −
          </button>
          <span className="zh-zoom-label">{Math.round(zoom * 100)}%</span>
          <button type="button" className="zh-btn" onClick={() => setZoom((z) => Math.min(2, +(z + 0.25).toFixed(2)))}>
            +
          </button>
        </div>
      </div>

      {renderStateBanner()}

      {status !== "failed" && imageUrl && (
        <div className="zh-canvas-viewport">
          <div
            className="zh-image-stage"
            style={{ transform: `scale(${zoom})`, transformOrigin: "top left" }}
          >
            <img src={imageUrl} alt={fileName} className="zh-base-image" />

            {showBoxes &&
              status !== "loading" &&
              visibleBoxes.map((box) => {
                const [x1, y1, x2, y2] = box.bbox_norm;
                const meta = ATTENTION_META[box.attention_level] || ATTENTION_META.auxiliary;
                const isSelected = selectedId === box.id;
                const isHovered = hoveredId === box.id;

                return (
                  <div
                    key={box.id}
                    role="button"
                    tabIndex={0}
                    className={`zh-bbox ${isSelected ? "zh-bbox-selected" : ""}`}
                    style={{
                      left: `${x1 * 100}%`,
                      top: `${y1 * 100}%`,
                      width: `${(x2 - x1) * 100}%`,
                      height: `${(y2 - y1) * 100}%`,
                      borderColor: meta.color,
                      backgroundColor: isSelected || isHovered ? meta.bg : "transparent",
                    }}
                    onClick={() => onSelect && onSelect(box)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" || e.key === " ") onSelect && onSelect(box);
                    }}
                    onMouseEnter={() => setHoveredId(box.id)}
                    onMouseLeave={() => setHoveredId(null)}
                  >
                    <span className="zh-bbox-badge" style={{ backgroundColor: meta.color }}>
                      {box.id} · {box.dimension} · {Math.round(box.score * 100)}%
                    </span>
                    {(isHovered || isSelected) && (
                      <div className="zh-bbox-tooltip">
                        <div className="zh-tooltip-title">
                          [{meta.text}] {box.label}
                        </div>
                        {box.source_ref && <div className="zh-tooltip-sub">参考：{box.source_ref}</div>}
                      </div>
                    )}
                  </div>
                );
              })}
          </div>
        </div>
      )}
    </div>
  );
}
