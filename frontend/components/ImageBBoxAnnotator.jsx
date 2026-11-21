import React, { useState } from "react";
import "./visual-components.css";

const ATTENTION_META = {
  focus: { text: "重点关注", color: "#DC2626", bg: "rgba(220, 38, 38, 0.16)" },
  moderate: { text: "一般关注", color: "#D97706", bg: "rgba(217, 119, 6, 0.15)" },
  auxiliary: { text: "辅助观察", color: "#2563EB", bg: "rgba(37, 99, 235, 0.12)" },
};

/**
 * 图片边界框标记组件（成员B Week 1 优化版 - 对齐《知衡 v2.4》与功能二规范）
 *
 * Props:
 * - imageUrl: string                  待检图片地址
 * - fileName?: string                 文件名
 * - referenceScope?: string           参考范围说明（智能自查底库版本 / 定向对比文件名）
 * - boxes: Array<{
 *     id: string,                     // 固定编号如 "01"
 *     dimension: string,              // 角色形象 | 动作与站位 | 色彩与画风 | 画面文字/Logo | 生成痕迹
 *     label: string,
 *     attention_level: "focus" | "moderate" | "auxiliary",
 *     score: number,
 *     bbox_norm: [number, number, number, number],
 *     location_desc?: string,
 *     fact_description?: string,
 *     source_ref?: string,
 *     evidence_note?: string
 *   }>
 * - selectedId?: string | null
 * - onSelect?: (box) => void
 */
export default function ImageBBoxAnnotator({
  imageUrl,
  fileName = "待检图片",
  referenceScope = "知识产权底库 v2.4 · 智能自查模式",
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

  return (
    <div className="zh-card">
      <div className="zh-toolbar">
        <div className="zh-toolbar-left">
          <span className="zh-title">{fileName}</span>
          <span className="zh-pill">{referenceScope}</span>
          <span className="zh-pill">定位 {boxes.length} 处</span>
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
          <button
            type="button"
            className="zh-btn"
            onClick={() => setShowBoxes((v) => !v)}
          >
            {showBoxes ? "隐藏标注框" : "显示标注框"}
          </button>
          <button
            type="button"
            className="zh-btn"
            onClick={() => setZoom((z) => Math.max(0.75, +(z - 0.25).toFixed(2)))}
          >
            −
          </button>
          <span className="zh-zoom-label">{Math.round(zoom * 100)}%</span>
          <button
            type="button"
            className="zh-btn"
            onClick={() => setZoom((z) => Math.min(2, +(z + 0.25).toFixed(2)))}
          >
            +
          </button>
        </div>
      </div>

      {boxes.length === 0 ? (
        <div className="zh-empty-state">
          当前参考范围内未发现匹配区域（不代表不存在其他未入库来源风险）
        </div>
      ) : (
        <div className="zh-canvas-viewport">
          <div
            className="zh-image-stage"
            style={{ transform: `scale(${zoom})`, transformOrigin: "top left" }}
          >
            <img src={imageUrl} alt={fileName} className="zh-base-image" />

            {showBoxes &&
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
                    <span
                      className="zh-bbox-badge"
                      style={{ backgroundColor: meta.color }}
                    >
                      {box.id} · {box.dimension} · {Math.round(box.score * 100)}%
                    </span>

                    {(isHovered || isSelected) && (
                      <div className="zh-bbox-tooltip">
                        <div className="zh-tooltip-title">
                          [{meta.text}] {box.label}
                        </div>
                        {box.source_ref && (
                          <div className="zh-tooltip-sub">参考：{box.source_ref}</div>
                        )}
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
