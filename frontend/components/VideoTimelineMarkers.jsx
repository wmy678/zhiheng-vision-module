import React, { useState } from "react";
import "./visual-components.css";

const ATTENTION_META = {
  focus: { text: "重点关注", color: "#DC2626" },
  moderate: { text: "一般关注", color: "#D97706" },
  auxiliary: { text: "辅助观察", color: "#2563EB" },
};

const TRACKS = [
  { key: "visual_reuse", label: "画面与搬运轨" },
  { key: "ai_anomaly", label: "AI 异常线索轨" },
  { key: "plot_av", label: "剧情/音画关联轨" },
];

function formatTime(sec) {
  const s = Math.max(0, Math.floor(sec));
  const m = Math.floor(s / 60);
  const rem = s % 60;
  return `${String(m).padStart(2, "0")}:${String(rem).padStart(2, "0")}`;
}

/**
 * 视频多轨时间轴色块组件（成员B Week 1 优化版 - 对齐《知衡 v2.4》与功能二规范）
 */
export default function VideoTimelineMarkers({
  durationSec = 60,
  currentTime = 0,
  segments = [],
  selectedId = null,
  onSelect,
  onSeek,
}) {
  const [filterLevel, setFilterLevel] = useState("all");
  const [hoveredSeg, setHoveredSeg] = useState(null);

  const safeDuration = Math.max(durationSec, 1);
  const playheadPct = Math.min(100, Math.max(0, (currentTime / safeDuration) * 100));

  const visibleSegments = segments.filter(
    (s) => filterLevel === "all" || s.attention_level === filterLevel
  );

  const handleTrackClick = (e) => {
    if (!onSeek) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const ratio = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
    onSeek(+(ratio * safeDuration).toFixed(2));
  };

  return (
    <div className="zh-card">
      <div className="zh-toolbar">
        <div className="zh-toolbar-left">
          <span className="zh-title">多轨风险时间轴</span>
          <span className="zh-pill">
            {formatTime(currentTime)} / {formatTime(safeDuration)}
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
        </div>
      </div>

      {segments.length === 0 ? (
        <div className="zh-empty-state">
          当前参考范围内未发现匹配片段（不代表不存在其他未入库来源风险）
        </div>
      ) : (
        <div className="zh-timeline-wrap">
          {hoveredSeg && (
            <div className="zh-timeline-popover">
              <strong>
                {hoveredSeg.id} · {hoveredSeg.label}
              </strong>
              <span>
                {formatTime(hoveredSeg.start_sec)} – {formatTime(hoveredSeg.end_sec)} · 匹配度{" "}
                {Math.round(hoveredSeg.score * 100)}%
              </span>
            </div>
          )}

          <div className="zh-tracks-box">
            {TRACKS.map((track) => {
              const trackSegs = visibleSegments.filter((s) => s.track === track.key);
              return (
                <div key={track.key} className="zh-track-row">
                  <div className="zh-track-name">{track.label}</div>
                  <div className="zh-timeline-track" onClick={handleTrackClick}>
                    {trackSegs.map((seg) => {
                      const leftPct = (seg.start_sec / safeDuration) * 100;
                      const widthPct = Math.max(
                        2,
                        ((seg.end_sec - seg.start_sec) / safeDuration) * 100
                      );
                      const meta = ATTENTION_META[seg.attention_level] || ATTENTION_META.auxiliary;
                      const isSelected = selectedId === seg.id;

                      return (
                        <button
                          key={seg.id}
                          type="button"
                          className={`zh-timeline-seg ${isSelected ? "zh-timeline-seg-selected" : ""}`}
                          style={{
                            left: `${leftPct}%`,
                            width: `${widthPct}%`,
                            backgroundColor: meta.color,
                          }}
                          onClick={(e) => {
                            e.stopPropagation();
                            onSelect && onSelect(seg);
                            onSeek && onSeek(seg.start_sec);
                          }}
                          onMouseEnter={() => setHoveredSeg(seg)}
                          onMouseLeave={() => setHoveredSeg(null)}
                        >
                          {seg.id}
                        </button>
                      );
                    })}
                    <div className="zh-playhead" style={{ left: `${playheadPct}%` }} />
                  </div>
                </div>
              );
            })}
          </div>

          <div className="zh-timeline-ticks">
            {[0, 0.25, 0.5, 0.75, 1].map((r) => (
              <span key={r}>{formatTime(safeDuration * r)}</span>
            ))}
          </div>
        </div>
      )}

      <div className="zh-seg-list">
        {visibleSegments.map((seg) => {
          const meta = ATTENTION_META[seg.attention_level] || ATTENTION_META.auxiliary;
          const isSelected = selectedId === seg.id;
          return (
            <button
              key={seg.id}
              type="button"
              className={`zh-seg-chip ${isSelected ? "zh-seg-chip-selected" : ""}`}
              onClick={() => {
                onSelect && onSelect(seg);
                onSeek && onSeek(seg.start_sec);
              }}
            >
              <span className="zh-dot" style={{ backgroundColor: meta.color }} />
              <span>
                {seg.id} · {formatTime(seg.start_sec)}–{formatTime(seg.end_sec)}
              </span>
              <span className="zh-chip-label">{seg.label}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
