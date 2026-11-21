"""
首批视觉底库数据入库与向量索引管理模块 (Week 1)
- 支持将本地图片目录、视频目录（自动抽帧）批量提取特征并写入向量底库。
- 优先使用 FAISS (IndexFlatIP，对应 L2 归一化向量的余弦相似度)；
  若当前环境尚未安装 faiss-cpu/faiss-gpu，自动降级为 NumPy 持久化索引 (.npz)，
  接口完全一致，方便第 1 周独立验证并在第 2 周无缝接入成员C的 FAISS 服务。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .feature_extractor import ImageFeatureExtractor
from .schemas import ReferenceMetaRecord
from .video_frame_extractor import extract_video_keyframes

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


class VisualVectorStore:
    """FAISS / NumPy 双兼容视觉向量底库"""

    def __init__(self, index_dir: str | Path, dim: int = 512) -> None:
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.dim = dim
        self.meta_path = self.index_dir / "metadata.jsonl"
        self.faiss_path = self.index_dir / "visual.faiss"
        self.numpy_path = self.index_dir / "visual_vectors.npz"

        self.records: List[ReferenceMetaRecord] = []
        self._vectors: List[np.ndarray] = []
        self._faiss = self._try_import_faiss()
        self._faiss_index = None

        if self._faiss is not None:
            self._faiss_index = self._faiss.IndexFlatIP(self.dim)

    @staticmethod
    def _try_import_faiss():
        try:
            import faiss  # type: ignore

            return faiss
        except Exception:
            return None

    @property
    def engine_name(self) -> str:
        return "faiss-IndexFlatIP" if self._faiss is not None else "numpy-cosine-fallback"

    def add(self, vector: np.ndarray, meta: ReferenceMetaRecord) -> int:
        vec = vector.astype(np.float32).reshape(1, self.dim)
        vector_id = len(self.records)
        meta.vector_id = vector_id
        self.records.append(meta)
        self._vectors.append(vec[0])
        if self._faiss_index is not None:
            self._faiss_index.add(vec)
        return vector_id

    def save(self) -> Dict[str, str | int]:
        """将向量索引与元数据持久化到磁盘。"""
        with open(self.meta_path, "w", encoding="utf-8") as f:
            for rec in self.records:
                f.write(rec.model_dump_json() + "\n")

        mat = (
            np.vstack(self._vectors).astype(np.float32)
            if self._vectors
            else np.zeros((0, self.dim), dtype=np.float32)
        )
        np.savez_compressed(self.numpy_path, vectors=mat)

        if self._faiss is not None and self._faiss_index is not None:
            self._faiss.write_index(self._faiss_index, str(self.faiss_path))

        return {
            "engine": self.engine_name,
            "total_vectors": len(self.records),
            "meta_path": str(self.meta_path),
            "index_path": str(self.faiss_path if self._faiss else self.numpy_path),
        }

    def load(self) -> int:
        """从磁盘恢复索引与元数据。"""
        self.records.clear()
        self._vectors.clear()
        if self.meta_path.exists():
            with open(self.meta_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.records.append(ReferenceMetaRecord.model_validate_json(line))

        if self._faiss is not None and self.faiss_path.exists():
            self._faiss_index = self._faiss.read_index(str(self.faiss_path))

        if self.numpy_path.exists():
            data = np.load(self.numpy_path)
            mat = data["vectors"]
            self._vectors = [mat[i] for i in range(mat.shape[0])]

        return len(self.records)

    def search(
        self, query_vector: np.ndarray, top_k: int = 5
    ) -> List[Tuple[ReferenceMetaRecord, float]]:
        """检索 Top-K 相似底库条目（返回元数据与余弦相似度）。"""
        if not self.records:
            return []
        q = query_vector.astype(np.float32).reshape(1, self.dim)
        k = min(top_k, len(self.records))

        if self._faiss_index is not None and self._faiss_index.ntotal > 0:
            scores, indices = self._faiss_index.search(q, k)
            results = []
            for idx, score in zip(indices[0], scores[0]):
                if idx >= 0:
                    results.append((self.records[int(idx)], float(score)))
            return results

        mat = np.vstack(self._vectors).astype(np.float32)
        sims = (mat @ q.T).flatten()
        top_idx = np.argsort(-sims)[:k]
        return [(self.records[int(i)], float(sims[int(i)])) for i in top_idx]


def ingest_directory(
    media_dir: str | Path,
    index_dir: str | Path,
    keyframes_dir: Optional[str | Path] = None,
    backend: str = "auto",
    sample_fps: float = 1.0,
    min_scene_diff: float = 0.08,
) -> Dict[str, object]:
    """
    批量扫描目录中的图片和视频，提取特征并入库。
    """
    src_dir = Path(media_dir)
    if not src_dir.exists():
        raise FileNotFoundError(f"底库素材目录不存在: {src_dir}")

    kf_root = Path(keyframes_dir) if keyframes_dir else (Path(index_dir) / "keyframes")
    extractor = ImageFeatureExtractor(backend=backend)
    store = VisualVectorStore(index_dir=index_dir, dim=extractor.dim)

    image_count = 0
    video_count = 0
    frame_count = 0

    for file_path in sorted(src_dir.rglob("*")):
        if not file_path.is_file():
            continue
        ext = file_path.suffix.lower()

        if ext in IMAGE_EXTS:
            rec = extractor.extract_file(file_path)
            meta = ReferenceMetaRecord(
                vector_id=0,
                source_id=f"img_{file_path.stem}",
                media_type="image",
                title=file_path.name,
                file_path=str(file_path),
                sha256=rec.sha256,
                frame_timestamp_sec=None,
                phash=rec.phash,
                model_name=rec.model_name,
            )
            store.add(np.array(rec.embedding, dtype=np.float32), meta)
            image_count += 1

        elif ext in VIDEO_EXTS:
            vid_res = extract_video_keyframes(
                video_path=file_path,
                output_dir=kf_root / file_path.stem,
                sample_fps=sample_fps,
                min_scene_diff=min_scene_diff,
            )
            video_count += 1
            for kf in vid_res.sampled_frames:
                frec = extractor.extract_file(kf.frame_path, item_id=f"{vid_res.video_id}_f{kf.frame_index}")
                meta = ReferenceMetaRecord(
                    vector_id=0,
                    source_id=f"vid_{vid_res.video_id}",
                    media_type="video",
                    title=file_path.name,
                    file_path=str(file_path),
                    sha256=vid_res.sha256,
                    frame_timestamp_sec=kf.timestamp_sec,
                    phash=kf.phash,
                    model_name=frec.model_name,
                )
                store.add(np.array(frec.embedding, dtype=np.float32), meta)
                frame_count += 1

    save_info = store.save()
    summary = {
        "extractor_backend": extractor.backend_name,
        "images_ingested": image_count,
        "videos_ingested": video_count,
        "video_keyframes_ingested": frame_count,
        **save_info,
    }
    summary_path = Path(index_dir) / "ingest_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
