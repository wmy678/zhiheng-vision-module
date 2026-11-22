"""
视觉底库增量入库与模型一致性校验模块 (v0.3)
改进点（对应评审意见 #2、#3）：
1. 明确区分 mode="incremental"（默认增量入库，保留旧库并按 sha256 去重）与 mode="rebuild"（全量重建）。
2. 保存 index_manifest.json，绑定 model_name / model_weights / preprocess_version / embedding_dim。
3. 每次 load()、增量 add()、search() 时强制校验模型签名一致以及向量数与元数据条数严格相等，杜绝跨模型混比或索引错位。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Literal, Optional, Set, Tuple

import numpy as np

from .feature_extractor import ImageFeatureExtractor, compute_sha256
from .schemas import IndexManifest, ReferenceMetaRecord
from .video_frame_extractor import extract_video_keyframes

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


class VisualVectorStore:
    """FAISS / NumPy 双兼容视觉向量底库（带模型签名与一致性校验）"""

    def __init__(
        self,
        index_dir: str | Path,
        dim: int = 512,
        model_name: Optional[str] = None,
        model_weights: Optional[str] = None,
        preprocess_version: Optional[str] = None,
    ) -> None:
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.dim = dim
        self.model_name = model_name
        self.model_weights = model_weights
        self.preprocess_version = preprocess_version

        self.meta_path = self.index_dir / "metadata.jsonl"
        self.manifest_path = self.index_dir / "index_manifest.json"
        self.faiss_path = self.index_dir / "visual.faiss"
        self.numpy_path = self.index_dir / "visual_vectors.npz"

        self.records: List[ReferenceMetaRecord] = []
        self._vectors: List[np.ndarray] = []
        self._existing_sha256: Set[str] = set()
        self.manifest: Optional[IndexManifest] = None

        self._faiss = self._try_import_faiss()
        self._faiss_index = self._faiss.IndexFlatIP(self.dim) if self._faiss is not None else None

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

    def has_sha256(self, sha256: str) -> bool:
        return sha256 in self._existing_sha256

    def verify_extractor_compatibility(self, extractor: ImageFeatureExtractor) -> None:
        """查询或增量入库前强制检查模型签名与现有索引一致。"""
        if self.manifest is None:
            return
        if (
            self.manifest.model_name != extractor.backend_name
            or self.manifest.model_weights != extractor.model_weights
            or self.manifest.preprocess_version != extractor.preprocess_version
            or self.manifest.embedding_dim != extractor.dim
        ):
            raise ValueError(
                "模型与底库索引签名不匹配，禁止跨模型直接比对！\n"
                f"  底库签名: {self.manifest.model_name}:{self.manifest.model_weights}:{self.manifest.preprocess_version}\n"
                f"  当前模型: {extractor.backend_name}:{extractor.model_weights}:{extractor.preprocess_version}\n"
                "请切换至一致的提取器，或使用 mode='rebuild' 重建索引。"
            )

    def add(self, vector: np.ndarray, meta: ReferenceMetaRecord) -> int:
        if vector.size != self.dim:
            raise ValueError(f"向量维度不匹配：期望 {self.dim}，实际 {vector.size}")
        vec = vector.astype(np.float32).reshape(1, self.dim)
        vector_id = len(self.records)
        meta.vector_id = vector_id
        self.records.append(meta)
        self._vectors.append(vec[0])
        self._existing_sha256.add(meta.sha256)
        if self._faiss_index is not None:
            self._faiss_index.add(vec)
        return vector_id

    def save(self) -> Dict[str, object]:
        if len(self.records) != len(self._vectors):
            raise RuntimeError(
                f"索引一致性校验失败：元数据 {len(self.records)} 条 != 向量 {len(self._vectors)} 条"
            )
        if self._faiss_index is not None and self._faiss_index.ntotal != len(self.records):
            raise RuntimeError(
                f"FAISS 索引一致性校验失败：ntotal={self._faiss_index.ntotal} != 元数据 {len(self.records)}"
            )

        with open(self.meta_path, "w", encoding="utf-8") as f:
            for rec in self.records:
                f.write(rec.to_json() + "\n")

        mat = (
            np.vstack(self._vectors).astype(np.float32)
            if self._vectors
            else np.zeros((0, self.dim), dtype=np.float32)
        )
        np.savez_compressed(self.numpy_path, vectors=mat)

        if self._faiss is not None and self._faiss_index is not None:
            self._faiss.write_index(self._faiss_index, str(self.faiss_path))

        m_name = self.model_name or (self.records[0].model_name if self.records else "unknown")
        m_weights = self.model_weights or (self.records[0].model_weights if self.records else "default")
        m_prep = self.preprocess_version or (self.records[0].preprocess_version if self.records else "v1")
        self.manifest = IndexManifest(
            model_name=m_name,
            model_weights=m_weights,
            preprocess_version=m_prep,
            embedding_dim=self.dim,
            engine=self.engine_name,
            total_vectors=len(self.records),
            unique_sha256_count=len(self._existing_sha256),
        )
        self.manifest_path.write_text(self.manifest.to_json(indent=2), encoding="utf-8")

        return {
            "engine": self.engine_name,
            "total_vectors": len(self.records),
            "unique_files": len(self._existing_sha256),
            "manifest_path": str(self.manifest_path),
            "meta_path": str(self.meta_path),
            "index_path": str(self.faiss_path if self._faiss else self.numpy_path),
        }

    def load(self, extractor: Optional[ImageFeatureExtractor] = None) -> int:
        self.records.clear()
        self._vectors.clear()
        self._existing_sha256.clear()

        if self.manifest_path.exists():
            self.manifest = IndexManifest.from_json(
                self.manifest_path.read_text(encoding="utf-8")
            )
            self.model_name = self.manifest.model_name
            self.model_weights = self.manifest.model_weights
            self.preprocess_version = self.manifest.preprocess_version
            self.dim = self.manifest.embedding_dim

        if extractor is not None:
            self.verify_extractor_compatibility(extractor)

        if self.meta_path.exists():
            with open(self.meta_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rec = ReferenceMetaRecord.from_json(line)
                        self.records.append(rec)
                        self._existing_sha256.add(rec.sha256)

        if self.numpy_path.exists():
            data = np.load(self.numpy_path)
            mat = data["vectors"]
            self._vectors = [mat[i] for i in range(mat.shape[0])]

        if self._faiss is not None:
            if self.faiss_path.exists():
                self._faiss_index = self._faiss.read_index(str(self.faiss_path))
            else:
                self._faiss_index = self._faiss.IndexFlatIP(self.dim)
                if self._vectors:
                    self._faiss_index.add(np.vstack(self._vectors).astype(np.float32))

        if len(self.records) != len(self._vectors):
            raise RuntimeError(
                f"加载索引失败：元数据 ({len(self.records)}) 与向量数量 ({len(self._vectors)}) 不一致！"
            )
        return len(self.records)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        extractor: Optional[ImageFeatureExtractor] = None,
    ) -> List[Tuple[ReferenceMetaRecord, float]]:
        if extractor is not None:
            self.verify_extractor_compatibility(extractor)
        if not self.records:
            return []
        q = query_vector.astype(np.float32).reshape(1, self.dim)
        k = min(top_k, len(self.records))

        if self._faiss_index is not None and self._faiss_index.ntotal > 0:
            scores, indices = self._faiss_index.search(q, k)
            return [
                (self.records[int(idx)], float(score))
                for idx, score in zip(indices[0], scores[0])
                if idx >= 0
            ]

        mat = np.vstack(self._vectors).astype(np.float32)
        sims = (mat @ q.T).flatten()
        top_idx = np.argsort(-sims)[:k]
        return [(self.records[int(i)], float(sims[int(i)])) for i in top_idx]


def ingest_directory(
    media_dir: str | Path,
    index_dir: str | Path,
    mode: Literal["incremental", "rebuild"] = "incremental",
    keyframes_dir: Optional[str | Path] = None,
    backend: str = "auto",
    sample_fps: float = 1.0,
    min_scene_diff: float = 0.06,
) -> Dict[str, object]:
    """
    批量扫描目录中的图片和视频并入库。
    - mode="incremental"（默认）：自动加载现有索引并按 SHA-256 跳过已入库文件，不覆盖旧数据。
    - mode="rebuild"：清空并重新构建全量索引。
    """
    src_dir = Path(media_dir)
    if not src_dir.exists():
        raise FileNotFoundError(f"底库素材目录不存在: {src_dir}")

    kf_root = Path(keyframes_dir) if keyframes_dir else (Path(index_dir) / "keyframes")
    extractor = ImageFeatureExtractor(backend=backend)
    store = VisualVectorStore(
        index_dir=index_dir,
        dim=extractor.dim,
        model_name=extractor.backend_name,
        model_weights=extractor.model_weights,
        preprocess_version=extractor.preprocess_version,
    )

    if mode == "incremental" and store.meta_path.exists():
        store.load(extractor=extractor)

    image_count = 0
    video_count = 0
    frame_count = 0
    skipped_duplicates = 0

    for file_path in sorted(src_dir.rglob("*")):
        if not file_path.is_file():
            continue
        ext = file_path.suffix.lower()
        if ext not in IMAGE_EXTS and ext not in VIDEO_EXTS:
            continue

        file_hash = compute_sha256(file_path)
        if mode == "incremental" and store.has_sha256(file_hash):
            skipped_duplicates += 1
            continue

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
                model_weights=rec.model_weights,
                preprocess_version=rec.preprocess_version,
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
                    model_weights=frec.model_weights,
                    preprocess_version=frec.preprocess_version,
                )
                store.add(np.array(frec.embedding, dtype=np.float32), meta)
                frame_count += 1

    save_info = store.save()
    summary = {
        "mode": mode,
        "extractor_signature": extractor.signature,
        "fallback_history": extractor.fallback_history,
        "new_images_ingested": image_count,
        "new_videos_ingested": video_count,
        "new_video_keyframes_ingested": frame_count,
        "skipped_duplicates": skipped_duplicates,
        **save_info,
    }
    summary_path = Path(index_dir) / "ingest_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
