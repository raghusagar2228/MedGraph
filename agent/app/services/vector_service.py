import os
import json
import logging
from typing import List, Dict, Any, Optional
import numpy as np

from app.config import settings
from app.schemas.evidence import ClinicalNote

logger = logging.getLogger(__name__)


class VectorService:
    """Native vector semantic search engine using local dense embeddings

    (FastEmbed / BAAI/bge-small-en-v1.5) and cosine similarity ranking.
    """

    def __init__(self, index_path: Optional[str] = None, model_name: Optional[str] = None):
        self.index_path = index_path or settings.vector_index_path
        self.model_name = model_name or settings.vector_model_name
        self._embedding_model = None
        self._documents: Dict[str, Dict[str, Any]] = {}
        self._load_index()

    def _get_embedding_model(self):
        """Lazy loader for the ONNX text embedding model."""
        if self._embedding_model is None:
            # Disable symlinks warning on Windows
            os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
            from fastembed import TextEmbedding
            logger.info(f"Initializing FastEmbed model: {self.model_name}")
            self._embedding_model = TextEmbedding(model_name=self.model_name)
        return self._embedding_model

    def _load_index(self) -> None:
        """Load persistent vector index from JSON file if it exists."""
        if os.path.exists(self.index_path):
            try:
                with open(self.index_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._documents = {doc["record_id"]: doc for doc in data}
                logger.info(f"Loaded {len(self._documents)} vector records from {self.index_path}")
            except Exception as e:
                logger.warning(f"Could not load vector index from {self.index_path}: {e}")
                self._documents = {}
        else:
            self._documents = {}

    def _save_index(self) -> None:
        """Persist vector index to disk."""
        try:
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            with open(self.index_path, "w", encoding="utf-8") as f:
                json.dump(list(self._documents.values()), f, indent=2)
            logger.debug(f"Saved {len(self._documents)} records to {self.index_path}")
        except Exception as e:
            logger.error(f"Failed to persist vector index to {self.index_path}: {e}")

    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate a dense normalized 384-dimensional vector embedding for text."""
        model = self._get_embedding_model()
        vectors = list(model.embed([text]))
        vec = vectors[0]
        # Normalize to unit vector for fast dot-product cosine similarity
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def index_notes(self, notes: List[ClinicalNote]) -> int:
        """Embed and store a collection of clinical notes into the persistent index."""
        if not notes:
            return 0

        model = self._get_embedding_model()
        texts = [n.content for n in notes]
        embeddings = list(model.embed(texts))

        indexed_count = 0
        for note, emb in zip(notes, embeddings):
            norm = np.linalg.norm(emb)
            norm_emb = (emb / norm if norm > 0 else emb).tolist()

            doc_entry = {
                "record_id": note.record_id,
                "patient_id": note.patient_id,
                "timestamp": note.timestamp,
                "author": note.author,
                "note_type": note.note_type,
                "content": note.content,
                "embedding": norm_emb,
                "source": "vector_native"
            }
            self._documents[note.record_id] = doc_entry
            indexed_count += 1

        self._save_index()
        logger.info(f"Successfully indexed {indexed_count} clinical notes in vector store.")
        return indexed_count

    def search(
        self,
        patient_id: str,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0
    ) -> List[ClinicalNote]:
        """Perform semantic cosine-similarity search against indexed clinical notes,

        strictly enforcing patient_id boundary.
        """
        # Ensure default synthetic notes are indexed if empty
        if not self._documents:
            self.ensure_seeded()

        # Generate normalized query vector
        query_vec = self.generate_embedding(query)

        scored_candidates = []
        for doc in self._documents.values():
            # Security: Strict patient isolation pre-filter
            if doc.get("patient_id") != patient_id:
                continue

            doc_vec = np.array(doc["embedding"], dtype=np.float32)
            # Dot product of normalized vectors equals cosine similarity
            sim = float(np.dot(query_vec, doc_vec))

            if sim >= min_score:
                scored_candidates.append((sim, doc))

        # Rank descending by cosine similarity
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        results = []
        for sim, doc in scored_candidates[:top_k]:
            results.append(
                ClinicalNote(
                    record_id=doc["record_id"],
                    patient_id=doc["patient_id"],
                    timestamp=doc.get("timestamp"),
                    author=doc.get("author"),
                    note_type=doc.get("note_type"),
                    content=doc["content"],
                    relevance_score=round(sim, 4),
                    source="vector_native"
                )
            )

        logger.info(f"Vector search for patient={patient_id}, query='{query}' returned {len(results)} matches.")
        return results

    def ensure_seeded(self) -> None:
        """Seed the vector index with default clinical notes from mock data if empty."""
        from app.services.mock_data import (
            P001_CLINICAL_NOTES,
            P002_CLINICAL_NOTES,
            P003_CLINICAL_NOTES,
            P004_CLINICAL_NOTES
        )
        all_notes = (
            P001_CLINICAL_NOTES +
            P002_CLINICAL_NOTES +
            P003_CLINICAL_NOTES +
            P004_CLINICAL_NOTES
        )
        logger.info(f"Seeding vector index with {len(all_notes)} default clinical notes.")
        self.index_notes(all_notes)

    def count(self) -> int:
        return len(self._documents)

    def clear(self) -> None:
        self._documents = {}
        if os.path.exists(self.index_path):
            try:
                os.remove(self.index_path)
            except OSError:
                pass


# Global singleton instance
vector_service = VectorService()
