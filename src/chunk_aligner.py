from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Tuple
import os
import numpy as np
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


class EmbeddingClient:
    """OpenAI-compatible embedding client.

    Works with any endpoint that implements the /v1/embeddings spec:
    local servers (e.g. bge-m3) and OpenAI alike.
    """

    def __init__(
        self,
        api_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 120.0
    ):
        self.api_url = api_url or os.getenv("EMBEDDING_API_URL")
        self.model = model or os.getenv("EMBEDDING_MODEL_NAME")
        self.api_key = api_key or os.getenv("EMBEDDING_API_KEY")
        self.timeout = timeout

    def __call__(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        if not texts:
            return np.array([], dtype=np.float32)

        if not HTTPX_AVAILABLE:
            raise RuntimeError("httpx is required for embeddings. Install: pip install httpx")

        if not self.api_url or not self.api_url.startswith(("http://", "https://")):
            raise ValueError(
                f"EMBEDDING_API_URL is not set or invalid: {self.api_url!r}. "
                "Set the EMBEDDING_API_URL environment variable to a valid URL."
            )

        endpoint = f"{self.api_url}/embeddings"
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                endpoint,
                headers=headers,
                json={"model": self.model, "input": texts}
            )
            response.raise_for_status()
            result = response.json()

            data = sorted(result.get("data", []), key=lambda x: x.get("index", 0))
            embeddings = [item["embedding"] for item in data]
            return np.array(embeddings, dtype=np.float32)


@dataclass
class Chunk:
    """Minimal chunk representation."""
    id: str
    text: str


@dataclass
class AlignerConfig:
    """Configuration for the aligner."""
    sem_weight: float = 0.6
    lex_weight: float = 0.4
    anchor_threshold: float = 0.5
    unchanged_threshold: float = 0.95
    embed_fn: Optional[Callable[[List[str]], np.ndarray]] = None


class ChunkAligner:
    """
    Vectorized chunk alignment engine.

    Phase 1: Build similarity matrices using vectorized operations
    Phase 2: Greedy diagonal alignment with gap merging
    Phase 3: Output pairs in standard format with scores and labels
    """

    def __init__(self, config: Optional[AlignerConfig] = None):
        self.config = config or AlignerConfig()
        # Store matrices for score lookup
        self._sem_matrix: Optional[np.ndarray] = None
        self._lex_matrix: Optional[np.ndarray] = None
        self._final_matrix: Optional[np.ndarray] = None

    def align(self, chunks_a: List[Chunk], chunks_b: List[Chunk]) -> Dict[str, Any]:
        """
        Align two lists of chunks and return pairs in standard format.

        Returns:
            Dict with 'pairs' list and 'summary' statistics
        """
        if not chunks_a and not chunks_b:
            return {"pairs": [], "summary": self._empty_summary()}

        if not chunks_a:
            return self._handle_all_added(chunks_b)

        if not chunks_b:
            return self._handle_all_deleted(chunks_a)

        # Phase 1: Build similarity matrices
        self._build_matrices(chunks_a, chunks_b)

        # Phase 2: Greedy alignment
        alignment = self._greedy_align(len(chunks_a), len(chunks_b))

        # Phase 3: Build output pairs with gap merging
        pairs = self._build_output_pairs(chunks_a, chunks_b, alignment)

        return {
            "pairs": pairs,
            "summary": self._compute_summary(pairs)
        }

    def _build_matrices(self, chunks_a: List[Chunk], chunks_b: List[Chunk]) -> None:
        """Build semantic and lexical similarity matrices."""
        texts_a = [c.text for c in chunks_a]
        texts_b = [c.text for c in chunks_b]

        # Semantic matrix
        self._sem_matrix = self._compute_semantic_matrix(texts_a, texts_b)

        # Lexical matrix
        self._lex_matrix = self._compute_lexical_matrix(texts_a, texts_b)

        # Combined final matrix
        w1, w2 = self.config.sem_weight, self.config.lex_weight
        self._final_matrix = (w1 * self._sem_matrix) + (w2 * self._lex_matrix)

    def _compute_semantic_matrix(self, texts_a: List[str], texts_b: List[str]) -> np.ndarray:
        """Compute cosine similarity matrix using torch.matmul."""
        if self.config.embed_fn is None:
            return np.zeros((len(texts_a), len(texts_b)), dtype=np.float32)

        emb_a = self.config.embed_fn(texts_a)
        emb_b = self.config.embed_fn(texts_b)

        if TORCH_AVAILABLE:
            emb_a_t = torch.from_numpy(emb_a).float()
            emb_b_t = torch.from_numpy(emb_b).float()
            emb_a_t = emb_a_t / (emb_a_t.norm(dim=1, keepdim=True) + 1e-8)
            emb_b_t = emb_b_t / (emb_b_t.norm(dim=1, keepdim=True) + 1e-8)
            sim = torch.matmul(emb_a_t, emb_b_t.T)
            return sim.numpy().clip(0, 1)
        else:
            emb_a_norm = emb_a / (np.linalg.norm(emb_a, axis=1, keepdims=True) + 1e-8)
            emb_b_norm = emb_b / (np.linalg.norm(emb_b, axis=1, keepdims=True) + 1e-8)
            return np.clip(emb_a_norm @ emb_b_norm.T, 0, 1).astype(np.float32)

    def _compute_lexical_matrix(self, texts_a: List[str], texts_b: List[str]) -> np.ndarray:
        """Compute lexical similarity matrix using rapidfuzz.cdist."""
        if RAPIDFUZZ_AVAILABLE:
            from rapidfuzz.process import cdist
            return cdist(texts_a, texts_b, scorer=fuzz.ratio, dtype=np.float32, workers=-1) / 100.0
        else:
            n, m = len(texts_a), len(texts_b)
            matrix = np.zeros((n, m), dtype=np.float32)
            for i, ta in enumerate(texts_a):
                for j, tb in enumerate(texts_b):
                    if ta == tb:
                        matrix[i, j] = 1.0
                    elif ta and tb:
                        max_len = max(len(ta), len(tb))
                        dist = sum(1 for a, b in zip(ta, tb) if a != b) + abs(len(ta) - len(tb))
                        matrix[i, j] = max(0.0, 1.0 - dist / max_len)
            return matrix

    def _greedy_align(self, n: int, m: int) -> List[Tuple[int, int, float]]:
        """Greedy alignment with lookahead to find anchor pairs."""
        threshold = self.config.anchor_threshold
        anchors = []
        i, j = 0, 0
        lookahead = 5

        while i < n and j < m:
            best_score, best_i, best_j = -1.0, i, j

            for di in range(min(lookahead, n - i)):
                for dj in range(min(lookahead, m - j)):
                    score = self._final_matrix[i + di, j + dj]
                    if score > best_score and score >= threshold:
                        best_score, best_i, best_j = score, i + di, j + dj

            if best_score >= threshold:
                anchors.append((best_i, best_j, best_score))
                i, j = best_i + 1, best_j + 1
            else:
                i += 1
                j += 1

        return anchors

    def _build_output_pairs(
        self,
        chunks_a: List[Chunk],
        chunks_b: List[Chunk],
        anchors: List[Tuple[int, int, float]]
    ) -> List[Dict[str, Any]]:
        """Build output pairs with gap merging into anchors.

        - Leading orphans (before first anchor): pair positionally (first-to-first, etc.),
          excess orphans merge into the first anchor's aggregated field
        - Orphans between anchors: merge into the PRECEDING anchor's aggregated field
        - Trailing orphans (after last anchor): merge into the last anchor's aggregated field
        """
        pairs = []
        n, m = len(chunks_a), len(chunks_b)
        prev_i, prev_j = 0, 0
        excess_leading_a: List[int] = []
        excess_leading_b: List[int] = []

        for ai, aj, _ in anchors:
            # Orphans before this anchor
            orphans_a = list(range(prev_i, ai))
            orphans_b = list(range(prev_j, aj))

            if pairs and (orphans_a or orphans_b):
                # Merge orphans into previous pair's aggregated field
                self._merge_orphans_into_pair(pairs[-1], chunks_a, chunks_b, orphans_a, orphans_b)
            elif orphans_a or orphans_b:
                # Leading orphans before first anchor - pair them positionally
                paired_leading = self._pair_leading_orphans(
                    chunks_a, chunks_b, orphans_a, orphans_b
                )
                pairs.extend(paired_leading["pairs"])
                excess_leading_a = paired_leading["excess_a"]
                excess_leading_b = paired_leading["excess_b"]

            # Create the anchor pair
            pairs.append(self._make_pair(chunks_a[ai], chunks_b[aj], ai, aj))

            # If we just created the first anchor and have excess leading orphans, merge them
            if excess_leading_a or excess_leading_b:
                self._merge_orphans_into_pair(pairs[-1], chunks_a, chunks_b, excess_leading_a, excess_leading_b)
                excess_leading_a = []
                excess_leading_b = []

            prev_i, prev_j = ai + 1, aj + 1

        # Handle trailing orphans
        trailing_a = list(range(prev_i, n))
        trailing_b = list(range(prev_j, m))

        if pairs and (trailing_a or trailing_b):
            self._merge_orphans_into_pair(pairs[-1], chunks_a, chunks_b, trailing_a, trailing_b)
        else:
            # No anchors found at all - all chunks are orphans, create DELETED/ADDED
            for idx_a in trailing_a:
                pairs.append(self._make_deleted_pair(chunks_a[idx_a], idx_a))
            for idx_b in trailing_b:
                pairs.append(self._make_added_pair(chunks_b[idx_b], idx_b))

        return pairs

    def _pair_leading_orphans(
        self,
        chunks_a: List[Chunk],
        chunks_b: List[Chunk],
        orphans_a: List[int],
        orphans_b: List[int]
    ) -> Dict[str, Any]:
        """Pair leading orphans positionally (first-to-first, second-to-second, etc.)."""
        paired = []
        min_len = min(len(orphans_a), len(orphans_b))

        for k in range(min_len):
            idx_a = orphans_a[k]
            idx_b = orphans_b[k]
            paired.append(self._make_pair(chunks_a[idx_a], chunks_b[idx_b], idx_a, idx_b))

        excess_a = orphans_a[min_len:]
        excess_b = orphans_b[min_len:]

        return {
            "pairs": paired,
            "excess_a": excess_a,
            "excess_b": excess_b
        }

    def _make_pair(self, chunk_a: Chunk, chunk_b: Chunk, i: int, j: int) -> Dict[str, Any]:
        """Create a matched pair with scores and label."""
        sem = float(self._sem_matrix[i, j])
        lex = float(self._lex_matrix[i, j])
        final = float(self._final_matrix[i, j])

        label = self._determine_label(sem, lex, final)

        return {
            "a_id": chunk_a.id,
            "b_id": chunk_b.id,
            "text_a": chunk_a.text,
            "text_b": chunk_b.text,
            "label": label,
            "scores": {"sem": sem, "lex": lex, "final": final},
            "aggregated": None
        }

    def _make_added_pair(self, chunk_b: Chunk, idx: int) -> Dict[str, Any]:
        """Create an ADDED pair (only in B)."""
        return {
            "a_id": None,
            "b_id": chunk_b.id,
            "text_a": None,
            "text_b": chunk_b.text,
            "label": "ADDED",
            "scores": {"sem": 0.0, "lex": 0.0, "final": 0.0},
            "aggregated": None
        }

    def _make_deleted_pair(self, chunk_a: Chunk, idx: int) -> Dict[str, Any]:
        """Create a DELETED pair (only in A)."""
        return {
            "a_id": chunk_a.id,
            "b_id": None,
            "text_a": chunk_a.text,
            "text_b": None,
            "label": "DELETED",
            "scores": {"sem": 0.0, "lex": 0.0, "final": 0.0},
            "aggregated": None
        }

    def _merge_orphans_into_pair(
        self,
        pair: Dict[str, Any],
        chunks_a: List[Chunk],
        chunks_b: List[Chunk],
        orphan_indices_a: List[int],
        orphan_indices_b: List[int]
    ) -> None:
        """Merge orphan chunks into the preceding pair's aggregated field."""
        if not orphan_indices_a and not orphan_indices_b:
            return

        if pair["aggregated"] is None:
            pair["aggregated"] = {"source_chunks": [], "target_chunks": []}

        for idx in orphan_indices_a:
            pair["aggregated"]["source_chunks"].append({
                "id": chunks_a[idx].id,
                "text": chunks_a[idx].text
            })

        for idx in orphan_indices_b:
            pair["aggregated"]["target_chunks"].append({
                "id": chunks_b[idx].id,
                "text": chunks_b[idx].text
            })

    def _determine_label(self, sem: float, lex: float, final: float) -> str:
        """Determine label based on scores."""
        if final >= self.config.unchanged_threshold and lex >= 0.98:
            return "UNCHANGED"
        elif final >= self.config.anchor_threshold:
            return "AMENDED"
        else:
            return "AMENDED"

    def _handle_all_added(self, chunks_b: List[Chunk]) -> Dict[str, Any]:
        """Handle case where all chunks are additions."""
        pairs = [self._make_added_pair(c, i) for i, c in enumerate(chunks_b)]
        return {"pairs": pairs, "summary": self._compute_summary(pairs)}

    def _handle_all_deleted(self, chunks_a: List[Chunk]) -> Dict[str, Any]:
        """Handle case where all chunks are deletions."""
        pairs = [self._make_deleted_pair(c, i) for i, c in enumerate(chunks_a)]
        return {"pairs": pairs, "summary": self._compute_summary(pairs)}

    @staticmethod
    def _compute_summary(pairs: List[Dict[str, Any]]) -> Dict[str, int]:
        """Compute summary statistics from pairs."""
        summary = {"unchanged": 0, "amended": 0, "added": 0, "deleted": 0}
        for p in pairs:
            label = p.get("label", "").lower()
            if label in summary:
                summary[label] += 1
        return summary

    @staticmethod
    def _empty_summary() -> Dict[str, int]:
        return {"unchanged": 0, "amended": 0, "added": 0, "deleted": 0}


# =============================================================================
# Factory Functions
# =============================================================================

def create_aligner(
    embed_fn: Optional[Callable[[List[str]], np.ndarray]] = None,
    sem_weight: float = 0.6,
    lex_weight: float = 0.4,
    anchor_threshold: float = 0.5,
    unchanged_threshold: float = 0.95,
    use_default_embeddings: bool = True
) -> ChunkAligner:
    """Factory function to create a configured ChunkAligner."""
    if embed_fn is None and use_default_embeddings:
        embed_fn = EmbeddingClient()

    config = AlignerConfig(
        sem_weight=sem_weight,
        lex_weight=lex_weight,
        anchor_threshold=anchor_threshold,
        unchanged_threshold=unchanged_threshold,
        embed_fn=embed_fn
    )
    return ChunkAligner(config)


def create_lexical_only_aligner(anchor_threshold: float = 0.5) -> ChunkAligner:
    """Create an aligner that uses only lexical similarity (no embeddings)."""
    config = AlignerConfig(
        sem_weight=0.0,
        lex_weight=1.0,
        anchor_threshold=anchor_threshold,
        embed_fn=None
    )
    return ChunkAligner(config)
