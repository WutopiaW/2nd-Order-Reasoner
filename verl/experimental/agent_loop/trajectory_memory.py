ye# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Global trajectory memory for memory-augmented agent-loop rollouts."""

from __future__ import annotations

import hashlib
import re
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

import numpy as np
import ray


class TextEmbedder(Protocol):
    def encode(self, text: str) -> list[float]: ...


class HashingTextEmbedder:
    """Dependency-free text embedder intended as a deterministic fallback.

    Production runs should normally configure ``HuggingFaceTextEmbedder`` or
    another semantic embedding implementation. Character n-grams keep this
    fallback useful for both whitespace-delimited and CJK text.
    """

    def __init__(self, dimension: int = 2048, min_ngram: int = 2, max_ngram: int = 4):
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}.")
        if min_ngram <= 0 or max_ngram < min_ngram:
            raise ValueError(f"Invalid n-gram range: {min_ngram=} {max_ngram=}.")
        self.dimension = dimension
        self.min_ngram = min_ngram
        self.max_ngram = max_ngram

    def encode(self, text: str) -> list[float]:
        normalized = re.sub(r"\s+", " ", text.casefold()).strip()
        vector = np.zeros(self.dimension, dtype=np.float32)
        for ngram_size in range(self.min_ngram, self.max_ngram + 1):
            for start in range(max(0, len(normalized) - ngram_size + 1)):
                ngram = normalized[start : start + ngram_size].encode("utf-8")
                digest = hashlib.blake2b(ngram, digest_size=8).digest()
                hashed = int.from_bytes(digest, byteorder="little", signed=False)
                index = hashed % self.dimension
                sign = 1.0 if (hashed >> 63) == 0 else -1.0
                vector[index] += sign
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= norm
        return vector.tolist()


class HuggingFaceTextEmbedder:
    """Mean-pooled Hugging Face encoder for production semantic retrieval."""

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        trust_remote_code: bool = False,
        max_length: int = 512,
    ):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = torch.device(device)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        self.model.eval().to(self.device)

    def encode(self, text: str) -> list[float]:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            hidden = self.model(**inputs).last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        pooled = self.torch.nn.functional.normalize(pooled.float(), dim=-1)
        return pooled[0].cpu().tolist()


@dataclass
class TrajectoryRecord:
    request_id: str
    prompt: str
    trajectory: str
    summary: str
    embedding: list[float] | np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


class TrajectoryMemory:
    """Bounded request-id keyed memory with cosine-similarity retrieval."""

    def __init__(self, capacity: int = 100_000):
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}.")
        self.capacity = capacity
        self._records: OrderedDict[str, TrajectoryRecord] = OrderedDict()
        self._embedding_dimension: int | None = None

    def __len__(self) -> int:
        return len(self._records)

    def upsert(self, record: TrajectoryRecord) -> None:
        if not record.request_id:
            raise ValueError("request_id must not be empty.")
        embedding = self._normalize_embedding(record.embedding)
        normalized_record = TrajectoryRecord(
            request_id=record.request_id,
            prompt=record.prompt,
            trajectory=record.trajectory,
            summary=record.summary,
            embedding=embedding.copy(),
            metadata=dict(record.metadata),
        )
        self._records.pop(record.request_id, None)
        self._records[record.request_id] = normalized_record
        while len(self._records) > self.capacity:
            self._records.popitem(last=False)

    def search(
        self,
        query_embedding: list[float],
        *,
        exclude_request_id: str | None = None,
    ) -> tuple[TrajectoryRecord | None, float | None]:
        if not self._records:
            return None, None
        query = self._normalize_embedding(query_embedding)
        best_record = None
        best_score = -float("inf")
        for request_id, record in self._records.items():
            if request_id == exclude_request_id:
                continue
            score = float(np.dot(query, record.embedding))
            if score > best_score:
                best_record = record
                best_score = score
        return best_record, best_score if best_record is not None else None

    def get(self, request_id: str) -> TrajectoryRecord | None:
        return self._records.get(request_id)

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "embedding_dimension": self._embedding_dimension,
            "records": [asdict(record) for record in self._records.values()],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        capacity = int(state["capacity"])
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}.")
        self.capacity = capacity
        self._embedding_dimension = state.get("embedding_dimension")
        self._records.clear()
        for record_data in state.get("records", []):
            self.upsert(TrajectoryRecord(**record_data))

    def _normalize_embedding(self, embedding: list[float]) -> np.ndarray:
        array = np.asarray(embedding, dtype=np.float32)
        if array.ndim != 1 or array.size == 0:
            raise ValueError(f"embedding must be a non-empty vector, got shape {array.shape}.")
        if not np.isfinite(array).all():
            raise ValueError("embedding contains non-finite values.")
        if self._embedding_dimension is None:
            self._embedding_dimension = int(array.size)
        elif array.size != self._embedding_dimension:
            raise ValueError(
                f"embedding dimension {array.size} does not match memory dimension {self._embedding_dimension}."
            )
        norm = float(np.linalg.norm(array))
        if norm == 0:
            return array
        return array / norm


@ray.remote
class TrajectoryMemoryActor:
    """Ray-owned global memory shared by all agent-loop workers."""

    def __init__(self, capacity: int = 100_000, embedder: dict[str, Any] | None = None):
        import hydra
        from omegaconf import OmegaConf

        self.memory = TrajectoryMemory(capacity=capacity)
        if embedder is None:
            self.embedder: TextEmbedder = HashingTextEmbedder()
        else:
            embedder_config = OmegaConf.create(embedder)
            self.embedder = hydra.utils.instantiate(embedder_config)

    def search(self, prompt: str, exclude_request_id: str | None = None) -> dict[str, Any] | None:
        embedding = self.embedder.encode(prompt)
        record, score = self.memory.search(embedding, exclude_request_id=exclude_request_id)
        if record is None:
            return None
        return {"record": asdict(record), "score": score}

    def upsert(
        self,
        request_id: str,
        prompt: str,
        trajectory: str,
        summary: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.memory.upsert(
            TrajectoryRecord(
                request_id=request_id,
                prompt=prompt,
                trajectory=trajectory,
                summary=summary,
                embedding=self.embedder.encode(prompt),
                metadata=metadata or {},
            )
        )

    def get(self, request_id: str) -> dict[str, Any] | None:
        record = self.memory.get(request_id)
        return asdict(record) if record is not None else None

    def size(self) -> int:
        return len(self.memory)

    def state_dict(self) -> dict[str, Any]:
        return self.memory.state_dict()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.memory.load_state_dict(state)
