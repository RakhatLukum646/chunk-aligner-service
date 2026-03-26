import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Literal, Optional, Union

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from chunk_aligner import Chunk, create_aligner

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Chunk Aligner Service",
    description="Aligns text chunks from two documents using semantic and lexical similarity.",
    version="1.0.0",
)


# -----------------------------------------------------------------------------
# Pydantic Models
# -----------------------------------------------------------------------------

class PairScores(BaseModel):
    sem: float
    lex: float
    final: float


class AggregatedChunk(BaseModel):
    id: str
    text: str


class AggregatedOrphans(BaseModel):
    source_chunks: List[AggregatedChunk] = Field(default_factory=list)
    target_chunks: List[AggregatedChunk] = Field(default_factory=list)


class AlignedPairOutput(BaseModel):
    a_id: Optional[str] = None
    b_id: Optional[str] = None
    text_a: Optional[str] = None
    text_b: Optional[str] = None
    label: Literal["UNCHANGED", "AMENDED", "ADDED", "DELETED"]
    scores: PairScores
    aggregated: Optional[AggregatedOrphans] = None


class AlignChunksSummary(BaseModel):
    unchanged: int = 0
    amended: int = 0
    added: int = 0
    deleted: int = 0


class AlignChunksResponse(BaseModel):
    pairs: List[AlignedPairOutput]
    summary: AlignChunksSummary


class AlignChunksDifyRequest(BaseModel):
    """Request model for Dify workflow - accepts either JSON strings or arrays."""
    chunks_a_json: Union[str, List[str]] = Field(
        ...,
        description="Chunk array from document A - either as JSON string or native array"
    )
    chunks_b_json: Union[str, List[str]] = Field(
        ...,
        description="Chunk array from document B - either as JSON string or native array"
    )
    sem_weight: float = Field(0.6, ge=0.0, le=1.0, description="Weight for semantic similarity (0.0-1.0)")
    lex_weight: float = Field(0.4, ge=0.0, le=1.0, description="Weight for lexical similarity (0.0-1.0)")
    anchor_threshold: float = Field(0.5, ge=0.0, le=1.0, description="Minimum score to consider a pair an anchor")
    skip_embeddings: bool = Field(False, description="Skip embeddings and use lexical-only comparison")


# -----------------------------------------------------------------------------
# Endpoint
# -----------------------------------------------------------------------------

@app.post(
    "/align-chunks-dify",
    summary="Align chunks (Dify format - JSON strings)",
    description=(
        "Dify-compatible endpoint that accepts JSON-encoded strings for chunk arrays. "
        "Use this when your workflow cannot pass native arrays. "
        "Pass chunks as JSON strings: '[\"chunk1\", \"chunk2\", ...]'"
    ),
    response_model=AlignChunksResponse,
    tags=["document-comparison"],
)
async def align_chunks_dify_endpoint(request: AlignChunksDifyRequest):
    """
    Align chunks from Dify workflow (JSON string input).

    Example request:
    ```json
    {
        "chunks_a_json": "[\"# ANNEX I\", \"## Terms\", \"Paragraph 1...\"]",
        "chunks_b_json": "[\"# ANNEX I\", \"## Modified Terms\", \"New paragraph...\"]",
        "sem_weight": 0.6,
        "lex_weight": 0.4,
        "anchor_threshold": 0.5,
        "skip_embeddings": false
    }
    ```
    """
    try:
        # Parse chunks_a - handle both string (JSON) and native array
        if isinstance(request.chunks_a_json, str):
            try:
                chunks_a_list = json.loads(request.chunks_a_json)
                if not isinstance(chunks_a_list, list):
                    raise HTTPException(
                        status_code=400,
                        detail=f"chunks_a_json must be a JSON array, got {type(chunks_a_list).__name__}"
                    )
            except json.JSONDecodeError as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid JSON in chunks_a_json: {str(e)}"
                )
        else:
            chunks_a_list = request.chunks_a_json

        # Parse chunks_b - handle both string (JSON) and native array
        if isinstance(request.chunks_b_json, str):
            try:
                chunks_b_list = json.loads(request.chunks_b_json)
                if not isinstance(chunks_b_list, list):
                    raise HTTPException(
                        status_code=400,
                        detail=f"chunks_b_json must be a JSON array, got {type(chunks_b_list).__name__}"
                    )
            except json.JSONDecodeError as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid JSON in chunks_b_json: {str(e)}"
                )
        else:
            chunks_b_list = request.chunks_b_json

        # Validate arrays
        if not chunks_a_list:
            raise HTTPException(status_code=400, detail="chunks_a_json cannot be empty array")
        if not chunks_b_list:
            raise HTTPException(status_code=400, detail="chunks_b_json cannot be empty array")

        logger.info(
            f"[Align-Dify] Processing {len(chunks_a_list)} source chunks, "
            f"{len(chunks_b_list)} target chunks"
        )

        # Convert to Chunk objects
        chunks_a = [
            Chunk(id=f"S{i:03d}", text=str(text).strip())
            for i, text in enumerate(chunks_a_list, start=1)
        ]
        chunks_b = [
            Chunk(id=f"S{i:03d}", text=str(text).strip())
            for i, text in enumerate(chunks_b_list, start=1)
        ]

        # Create aligner
        aligner = create_aligner(
            sem_weight=request.sem_weight,
            lex_weight=request.lex_weight,
            anchor_threshold=request.anchor_threshold,
            use_default_embeddings=not request.skip_embeddings
        )

        # Run alignment
        result: Dict[str, Any] = await asyncio.to_thread(
            aligner.align, chunks_a, chunks_b
        )

        # Convert to response format
        response_pairs = []
        for p in result["pairs"]:
            aggregated = None
            if p.get("aggregated"):
                aggregated = AggregatedOrphans(
                    source_chunks=[
                        AggregatedChunk(id=c["id"], text=c["text"])
                        for c in p["aggregated"].get("source_chunks", [])
                    ],
                    target_chunks=[
                        AggregatedChunk(id=c["id"], text=c["text"])
                        for c in p["aggregated"].get("target_chunks", [])
                    ]
                )

            response_pairs.append(AlignedPairOutput(
                a_id=p.get("a_id"),
                b_id=p.get("b_id"),
                text_a=p.get("text_a"),
                text_b=p.get("text_b"),
                label=p["label"],
                scores=PairScores(
                    sem=p["scores"]["sem"],
                    lex=p["scores"]["lex"],
                    final=p["scores"]["final"]
                ),
                aggregated=aggregated
            ))

        summary = result.get("summary", {})

        logger.info(f"[Align-Dify] Complete: {len(response_pairs)} pairs, summary={summary}")

        return AlignChunksResponse(
            pairs=response_pairs,
            summary=AlignChunksSummary(
                unchanged=summary.get("unchanged", 0),
                amended=summary.get("amended", 0),
                added=summary.get("added", 0),
                deleted=summary.get("deleted", 0)
            )
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("[Align-Dify] Chunk alignment failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Chunk alignment failed: {str(exc)}"
        ) from exc


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
