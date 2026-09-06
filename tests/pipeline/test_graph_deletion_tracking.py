"""Manual graph deletion must remove persisted provenance before a new generation."""

import asyncio
import json
import sys
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from lightrag import LightRAG
from lightrag.base import DeletionResult
from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.utils import EmbeddingFunc, Tokenizer, make_relation_chunk_key

pytestmark = pytest.mark.offline


class _Characters:
    def encode(self, content):
        return [ord(c) for c in content]

    def decode(self, tokens):
        return "".join(chr(c) for c in tokens)


async def _embed(texts):
    return np.ones((len(texts), 8))


async def _no_llm(*args, **kwargs):
    raise AssertionError("This lifecycle fixture must not call an LLM")


async def _extract(chunks, *args, **kwargs):
    results = []
    for chunk_id, payload in chunks.items():
        entities = {
            name: [
                {
                    "entity_name": name,
                    "entity_type": "company",
                    "description": f"{name} company",
                    "source_id": chunk_id,
                    "file_path": payload["file_path"],
                    "timestamp": 1,
                }
            ]
            for name in ("Atlas", "Borealis")
        }
        relations = {
            ("Atlas", "Borealis"): [
                {
                    "src_id": "Atlas",
                    "tgt_id": "Borealis",
                    "weight": 1.0,
                    "description": "Atlas cooperates with Borealis",
                    "keywords": "cooperates",
                    "source_id": chunk_id,
                    "file_path": payload["file_path"],
                    "timestamp": 1,
                }
            ]
        }
        results.append((entities, relations))
    return results


@pytest.mark.asyncio
@pytest.mark.parametrize("entrypoint", ["sdk", "rest"])
@pytest.mark.parametrize("kind", ["entity", "relation", "relation-reversed"])
async def test_delete_removes_tracking_before_reinsert(
    tmp_path, monkeypatch, kind, entrypoint
):
    rag = LightRAG(
        working_dir=str(tmp_path),
        workspace=f"tracking_{uuid4().hex}",
        llm_model_func=_no_llm,
        embedding_func=EmbeddingFunc(embedding_dim=8, func=_embed),
        tokenizer=Tokenizer("characters", _Characters()),
        max_parallel_insert=1,
    )
    await rag.initialize_storages()
    rag._process_extract_entities = _extract
    try:
        await rag.ainsert(
            "Atlas and Borealis: old harbor", ids=["old"], file_paths=["old.txt"]
        )
        old_chunk = (await rag.doc_status.get_by_id("old"))["chunks_list"][0]
        relation_key = make_relation_chunk_key("Atlas", "Borealis")
        assert (await rag.relation_chunks.get_by_id(relation_key))["chunk_ids"] == [
            old_chunk
        ]
        names = (
            ("Borealis", "Atlas")
            if kind.endswith("reversed")
            else ("Atlas", "Borealis")
        )
        if entrypoint == "rest":
            # Router imports parse argv; do not hand them pytest's options.
            monkeypatch.setattr(sys, "argv", [sys.argv[0]])
            from fastapi import FastAPI
            from httpx import ASGITransport, AsyncClient
            from lightrag.api.routers.graph_routes import create_graph_routes

            app = FastAPI()
            app.include_router(create_graph_routes(rag, api_key="fixture-key"))
            resource = "entity" if kind == "entity" else "relation"
            payload = (
                {"entity_name": "Atlas"}
                if kind == "entity"
                else {"source_entity": names[0], "target_entity": names[1]}
            )
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.request(
                    "DELETE",
                    f"/graph/{resource}/delete",
                    json=payload,
                    headers={"X-API-Key": "fixture-key"},
                )
            assert response.status_code == 200, response.text
            assert response.json()["status"] == "success"
        else:
            result = (
                await rag.adelete_by_entity("Atlas")
                if kind == "entity"
                else await rag.adelete_by_relation(*names)
            )
            assert result.status == "success"
        assert await rag.relation_chunks.get_by_id(relation_key) is None
        assert relation_key not in json.loads(
            Path(rag.relation_chunks._file_name).read_text()
        )
        if kind == "entity":
            assert await rag.entity_chunks.get_by_id("Atlas") is None
            assert "Atlas" not in json.loads(
                Path(rag.entity_chunks._file_name).read_text()
            )
        # Unrelated entity provenance must survive both deletion paths.
        assert (await rag.entity_chunks.get_by_id("Borealis"))["chunk_ids"] == [
            old_chunk
        ]

        await rag.ainsert(
            "Atlas and Borealis: new mountain", ids=["new"], file_paths=["new.txt"]
        )
        new_chunk = (await rag.doc_status.get_by_id("new"))["chunks_list"][0]
        edge = await rag.chunk_entity_relation_graph.get_edge("Atlas", "Borealis")
        assert edge["source_id"].split(GRAPH_FIELD_SEP) == [new_chunk]
        assert (await rag.relation_chunks.get_by_id(relation_key))["chunk_ids"] == [
            new_chunk
        ]
        if kind == "entity":
            node = await rag.chunk_entity_relation_graph.get_node("Atlas")
            assert node["source_id"].split(GRAPH_FIELD_SEP) == [new_chunk]
            assert (await rag.entity_chunks.get_by_id("Atlas"))["chunk_ids"] == [
                new_chunk
            ]
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_sdk_deletion_reserves_workspace_writer_slot(tmp_path, monkeypatch):
    """A deletion excludes both the pipeline and disjoint admin writers.

    Per-entity locks do not collide for Atlas and Borealis, so this proves the
    workspace reservation rather than accidentally relying on a shared key.
    """
    from lightrag.kg.shared_storage import get_namespace_data
    from lightrag import utils_graph

    rag = LightRAG(
        working_dir=str(tmp_path),
        workspace=f"tracking_slot_{uuid4().hex}",
        llm_model_func=_no_llm,
        embedding_func=EmbeddingFunc(embedding_dim=8, func=_embed),
        tokenizer=Tokenizer("characters", _Characters()),
    )
    await rag.initialize_storages()
    entered = asyncio.Event()
    release = asyncio.Event()

    async def _blocked_delete(*args, **kwargs):
        entered.set()
        await release.wait()
        return DeletionResult(
            status="success", doc_id=args[3], message="deleted", status_code=200
        )

    monkeypatch.setattr(utils_graph, "adelete_by_entity", _blocked_delete)
    first = asyncio.create_task(rag.adelete_by_entity("Atlas"))
    try:
        await entered.wait()
        pipeline_status = await get_namespace_data(
            "pipeline_status", workspace=rag.workspace
        )
        assert pipeline_status["busy"] is True
        assert pipeline_status["operation_record"] == {
            "kind": "graph_mutation",
            "operation": "entity deletion",
        }

        with pytest.raises(RuntimeError, match="Pipeline is busy"):
            await rag.adelete_by_entity("Borealis")

        release.set()
        assert (await first).status == "success"
        assert pipeline_status["busy"] is False
        assert pipeline_status["busy_owner"] is None
    finally:
        release.set()
        if not first.done():
            await first
        await rag.finalize_storages()
