"""NetworkXStorage drop status follows the durable operation, not notification."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lightrag.kg import networkx_impl
from lightrag.kg.networkx_impl import NetworkXStorage
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
from lightrag.utils import EmbeddingFunc

pytestmark = pytest.mark.offline


@pytest.fixture(autouse=True)
def _shared_data():
    finalize_share_data()
    initialize_share_data()
    yield
    finalize_share_data()


async def _embed(texts):
    return np.random.rand(len(texts), 8)


def _make_storage(tmp_path) -> NetworkXStorage:
    return NetworkXStorage(
        namespace="test_graph",
        workspace="ws",
        global_config={
            "working_dir": str(tmp_path),
            "embedding_batch_num": 10,
            "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.5},
        },
        embedding_func=EmbeddingFunc(embedding_dim=8, max_token_size=512, func=_embed),
    )


@pytest.mark.asyncio
async def test_drop_stays_successful_when_peer_notification_fails(
    tmp_path, monkeypatch
):
    storage = _make_storage(tmp_path)
    await storage.initialize()
    try:
        await storage.upsert_node("n1", {"entity_id": "n1"})
        await storage.index_done_callback()
        assert Path(storage._graphml_xml_file).exists()

        async def notification_boom(namespace, workspace=None):
            raise RuntimeError("notification boom")

        monkeypatch.setattr(networkx_impl, "set_all_update_flags", notification_boom)
        logged_errors = []
        monkeypatch.setattr(networkx_impl.logger, "error", logged_errors.append)
        result = await storage.drop()

        assert result == {"status": "success", "message": "data dropped"}
        assert not Path(storage._graphml_xml_file).exists()
        assert storage._graph.number_of_nodes() == 0
        assert storage.storage_updated.value is False
        assert any("some processes may not reload" in msg for msg in logged_errors)
        assert any("notification boom" in msg for msg in logged_errors)
    finally:
        await storage.finalize()


@pytest.mark.asyncio
async def test_drop_reports_a_destructive_file_failure(tmp_path, monkeypatch):
    storage = _make_storage(tmp_path)
    await storage.initialize()
    try:
        await storage.upsert_node("n1", {"entity_id": "n1"})
        await storage.index_done_callback()

        def remove_boom(path):
            raise OSError("delete boom")

        monkeypatch.setattr(networkx_impl.os, "remove", remove_boom)

        result = await storage.drop()

        assert result == {"status": "error", "message": "delete boom"}
        assert Path(storage._graphml_xml_file).exists()
        assert storage._graph.has_node("n1")
    finally:
        await storage.finalize()
