"""Chunk-tracking cleanup must never outlive-order the graph object it describes.

`adelete_by_entity` / `adelete_by_relation` remove the graph object first and its
tracking rows afterwards. The reverse order -- which the helpers used before this
file existed -- breaks the purge recovery contract: a tracking row is the
authoritative attribution carrier, so dropping it while a transient vector or
graph failure leaves the object alive degrades that object's provenance to the
truncated graph `source_id`, from which `_purge_kg_contributions` can conclude
"no remaining sources" and delete an entity other documents still reference.

`TestFailureLeavesProvenanceIntact` are the fix proofs: they inject a vector-store
failure at the point where the pre-fix code had already deleted the tracking rows
and assert the rows survive alongside the object. They fail behaviourally on the
pre-fix ordering (the row is gone), not by importing a symbol the fix adds.

`TestOrphanRowsConverge` covers the residue of the chosen order -- a row whose
object is already gone -- and pins that a repeat deletion sweeps it, which is what
makes a partial failure recoverable instead of permanent.

`TestDurableCommitOrdering` covers the second half of the same invariant. On the
deferred backends the calls above are all in-memory and `index_done_callback` is
the only durable commit, so sequencing the *calls* proves nothing there: flushing
every store in one `asyncio.gather` leaves the durable order unconstrained, and a
failed GraphML commit next to a successful tracking commit puts the forbidden
state on disk. The flush is therefore split into two ordered phases.

The doubles are deliberately split by write timing, because the two halves of the
invariant fail on different backends: immediate-write (Redis/PG/Mongo KV, Neo4j/PG
graph) for the call ordering, deferred-commit (NetworkX/JSON) for the flush
ordering. tests/pipeline/test_graph_deletion_tracking.py runs the real deferred
stack and is green under every ordering, which is exactly why it cannot stand in
for either group here.
"""

from __future__ import annotations

from copy import deepcopy

import pytest

from lightrag import utils_graph
from lightrag.kg.networkx_impl import NetworkXStorage
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
from lightrag.utils import make_relation_chunk_key

pytestmark = pytest.mark.offline

ENTITY = "ATLAS"
OTHER = "BOREALIS"
RELATION_KEY = make_relation_chunk_key(*sorted([ENTITY, OTHER]))
CHUNKS = {"chunk_ids": ["chunk-1"], "count": 1}


@pytest.fixture(autouse=True)
def _shared_data():
    finalize_share_data()
    initialize_share_data()
    yield
    finalize_share_data()


class _Boom(RuntimeError):
    """Injected backend failure."""


class _KVStorage:
    """Immediate-write KV double: `delete` takes effect before the next await."""

    def __init__(self):
        self.records: dict = {}
        self.flushes = 0
        self.fail_delete_times = 0

    async def get_by_id(self, key):
        return deepcopy(self.records.get(key))

    async def upsert(self, data):
        self.records.update(deepcopy(data))

    async def delete(self, ids):
        if self.fail_delete_times > 0:
            self.fail_delete_times -= 1
            raise _Boom("tracking delete failed")
        for key in ids:
            self.records.pop(key, None)

    async def index_done_callback(self):
        self.flushes += 1


class _VectorStorage:
    def __init__(self, global_config):
        self.global_config = global_config
        self.fail = False

    def _check(self):
        if self.fail:
            raise _Boom("vector backend unavailable")

    async def delete(self, ids):
        self._check()

    async def delete_entity(self, entity_name):
        self._check()

    async def delete_entity_relation(self, entity_name):
        self._check()

    async def index_done_callback(self):
        return None


class _Fixture:
    """A real NetworkXStorage so node/edge survival is observed, not simulated."""

    def __init__(self, tmp_path):
        self.global_config = {
            "working_dir": str(tmp_path),
            "workspace": "",
            "embedding_batch_num": 10,
        }
        self.graph = NetworkXStorage(
            namespace="chunk_entity_relation",
            workspace="",
            global_config=self.global_config,
            embedding_func=None,
        )
        self.entities_vdb = _VectorStorage(self.global_config)
        self.relationships_vdb = _VectorStorage(self.global_config)
        self.entity_chunks = _KVStorage()
        self.relation_chunks = _KVStorage()

    async def start(self):
        await self.graph.initialize()
        for name in (ENTITY, OTHER):
            await self.graph.upsert_node(
                name, {"entity_id": name, "description": "d", "source_id": "chunk-1"}
            )
        await self.graph.upsert_edge(
            ENTITY, OTHER, {"description": "d", "weight": 1.0, "source_id": "chunk-1"}
        )
        await self.entity_chunks.upsert({ENTITY: dict(CHUNKS), OTHER: dict(CHUNKS)})
        await self.relation_chunks.upsert({RELATION_KEY: dict(CHUNKS)})
        return self

    async def delete_entity(self):
        return await utils_graph.adelete_by_entity(
            self.graph,
            self.entities_vdb,
            self.relationships_vdb,
            ENTITY,
            entity_chunks_storage=self.entity_chunks,
            relation_chunks_storage=self.relation_chunks,
        )

    async def delete_relation(self):
        return await utils_graph.adelete_by_relation(
            self.graph,
            self.relationships_vdb,
            ENTITY,
            OTHER,
            relation_chunks_storage=self.relation_chunks,
        )


@pytest.fixture
async def rag(tmp_path):
    fixture = await _Fixture(tmp_path).start()
    yield fixture
    await fixture.graph.finalize()


class TestFailureLeavesProvenanceIntact:
    """Fix proofs: a live object must never be left without its tracking row."""

    @pytest.mark.asyncio
    async def test_entity_vector_failure_keeps_all_tracking_rows(self, rag):
        rag.entities_vdb.fail = True

        result = await rag.delete_entity()

        assert result.status == "fail"
        # The object survived the failure, so its provenance must survive too.
        assert await rag.graph.has_node(ENTITY)
        assert rag.entity_chunks.records[ENTITY] == CHUNKS
        assert rag.relation_chunks.records[RELATION_KEY] == CHUNKS

    @pytest.mark.asyncio
    async def test_relation_vector_failure_keeps_tracking_row(self, rag):
        rag.relationships_vdb.fail = True

        result = await rag.delete_relation()

        assert result.status == "fail"
        assert await rag.graph.has_edge(ENTITY, OTHER)
        assert rag.relation_chunks.records[RELATION_KEY] == CHUNKS


class TestOrphanRowsConverge:
    """The residue of this order -- an orphan row -- must be recoverable."""

    @pytest.mark.asyncio
    async def test_entity_tracking_failure_converges_on_retry(self, rag):
        # Graph node and relation row go first; the entity row delete blows up.
        rag.entity_chunks.fail_delete_times = 1

        first = await rag.delete_entity()

        assert first.status == "fail"
        assert not await rag.graph.has_node(ENTITY)
        assert RELATION_KEY not in rag.relation_chunks.records
        orphan_left_behind = ENTITY in rag.entity_chunks.records
        assert orphan_left_behind

        second = await rag.delete_entity()

        assert second.status == "not_found"
        assert ENTITY not in rag.entity_chunks.records
        assert rag.entity_chunks.flushes == 1
        # Sweeping the orphan must not touch an unrelated entity's provenance.
        assert rag.entity_chunks.records[OTHER] == CHUNKS

    @pytest.mark.asyncio
    async def test_relation_tracking_failure_converges_on_retry(self, rag):
        rag.relation_chunks.fail_delete_times = 1

        first = await rag.delete_relation()

        assert first.status == "fail"
        assert not await rag.graph.has_edge(ENTITY, OTHER)
        assert RELATION_KEY in rag.relation_chunks.records

        second = await rag.delete_relation()

        assert second.status == "not_found"
        assert RELATION_KEY not in rag.relation_chunks.records
        assert rag.relation_chunks.flushes == 1

    @pytest.mark.asyncio
    async def test_not_found_without_orphan_row_skips_the_flush(self, rag):
        result = await utils_graph.adelete_by_entity(
            rag.graph,
            rag.entities_vdb,
            rag.relationships_vdb,
            "NEVER_EXISTED",
            entity_chunks_storage=rag.entity_chunks,
            relation_chunks_storage=rag.relation_chunks,
        )

        assert result.status == "not_found"
        assert rag.entity_chunks.flushes == 0
        assert set(rag.entity_chunks.records) == {ENTITY, OTHER}

    @pytest.mark.asyncio
    async def test_orphan_sweep_handles_a_legacy_shaped_row(self, rag):
        # A partial/legacy row is still stored attribution for a gone object.
        await rag.entity_chunks.upsert({"GHOST": {"count": 0}})

        result = await utils_graph.adelete_by_entity(
            rag.graph,
            rag.entities_vdb,
            rag.relationships_vdb,
            "GHOST",
            entity_chunks_storage=rag.entity_chunks,
            relation_chunks_storage=rag.relation_chunks,
        )

        assert result.status == "not_found"
        assert "GHOST" not in rag.entity_chunks.records


class _DeferredKVStorage:
    """Deferred-commit KV double: `delete` is in-memory, `index_done_callback` commits."""

    def __init__(self, name: str, commit_log: list[str]):
        self.name = name
        self.commit_log = commit_log
        self.records: dict = {}
        self.disk: dict = {}

    async def get_by_id(self, key):
        return deepcopy(self.records.get(key))

    async def upsert(self, data):
        self.records.update(deepcopy(data))

    async def delete(self, ids):
        for key in ids:
            self.records.pop(key, None)

    async def index_done_callback(self):
        self.commit_log.append(self.name)
        self.disk = deepcopy(self.records)


@pytest.fixture
async def deferred(tmp_path):
    """Real NetworkX graph plus deferred-commit tracking doubles."""
    fixture = _Fixture(tmp_path)
    fixture.commit_log: list[str] = []
    fixture.entity_chunks = _DeferredKVStorage("entity_chunks", fixture.commit_log)
    fixture.relation_chunks = _DeferredKVStorage("relation_chunks", fixture.commit_log)
    await fixture.start()
    # Seed the committed baseline: this is what a restart would read back.
    await fixture.entity_chunks.index_done_callback()
    await fixture.relation_chunks.index_done_callback()
    fixture.commit_log.clear()
    yield fixture
    await fixture.graph.finalize()


def _log_graph_commit(fixture, monkeypatch, *, fail: bool):
    original = fixture.graph.index_done_callback

    async def _commit():
        fixture.commit_log.append("graph")
        if fail:
            raise _Boom("graph commit failed")
        return await original()

    monkeypatch.setattr(fixture.graph, "index_done_callback", _commit)


class TestDurableCommitOrdering:
    """The graph must reach disk before the tracking rows do."""

    @pytest.mark.asyncio
    async def test_entity_graph_commit_failure_keeps_rows_on_disk(
        self, deferred, monkeypatch
    ):
        _log_graph_commit(deferred, monkeypatch, fail=True)

        result = await deferred.delete_entity()

        assert result.status == "fail"
        # The graph never committed, so on restart the entity is still live --
        # its tracking rows must still be on disk with it.
        assert deferred.entity_chunks.disk[ENTITY] == CHUNKS
        assert deferred.relation_chunks.disk[RELATION_KEY] == CHUNKS
        assert deferred.commit_log == ["graph"]

    @pytest.mark.asyncio
    async def test_relation_graph_commit_failure_keeps_row_on_disk(
        self, deferred, monkeypatch
    ):
        _log_graph_commit(deferred, monkeypatch, fail=True)

        result = await deferred.delete_relation()

        assert result.status == "fail"
        assert deferred.relation_chunks.disk[RELATION_KEY] == CHUNKS
        assert deferred.commit_log == ["graph"]

    @pytest.mark.asyncio
    async def test_successful_entity_delete_commits_graph_first(
        self, deferred, monkeypatch
    ):
        # Stability, not a fix proof: a single gather also happens to start the
        # graph commit first, so only the two failure cases above go red on the
        # unordered flush. This one pins the happy-path order against a future
        # reshuffle of the phases.
        _log_graph_commit(deferred, monkeypatch, fail=False)

        result = await deferred.delete_entity()

        assert result.status == "success"
        assert deferred.commit_log[0] == "graph"
        assert set(deferred.commit_log[1:]) == {"entity_chunks", "relation_chunks"}
        assert ENTITY not in deferred.entity_chunks.disk
        assert RELATION_KEY not in deferred.relation_chunks.disk
