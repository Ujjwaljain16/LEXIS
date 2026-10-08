"""
Regression tests for PostgresClient.get_citation().

Bug fixed: the SELECT query was missing its `$1` bind placeholder
(`"...WHERE pqac_id = "`), so asyncpg never bound the parameter. Every call
raised inside asyncpg, was swallowed by the surrounding `except Exception`,
and get_citation() silently returned None for every lookup, valid or not.

These tests exercise PostgresClient against a fake asyncpg connection (no
live Postgres is available in this environment) and assert on the exact
query text and bound arguments passed to `fetchrow`, so a regression to the
missing-placeholder bug would fail these tests immediately rather than being
swallowed again.
"""
import pytest

from lexis.indexing.pg_client import PostgresClient, BoundingBox


class FakeConnection:
    """Minimal stand-in for an asyncpg connection."""

    def __init__(self, fetchrow_result=None, raise_exc=None):
        self.fetchrow_result = fetchrow_result
        self.raise_exc = raise_exc
        self.calls = []  # list of (query, args)

    async def fetchrow(self, query, *args):
        self.calls.append((query, args))
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.fetchrow_result


class FakeAcquireContext:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, exc_type, exc, tb):
        return False


class FakePool:
    def __init__(self, conn):
        self._conn = conn

    def acquire(self):
        return FakeAcquireContext(self._conn)


def make_client(conn: FakeConnection) -> PostgresClient:
    """Builds a PostgresClient wired to a fake pool, bypassing real connect()."""
    client = PostgresClient(dsn="postgresql://fake-host/fake-db")
    client.pool = FakePool(conn)

    async def _fake_connect():
        # pool is already set; real connect() must not be invoked in these tests.
        return None

    client.connect = _fake_connect
    return client


@pytest.mark.asyncio
async def test_get_citation_binds_parameter_with_placeholder_not_concatenation():
    """The exact bug: query text must contain a real bind placeholder, and the
    identifier must be passed as a bound argument, never interpolated into the
    SQL string itself."""
    row = {
        "pqac_id": "pqac-9f8e7d6c",
        "document_id": "doc-alpha-1",
        "document_version": 2,
        "document_hash": "abc123",
        "page": 5,
        "x0": 10.0, "y0": 20.0, "x1": 110.0, "y1": 40.0,
        "text_span": "an arbitrary passage of text",
        "chunk_id": "chunk-xyz",
        "citation_confidence": 0.87,
    }
    conn = FakeConnection(fetchrow_result=row)
    client = make_client(conn)

    result = await client.get_citation("pqac-9f8e7d6c")

    assert len(conn.calls) == 1
    query, args = conn.calls[0]
    assert "$1" in query, "query must use a real asyncpg bind placeholder"
    assert "pqac-9f8e7d6c" not in query, "the identifier must never be string-interpolated into SQL"
    assert args == ("pqac-9f8e7d6c",)

    assert result is not None
    assert result.pqac_id == "pqac-9f8e7d6c"
    assert result.document_id == "doc-alpha-1"
    assert result.page == 5
    assert result.bbox == BoundingBox(x0=10.0, y0=20.0, x1=110.0, y1=40.0)
    assert result.citation_confidence == 0.87


@pytest.mark.asyncio
async def test_get_citation_returns_none_for_unknown_id_without_raising():
    conn = FakeConnection(fetchrow_result=None)
    client = make_client(conn)

    result = await client.get_citation("pqac-does-not-exist")

    assert result is None
    assert len(conn.calls) == 1
    query, args = conn.calls[0]
    assert "$1" in query
    assert args == ("pqac-does-not-exist",)


@pytest.mark.asyncio
async def test_get_citation_different_ids_bind_different_values():
    """Guards against a hardcoded/constant bound value masking the bug."""
    row = {
        "pqac_id": "pqac-second", "document_id": "doc-2", "document_version": 1,
        "document_hash": "", "page": 1, "x0": 0.0, "y0": 0.0, "x1": 1.0, "y1": 1.0,
        "text_span": "text", "chunk_id": "c2", "citation_confidence": None,
    }
    conn = FakeConnection(fetchrow_result=row)
    client = make_client(conn)

    await client.get_citation("pqac-first")
    await client.get_citation("pqac-second")

    assert [args for _, args in conn.calls] == [("pqac-first",), ("pqac-second",)]


@pytest.mark.asyncio
async def test_get_citation_still_swallows_genuine_db_errors_and_logs(caplog):
    """Preserves the existing best-effort contract (matches get_document's pattern)
    for real failures (e.g. connection drop) -- only the SQL-binding bug is fixed,
    not the surrounding error-handling behaviour."""
    conn = FakeConnection(raise_exc=ConnectionError("simulated connection drop"))
    client = make_client(conn)

    with caplog.at_level("WARNING"):
        result = await client.get_citation("pqac-whatever")

    assert result is None
    assert any("simulated connection drop" in record.message for record in caplog.records)
