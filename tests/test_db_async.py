"""Async DB function tests — asyncpg fully mocked."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call
import asyncpg


def _make_conn(fetch_return=None, execute_return=None):
    """Build a mock asyncpg connection."""
    conn = AsyncMock()
    conn.fetch = AsyncMock(return_value=fetch_return or [])
    conn.execute = AsyncMock(return_value=execute_return)
    conn.close = AsyncMock()

    # transaction() must be an async context manager
    txn = AsyncMock()
    txn.__aenter__ = AsyncMock(return_value=None)
    txn.__aexit__ = AsyncMock(return_value=False)
    conn.transaction = MagicMock(return_value=txn)

    return conn


class TestFetchRows:
    @pytest.mark.asyncio
    async def test_fetch_rows_returns_list(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@host:5432/db")
        mock_row = {"id": "1", "cb_link_id": "B-IN-001", "login_service": "Net", "login_url": "https://x.com", "display_name": "Bank", "sorting_order": 1, "status": "active"}
        conn = _make_conn(fetch_return=[mock_row])

        from src.db import fetch_rows
        with patch("asyncpg.connect", new=AsyncMock(return_value=conn)):
            result = await fetch_rows("B-IN-001")

        assert result == [mock_row]
        conn.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_fetch_rows_active_only(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@host:5432/db")
        conn = _make_conn(fetch_return=[])

        from src.db import fetch_rows
        with patch("asyncpg.connect", new=AsyncMock(return_value=conn)):
            result = await fetch_rows("B-IN-001", include_inactive=False)

        assert result == []
        # Query should include $2 param for "active" status filter
        call_args = conn.fetch.call_args
        assert "active" in call_args.args

    @pytest.mark.asyncio
    async def test_fetch_rows_closes_conn_on_exception(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@host:5432/db")
        conn = _make_conn()
        conn.fetch = AsyncMock(side_effect=RuntimeError("db error"))

        from src.db import fetch_rows
        with patch("asyncpg.connect", new=AsyncMock(return_value=conn)):
            with pytest.raises(RuntimeError):
                await fetch_rows("B-IN-001")

        conn.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_fetch_rows_no_env_raises(self, monkeypatch):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        from src.db import fetch_rows
        with pytest.raises(RuntimeError, match="DATABASE_URL"):
            await fetch_rows("B-IN-001")


class TestCreateActivityRun:
    @pytest.mark.asyncio
    async def test_create_returns_uuid_string(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@host:5432/db")
        conn = _make_conn()

        from src.db import create_activity_run
        with patch("asyncpg.connect", new=AsyncMock(return_value=conn)):
            run_id = await create_activity_run(["B-IN-001", "B-IN-002"])

        assert isinstance(run_id, str)
        assert len(run_id) == 36  # UUID format

    @pytest.mark.asyncio
    async def test_create_executes_insert_for_each_id(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@host:5432/db")
        conn = _make_conn()

        from src.db import create_activity_run
        with patch("asyncpg.connect", new=AsyncMock(return_value=conn)):
            await create_activity_run(["B-IN-001", "B-IN-002"], triggered_by="test")

        # 1 insert for activity_runs + 2 for activity_run_items
        assert conn.execute.await_count == 3

    @pytest.mark.asyncio
    async def test_create_closes_conn(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@host:5432/db")
        conn = _make_conn()

        from src.db import create_activity_run
        with patch("asyncpg.connect", new=AsyncMock(return_value=conn)):
            await create_activity_run(["B-IN-001"])

        conn.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_create_no_env_raises(self, monkeypatch):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        from src.db import create_activity_run
        with pytest.raises(RuntimeError, match="DATABASE_URL"):
            await create_activity_run(["B-IN-001"])


class TestStartActivityRun:
    @pytest.mark.asyncio
    async def test_start_updates_status(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@host:5432/db")
        conn = _make_conn()

        from src.db import start_activity_run
        with patch("asyncpg.connect", new=AsyncMock(return_value=conn)):
            await start_activity_run("run-uuid-123")

        conn.execute.assert_awaited_once()
        # SQL should contain 'running'
        sql = conn.execute.call_args.args[0]
        assert "running" in sql

    @pytest.mark.asyncio
    async def test_start_no_env_raises(self, monkeypatch):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        from src.db import start_activity_run
        with pytest.raises(RuntimeError, match="DATABASE_URL"):
            await start_activity_run("run-uuid-123")

    @pytest.mark.asyncio
    async def test_start_closes_conn(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@host:5432/db")
        conn = _make_conn()

        from src.db import start_activity_run
        with patch("asyncpg.connect", new=AsyncMock(return_value=conn)):
            await start_activity_run("run-uuid-123")

        conn.close.assert_awaited_once()


class TestUpsertRunItemResult:
    @pytest.mark.asyncio
    async def test_upsert_succeeded(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@host:5432/db")
        conn = _make_conn()

        from src.db import upsert_run_item_result
        with patch("asyncpg.connect", new=AsyncMock(return_value=conn)):
            await upsert_run_item_result("run-id", "B-IN-001", {"key": "val"}, succeeded=True)

        sql, status = conn.execute.call_args.args[0], conn.execute.call_args.args[1]
        assert "completed" in status

    @pytest.mark.asyncio
    async def test_upsert_failed(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@host:5432/db")
        conn = _make_conn()

        from src.db import upsert_run_item_result
        with patch("asyncpg.connect", new=AsyncMock(return_value=conn)):
            await upsert_run_item_result("run-id", "B-IN-001", {}, succeeded=False)

        status = conn.execute.call_args.args[1]
        assert "failed" in status

    @pytest.mark.asyncio
    async def test_upsert_no_env_raises(self, monkeypatch):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        from src.db import upsert_run_item_result
        with pytest.raises(RuntimeError, match="DATABASE_URL"):
            await upsert_run_item_result("run-id", "cb", {}, succeeded=True)

    @pytest.mark.asyncio
    async def test_upsert_closes_conn(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@host:5432/db")
        conn = _make_conn()

        from src.db import upsert_run_item_result
        with patch("asyncpg.connect", new=AsyncMock(return_value=conn)):
            await upsert_run_item_result("run-id", "cb", {}, succeeded=True)

        conn.close.assert_awaited_once()


class TestFinalizeActivityRun:
    @pytest.mark.asyncio
    async def test_finalize_all_success(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@host:5432/db")
        conn = _make_conn()

        from src.db import finalize_activity_run
        with patch("asyncpg.connect", new=AsyncMock(return_value=conn)):
            await finalize_activity_run("run-id", success_items=5, failed_items=0)

        status = conn.execute.call_args.args[1]
        assert "completed" in status

    @pytest.mark.asyncio
    async def test_finalize_all_failed(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@host:5432/db")
        conn = _make_conn()

        from src.db import finalize_activity_run
        with patch("asyncpg.connect", new=AsyncMock(return_value=conn)):
            await finalize_activity_run("run-id", success_items=0, failed_items=3)

        status = conn.execute.call_args.args[1]
        assert "failed" in status

    @pytest.mark.asyncio
    async def test_finalize_partial(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@host:5432/db")
        conn = _make_conn()

        from src.db import finalize_activity_run
        with patch("asyncpg.connect", new=AsyncMock(return_value=conn)):
            await finalize_activity_run("run-id", success_items=3, failed_items=2)

        status = conn.execute.call_args.args[1]
        assert "partial" in status

    @pytest.mark.asyncio
    async def test_finalize_no_env_raises(self, monkeypatch):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        from src.db import finalize_activity_run
        with pytest.raises(RuntimeError, match="DATABASE_URL"):
            await finalize_activity_run("run-id", 0, 0)

    @pytest.mark.asyncio
    async def test_finalize_closes_conn(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@host:5432/db")
        conn = _make_conn()

        from src.db import finalize_activity_run
        with patch("asyncpg.connect", new=AsyncMock(return_value=conn)):
            await finalize_activity_run("run-id", 1, 0)

        conn.close.assert_awaited_once()
