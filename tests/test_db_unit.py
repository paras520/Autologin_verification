"""Unit tests for src/db.py pure helpers."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.db import _build_connect_kwargs


def _make_mock_conn():
    """Build a fully-mocked asyncpg connection with transaction support."""
    mock_conn = AsyncMock()
    mock_conn.close = AsyncMock()
    mock_conn.fetch = AsyncMock(return_value=[{
        "id": "row-1",
        "cb_link_id": "B-IN-abc123",
        "login_service": "NetBanking",
        "login_url": "https://netbanking.example.com/login",
        "display_name": "Example NetBanking",
        "sorting_order": 1,
        "status": "active",
    }])
    mock_conn.execute = AsyncMock()
    mock_tx = MagicMock()
    mock_tx.__aenter__ = AsyncMock(return_value=None)
    mock_tx.__aexit__ = AsyncMock(return_value=None)
    mock_conn.transaction = MagicMock(return_value=mock_tx)
    return mock_conn


class TestBuildConnectKwargs:
    def test_no_sslmode_no_ssl(self):
        dsn = "postgresql://user:pass@host:5432/db"
        result = _build_connect_kwargs(dsn)
        assert "ssl" not in result
        assert result["dsn"] == "postgresql://user:pass@host:5432/db"

    def test_sslmode_require_adds_ssl_true(self):
        dsn = "postgresql://user:pass@host:5432/db?sslmode=require"
        result = _build_connect_kwargs(dsn)
        assert result["ssl"] is True

    def test_sslmode_require_strips_query_from_dsn(self):
        dsn = "postgresql://user:pass@host:5432/db?sslmode=require"
        result = _build_connect_kwargs(dsn)
        assert "sslmode" not in result["dsn"]
        assert "?" not in result["dsn"]

    def test_other_query_params_stripped(self):
        dsn = "postgresql://user:pass@host:5432/db?connect_timeout=10"
        result = _build_connect_kwargs(dsn)
        assert "connect_timeout" not in result["dsn"]

    def test_sslmode_prefer_no_ssl_kwarg(self):
        dsn = "postgresql://user:pass@host:5432/db?sslmode=prefer"
        result = _build_connect_kwargs(dsn)
        assert "ssl" not in result

    def test_returns_dict_with_dsn_key(self):
        result = _build_connect_kwargs("postgresql://localhost/test")
        assert "dsn" in result
        assert isinstance(result, dict)


class TestFetchRowsAsync:
    @pytest.mark.asyncio
    async def test_fetch_rows_raises_without_database_url(self, monkeypatch):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        from src.db import fetch_rows
        with pytest.raises(RuntimeError, match="DATABASE_URL"):
            await fetch_rows("test-id")

    @pytest.mark.asyncio
    async def test_create_activity_run_raises_without_database_url(self, monkeypatch):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        from src.db import create_activity_run
        with pytest.raises(RuntimeError, match="DATABASE_URL"):
            await create_activity_run(["test-id"])

    @pytest.mark.asyncio
    async def test_fetch_rows_success(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@host:5432/db")
        mock_conn = _make_mock_conn()
        with patch("asyncpg.connect", new=AsyncMock(return_value=mock_conn)):
            from src.db import fetch_rows
            rows = await fetch_rows("B-IN-abc123")
        assert len(rows) == 1
        assert rows[0]["cb_link_id"] == "B-IN-abc123"
        mock_conn.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetch_rows_active_only(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@host:5432/db")
        mock_conn = _make_mock_conn()
        with patch("asyncpg.connect", new=AsyncMock(return_value=mock_conn)):
            from src.db import fetch_rows
            rows = await fetch_rows("B-IN-abc123", include_inactive=False)
        # fetch was called with two positional params (cb_link_id + "active")
        call_args = mock_conn.fetch.call_args
        assert "active" in call_args.args or (len(call_args.args) > 2 and call_args.args[2] == "active")
        assert isinstance(rows, list)

    @pytest.mark.asyncio
    async def test_create_activity_run_success(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@host:5432/db")
        mock_conn = _make_mock_conn()
        with patch("asyncpg.connect", new=AsyncMock(return_value=mock_conn)):
            from src.db import create_activity_run
            run_id = await create_activity_run(["B-IN-1", "B-IN-2"], triggered_by="test")
        assert isinstance(run_id, str)
        assert len(run_id) == 36  # UUID string length
        mock_conn.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_activity_run_success(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@host:5432/db")
        mock_conn = _make_mock_conn()
        with patch("asyncpg.connect", new=AsyncMock(return_value=mock_conn)):
            from src.db import start_activity_run
            await start_activity_run("some-run-uuid")
        mock_conn.execute.assert_called_once()
        mock_conn.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_activity_run_raises_without_database_url(self, monkeypatch):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        from src.db import start_activity_run
        with pytest.raises(RuntimeError, match="DATABASE_URL"):
            await start_activity_run("some-run-uuid")

    @pytest.mark.asyncio
    async def test_upsert_run_item_result_success(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@host:5432/db")
        mock_conn = _make_mock_conn()
        with patch("asyncpg.connect", new=AsyncMock(return_value=mock_conn)):
            from src.db import upsert_run_item_result
            await upsert_run_item_result("run-uuid", "B-IN-1", {"score": 90}, succeeded=True)
        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args.args
        assert "completed" in call_args[1]  # status
        mock_conn.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_upsert_run_item_result_failed(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@host:5432/db")
        mock_conn = _make_mock_conn()
        with patch("asyncpg.connect", new=AsyncMock(return_value=mock_conn)):
            from src.db import upsert_run_item_result
            await upsert_run_item_result("run-uuid", "B-IN-1", {"score": 0}, succeeded=False)
        call_args = mock_conn.execute.call_args.args
        assert "failed" in call_args[1]  # status

    @pytest.mark.asyncio
    async def test_upsert_run_item_result_raises_without_database_url(self, monkeypatch):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        from src.db import upsert_run_item_result
        with pytest.raises(RuntimeError, match="DATABASE_URL"):
            await upsert_run_item_result("run-uuid", "B-IN-1", {}, succeeded=True)

    @pytest.mark.asyncio
    async def test_finalize_activity_run_completed(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@host:5432/db")
        mock_conn = _make_mock_conn()
        with patch("asyncpg.connect", new=AsyncMock(return_value=mock_conn)):
            from src.db import finalize_activity_run
            await finalize_activity_run("run-uuid", success_items=5, failed_items=0)
        call_args = mock_conn.execute.call_args.args
        assert "completed" in call_args[1]

    @pytest.mark.asyncio
    async def test_finalize_activity_run_failed(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@host:5432/db")
        mock_conn = _make_mock_conn()
        with patch("asyncpg.connect", new=AsyncMock(return_value=mock_conn)):
            from src.db import finalize_activity_run
            await finalize_activity_run("run-uuid", success_items=0, failed_items=3)
        call_args = mock_conn.execute.call_args.args
        assert "failed" in call_args[1]

    @pytest.mark.asyncio
    async def test_finalize_activity_run_partial(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@host:5432/db")
        mock_conn = _make_mock_conn()
        with patch("asyncpg.connect", new=AsyncMock(return_value=mock_conn)):
            from src.db import finalize_activity_run
            await finalize_activity_run("run-uuid", success_items=3, failed_items=2)
        call_args = mock_conn.execute.call_args.args
        assert "partial" in call_args[1]

    @pytest.mark.asyncio
    async def test_finalize_activity_run_raises_without_database_url(self, monkeypatch):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        from src.db import finalize_activity_run
        with pytest.raises(RuntimeError, match="DATABASE_URL"):
            await finalize_activity_run("run-uuid", success_items=1, failed_items=0)
