"""Unit tests for src/db.py pure helpers."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.db import _build_connect_kwargs


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
