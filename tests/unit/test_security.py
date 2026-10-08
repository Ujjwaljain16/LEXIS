"""serving/security.py: fail-closed auth, constant-time key match, per-tenant rate limiting, and the
ingest path guard."""
import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient

from lexis.config import settings
from lexis.serving.security import (
    SlidingWindowLimiter, authenticate, parse_api_keys, rate_limit_dependency, require_tenant, resolve_ingest_path,
)


def test_parse_api_keys():
    assert parse_api_keys("k1:acme, k2:globex ,bare, :nokey,,") == {"k1": "acme", "k2": "globex", "bare": "default"}
    assert parse_api_keys("") == {}


def test_authenticate():
    keys = {"secret-1": "acme", "secret-2": "globex"}
    assert authenticate("secret-2", keys) == "globex"
    assert authenticate("secret", keys) is None
    assert authenticate("", keys) is None and authenticate(None, keys) is None


def build_app(limiter=None):
    app = FastAPI()
    deps = [Depends(rate_limit_dependency(limiter))] if limiter else [Depends(require_tenant)]

    @app.get("/x", dependencies=deps)
    async def x():
        return {"ok": True}

    return TestClient(app)


def test_fails_closed_when_no_keys_configured(monkeypatch):
    monkeypatch.setattr(settings, "lexis_api_keys", "")
    monkeypatch.setattr(settings, "auth_disabled", False)
    assert build_app().get("/x", headers={"X-API-Key": "anything"}).status_code == 503


def test_rejects_missing_and_wrong_key_accepts_right_key(monkeypatch):
    monkeypatch.setattr(settings, "lexis_api_keys", "good:acme")
    monkeypatch.setattr(settings, "auth_disabled", False)
    client = build_app()
    assert client.get("/x").status_code == 401
    assert client.get("/x", headers={"X-API-Key": "bad"}).status_code == 401
    assert client.get("/x", headers={"X-API-Key": "good"}).status_code == 200


def test_no_default_dev_key_is_accepted(monkeypatch):
    monkeypatch.setattr(settings, "lexis_api_keys", "good:acme")
    monkeypatch.setattr(settings, "auth_disabled", False)
    assert build_app().get("/x", headers={"X-API-Key": "dev_secret_key"}).status_code == 401


def test_auth_disabled_is_explicit_opt_in(monkeypatch):
    monkeypatch.setattr(settings, "lexis_api_keys", "")
    monkeypatch.setattr(settings, "auth_disabled", True)
    assert build_app().get("/x").status_code == 200


def test_sliding_window_limiter_allows_then_blocks_then_recovers():
    now = [0.0]
    limiter = SlidingWindowLimiter(limit=2, window_s=10, clock=lambda: now[0])
    assert limiter.check("t") is None
    now[0] = 1
    assert limiter.check("t") is None
    now[0] = 2
    assert limiter.check("t") == pytest.approx(8.0)       # next slot frees when the first hit ages out
    now[0] = 10
    assert limiter.check("t") is None                      # first hit (t=0) has left the window


def test_limiter_is_per_tenant():
    limiter = SlidingWindowLimiter(limit=1, window_s=60, clock=lambda: 0.0)
    assert limiter.check("a") is None
    assert limiter.check("a") is not None
    assert limiter.check("b") is None


def test_rate_limit_returns_429_with_retry_after(monkeypatch):
    monkeypatch.setattr(settings, "lexis_api_keys", "good:acme")
    monkeypatch.setattr(settings, "auth_disabled", False)
    client = build_app(SlidingWindowLimiter(limit=1, window_s=60))
    assert client.get("/x", headers={"X-API-Key": "good"}).status_code == 200
    r = client.get("/x", headers={"X-API-Key": "good"})
    assert r.status_code == 429 and int(r.headers["Retry-After"]) >= 1


def test_ingest_path_accepts_files_inside_root(tmp_path):
    (tmp_path / "sub").mkdir()
    f = tmp_path / "sub" / "a.pdf"
    f.write_text("x")
    assert resolve_ingest_path(str(tmp_path), "sub/a.pdf") == f.resolve()


@pytest.mark.parametrize("bad", ["../outside.txt", "sub/../../outside.txt"])
def test_ingest_path_rejects_traversal(tmp_path, bad):
    (tmp_path.parent / "outside.txt").write_text("secret")
    (tmp_path / "sub").mkdir(exist_ok=True)
    with pytest.raises(HTTPException) as e:
        resolve_ingest_path(str(tmp_path), bad)
    assert e.value.status_code == 400


def test_ingest_path_rejects_absolute_path_elsewhere(tmp_path):
    outside = tmp_path.parent / "abs_outside.txt"
    outside.write_text("secret")
    with pytest.raises(HTTPException) as e:
        resolve_ingest_path(str(tmp_path), str(outside))
    assert e.value.status_code == 400


def test_ingest_path_404_for_missing_file(tmp_path):
    with pytest.raises(HTTPException) as e:
        resolve_ingest_path(str(tmp_path), "nope.pdf")
    assert e.value.status_code == 404
