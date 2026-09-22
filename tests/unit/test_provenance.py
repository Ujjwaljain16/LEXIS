import json

from lexis.evaluation.provenance import (
    REDACTED,
    build_provenance,
    config_hash,
    file_sha256,
    git_sha,
    package_versions,
    redact_secrets,
)


def test_redacts_secret_keys_case_insensitively():
    out = redact_secrets({"gemini_api_key": "abc", "QDRANT_API_KEY": "x", "model": "m", "db_password": "p"})
    assert out["gemini_api_key"] == REDACTED and out["QDRANT_API_KEY"] == REDACTED
    assert out["db_password"] == REDACTED and out["model"] == "m"


def test_redacts_credentials_inside_urls_but_keeps_host():
    out = redact_secrets({"postgres_url": "postgresql://user:hunter2@host.example:5432/db?sslmode=require"})
    assert "hunter2" not in out["postgres_url"] and "user" not in out["postgres_url"].split("@")[0].replace("postgresql://", "")
    assert "host.example" in out["postgres_url"]


def test_url_without_credentials_untouched():
    assert redact_secrets({"u": "https://example.org/x"})["u"] == "https://example.org/x"


def test_redaction_recurses_into_nested_structures():
    out = redact_secrets({"a": [{"token": "t"}, "ok"], "b": {"secret": "s"}})
    assert out["a"][0]["token"] == REDACTED and out["b"]["secret"] == REDACTED and out["a"][1] == "ok"


def test_empty_secret_value_not_marked():
    assert redact_secrets({"api_key": ""})["api_key"] == ""


def test_config_hash_ignores_secret_values_and_key_order():
    a = {"k": 1, "api_key": "AAA", "z": [1, 2]}
    b = {"z": [1, 2], "api_key": "BBB", "k": 1}
    assert config_hash(a) == config_hash(b)


def test_config_hash_changes_when_nonsecret_changes():
    assert config_hash({"k": 1}) != config_hash({"k": 2})


def test_file_sha256_matches_content(tmp_path):
    p = tmp_path / "f.txt"
    p.write_bytes(b"hello")
    assert file_sha256(str(p)) == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"


def test_git_sha_none_outside_a_repo(tmp_path, monkeypatch):
    # stop git from discovering a repo in a parent of the temp dir
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    assert git_sha(str(tmp_path)) is None


def test_package_versions_handles_missing():
    v = package_versions(["numpy", "definitely-not-a-package-xyz"])
    assert v["numpy"] and v["definitely-not-a-package-xyz"] is None


def test_build_provenance_is_json_serializable_and_secret_free(tmp_path):
    data = tmp_path / "d.json"
    data.write_text("{}")
    prov = build_provenance(
        {"gemini_api_key": "SECRET123", "rrf_k": 61},
        data_paths=[str(data), str(tmp_path / "missing.json")],
        packages=["numpy"],
        extra={"note": "x", "token": "T"},
    )
    text = json.dumps(prov)
    assert "SECRET123" not in text and '"T"' not in text
    assert prov["data_missing"] == [str(tmp_path / "missing.json")]
    assert str(data) in prov["data_sha256"] and prov["config_hash"]
