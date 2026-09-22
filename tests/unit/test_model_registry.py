import pytest
from pydantic import ValidationError

from lexis.registry.models import LicenseError, ModelRegistry, ModelSpec


def spec(**kw):
    base = dict(name="org/x", kind="reranker", license="Apache-2.0", license_verified=True,
                commercial_ok=True, hardware="cpu")
    base.update(kw)
    return base


def registry():
    return ModelRegistry.from_mapping({"models": {
        "good": spec(),
        "nc": spec(license="CC-BY-NC-4.0", commercial_ok=False),
        "unverified": spec(license_verified=False),
        "emb": spec(kind="embedder", dim=8),
    }})


def test_get_returns_spec_with_key():
    assert registry().get("good").key == "good"


def test_unknown_key_lists_known_keys():
    with pytest.raises(KeyError, match="known"):
        registry().get("nope")


def test_keys_filter_by_kind_and_sorted():
    r = registry()
    assert r.keys("reranker") == ["good", "nc", "unverified"]
    assert r.keys("embedder") == ["emb"]
    assert r.keys() == ["emb", "good", "nc", "unverified"]


def test_contains():
    assert "good" in registry() and "nope" not in registry()


def test_resolve_wrong_kind_raises_type_error():
    with pytest.raises(TypeError):
        registry().resolve("good", "embedder")


def test_noncommercial_profile_allows_everything():
    r = registry()
    for key in ("good", "nc", "unverified"):
        assert r.resolve(key, "reranker", commercial=False).key == key


def test_commercial_profile_blocks_noncommercial_licence():
    with pytest.raises(LicenseError, match="non-commercial"):
        registry().resolve("nc", "reranker", commercial=True)


def test_commercial_profile_blocks_unverified_licence():
    with pytest.raises(LicenseError, match="not yet verified"):
        registry().resolve("unverified", "reranker", commercial=True)


def test_commercial_profile_allows_verified_commercial_model():
    assert registry().resolve("good", "reranker", commercial=True).key == "good"


def test_embedder_requires_dim():
    with pytest.raises(ValidationError):
        ModelSpec(key="e", **spec(kind="embedder"))


def test_dim_must_be_positive():
    with pytest.raises(ValidationError):
        ModelSpec(key="e", **spec(kind="embedder", dim=0))


def test_languages_are_lowercased():
    assert ModelSpec(key="k", **spec(languages=["EN", "Hi"])).languages == ["en", "hi"]


def test_empty_registry_rejected():
    with pytest.raises(ValueError):
        ModelRegistry.from_mapping({"models": {}})
    with pytest.raises(ValueError):
        ModelRegistry.from_mapping({})


def test_invalid_kind_rejected():
    with pytest.raises(ValidationError):
        ModelSpec(key="k", **spec(kind="banana"))
