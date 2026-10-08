"""
Model registry (plan section 3.1, extension point 4).

Embedders, rerankers, NLI verifiers and LLMs are DATA (config/models.yaml),
referenced everywhere by a short key -- never by a hardcoded model-name
string in code. Each entry carries the facts that decide whether a model
may be used in a given deployment profile: licence, whether that licence
has actually been verified, and whether commercial use is allowed. A
"commercial" profile refuses any model that is non-commercial OR whose
licence is unverified, so an unchecked model cannot slip into a commercial
demo by accident.
"""
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

Kind = Literal["embedder", "reranker", "nli", "llm"]
Hardware = Literal["cpu", "gpu", "api"]


class LicenseError(Exception):
    """A model is not permitted in the requested deployment profile."""


class ModelSpec(BaseModel):
    key: str = Field(min_length=1)
    name: str = Field(min_length=1)                 # provider / hub identifier
    kind: Kind
    dim: Optional[int] = Field(default=None, gt=0)  # embedders only
    max_length: Optional[int] = Field(default=None, gt=0)
    license: str = Field(min_length=1)
    license_verified: bool
    commercial_ok: bool
    hardware: Hardware
    query_prefix: str = ""
    doc_prefix: str = ""
    languages: List[str] = Field(default_factory=list)
    params: Dict[str, Any] = Field(default_factory=dict)
    notes: str = ""

    @field_validator("languages")
    @classmethod
    def _lower_languages(cls, v: List[str]) -> List[str]:
        return [x.lower() for x in v]

    @model_validator(mode="after")
    def _embedder_needs_dim(self) -> "ModelSpec":
        if self.kind == "embedder" and self.dim is None:
            raise ValueError(f"embedder {self.key!r} must declare dim")
        return self

    def allowed_in(self, commercial: bool) -> bool:
        return (not commercial) or (self.commercial_ok and self.license_verified)


class ModelRegistry:
    def __init__(self, specs: Mapping[str, ModelSpec]):
        self._specs = dict(specs)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ModelRegistry":
        raw = data.get("models")
        if not isinstance(raw, Mapping) or not raw:
            raise ValueError("registry file must contain a non-empty 'models' mapping")
        return cls({key: ModelSpec(key=key, **spec) for key, spec in raw.items()})

    @classmethod
    def from_file(cls, path: Path) -> "ModelRegistry":
        return cls.from_mapping(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})

    def __contains__(self, key: str) -> bool:
        return key in self._specs

    def get(self, key: str) -> ModelSpec:
        try:
            return self._specs[key]
        except KeyError:
            raise KeyError(f"unknown model key {key!r}; known: {sorted(self._specs)}") from None

    def keys(self, kind: Optional[Kind] = None) -> List[str]:
        return sorted(k for k, s in self._specs.items() if kind is None or s.kind == kind)

    def resolve(self, key: str, kind: Kind, commercial: bool = False) -> ModelSpec:
        """Fetch a model, checking it is the expected kind and permitted in the profile."""
        spec = self.get(key)
        if spec.kind != kind:
            raise TypeError(f"model {key!r} is a {spec.kind}, expected {kind}")
        if not spec.allowed_in(commercial):
            reason = "non-commercial licence" if not spec.commercial_ok else "licence not yet verified"
            raise LicenseError(f"model {key!r} ({spec.license}) is not allowed in a commercial profile: {reason}")
        return spec
