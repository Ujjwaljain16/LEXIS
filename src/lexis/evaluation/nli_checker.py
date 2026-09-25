"""
NLI (Natural Language Inference) checking for hallucination detection.

Rationale: given a premise (e.g. a retrieved chunk's text) and a hypothesis
(e.g. one atomic claim from a generated answer), decide whether the
premise actually entails the hypothesis -- the core primitive behind any
faithfulness/support metric (plan section 5, "Real NLI (replace
`return True`)"). Previously check_entailment() unconditionally returned
True, so every generated claim was reported as faithful regardless of
whether the retrieved context said anything of the kind.

Model choice is a registry key (config/models.yaml, kind="nli"), never a
hardcoded model name -- see registry/models.py and packs/schema.py's
ModelRefs.nli. The active jurisdiction pack isn't yet threaded through
every call site in this codebase, so a caller that wants a specific
model passes model_name explicitly; NLIChecker() with no arguments
resolves the us_contracts pack's chosen NLI model, matching the same
default-pack convention used elsewhere (e.g. ingestion/parser.py's
doc-type patterns).
"""
import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

from sentence_transformers import CrossEncoder

from lexis.config import settings
from lexis.packs.loader import load_pack
from lexis.registry.models import ModelRegistry

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _default_model_name() -> str:
    pack = load_pack(Path("config") / "packs" / "us_contracts.yaml")
    registry = ModelRegistry.from_file(Path("config") / "models.yaml")
    spec = registry.resolve(pack.models.nli, kind="nli")
    return spec.name


class NLIChecker:
    def __init__(self, model_name: Optional[str] = None, model: Optional[CrossEncoder] = None):
        # model= lets a test inject a fake without downloading/loading a real model.
        self.model = model if model is not None else CrossEncoder(model_name or _default_model_name())
        self._entailment_idx = self._resolve_entailment_index(self.model)

    @staticmethod
    def _resolve_entailment_index(model) -> int:
        """NLI checkpoints do not agree on label order (config/models.yaml documents this
        explicitly for nli-deberta-v3-base: some put entailment at index 0, others at 1 or 2)
        -- the index must be read from the model's own config, never assumed fixed."""
        id2label = getattr(getattr(model, "config", None), "id2label", None)
        if not id2label:
            raise ValueError(
                "NLI model has no config.id2label -- cannot determine which output index is "
                "'entailment'. This model is not usable as an NLIChecker backend."
            )
        for idx, label in id2label.items():
            if str(label).strip().lower() == "entailment":
                return int(idx)
        raise ValueError(f"NLI model's id2label has no 'entailment' label: {id2label}")

    def entailment_score(self, premise: str, hypothesis: str) -> float:
        """Probability that `premise` entails `hypothesis`, in [0, 1]."""
        scores = self.model.predict([(premise, hypothesis)], apply_softmax=True)
        return float(scores[0][self._entailment_idx])

    def check_entailment(self, premise: str, hypothesis: str, threshold: Optional[float] = None) -> bool:
        """True if `premise` (e.g. a retrieved chunk) entails `hypothesis` (e.g. one claim
        from a generated answer) with probability >= threshold (default:
        settings.nli_entailment_threshold). A legal RAG system should treat an ambiguous,
        low-confidence claim as unsupported rather than faithful; raise the threshold for a
        stricter guardrail, never lower it without a measured justification (see
        config/models.yaml's per-model accuracy notes)."""
        if threshold is None:
            threshold = settings.nli_entailment_threshold
        return self.entailment_score(premise, hypothesis) >= threshold
