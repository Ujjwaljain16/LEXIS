"""
Run provenance for evaluation artifacts (plan P0).

Every result JSON should embed build_provenance(...) so a number can always
be traced to: the fully resolved config, the code version (git SHA when
available), the exact bytes of the data files it ran on, and library
versions. Secrets are redacted before anything is recorded -- an artifact
must never contain an API key or a credentialed connection string.
"""
import hashlib
import json
import re
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

_SECRET_KEY_RE = re.compile(r"(api[_-]?key|secret|token|password|passwd|credential|private[_-]?key)", re.I)
_CRED_URL_RE = re.compile(r"(?P<scheme>[A-Za-z][A-Za-z0-9+.-]*://)[^/@\s:]+:[^/@\s]+@")
REDACTED = "***REDACTED***"


def redact_secrets(obj: Any) -> Any:
    """Recursively masks values under secret-looking keys and credentials
    embedded in URLs (user:pass@host)."""
    if isinstance(obj, Mapping):
        return {k: (REDACTED if _SECRET_KEY_RE.search(str(k)) and v else redact_secrets(v)) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [redact_secrets(v) for v in obj]
    if isinstance(obj, str):
        return _CRED_URL_RE.sub(lambda m: f"{m.group('scheme')}{REDACTED}@", obj)
    return obj


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def config_hash(config: Mapping[str, Any]) -> str:
    """Hash of the REDACTED, canonicalised config, so hashes are stable and
    never depend on secret values."""
    return hashlib.sha256(canonical_json(redact_secrets(config)).encode("utf-8")).hexdigest()


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_sha(repo_dir: Optional[str] = None) -> Optional[str]:
    """Current commit SHA, or None if not a git repo / git unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else None
    except (OSError, subprocess.SubprocessError):
        return None


def package_versions(names: Iterable[str]) -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def build_provenance(
    config: Mapping[str, Any],
    data_paths: Iterable[str] = (),
    packages: Iterable[str] = (),
    repo_dir: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    data_hashes = {str(p): file_sha256(str(p)) for p in data_paths if Path(p).is_file()}
    missing = [str(p) for p in data_paths if not Path(p).is_file()]
    return {
        "config": redact_secrets(dict(config)),
        "config_hash": config_hash(config),
        "git_sha": git_sha(repo_dir),
        "data_sha256": data_hashes,
        "data_missing": missing,
        "package_versions": package_versions(packages),
        "extra": redact_secrets(dict(extra)) if extra else {},
    }
