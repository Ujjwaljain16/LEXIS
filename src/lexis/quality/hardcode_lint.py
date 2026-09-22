"""
Zero-hardcoding lint (plan section 3.2).

Scans the source tree with the AST and reports:
  * string literals that look like URLs, model identifiers, or repo ids
  * numeric literals used as tunable parameters: function defaults, call
    keyword arguments, comparison operands, and module-level constants

Both rule sets live in quality/hardcode_rules.json (data, not code).

Ratchet policy: the current violations are recorded per file and rule in
quality/hardcode_baseline.json. The check FAILS if any file exceeds its
baseline count for any rule, or a file with no baseline gains a violation.
Counts may only go down; regenerating the baseline refuses to raise a count
unless explicitly allowed. This turns "we should stop hardcoding" into an
enforced, monotonic property without demanding a big-bang cleanup.
"""
import ast
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Violation:
    file: str
    line: int
    rule: str
    value: str


def load_rules(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _docstring_nodes(tree: ast.AST) -> set:
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                    and isinstance(body[0].value.value, str):
                ids.add(id(body[0].value))
    return ids


def _is_number(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool)


def _signed_value(node: ast.AST):
    if _is_number(node):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)) and _is_number(node.operand):
        return -node.operand.value if isinstance(node.op, ast.USub) else node.operand.value
    return None


def scan_source(source: str, rel_path: str, rules: dict) -> List[Violation]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    allowed = set(rules.get("allowed_numbers", []))
    contexts = set(rules.get("numeric_contexts", []))
    patterns = {name: re.compile(p) for name, p in rules.get("string_patterns", {}).items()}
    docstrings = _docstring_nodes(tree)
    found: List[Violation] = []

    def numeric(node: ast.AST, context: str):
        if context not in contexts:
            return
        val = _signed_value(node)
        if val is not None and val not in allowed:
            found.append(Violation(rel_path, node.lineno, f"numeric:{context}", repr(val)))

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
            for name, pat in patterns.items():
                if pat.search(node.value):
                    found.append(Violation(rel_path, node.lineno, f"string:{name}", node.value[:60]))
                    break
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for d in [*node.args.defaults, *[k for k in node.args.kw_defaults if k is not None]]:
                numeric(d, "function_default")
        elif isinstance(node, ast.Call):
            for kw in node.keywords:
                numeric(kw.value, "call_keyword")
        elif isinstance(node, ast.Compare):
            for c in node.comparators:
                numeric(c, "comparison")
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id.isupper() for t in node.targets):
            numeric(node.value, "module_constant")
    return found


def scan_tree(repo_root: Path, rules: dict) -> List[Violation]:
    src_root = repo_root / rules["source_root"]
    excludes = rules.get("exclude_paths", [])
    out: List[Violation] = []
    for path in sorted(src_root.rglob("*.py")):
        rel = path.relative_to(src_root).as_posix()
        if any(rel == e or (e.endswith("/") and rel.startswith(e)) for e in excludes):
            continue
        out.extend(scan_source(path.read_text(encoding="utf-8"), rel, rules))
    return out


def summarize(violations: List[Violation]) -> Dict[str, Dict[str, int]]:
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for v in violations:
        counts[v.file][v.rule] += 1
    return {f: dict(sorted(r.items())) for f, r in sorted(counts.items())}


def compare_to_baseline(current: Dict[str, Dict[str, int]], baseline: Dict[str, Dict[str, int]]) -> List[str]:
    """Human-readable regressions: any (file, rule) count above its baseline."""
    problems = []
    for f, rules in current.items():
        for rule, n in rules.items():
            allowed = baseline.get(f, {}).get(rule, 0)
            if n > allowed:
                problems.append(f"{f}: {rule} count {n} exceeds baseline {allowed}")
    return problems


def total(summary: Dict[str, Dict[str, int]]) -> int:
    return sum(n for rules in summary.values() for n in rules.values())


def write_baseline(current: Dict[str, Dict[str, int]], path: Path, existing: Optional[dict] = None,
                   allow_increase: bool = False) -> None:
    if existing and not allow_increase:
        problems = compare_to_baseline(current, existing)
        if problems:
            raise ValueError("refusing to raise the baseline:\n" + "\n".join(problems))
    path.write_text(json.dumps(current, indent=2, sort_keys=True), encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Zero-hardcoding ratchet lint.")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--rules", default="quality/hardcode_rules.json")
    ap.add_argument("--baseline", default="quality/hardcode_baseline.json")
    ap.add_argument("--update-baseline", action="store_true", help="rewrite baseline (only allowed to lower counts)")
    ap.add_argument("--allow-increase", action="store_true")
    args = ap.parse_args(argv)

    root = Path(args.repo_root)
    rules = load_rules(root / args.rules)
    current = summarize(scan_tree(root, rules))
    baseline_path = root / args.baseline
    baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else None

    if args.update_baseline:
        write_baseline(current, baseline_path, baseline, args.allow_increase)
        print(f"baseline written: {total(current)} violations across {len(current)} files")
        return 0
    problems = compare_to_baseline(current, baseline or {})
    print(f"violations: {total(current)} (baseline {total(baseline or {})})")
    for p in problems:
        print("REGRESSION:", p)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
