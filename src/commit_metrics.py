import math
import re
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Set

import lizard

PYTHON_COMMENT_RE = re.compile(r"^\s*#")
JAVA_COMMENT_RE = re.compile(r"^\s*//")
TOKEN_RE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*|\d+\.\d+|\d+|==|!=|<=|>=|&&|\|\||<<|>>|->|::|[^\s]"
)
OPERATORS = {
    "+",
    "-",
    "*",
    "/",
    "%",
    "=",
    "==",
    "!=",
    "<",
    ">",
    "<=",
    ">=",
    "&&",
    "||",
    "!",
    "&",
    "|",
    "^",
    "~",
    "<<",
    ">>",
    "+=",
    "-=",
    "*=",
    "/=",
    "%=",
    "++",
    "--",
    "?",
    ":",
    "->",
    "::",
}


def supported_commit_feature_names() -> Set[str]:
    """Return the set of PROMISE-style feature names supported by commit-level proxies."""
    return set(_feature_alias_map().keys())


def _feature_alias_map() -> Dict[str, str]:
    return {
        "loc": "loc",
        "lOCode": "lOCode",
        "lOBlank": "lOBlank",
        "lOComment": "lOComment",
        "locCodeAndComment": "locCodeAndComment",
        "branchCount": "branchCount",
        "v(g)": "v_g",
        "ev(g)": "ev_g",
        "iv(g)": "iv_g",
        "uniq_Op": "uniq_Op",
        "uniq_Opnd": "uniq_Opnd",
        "total_Op": "total_Op",
        "total_Opnd": "total_Opnd",
        "n": "n",
        "n1": "n1",
        "n2": "n2",
        "N": "N",
        "N1": "N1",
        "N2": "N2",
        "v": "v",
        "d": "d",
        "e": "e",
        "i": "i",
        "l": "l",
        "b": "b",
        "t": "t",
    }


def _count_comment_lines(lines: List[str], suffix: str) -> int:
    in_block_comment = False
    count = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        if suffix == ".py":
            if PYTHON_COMMENT_RE.match(line):
                count += 1
            continue

        if in_block_comment:
            count += 1
            if "*/" in stripped:
                in_block_comment = False
            continue

        if stripped.startswith("/*"):
            count += 1
            if "*/" not in stripped:
                in_block_comment = True
            continue

        if JAVA_COMMENT_RE.match(line) or stripped.startswith("*"):
            count += 1

    return count


def _tokenize_source(source: str) -> Iterable[str]:
    return TOKEN_RE.findall(source)


def _extract_halstead_like_metrics(tokens: Iterable[str]) -> Dict[str, float]:
    operators = [token for token in tokens if token in OPERATORS]
    operands = [token for token in tokens if token not in OPERATORS and not token.isspace()]

    uniq_operators = set(operators)
    uniq_operands = set(operands)

    total_op = len(operators)
    total_opnd = len(operands)
    uniq_op = len(uniq_operators)
    uniq_opnd = len(uniq_operands)

    vocabulary = uniq_op + uniq_opnd
    length = total_op + total_opnd
    volume = length * math.log2(vocabulary) if vocabulary > 1 and length > 0 else 0.0
    difficulty = (
        (uniq_op / 2.0) * (total_opnd / max(uniq_opnd, 1))
        if uniq_op > 0 and uniq_opnd > 0
        else 0.0
    )
    effort = difficulty * volume
    level = 1.0 / difficulty if difficulty > 0 else 0.0
    intelligence = volume / difficulty if difficulty > 0 else 0.0
    bugs = volume / 3000.0 if volume > 0 else 0.0
    time_required = effort / 18.0 if effort > 0 else 0.0

    return {
        "uniq_Op": float(uniq_op),
        "uniq_Opnd": float(uniq_opnd),
        "total_Op": float(total_op),
        "total_Opnd": float(total_opnd),
        "n": float(vocabulary),
        "n1": float(uniq_op),
        "n2": float(uniq_opnd),
        "N": float(length),
        "N1": float(total_op),
        "N2": float(total_opnd),
        "v": float(volume),
        "d": float(difficulty),
        "e": float(effort),
        "i": float(intelligence),
        "l": float(level),
        "b": float(bugs),
        "t": float(time_required),
    }


def extract_file_metrics(path: Path) -> Dict[str, float]:
    """
    Extract commit-level proxy metrics for a source file.

    Supported by lizard for C/C++, Java, and Python.
    """
    path = Path(path)
    source = path.read_text(encoding="utf-8", errors="ignore")
    lines = source.splitlines()
    blank_lines = sum(1 for line in lines if not line.strip())
    comment_lines = _count_comment_lines(lines, path.suffix.lower())

    file_info = next(iter(lizard.analyze([str(path)])), None)
    if file_info is None:
        raise ValueError(f"Could not analyze file with lizard: {path}")

    total_cc = float(getattr(file_info, "CCN", 0) or 0)
    avg_cc = float(getattr(file_info, "average_cyclomatic_complexity", 0) or 0)
    token_count = float(getattr(file_info, "token_count", 0) or 0)
    nloc = float(getattr(file_info, "nloc", 0) or 0)

    tokens = list(_tokenize_source(source))
    halstead_like = _extract_halstead_like_metrics(tokens)

    raw_metrics = {
        "loc": nloc,
        "lOCode": nloc,
        "lOBlank": float(blank_lines),
        "lOComment": float(comment_lines),
        "locCodeAndComment": nloc + float(comment_lines),
        "branchCount": max(total_cc - 1.0, 0.0),
        "v_g": total_cc,
        "ev_g": max(total_cc - 1.0, 0.0),
        "iv_g": max(avg_cc, 0.0),
        **halstead_like,
        "token_count": token_count,
    }

    alias_map = _feature_alias_map()
    return {feature_name: raw_metrics[raw_key] for feature_name, raw_key in alias_map.items()}


def extract_source_metrics(source_code: str, suffix: str = ".c") -> Dict[str, float]:
    """
    Extract commit-level proxy metrics directly from source text.

    The suffix controls language parsing behavior (for example, .c, .cpp, .java, .py).
    """
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=suffix,
        delete=False,
    ) as tmp_file:
        tmp_file.write(source_code)
        temp_path = Path(tmp_file.name)

    try:
        return extract_file_metrics(temp_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def coverage_for_features(feature_names: Iterable[str]) -> Dict[str, List[str]]:
    """Return which requested features are supported by the commit-level proxy layer."""
    supported = supported_commit_feature_names()
    requested = list(feature_names)
    return {
        "supported": sorted([name for name in requested if name in supported]),
        "missing": sorted([name for name in requested if name not in supported]),
    }
