from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

WHITESPACE_RE = re.compile(r"\s+")
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")

def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)

def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def read_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = text.replace("\u00a0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = WHITESPACE_RE.sub(" ", text)
    text = text.replace(" \n ", "\n").replace(" \n", "\n").replace("\n ", "\n")
    text = MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()

def split_lines(text: str) -> List[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip()]

def remove_repeating_headers_footers(
    pages: List[str],
    top_n: int = 3,
    bottom_n: int = 3,
    min_ratio: float = 0.6
) -> List[str]:
    if not pages:
        return pages
    total = len(pages)

    top_counts: Dict[str, int] = {}
    bottom_counts: Dict[str, int] = {}
    per_lines = [split_lines(p) for p in pages]

    for lines in per_lines:
        for ln in lines[:top_n]:
            top_counts[ln] = top_counts.get(ln, 0) + 1
        for ln in lines[-bottom_n:]:
            bottom_counts[ln] = bottom_counts.get(ln, 0) + 1

    top_drop = {ln for ln, c in top_counts.items() if c / total >= min_ratio}
    bottom_drop = {ln for ln, c in bottom_counts.items() if c / total >= min_ratio}

    cleaned: List[str] = []
    for lines in per_lines:
        new_lines: List[str] = []
        for i, ln in enumerate(lines):
            if i < top_n and ln in top_drop:
                continue
            if i >= max(0, len(lines) - bottom_n) and ln in bottom_drop:
                continue
            new_lines.append(ln)
        cleaned.append("\n".join(new_lines))
    return cleaned

def safe_doc_id_from_filename(path: str | Path) -> str:
    orig = Path(path).stem  # 원본(한글 포함)
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", orig).strip("_")
    if not safe:
        safe = "doc"
    h = hashlib.sha256(orig.encode("utf-8")).hexdigest()[:8]
    base = safe[:71]  # 71 + 1 + 8 = 80
    return f"{base}_{h}"


def looks_like_heading(line: str) -> bool:
    if not line:
        return False
    if len(line) > 24:
        return False
    if re.match(r"^\d+\s*[\.\)]\s*.+", line):
        return True
    if re.search(r"[.!?]", line):
        return False
    if re.match(r"^[가-힣A-Za-z0-9\s·-]+$", line) and re.search(r"[가-힣]", line):
        return True
    return False

def byte_size_for_f16(count: int, dim: int) -> int:
    return count * dim * 2  # float16 = 2 bytes
