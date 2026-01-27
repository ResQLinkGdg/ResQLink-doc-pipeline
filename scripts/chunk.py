from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from utils import looks_like_heading, normalize_text, read_jsonl, write_jsonl


def chunk_pages(
    pages: List[Dict[str, Any]],
    chunk_size: int,
    overlap: int,
    min_chars: int
) -> List[Dict[str, Any]]:
    pages.sort(key=lambda r: int(r.get("pageNo", 0)))
    out: List[Dict[str, Any]] = []

    buf = ""
    buf_start_page: Optional[int] = None
    current_section: Optional[str] = None

    def flush(end_page: int, final: bool) -> None:
        nonlocal buf, buf_start_page
        txt = buf.strip()
        if buf_start_page is not None and len(txt) >= min_chars:
            out.append({
                "docId": pages[0]["docId"],
                "docTitle": pages[0].get("docTitle"),
                "section": current_section,
                "pageStart": buf_start_page,
                "pageEnd": end_page,
                "content": txt
            })
        if final:
            buf = ""
            buf_start_page = None
            return
        buf = txt[-overlap:] if overlap > 0 else ""
        buf_start_page = None

    for p in pages:
        page_no = int(p.get("pageNo", 0))
        text = normalize_text(p.get("text", ""))
        if not text:
            continue

        # section heuristic: first line looks like heading
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if lines and looks_like_heading(lines[0]):
            current_section = lines[0]
            text = "\n".join(lines[1:]).strip()

        if not text:
            continue

        if buf_start_page is None:
            buf_start_page = page_no

        addition = f"\n\n[page {page_no}]\n" + text

        # flush if overflow and buffer has something
        if len(buf) + len(addition) > chunk_size and buf.strip():
            flush(end_page=page_no, final=False)
            if buf_start_page is None:
                buf_start_page = page_no

        # if one page is too large and buffer is empty -> hard slicing
        if len(addition) > chunk_size and not buf.strip():
            start = 0
            while start < len(addition):
                end = min(start + chunk_size, len(addition))
                part = addition[start:end].strip()
                if len(part) >= min_chars:
                    out.append({
                        "docId": pages[0]["docId"],
                        "docTitle": pages[0].get("docTitle"),
                        "section": current_section,
                        "pageStart": page_no,
                        "pageEnd": page_no,
                        "content": part
                    })
                if end == len(addition):
                    break
                start = end - overlap if overlap > 0 else end
            buf = ""
            buf_start_page = None
            continue

        buf += addition

        if len(buf) >= chunk_size:
            flush(end_page=page_no, final=False)

    if buf.strip():
        flush(end_page=int(pages[-1].get("pageNo", 0)), final=True)

    # assign chunkId sequentially per doc
    for i, c in enumerate(out, start=1):
        c["chunkId"] = f"{pages[0]['docId']}_c{i:06d}"

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk_size", type=int, default=650)
    ap.add_argument("--overlap", type=int, default=120)
    ap.add_argument("--min_chars", type=int, default=250)
    args = ap.parse_args()

    if args.overlap >= args.chunk_size:
        raise SystemExit("overlap must be < chunk_size (otherwise hard-slicing can loop forever)")


    by_doc: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(args.input):
        by_doc[row["docId"]].append(row)

    all_chunks: List[Dict[str, Any]] = []
    for doc_id, pages in tqdm(by_doc.items(), desc="chunk"):
        print(f"[chunk] doc={doc_id}, pages={len(pages)}")
        all_chunks.extend(chunk_pages(pages, args.chunk_size, args.overlap, args.min_chars))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out, all_chunks)


if __name__ == "__main__":
    main()
