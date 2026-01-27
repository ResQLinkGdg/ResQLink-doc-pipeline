from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

from utils import normalize_text, read_jsonl, remove_repeating_headers_footers, write_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top_n", type=int, default=3)
    ap.add_argument("--bottom_n", type=int, default=3)
    ap.add_argument("--min_ratio", type=float, default=0.6)
    args = ap.parse_args()

    by_doc: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(args.input):
        by_doc[row["docId"]].append(row)

    cleaned_rows: List[Dict[str, Any]] = []
    for _, pages in tqdm(by_doc.items(), desc="clean"):
        pages.sort(key=lambda r: int(r.get("pageNo", 0)))

        raw = [normalize_text(p.get("text", "")) for p in pages]
        stripped = remove_repeating_headers_footers(
            raw,
            top_n=args.top_n,
            bottom_n=args.bottom_n,
            min_ratio=args.min_ratio
        )

        for p, text in zip(pages, stripped):
            cleaned_rows.append({
                "docId": p["docId"],
                "docTitle": p.get("docTitle"),
                "source": p.get("source"),
                "file": p.get("file"),
                "pageNo": p.get("pageNo"),
                "text": text
            })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out, cleaned_rows)


if __name__ == "__main__":
    main()
