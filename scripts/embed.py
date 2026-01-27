from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from tqdm import tqdm

from utils import byte_size_for_f16, ensure_dir, read_jsonl, sha256_file


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out_bin", required=True)
    ap.add_argument("--out_meta", required=True)
    ap.add_argument("--model", required=True, help="sentence-transformers model name/path")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--normalize", action="store_true")
    args = ap.parse_args()

    chunks: List[Dict[str, Any]] = list(read_jsonl(args.input))
    if not chunks:
        raise SystemExit("No chunks found. Run chunk step first.")

    from sentence_transformers import SentenceTransformer  # type: ignore
    model = SentenceTransformer(args.model)

    texts = [c.get("content", "") for c in chunks]

    emb_batches: List[np.ndarray] = []
    for i in tqdm(range(0, len(texts), args.batch_size), desc="embed"):
        batch = texts[i:i + args.batch_size]
        emb = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        if emb.dtype != np.float32:
            emb = emb.astype(np.float32)
        emb_batches.append(emb)

    embs = np.vstack(emb_batches)
    if args.normalize:
        embs = l2_normalize(embs)

    count, dim = embs.shape

    out_bin = Path(args.out_bin)
    ensure_dir(out_bin.parent)

    # write float16 binary
    embs_f16 = embs.astype(np.float16)
    with open(out_bin, "wb") as f:
        f.write(embs_f16.tobytes(order="C"))

    meta = {
        "dtype": "f16",
        "dim": int(dim),
        "count": int(count),
        "normalized": bool(args.normalize),
        "model": args.model,
        "order": "chunks.jsonl line order",
        "fileSize": int(out_bin.stat().st_size),
        "expectedFileSize": int(byte_size_for_f16(int(count), int(dim))),
        "sha256": sha256_file(out_bin),
    }
    meta["fileSizeOk"] = (meta["fileSize"] == meta["expectedFileSize"])

    out_meta = Path(args.out_meta)
    ensure_dir(out_meta.parent)
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    if not meta["fileSizeOk"]:
        raise SystemExit(
            f"Embedding bin size mismatch: got {meta['fileSize']} expected {meta['expectedFileSize']}"
        )


if __name__ == "__main__":
    main()
