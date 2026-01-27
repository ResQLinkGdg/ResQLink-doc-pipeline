from __future__ import annotations

import argparse
import json
import sys
import runpy
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from utils import ensure_dir, read_jsonl, byte_size_for_f16, safe_doc_id_from_filename

SCRIPTS_DIR = Path(__file__).resolve().parent


def run_step(script_name: str, argv: List[str]) -> None:
    script_path = SCRIPTS_DIR / f"{script_name}.py"
    if not script_path.exists():
        raise SystemExit(f"Missing script: {script_path}")

    old_argv = sys.argv
    # ensure scripts dir import works (utils)
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))

    try:
        sys.argv = [str(script_path)] + argv
        runpy.run_path(str(script_path), run_name="__main__")
    finally:
        sys.argv = old_argv


def count_jsonl_lines(path: Path) -> int:
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def validate_pack(pack_dir: Path) -> None:
    chunks_path = pack_dir / "chunks.jsonl"
    meta_path = pack_dir / "embeddings_meta.json"
    bin_path = pack_dir / "embeddings.f16.bin"

    if not chunks_path.exists():
        raise SystemExit("Missing pack chunks.jsonl")
    if not meta_path.exists():
        raise SystemExit("Missing pack embeddings_meta.json")
    if not bin_path.exists():
        raise SystemExit("Missing pack embeddings.f16.bin")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    count = int(meta["count"])
    dim = int(meta["dim"])

    expected_size = byte_size_for_f16(count, dim)
    actual_size = bin_path.stat().st_size
    if actual_size != expected_size:
        raise SystemExit(f"embeddings.f16.bin size mismatch: {actual_size} != {expected_size}")

    line_count = count_jsonl_lines(chunks_path)
    if line_count != count:
        raise SystemExit(f"chunks.jsonl line mismatch: {line_count} != meta.count({count})")

    # basic chunk schema validation (first 50)
    required = {"chunkId", "docId", "content"}
    checked = 0
    for row in read_jsonl(chunks_path):
        missing = required - set(row.keys())
        if missing:
            raise SystemExit(f"Chunk missing fields {missing}: {row.get('chunkId')}")
        checked += 1
        if checked >= 50:
            break


def generate_docs_from_input_pdfs(input_dir: Path) -> List[Dict[str, Any]]:
    docs = []
    for pdf in sorted(input_dir.glob("**/*.pdf")):
        if not pdf.is_file():
            continue
        doc_id = safe_doc_id_from_filename(pdf)
        docs.append({
            "docId": doc_id,
            "title": pdf.stem,
            "publisher": "UNKNOWN",
            "license": "UNKNOWN",
            "sourceUrl": ""
        })
    return docs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--work_dir", required=True)
    ap.add_argument("--pack_dir", required=True)
    ap.add_argument("--pack_id", required=True)
    ap.add_argument("--pack_version", default="v1")
    ap.add_argument("--source", default="UNKNOWN")
    ap.add_argument("--embed_model", required=True)
    ap.add_argument("--chunk_size", type=int, default=650)
    ap.add_argument("--overlap", type=int, default=120)
    ap.add_argument("--min_chars", type=int, default=250)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--skip_embed", action="store_true")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    work_dir = Path(args.work_dir)
    pack_dir = Path(args.pack_dir)

    ensure_dir(work_dir)
    ensure_dir(pack_dir)

    extracted = work_dir / "extracted_pages.jsonl"
    cleaned = work_dir / "cleaned_pages.jsonl"
    chunks_stage = work_dir / "chunks_stage.jsonl"

    # 1) Extract -> Clean -> Chunk
    run_step("extract", ["--input_dir", str(input_dir), "--out", str(extracted), "--source", args.source])
    run_step("clean", ["--input", str(extracted), "--out", str(cleaned)])
    run_step("chunk", [
        "--input", str(cleaned),
        "--out", str(chunks_stage),
        "--chunk_size", str(args.chunk_size),
        "--overlap", str(args.overlap),
        "--min_chars", str(args.min_chars),
    ])

    # 2) Final chunks to pack
    pack_chunks = pack_dir / "chunks.jsonl"
    shutil.copyfile(chunks_stage, pack_chunks)

    # 3) Embeddings
    if not args.skip_embed:
        out_bin = pack_dir / "embeddings.f16.bin"
        out_meta = pack_dir / "embeddings_meta.json"
        embed_argv = [
            "--input", str(pack_chunks),
            "--out_bin", str(out_bin),
            "--out_meta", str(out_meta),
            "--model", args.embed_model,
            "--batch_size", str(args.batch_size),
        ]
        if args.normalize:
            embed_argv.append("--normalize")
        run_step("embed", embed_argv)
    else:
        print("[WARN] skip_embed enabled - embeddings will not be generated.")

    # 4) licenses + manifest
    docs = generate_docs_from_input_pdfs(input_dir)
    (pack_dir / "licenses.json").write_text(json.dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8")

    # manifest: read meta if exists
    meta_path = pack_dir / "embeddings_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        dim = int(meta.get("dim", 0))
        dtype = meta.get("dtype", "f16")
        normalized = bool(meta.get("normalized", False))
    else:
        dim, dtype, normalized = 0, "", False

    manifest = {
        "packId": args.pack_id,
        "version": args.pack_version,
        "createdAt": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "embedding": {
            "model": args.embed_model,
            "dim": dim,
            "dtype": dtype,
            "normalized": normalized
        },
        "chunking": {
            "chunkSize": args.chunk_size,
            "overlap": args.overlap,
            "minChars": args.min_chars
        },
        "docs": docs
    }
    (pack_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # 5) Validate
    if not args.skip_embed:
        validate_pack(pack_dir)
        print("[OK] Pack validated.")
    print(f"[DONE] Pack generated at: {pack_dir}")


if __name__ == "__main__":
    main()
