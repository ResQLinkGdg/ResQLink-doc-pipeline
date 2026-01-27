from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

from utils import ensure_dir, safe_doc_id_from_filename, write_jsonl


def list_pdfs(input_dir: Path) -> List[Path]:
    return sorted([p for p in input_dir.glob("**/*.pdf") if p.is_file()])


def extract_pdf_text(pdf_path: Path) -> List[str]:
    # Prefer pdfplumber
    try:
        import pdfplumber  # type: ignore
        out: List[str] = []
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                out.append(page.extract_text() or "")
        return out
    except Exception:
        # Fallback: pypdf
        from pypdf import PdfReader  # type: ignore
        reader = PdfReader(str(pdf_path))
        out = []
        for page in reader.pages:
            out.append(page.extract_text() or "")
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", default="UNKNOWN")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    pdfs = list_pdfs(input_dir)
    if not pdfs:
        raise SystemExit(f"No PDF found under: {input_dir}")

    rows: List[Dict[str, Any]] = []
    for pdf_path in tqdm(pdfs, desc="extract"):
        doc_id = safe_doc_id_from_filename(pdf_path)
        doc_title = pdf_path.stem
        pages = extract_pdf_text(pdf_path)

        for i, text in enumerate(pages, start=1):
            rows.append({
                "docId": doc_id,
                "docTitle": doc_title,
                "source": args.source,
                "file": str(pdf_path.as_posix()),
                "pageNo": i,
                "text": text
            })

    ensure_dir(Path(args.out).parent)
    write_jsonl(args.out, rows)


if __name__ == "__main__":
    main()
