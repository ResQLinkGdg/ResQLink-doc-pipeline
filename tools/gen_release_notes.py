import json
from collections import Counter
from pathlib import Path
from datetime import datetime

PACK_DIR = Path("pack/v1")

def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def main():
    pack_dir = PACK_DIR

    manifest = read_json(pack_dir / "manifest.json") if (pack_dir / "manifest.json").exists() else {}
    meta = read_json(pack_dir / "embeddings_meta.json")
    chunks = list(read_jsonl(pack_dir / "chunks.jsonl"))

    chunk_cnt = len(chunks)

    per_doc_chunks = Counter()
    for c in chunks:
        per_doc_chunks[c.get("docId")] += 1

    total_docs = len(per_doc_chunks)

    created_at = manifest.get("createdAt") or datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    pack_id = manifest.get("packId", "resqlink-v1")
    pack_version = manifest.get("version", "v1")

    lines = []
    lines.append(f"## ResQLink 문서팩 {pack_version}")
    lines.append("")
    lines.append("### 요약")
    lines.append(f"- pack_id: `{pack_id}`")
    lines.append(f"- 생성시각(UTC): `{created_at}`")
    lines.append(f"- 임베딩 모델: `{meta.get('model')}`")
    lines.append(f"- 문서 수: **{total_docs}**, 청크 수: **{chunk_cnt}**")
    lines.append(f"- 임베딩 차원: **{meta.get('dim')}**, dtype: `{meta.get('dtype')}`")
    lines.append("")
    lines.append("### 포함 파일")
    lines.append("- `chunks.jsonl` (근거 텍스트 청크)")
    lines.append("- `embeddings.f16.bin` (청크 임베딩)")
    lines.append("- `embeddings_meta.json` (임베딩 메타)")
    if (pack_dir / "manifest.json").exists():
        lines.append("- `manifest.json` (팩 메타)")
    if (pack_dir / "licenses.json").exists():
        lines.append("- `licenses.json` (라이선스)")
    lines.append("")
    lines.append("### Kotlin 사용 흐름")
    lines.append("1) 앱 assets에 `pack/v1/` 폴더 그대로 추가")
    lines.append("2) 쿼리 임베딩 생성 → embeddings와 유사도 Top-K")
    lines.append("3) Top-K 인덱스로 `chunks.jsonl`에서 근거 텍스트 표시")
    lines.append("")
    lines.append("### 문서별 청크 수")
    for doc_id, cnt in per_doc_chunks.most_common():
        lines.append(f"- `{doc_id}`: {cnt}")

    print("\n".join(lines))

if __name__ == "__main__":
    main()
