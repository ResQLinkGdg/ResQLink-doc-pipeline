"""v1 vs v2 문서팩 검색 품질 비교"""
import json
import numpy as np
from sentence_transformers import SentenceTransformer

QUERIES = [
    "심정지 환자 심폐소생술 방법",
    "화상 응급처치",
    "골절 부목 고정법",
]
TOP_K = 5


def load_pack(pack_dir):
    chunks = [json.loads(l) for l in open(f"{pack_dir}/chunks.jsonl", encoding="utf-8")]
    meta = json.load(open(f"{pack_dir}/embeddings_meta.json", encoding="utf-8"))
    count, dim = meta["count"], meta["dim"]
    emb = np.fromfile(f"{pack_dir}/embeddings.f16.bin", dtype=np.float16).astype(np.float32)
    emb = emb.reshape(count, dim)
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    return chunks, emb, meta


def search(chunks, emb, q_vec, top_k=TOP_K):
    scores = emb @ q_vec
    top = scores.argsort()[-top_k:][::-1]
    results = []
    for idx in top:
        c = chunks[int(idx)]
        results.append((float(scores[int(idx)]), c))
    return results


# Load v1
print("=" * 70)
print("Loading v1 (intfloat/multilingual-e5-small, dim=384)")
print("=" * 70)
chunks_v1, emb_v1, meta_v1 = load_pack("pack/v1")
model_v1 = SentenceTransformer(meta_v1["model"])

# v2는 tflite라 여기서 쿼리 임베딩 불가 → v2 결과는 이미 위에서 확인함
# v1만 테스트

for q in QUERIES:
    print(f"\n--- Query: \"{q}\" ---")
    # e5 모델은 query prefix 필요
    q_text = f"query: {q}" if "e5" in meta_v1["model"] else q
    q_vec = model_v1.encode([q_text], convert_to_numpy=True).astype(np.float32)
    q_vec = q_vec / (np.linalg.norm(q_vec, axis=1, keepdims=True) + 1e-12)

    results = search(chunks_v1, emb_v1, q_vec[0])
    for i, (score, c) in enumerate(results, 1):
        snippet = c["content"].replace("\n", " ")[:120]
        print(f"  {i}. [{score:.4f}] doc={c['docId']}  p{c.get('pageStart')}-{c.get('pageEnd')}")
        print(f"     {snippet}...")
