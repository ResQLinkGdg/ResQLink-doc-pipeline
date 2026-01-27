import json
import numpy as np
from sentence_transformers import SentenceTransformer

chunks = [json.loads(l) for l in open("pack/v1/chunks.jsonl", encoding="utf-8")]
meta = json.load(open("pack/v1/embeddings_meta.json", encoding="utf-8"))
count, dim = meta["count"], meta["dim"]

emb = np.fromfile("pack/v1/embeddings.f16.bin", dtype=np.float16).astype(np.float32)
emb = emb.reshape(count, dim)
emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)

model = SentenceTransformer(meta["model"])
query = "심정지 환자 심폐소생술 방법"
q = model.encode([query], convert_to_numpy=True).astype(np.float32)
q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)

scores = emb @ q[0]
top = scores.argsort()[-5:][::-1]

for i, idx in enumerate(top, 1):
    c = chunks[int(idx)]
    snippet = c["content"].replace("\n", " ")[:140]
    print(i, f"{scores[int(idx)]:.4f}", c["docId"], f"p{c.get('pageStart')}-{c.get('pageEnd')}", snippet)
