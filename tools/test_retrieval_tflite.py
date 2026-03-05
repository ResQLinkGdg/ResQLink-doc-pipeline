"""
v2 문서팩 검색 테스트 (tflite USE-QA 모델 사용)

Docker에서 실행:
  docker run --rm \
    -v "$PWD/pack:/app/pack:ro" \
    -v "$PWD/models:/app/models:ro" \
    -v "$PWD/tools:/app/tools:ro" \
    resqlink-docpipeline --help  # (entrypoint 우회 필요, 아래 참고)
"""
import json
import sys
import numpy as np

PACK_DIR = sys.argv[1] if len(sys.argv) > 1 else "pack/v2"
MODEL_PATH = sys.argv[2] if len(sys.argv) > 2 else "models/universal_sentence_encoder.tflite"
QUERY = sys.argv[3] if len(sys.argv) > 3 else "심정지 환자 심폐소생술 방법"
TOP_K = 5

# 1) 청크 + 임베딩 로드
chunks = [json.loads(l) for l in open(f"{PACK_DIR}/chunks.jsonl", encoding="utf-8")]
meta = json.load(open(f"{PACK_DIR}/embeddings_meta.json", encoding="utf-8"))
count, dim = meta["count"], meta["dim"]

emb = np.fromfile(f"{PACK_DIR}/embeddings.f16.bin", dtype=np.float16).astype(np.float32)
emb = emb.reshape(count, dim)
emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)

print(f"[INFO] chunks={count}, dim={dim}, query=\"{QUERY}\"")

# 2) 쿼리 임베딩 (tflite USE-QA: query_encoding 사용)
try:
    from tflite_support.task import text as ttext
    from tflite_support.task.core import BaseOptions

    options = ttext.TextEmbedderOptions(
        base_options=BaseOptions(file_name=MODEL_PATH, num_threads=4)
    )
    embedder = ttext.TextEmbedder.create_from_options(options)
    r = embedder.embed(QUERY)
    # query_encoding = first embedding, response_encoding = last
    emb_obj = r.embeddings[0]
    fv = emb_obj.feature_vector
    vec = fv.value_float if hasattr(fv, "value_float") else fv.value if hasattr(fv, "value") else fv
    q = np.asarray(vec, dtype=np.float32).reshape(1, -1)
except ImportError:
    import tensorflow_text as tft  # noqa
    import tensorflow as tf

    interp = tf.lite.Interpreter(
        model_path=MODEL_PATH,
        num_threads=4,
        custom_op_registerers=tft.tflite_registrar.SELECT_TFTEXT_OPS,
    )
    interp.allocate_tensors()
    inp = interp.get_input_details()
    out = interp.get_output_details()

    for d in inp:
        interp.resize_tensor_input(d["index"], [1])
    interp.allocate_tensors()

    interp.set_tensor(inp[0]["index"], np.array([QUERY]))       # query
    interp.set_tensor(inp[1]["index"], np.array([""]))           # context
    interp.set_tensor(inp[2]["index"], np.array([""]))           # response
    interp.invoke()

    q = interp.get_tensor(out[0]["index"]).astype(np.float32)   # query_encoding

q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)

# 3) 코사인 유사도 Top-K
scores = emb @ q[0]
top = scores.argsort()[-TOP_K:][::-1]

print(f"\n--- Top {TOP_K} results ---")
for i, idx in enumerate(top, 1):
    c = chunks[int(idx)]
    snippet = c["content"].replace("\n", " ")[:140]
    print(f"{i}. [{scores[int(idx)]:.4f}] doc={c['docId']}  p{c.get('pageStart')}-{c.get('pageEnd')}")
    print(f"   {snippet}...")
    print()
