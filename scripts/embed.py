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


def embed_with_tflite(model_path: Path, texts: List[str], num_threads: int, batch_size: int) -> np.ndarray:
    """
    USE-QA .tflite 모델의 response encoding을 사용해 임베딩을 생성합니다.
    tflite-support 0.4+ (Linux) 또는 tensorflow + tensorflow-text 환경이 필요합니다.

    모델 구조 (USE-QA on-device):
      - Input 0: query text
      - Input 1: response context
      - Input 2: response text
      - Output 0: query_encoding  [N, 100]
      - Output 1: response_encoding [N, 100]

    문서 청크는 response_encoding으로 임베딩합니다.
    """
    if not model_path.exists():
        raise SystemExit(f"TFLite model not found: {model_path}")

    # tensorflow-text 커스텀 op 등록 후 tf.lite.Interpreter 사용
    try:
        import tensorflow_text as tft  # noqa: F401 – registers SentencePiece ops
        import tensorflow as tf

        interp = tf.lite.Interpreter(
            model_path=str(model_path),
            num_threads=num_threads,
            custom_op_registerers=tft.tflite_registrar.SELECT_TFTEXT_OPS,
        )
    except (ImportError, TypeError):
        # fallback: tflite-support Task Library (Linux tflite-support>=0.4)
        try:
            from tflite_support.task import text as ttext
            from tflite_support.task.core import BaseOptions
        except ImportError as e:
            raise SystemExit(
                "Cannot load TFLite model. Need one of:\n"
                "  1) tensorflow + tensorflow-text  (pip install tensorflow tensorflow-text)\n"
                "  2) tflite-support>=0.4           (pip install tflite-support)\n"
                f"Original error: {e}"
            )

        return _embed_with_tflite_support(model_path, texts, num_threads, ttext, BaseOptions)

    interp.allocate_tensors()
    inp = interp.get_input_details()
    out = interp.get_output_details()

    # response_encoding 출력 인덱스 찾기
    response_out_idx = None
    for d in out:
        if "Result" in d["name"] or "response" in d["name"].lower():
            response_out_idx = d["index"]
            break
    if response_out_idx is None:
        response_out_idx = out[-1]["index"]  # fallback: 마지막 출력

    embs: List[np.ndarray] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="embed(tflite)"):
        batch = texts[i : i + batch_size]
        bs = len(batch)

        for d in inp:
            interp.resize_tensor_input(d["index"], [bs])
        interp.allocate_tensors()

        query_arr = np.array([""] * bs)
        context_arr = np.array([""] * bs)
        response_arr = np.array(batch)

        interp.set_tensor(inp[0]["index"], query_arr)
        interp.set_tensor(inp[1]["index"], context_arr)
        interp.set_tensor(inp[2]["index"], response_arr)
        interp.invoke()

        emb = interp.get_tensor(response_out_idx).astype(np.float32)
        embs.append(emb)

    return np.vstack(embs).astype(np.float32)


def _embed_with_tflite_support(model_path, texts, num_threads, ttext, BaseOptions):
    """tflite-support Task Library 기반 fallback (Linux tflite-support>=0.4)."""
    options = ttext.TextEmbedderOptions(
        base_options=BaseOptions(file_name=str(model_path), num_threads=num_threads)
    )
    embedder = ttext.TextEmbedder.create_from_options(options)

    embs: List[np.ndarray] = []
    for s in tqdm(texts, desc="embed(tflite-support)"):
        r = embedder.embed(s)
        if not hasattr(r, "embeddings") or not r.embeddings:
            raise SystemExit("TextEmbedder returned no embeddings.")

        emb_obj = r.embeddings[-1]
        vec = None
        if hasattr(emb_obj, "feature_vector"):
            fv = emb_obj.feature_vector
            # FeatureVector 객체에서 실제 값 추출
            if hasattr(fv, "value_float"):
                vec = fv.value_float
            elif hasattr(fv, "value"):
                vec = fv.value
            else:
                vec = fv
        elif hasattr(emb_obj, "embedding"):
            vec = emb_obj.embedding
        else:
            raise SystemExit("Unsupported embedding object structure from tflite-support.")

        vec_np = np.asarray(vec, dtype=np.float32)
        if vec_np.ndim != 1:
            vec_np = vec_np.reshape(-1).astype(np.float32)
        embs.append(vec_np)

    if not embs:
        raise SystemExit("No embeddings computed.")
    return np.vstack(embs).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="chunks.jsonl path")
    ap.add_argument("--out_bin", required=True, help="output embeddings.f16.bin path")
    ap.add_argument("--out_meta", required=True, help="output embeddings_meta.json path")
    ap.add_argument(
        "--model",
        required=True,
        help="Embedding model: .tflite path (USE-QA) or sentence-transformers model name",
    )
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_threads", type=int, default=4, help="TFLite interpreter threads")
    ap.add_argument("--normalize", action="store_true", help="L2 normalize embeddings")
    args = ap.parse_args()

    chunks: List[Dict[str, Any]] = list(read_jsonl(args.input))
    if not chunks:
        raise SystemExit("No chunks found. Run chunk step first.")

    texts = [c.get("content", "") for c in chunks]

    model_arg = args.model.strip()
    if model_arg.lower().endswith(".tflite"):
        embs = embed_with_tflite(Path(model_arg), texts, args.num_threads, args.batch_size)
    else:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise SystemExit(
                "sentence-transformers is not installed.\n"
                "  pip install sentence-transformers\n"
                f"Original error: {e}"
            )
        model = SentenceTransformer(model_arg)
        emb_batches: List[np.ndarray] = []
        for i in tqdm(range(0, len(texts), args.batch_size), desc="embed(st)"):
            batch = texts[i : i + args.batch_size]
            emb = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
            emb = np.asarray(emb, dtype=np.float32)
            if emb.ndim == 1:
                emb = emb[None, :]
            emb_batches.append(emb)
        embs = np.vstack(emb_batches).astype(np.float32)

    if args.normalize:
        embs = l2_normalize(embs)

    count, dim = embs.shape

    out_bin = Path(args.out_bin)
    ensure_dir(out_bin.parent)

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
    meta["fileSizeOk"] = meta["fileSize"] == meta["expectedFileSize"]

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
