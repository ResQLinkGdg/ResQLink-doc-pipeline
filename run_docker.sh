#!/usr/bin/env bash
# ResQLink doc-pipeline: Docker로 문서팩 생성
#
# 사용법:
#   ./run_docker.sh
#
# 필요 마운트:
#   input/pdf/   - 원본 PDF 파일
#   models/      - universal_sentence_encoder.tflite
#
# 결과물:
#   pack/v1/     - chunks.jsonl, embeddings.f16.bin, manifest.json 등

set -euo pipefail

IMAGE_NAME="resqlink-docpipeline"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Docker 이미지 빌드 (캐시 활용)
echo "[1/2] Building Docker image..."
docker build -t "$IMAGE_NAME" "$SCRIPT_DIR"

# 파이프라인 실행
echo "[2/2] Running pipeline..."
docker run --rm \
  -v "$SCRIPT_DIR/input:/app/input:ro" \
  -v "$SCRIPT_DIR/models:/app/models:ro" \
  -v "$SCRIPT_DIR/pack:/app/pack" \
  -v "$SCRIPT_DIR/work:/app/work" \
  "$IMAGE_NAME" \
  --input_dir input/pdf \
  --work_dir work \
  --pack_dir pack/v1 \
  --pack_id firstaid_kr_v1_use \
  --embed_model models/universal_sentence_encoder.tflite \
  --batch_size 32 \
  --num_threads 4 \
  --normalize

echo "[DONE] Output: pack/v1/"
