# DOCPACK (Offline RAG Document Pack Builder)

공식 PDF(텍스트 PDF)를 **오프라인 RAG용 문서팩(docpack)** 으로 변환하는 파이프라인입니다.

## 폴더 구조
- `input/pdf/` : 원본 PDF 넣는 곳
- `work/` : 중간 산출물 (페이지 추출/정제/청킹 stage)
- `pack/v1/` : 최종 산출물 (앱에 넣는 파일들)
- `scripts/` : 파이프라인 스크립트

## 최종 산출물 (pack/v1)
- `manifest.json` : 문서팩 메타(팩ID, 버전, embedding/chunking 스펙, 문서 목록)
- `licenses.json` : 출처/라이선스 표기
- `chunks.jsonl` : 1줄=1청크(메타+텍스트)
- `embeddings.f16.bin` : float16 임베딩 바이너리 (chunks.jsonl 라인 순서와 1:1)
- `embeddings_meta.json` : count/dim/dtype/sha256 등 로딩용 메타

## 설치
```bash
.\.venv\Scripts\Activate.ps1
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
