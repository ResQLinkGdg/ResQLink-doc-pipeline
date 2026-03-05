
```md
# RESQLINK-DOC-PIPELINE

PDF 문서를 **오프라인 RAG 문서팩(docpack)** 형태로 변환하는 파이프라인입니다.  
최종 산출물은 `pack/v1/` 아래에 생성되며, 앱(예: Kotlin Android)에서 그대로 포함/배포할 수 있게 구성합니다.

---

## 폴더 구조

```

RESQLINK-DOC-PIPELINE/
├─ pack/
│  └─ v1/
│     ├─ chunks.jsonl
│     ├─ embeddings_meta.json
│     ├─ licenses.json
│     └─ manifest.json
├─ scripts/
│  ├─ build.py
│  ├─ chunk.py
│  ├─ clean.py
│  ├─ embed.py
│  ├─ extract.py
│  └─ utils.py
├─ tools/
│  ├─ gen_release_notes.py
│  └─ test_retrieval.py
├─ .gitignore
├─ README.md
├─ RELEASE_NOTES.md
└─ requirements.txt

````

> 참고: 원본 PDF 입력 폴더는 기본적으로 `input/pdf` 같은 경로를 사용하도록 설계하는 걸 권장합니다.  
> (폴더가 없으면 만들어서 넣으면 됩니다.)

---

## 최종 산출물(pack/v1)

- `chunks.jsonl`  
  - 1줄 = 1 청크(JSON)  
  - 필드 예: `chunkId`, `docId`, `content`, `meta...`
- `embeddings.f16.bin` *(생성되는 경우)*  
  - 청크 임베딩 float16 바이너리 (라인 순서 = `chunks.jsonl` 순서)
- `embeddings_meta.json`  
  - 임베딩 메타: `count`, `dim`, `dtype`, `sha256`, `normalized` 등
- `licenses.json`  
  - 문서 출처/라이선스 표기용
- `manifest.json`  
  - 문서팩 전체 메타(팩 ID, 버전, 임베딩/청킹 스펙 등)

> 지금 캡쳐에는 `embeddings.f16.bin` 파일이 안 보이는데,  
> 임베딩 단계가 돌아가면 보통 같이 생성됩니다. (`--skip_embed` 쓰면 안 생김)

---

## 설치

### 1) 가상환경 (권장)
macOS / Linux:
```bash
python -m venv .venv
source .venv/bin/activate
````

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) 의존성 설치

```bash
pip install -r requirements.txt
```

---

## 임베딩 모델 (universal_sentence_encoder.tflite)

이 파이프라인은 임베딩 단계에서 `.tflite` 모델을 사용할 수 있습니다.
예: `universal_sentence_encoder.tflite`

권장 배치 방법:

* 프로젝트 루트에 `models/` 폴더를 하나 만들고 넣기 (gitignore 권장)

  * `models/universal_sentence_encoder.tflite`

예시:

```
models/
└─ universal_sentence_encoder.tflite
```

---

## 문서팩 생성(Quick Start)

아래 예시는 입력 PDF가 `input/pdf`에 있다고 가정합니다.

```bash
python scripts/build.py \
  --input_dir input/pdf \
  --work_dir work \
  --pack_dir pack/v1 \
  --pack_id firstaid_kr_v1_use \
  --embed_model models/universal_sentence_encoder.tflite \
  --batch_size 32 \
  --num_threads 4 \
  --normalize
```

옵션 설명(자주 쓰는 것만):

* `--input_dir` : 원본 PDF 폴더
* `--work_dir` : 중간 산출물 폴더(추출/정제/청킹 stage)
* `--pack_dir` : 최종 산출물 폴더 (보통 `pack/v1`)
* `--pack_id` : 문서팩 ID (앱에서 구분용)
* `--embed_model` : `.tflite` 모델 경로 (USE 등)
* `--batch_size` : 임베딩 배치 크기
* `--num_threads` : TFLite 추론 스레드 수
* `--normalize` : L2 정규화(코사인 유사도 검색에 유리)
* `--skip_embed` : 임베딩 생성 스킵(청크만 만들 때)

---

## 파이프라인 흐름

`build.py` 한 방에 아래 순서로 실행됩니다.

1. `extract.py` : PDF → 페이지 단위 텍스트 추출
2. `clean.py` : 정제(불필요한 공백/깨짐/노이즈 처리)
3. `chunk.py` : 길이 기준 청킹(오버랩 포함 가능)
4. `embed.py` : 청크 임베딩 생성 (TFLite 또는 ST 방식)
5. `manifest.json`, `licenses.json` 생성 + 산출물 검증

---

## 도구(tools)

* `tools/test_retrieval.py`

  * 문서팩이 잘 검색되는지 간단히 테스트(Top-K 등) 용도
* `tools/gen_release_notes.py`

  * 릴리즈 노트/변경사항 정리 자동화 용도

---

## 트러블슈팅

### Q1. `chunk` 단계에서 멈춘 것처럼 보여요

* 청킹 입력(`cleaned_pages.jsonl`)이 엄청 크면 시간이 꽤 걸릴 수 있습니다.
* `work/` 아래 중간 파일 크기 확인해보고,
* 필요하면 `--chunk_size`, `--min_chars` 조절해서 청크 수를 줄여보이소.

### Q2. 임베딩 파일이 안 생겨요

* `--skip_embed` 옵션이 켜져 있는지 확인
* `--embed_model` 경로가 맞는지 확인
* (TFLite 사용 시) `tensorflow` 또는 `tflite-runtime` 설치 확인

---

## 라이선스 / 출처

문서 출처/라이선스는 `pack/v1/licenses.json`에 기록됩니다.
앱 배포 시 해당 파일을 함께 포함하는 걸 권장합니다.

```