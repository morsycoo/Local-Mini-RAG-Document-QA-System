# Local Mini RAG Document QA System

> **Temporary consolidated project README — Days 1–3**  
> This consolidates the supplied Day 1 and Day 2 README files and the Day 3 results recoverable from the referenced “Mini RAG” conversation. It does not modify implementation files or Git state.

## Evidence and status convention

- **Measured / recorded** means explicitly recorded in the supplied Day 1 or Day 2 README, or the Day 3 experiment write-up in the conversation. These are historical results, not fresh executions in this workspace.
- **Implementation description** means the source README documents the architecture or behavior as implemented and tested.
- **Not evaluated** means no result is available and no quality claim should be inferred.
- The original Day 3 `README_03_embeddings_and_vector_database.md` was reported as a 24.6 KB, 683-line Library artifact. Its contents were not available through the connected Library in this task. Day 3 figures below are transcribed from the detailed Day 3 write-up in the conversation, not independently checked against that original file. The two uploaded Day 1–2 README files were available and used directly.

## Project objective

Build a local, reproducible document question-answering foundation by validating each data-processing stage before relying on retrieval or answer generation. The first three days cover document ingestion, Markdown-aware chunking, embedding generation, and persistent vector indexing.

The evidence establishes data extraction and integrity, chunk coverage and structure, embedding artifact integrity, and vector-store persistence. **Retrieval evaluation has not yet been done.** No measured retrieval relevance, recall, ranking, answer-quality, or end-to-end QA results are available here.

## Three-day pipeline

```text
Raw source documents
  -> discover, extract, normalize, hash, and persist document/page records
  -> parse Markdown structure and create traceable chunks
  -> audit model token lengths
  -> generate and validate document embeddings
  -> persist chunk text, metadata, and vectors in ChromaDB
  -> [next stage] query embedding, similarity retrieval, and retrieval evaluation
```

| Day | Scope | Recorded outcome |
|---|---|---|
| 1 | Document loading and text extraction | Six Markdown documents; 118,204 extracted characters; normalized text match 6/6; repeat run skipped all 6 unchanged. |
| 2 | Chunking | Six pages, 228 selected recursive Markdown chunks, 100% source coverage, 98.78% structural-block preservation, zero integrity errors. |
| 3 | Token audit, embeddings, ChromaDB | 228 chunks audited with zero token overflow; 228 vectors of 768 dimensions; 228 persistent records; historical full-suite result 727 passed, 0 failed, 23 warnings. |

---

# Day 1 — Document Loading and Text Extraction

## Scope and architecture

The local ingestion layer discovers supported files, extracts text, preserves document and page provenance, validates records, detects duplicates and changes, and persists JSON Lines data. It is independent of the LLM runtime: it does not start Ollama, load a model, generate embeddings, or make network requests.

```text
Knowledge sources -> discovery/safety filters -> TXT/Markdown/native-text PDF extraction
                  -> validated document/page records -> documents.jsonl and pages.jsonl
                  -> input to subsequent chunking
```

Recorded layout:

```text
app/ingestion/{models.py,discovery.py,extractors.py,pipeline.py,cli.py}
data/knowledge_base/{raw/,processed/documents.jsonl,processed/pages.jsonl}
notebooks/01_document_loading_and_text_extraction.ipynb
tests/ingestion/{test_discovery.py,test_models.py,test_pdf_extraction.py,test_pipeline.py,
                 test_process_document.py,test_rollback.py,test_text_extraction.py}
```

Responsibilities: `models.py` defines Pydantic schemas/status values; `discovery.py` performs deterministic safe discovery; `extractors.py` handles TXT, Markdown, and PDF; `pipeline.py` handles IDs, hashes, classification, records, persistence, and rollback; `cli.py` provides a local CLI.

## Inputs, records, identity

| Extension | Behavior | Page granularity |
|---|---|---|
| `.txt` | UTF-8 with BOM support (`utf-8-sig`) | One logical page per file |
| `.md` | UTF-8 with BOM support; preserves Markdown syntax | One logical page per file |
| `.pdf` | PyMuPDF native text extraction, one-based page numbers | One record per PDF page |

Recursive deterministic discovery filters hidden files, symlinks, unsupported extensions, non-regular files, unsafe/inaccessible paths, empty files, and files above **25 MiB**. Scanned PDFs are not OCRed.

Three fields serve distinct purposes:

1. `document_id` identifies a relative source path. It is SHA-256 of the UTF-8 relative path, truncated to the first 16 hexadecimal characters.
2. `source_hash` is SHA-256 of original file bytes.
3. `content_hash` is SHA-256 of normalized extracted text encoded as UTF-8.

Normalization is conservative and normalizes line endings. It does not intentionally rewrite or summarize content. `DocumentRecord` stores source identity, extracted content, hashes, file size, timestamp, and metadata. `PageRecord` stores parent ID, one-based page number, page content, and metadata. `IngestionResult` reports status, page count, and optional error. Pages refer to documents by `document_id`.

| Input condition | Status |
|---|---|
| New path and content | `NEW_DOCUMENT` |
| Existing path, unchanged hashes | `SKIPPED_UNCHANGED` |
| Existing path, changed content | `UPDATED_DOCUMENT` |
| Different path, existing content | `DUPLICATE_CONTENT` |
| Extraction/processing error | `FAILED` |

Storage I/O errors propagate rather than being mislabeled as extraction failures.

## Persistence and recovery

`data/knowledge_base/processed/documents.jsonl` and `pages.jsonl` store one record per line. Each file is written to a temporary file in its destination directory, flushed and synchronized, then atomically replaced with `os.replace()`. The two-file sequence uses compensating rollback: save old page store, write pages, write documents, and restore old pages if document persistence fails. If rollback also fails, raise a `RuntimeError` requiring manual recovery.

This is not a fully transactional two-file database: forced termination or power loss between replacements can leave inconsistent files, and no multi-writer lock is provided.

## Day 1 measured results

### Controlled extraction fixtures

| Fixture | Source size | Extracted characters | Words | Lines | Result |
|---|---:|---:|---:|---:|---|
| `sample.txt` | 446 bytes | 436 | 59 | 11 | Matched normalized source |
| `sample.md` | 477 bytes | 465 | 64 | 13 | Content and Markdown headings preserved |
| `sample.pdf` | 1,636 bytes | 436 | 59 | 13 | Normalized match; two page boundaries retained |

The PDF fixture was text-based; OCR/image-only extraction was not tested.

### Unicode check

For `Hello مرحبا 😊`, recorded lengths were **21 bytes UTF-8**, **30 bytes UTF-16**, and **56 bytes UTF-32** (UTF-16/32 include the BOM emitted by Python's generic codecs). UTF-8 BOM bytes were `[239, 187, 191]`; `utf-8-sig` returned the intended text; ASCII decoding raised `UnicodeDecodeError` as expected.

### Six-document corpus

| Source file | Bytes | Extracted characters |
|---|---:|---:|
| `README_03_Local_Runtime_Engineering.md` | 14,492 | 14,492 |
| `README_04_prompt_construction_and_structured_output.md` | 9,608 | 9,608 |
| `README_05_end_to_end_local_ai_assistant.md` | 32,510 | 32,360 |
| `README_06_prompt_regression_testing.md` | 23,258 | 22,690 |
| `README_07_hadota_web_chat.md` | 9,503 | 9,497 |
| `README_08_HADOTA_Local_AI_Assistant.md` | 29,580 | 29,557 |
| **Total** | **118,951** | **118,204** |

Discovery: **6 found, 0 skipped**, deterministic ordering passed; **118,951 bytes** (approximately **116.16 KiB**). Extraction: **6 successful, 0 failed, 0 empty, 6 logical pages**, **118,204 characters**.

Content comparison: **5/6 exact (83.33%)**, **6/6 normalized (100%)**. The one exact mismatch was `README_06_prompt_regression_testing.md`: source had **565 CRLF** line endings, text-mode reading normalized them to **565 LF**. Source decoded text was **23,255 characters**, extracted text **22,690**. Normalized comparison matched; the recorded explanation is line-ending normalization, not content loss.

### Round-trip and idempotency

- Controlled JSONL round-trip: valid Pydantic document/page records passed; invalid cases were rejected (one negative group yielded **4 validation errors**); **2 records** restored equal to originals.
- Fixture first run: **3 discovered**, `NEW_DOCUMENT=3`, other statuses zero, **3 documents / 4 pages** (one two-page PDF).
- Fixture unchanged rerun: `SKIPPED_UNCHANGED=3`; all other statuses zero.
- After modifying `sample.txt`: `UPDATED_DOCUMENT=1`, `SKIPPED_UNCHANGED=2`, failures zero.
- After adding same-content `duplicate.txt` at another path: `DUPLICATE_CONTENT=1`, `SKIPPED_UNCHANGED=3`, failures zero. Final fixture store: **3 documents / 4 pages**, unique IDs and `(document_id,page_number)` keys.
- Real corpus first run: **6 discovered**, `NEW_DOCUMENT=6`, other statuses zero; **6 documents / 6 pages**, **118,204 characters**.
- Real corpus unchanged rerun: `SKIPPED_UNCHANGED=6`, other statuses zero.

## Day 1 test record and limitations

| Historical run | Result |
|---|---:|
| Focused processing/rollback tests | **10 passed** in 0.53 s |
| Ingestion suite | **96 passed** in 2.42 s |
| Full project suite | **462 passed** in 52.78 s |
| `compileall` for ingestion/tests | Exit code **0** |

These are historical implementation runs, not reruns for this README. Limitations: no OCR for scanned PDFs; complex PDF reading order/tables may be imperfect; TXT/Markdown pages are logical pages; JSONL pair is not crash-atomic and has no concurrent-writer coordination. The measured corpus was six Markdown files plus controlled fixtures. Day 1 did not implement chunking, embeddings, vector search, retrieval, reranking, answer generation, OCR, or HTML/Word/spreadsheet ingestion.

---

# Day 2 — Text Chunking

## Input, design, and implementation

Day 2 reused **6 documents**, **6 pages**, and **118,204 characters** from Day 1. `recursive_markdown` was selected; fixed-window remained the baseline.

Recorded related files:

```text
app/chunking/{models.py,strategies.py,markdown.py,markdown_parser.py,recursive.py}
data/knowledge_base/processed/chunks.jsonl
reports/chunking_comparison.json
notebooks/02_text_chunking.ipynb
tests/chunking/{test_markdown.py,test_models.py,test_recursive.py,test_strategies.py}
```

`ChunkRecord` tracks chunk ID, source document/page, text, original-page character offsets, content hash, Markdown `section_path`, strategy, config signature/version, and metadata. Offsets and text were validated against source for all reloaded records.

The parser covers heading hierarchy, paragraphs, ordered/unordered lists, tables, fenced code, and horizontal rules. Code-fence contents are not misread as headings. `parse_markdown()` in `markdown.py` was selected for the recursive path; `parse_markdown_blocks()` is an alternative inspected parser.

Configuration: `chunk_size=1200` characters for both strategies; fixed-window `overlap=180`; recursive `min_chunk_size=100`; `chunking_version=1`. Recorded configuration signature:

```text
555f1d45d63fb9e67e25d4493aabc970af4bc72c4e04c2c698445f4935d2944f
```

Fixed-window uses character windows with overlap. Recursive Markdown groups structural blocks and splits at paragraph/newline and sentence boundaries, with a hard fallback for oversized segments. It carries content, offsets, section path, and block type. Recursive overlap was not exposed by the inspected API; the 180-character overlap applies only to fixed-window. The recursive 100-character minimum was not a strict guarantee: **19 chunks were shorter**, minimum **22 characters**, and were retained rather than merged across structure.

## Measured comparison

Both strategies processed the same corpus. Processing time is five-sample timing specific to this machine and run, not a general performance guarantee.

| Metric | Fixed-window | Recursive Markdown |
|---|---:|---:|
| Documents / pages | 6 / 6 | 6 / 6 |
| Source characters | 118,204 | 118,204 |
| Chunks | 119 | 228 |
| Mean / median size | 1,164.24 / 1,200 | 518.44 / 457.5 |
| Min / max size | 212 / 1,200 | 22 / 1,185 |
| Under 100 chars | 0 | 19 |
| 100–399 / 400–799 / 800–1,200 chars | 3 / 2 / 114 | 76 / 85 / 48 |
| Over 1,200 chars | 0 | 0 |
| Coverage | 118,204 (100.00%) | 118,204 (100.00%) |
| Structural blocks / contained in one chunk | 740 / 699 | 740 / 731 |
| Structural preservation | 94.46% | 98.78% |
| Integrity errors | 0 | 0 |
| Median / min / max time | 0.000070 / 0.000069 / 0.000073 s | 0.011590 / 0.007432 / 0.013968 s |

| Source | Characters | Fixed chunks | Fixed preservation | Recursive chunks | Recursive preservation |
|---|---:|---:|---:|---:|---:|
| `README_03_Local_Runtime_Engineering.md` | 14,492 | 15 | 97.46% | 32 | 100.00% |
| `README_04_prompt_construction_and_structured_output.md` | 9,608 | 10 | 97.47% | 22 | 100.00% |
| `README_05_end_to_end_local_ai_assistant.md` | 32,360 | 32 | 93.68% | 69 | 98.95% |
| `README_06_prompt_regression_testing.md` | 22,690 | 23 | 98.64% | 48 | 100.00% |
| `README_07_hadota_web_chat.md` | 9,497 | 10 | 77.78% | 15 | 88.89% |
| `README_08_HADOTA_Local_AI_Assistant.md` | 29,557 | 29 | 85.71% | 42 | 96.19% |
| **Total** | **118,204** | **119** | — | **228** | — |

Structural preservation means a parsed Markdown block stayed fully within one chunk; it is not a semantic-coherence or retrieval-quality metric.

## Selection, persistence, and tests

`recursive_markdown` was chosen for higher measured structural preservation, full coverage, and zero integrity errors. It generated 228 chunks versus 119 and took longer in the recorded samples; accepted for this corpus. Revisit if retrieval evaluation favors another setup.

All **228** recursive records were persisted to `data/knowledge_base/processed/chunks.jsonl` and reloaded. File/report existence, reload, unique IDs, offsets/text, hashes, strategy/config metadata passed; **0 validation errors**. Chunk JSONL size: **290,032 bytes**; report size: **9,103 bytes**. Notebook audit: **126 cells**, 93 Markdown, 33 code, all 33 code cells executed.

Historical tests: `python -m pytest tests/chunking -v` — **53 passed**; `python -m pytest -q` — **515 passed**. Not rerun for this artifact.

Limitations: 19 recursive chunks are under 100 characters; recursive overlap is not implemented; timings are environment-dependent; corpus is six Markdown pages. Structural preservation does not evaluate retrieval. Day 2 excluded embedding generation, vector DB, similarity search, context assembly, reranking, answer generation, end-to-end RAG evaluation, and a chunking CLI/pipeline.

---

# Day 3 — Embeddings and Persistent Vector Database

> **Source note:** The original Day 3 README was not retrievable from the connected Library in this task. This section uses the detailed experiment write-up in the referenced conversation; omitted file-specific inventory is not independently verified here.

## 1. Tokenization audit and model input validation

The **228 Day 2 chunks** were checked against the selected model's **512-token** maximum. The audit included the `passage: ` prefix and special tokens and disabled truncation, measuring the intended full model input rather than character count.

| Parameter/result | Recorded value |
|---|---:|
| Model | `intfloat/multilingual-e5-base` |
| Pinned revision | `ea6fe3baa7f6bfe27fbc99fde19d01b3e2bca610` |
| Maximum length | 512 tokens |
| Prefix / special tokens | `passage: ` / included |
| Truncation | Disabled |
| Chunks audited | 228 |
| Min / median tokens | 8 / 109.5 |
| Mean | 139.19298245614036 |
| 95th percentile | 326.3 |
| Maximum | 452 |
| Overflow / within limit | 0 / 228 |

Transformers tokenizer loading first encountered Windows `OSError: [WinError 1455]` (“The paging file is too small for this operation to complete”) during a CUDA-related import. The recorded workaround used `tokenizers` directly with the local pinned snapshot's `tokenizer.json`, avoiding another PyTorch initialization. This was a tokenizer-only audit, not another embedding run.

All 228 inputs fit; no truncation or additional token-aware subdivision was needed, so the existing chunks were retained. This establishes model input-length compatibility, not optimal chunking or semantic completeness.

## 2. Embedding generation and vector integrity

Recorded configuration: `intfloat/multilingual-e5-base`, revision `ea6fe3baa7f6bfe27fbc99fde19d01b3e2bca610`, **768 dimensions**, max length **512**, `float32`, normalization enabled, document prefix `passage: `, future query prefix `query: `, local files only. Day 3 generated document-side embeddings; query-side configuration was noted for retrieval.

```text
data/knowledge_base/processed/embeddings/
├── embeddings.npz  # matrix and chunk IDs
└── manifest.json   # model, source, config, and artifact details/hashes
```

| Validation | Recorded result |
|---|---:|
| Matrix shape | `(228, 768)` |
| dtype | `float32` |
| All values finite | `True` |
| Zero vectors | 0 |
| Mean L2 norm | 1.0 |
| Minimum L2 norm | 0.9999999403953552 |
| Maximum L2 norm | 1.0000001192092896 |
| Chunk ID alignment | `True` |

Recorded SHA-256:

```text
Embedding artifact: 1587343fa3e0c2b0bdca75999572684de5d6f1a247f08842c86cefcacfdc773e
Manifest:           a17c6e49eda82c7bc759b74b5fd7cf9dd70fbf16e9ef5ec272523d2bc3baa073
```

The conversation reports artifact-hash match to the manifest and source-chunks-hash match to the recorded source hash. Actual generation device, generation batch size, and generation duration were **not independently recorded**. CUDA availability and an RTX 5070 Laptop GPU were mentioned, but GPU use should not be claimed without selected-device evidence.

## 3. ChromaDB persistence and indexing integrity

The local vector store uses ChromaDB `PersistentClient`, with precomputed vectors supplied directly (`embedding_function=None`). ChromaDB handles persistence/indexing; model embeddings remain controlled by the project.

| Setting | Recorded value |
|---|---|
| Database path | `data/knowledge_base/vector_store/chroma/` |
| Collection | `hadota_chunks_e5_base_v1` |
| Metric | cosine |
| Dimension / embedding function | 768 / None |
| Records | 228 |
| HNSW `space` | `cosine` |
| `ef_construction` / `ef_search` | 100 / 100 |
| `max_neighbors` | 16 |
| `resize_factor` | 1.2 |
| `sync_threshold` | 1000 |

These are inspected settings, not benchmark-proven optimal values.

```text
app/vector_store/
├── __init__.py
├── config.py       # database and collection settings
├── exceptions.py   # vector-store errors
├── chroma_store.py # PersistentClient and collection access
├── metadata.py     # metadata conversion
└── indexer.py      # validation and indexing
```

Before indexing, the recorded indexer validates files, chunk/embedding ID alignment and uniqueness, shape/dimension, `float32`, finite values, normalization, source and artifact hashes against the manifest, and model/revision settings. Index batch size was **64**, producing four batches for 228 records. Each record contains chunk ID, text, vector, and source metadata. Final collection count: **228**.

### Persistence and re-indexing results

A new persistent client reopened the database:

| Validation | Result |
|---|---:|
| Collection count | 228 |
| Stored IDs | 228 |
| Missing / unexpected IDs | 0 / 0 |

Sample indices **0, 100, 227** returned **3 IDs**, **3 documents**, **3 metadata objects**, and embedding shape **`(3, 768)`**. Text, metadata, and embedding comparisons were on those three samples, not all 228 records. Full ID/count checks were reported in indexer tests.

Re-indexing the same corpus kept the collection at **228** and created **0 duplicate records**. This is an idempotency result for that recorded repeat run.

### Test status

| Focused vector-store test area | Passed |
|---|---:|
| Configuration | 19 |
| Exceptions | 11 |
| ChromaDB store | 17 |
| Metadata mapping | 62 |
| Indexer | 37 |
| **Total** | **146** |

The conversation reports a historical full-suite result of **727 passed, 0 failed, 23 warnings**, in **55.88 seconds**. It states this was a previously recorded run, not rerun by report generation. The warnings were described as ChromaDB deprecation warnings. This is not fresh verification of the current source tree.

## Day 3 limitations and handoff

- Token-length validity does not prove semantic quality or optimal chunk size.
- Vector integrity and persistence checks do not establish that the model is best.
- ChromaDB settings were inspected and exercised, not tuned against retrieval benchmarks.
- Embedding generation device, batch size, and duration were not independently recorded; index batch size **64** is separately recorded.
- Retrieval relevance, recall, ranking, context sufficiency, answer correctness, and end-to-end RAG quality have **not been evaluated**.
- Day 4 was identified as Semantic Retrieval & Similarity Search: embed queries with compatible model settings and `query: `, retrieve from the existing collection, and evaluate against an explicit test set. The plan was to reuse existing embeddings/collection, not rebuild them.

---

# Consolidated status and reproducibility

| Deliverable | Recorded status |
|---|---|
| Day 1 ingestion and extraction | Implemented and tested; historical full suite 462 passed |
| Day 2 chunking | Implemented; recursive Markdown selected; 228 chunks persisted/reloaded; historical full suite 515 passed |
| Day 3 tokenization audit | Completed; 228 audited; 0 overflow |
| Day 3 embeddings | 228 × 768 artifact and manifest described; integrity checks recorded |
| Day 3 ChromaDB indexing | 228 persistent records; IDs complete; re-indexing created no duplicates |
| Retrieval evaluation | **Not yet done** |
| Answer generation / QA evaluation | **Not established by Day 1–3 evidence** |

The READMEs list historical commands including `python -m pytest tests/ingestion/ -v`, `python -m pytest tests/chunking -v`, and `python -m pytest -q`. Their recorded outcomes apply to their respective implementation runs. They were not run while producing this artifact and are not pass results for a later repository state.

## Key Day 3 handoff facts

- **6** source Markdown documents; **6** logical pages; **118,204** extracted characters.
- **228** selected recursive Markdown chunks; **100.00%** coverage; **98.78%** structural-block preservation in Day 2 comparison.
- **228** chunks within a **512-token** limit; maximum observed input **452**; **0** overflow.
- **228 × 768** `float32` normalized embeddings; IDs aligned; all finite; **0** zero vectors.
- **228** ChromaDB records after reopen; **0** missing/unexpected IDs; **0** duplicates after repeat indexing.
- Retrieval evaluation: **not yet done**.
