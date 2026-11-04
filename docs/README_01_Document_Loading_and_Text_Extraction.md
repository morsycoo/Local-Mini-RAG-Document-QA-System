# Document Loading and Text Extraction

A reproducible, local ingestion layer for discovering PDF, Markdown, and plain-text knowledge sources, extracting their text, preserving document and page provenance, and persisting validated records for subsequent RAG processing.

> **Project:** Local Mini RAG Document QA System  
> **Component:** Document ingestion  
> **Implementation status:** Implemented and tested  
> **Python:** 3.13.0

---

## 1. Overview

The ingestion component converts a directory of source documents into structured, traceable records. It is deliberately independent of the LLM runtime: loading and extracting documents does not require starting Ollama, loading a model, generating embeddings, or making network requests.

The implemented component provides:

- Recursive discovery of supported source files.
- File filtering and safety checks before extraction.
- Text extraction from `.txt`, `.md`, and native-text `.pdf` files.
- Preservation of Markdown structure and PDF page boundaries.
- Validated Pydantic models for documents, pages, and ingestion outcomes.
- Deterministic document identifiers and SHA-256 source/content hashes.
- Duplicate detection, unchanged-file detection, and update classification.
- JSON Lines (`.jsonl`) persistence for document and page records.
- Atomic replacement of each individual JSONL file and compensating rollback for the two-file persistence sequence.
- A local command-line entry point.
- Unit and integration tests covering the ingestion behavior and failure paths.

The output is intended to be consumed by later processing stages. This component does **not** split text into chunks, create embeddings, build a vector index, retrieve passages, or generate answers.

## 2. Position in the RAG Workflow

```text
Knowledge Sources
       |
       v
Document Discovery
       |
       v
Text Extraction
       |
       v
Validated Document and Page Records
       |
       v
JSONL Persistence
       |
       +----> Input for subsequent chunking and retrieval stages
```

Keeping ingestion deterministic and separate from model inference makes it possible to test extraction and storage independently from LLM behavior. It also makes the source corpus reproducible: processed records can be regenerated from the original files.

## 3. Implemented Project Structure

The ingestion implementation is organized into a dedicated package:

```text
app/
└── ingestion/
    ├── __init__.py
    ├── models.py
    ├── discovery.py
    ├── extractors.py
    ├── pipeline.py
    └── cli.py

data/
├── experiments/
│   └── ...
└── knowledge_base/
    ├── raw/
    │   └── ...                  # Original source documents
    └── processed/
        ├── documents.jsonl
        └── pages.jsonl

docs/
└── README_01_document_loading_and_text_extraction.md

notebooks/
└── 01_document_loading_and_text_extraction.ipynb

tests/
└── ingestion/
    ├── test_discovery.py
    ├── test_models.py
    ├── test_pdf_extraction.py
    ├── test_pipeline.py
    ├── test_process_document.py
    ├── test_rollback.py
    └── test_text_extraction.py
```

### Module responsibilities

| Module | Responsibility |
|---|---|
| `models.py` | Pydantic schemas, validation rules, and ingestion status values. |
| `discovery.py` | Safe and deterministic discovery of supported source files. |
| `extractors.py` | TXT, Markdown, and PDF text extraction. |
| `pipeline.py` | Hashing, ID generation, record construction, duplicate/update classification, JSONL persistence, and rollback handling. |
| `cli.py` | Local command-line interface for running ingestion without starting the web application or an LLM runtime. |

## 4. Runtime and Dependencies

The implementation was developed and tested in the following environment:

| Component | Observed version |
|---|---:|
| Python | `3.13.0` |
| PyMuPDF | `1.28.2` |
| Pydantic | `2.13.4` |
| pytest | `9.0.3` |

The project's dependency constraints are:

```text
pydantic>=2.8,<3.0
pymupdf>=1.25,<2.0
```

The PDF extractor imports PyMuPDF using its current package/module name:

```python
import pymupdf
```

No vector database, embedding model, OCR package, or additional service is required by the implemented ingestion layer.

## 5. Supported Formats and Extraction Behavior

The discovery layer supports the following extensions:

| Extension | Extraction behavior | Record granularity |
|---|---|---|
| `.txt` | Reads UTF-8 text with UTF-8 BOM support. | One logical page per file. |
| `.md` | Reads UTF-8 text with UTF-8 BOM support and preserves Markdown syntax. | One logical page per file. |
| `.pdf` | Extracts embedded/native text using PyMuPDF. | One record per PDF page. |

### 5.1 Plain text

Text files are read with UTF-8 BOM-aware decoding (`utf-8-sig`). Their line breaks and text content are preserved, subject to normal text-mode newline handling by the Python file API.

### 5.2 Markdown

Markdown files are read as text. The extractor does not render Markdown into HTML or flatten it into prose. Headings, list markers, table syntax, code fences, links, and other source markup remain available to later processing stages.

### 5.3 PDF

PDF extraction uses PyMuPDF and retains page-level boundaries. Page numbering is one-based in the resulting `PageRecord` objects. A two-page PDF therefore produces two page records associated with the same `document_id`.

This is native text extraction, not OCR. Scanned or image-only PDF pages may contain no extractable text. Complex multi-column reading order, tables, and page layouts may require additional processing and manual inspection; successful file opening does not guarantee perfect semantic extraction.

### 5.4 Discovery filters

The discovery implementation uses:

- Supported extensions: `.txt`, `.md`, and `.pdf`.
- Maximum file size: **25 MiB**.
- Recursive discovery with deterministic ordering.
- Checks that filter hidden files, symbolic links, unsupported extensions, non-regular files, unsafe or inaccessible paths, empty files, and files above the size limit.

Discovery returns structured information about discovered files and skipped-file issues rather than treating every filesystem entry as a valid document.

## 6. Data Models and Validation

The schemas are defined in `app/ingestion/models.py`. They use Pydantic validation and reject unexpected fields where configured.

### 6.1 `DocumentRecord`

Represents the source document as a whole.

| Field | Type / constraint | Meaning |
|---|---|---|
| `document_id` | Non-empty string | Deterministic identifier for the source path. |
| `file_name` | Non-empty string | Original file name. |
| `source_path` | Non-empty string | Source path recorded for the document. |
| `file_type` | Non-empty string | File extension/type. |
| `content` | String | Full extracted document text. |
| `source_hash` | 64-character hexadecimal SHA-256 | Hash of the original source-file bytes. |
| `content_hash` | 64-character hexadecimal SHA-256 | Hash of normalized extracted text. |
| `file_size` | Integer greater than or equal to zero | Source file size in bytes. |
| `ingested_at` | UTC-aware datetime by default | Time the record was created. |
| `metadata` | Dictionary | Additional document metadata. |

The model preserves content as provided; it does not itself define document ID generation or text normalization. Those responsibilities belong to the pipeline.

### 6.2 `PageRecord`

Represents extracted text from an individual page or logical page.

| Field | Type / constraint | Meaning |
|---|---|---|
| `document_id` | Non-empty string | Links the page to its parent document. |
| `page_number` | Integer greater than or equal to one | One-based page number. |
| `content` | String | Extracted page text. |
| `metadata` | Dictionary | Page-specific metadata. |

Pages are stored as separate records linked by `document_id`; they are not nested inside the `DocumentRecord` schema.

For TXT and Markdown, the extractor emits one logical page. For PDF, it emits one page record per PDF page.

### 6.3 `IngestionResult`

Represents the outcome of processing one source file.

| Field | Type / constraint | Meaning |
|---|---|---|
| `document_id` | Non-empty string | Identifier associated with the source. |
| `source_path` | Non-empty string | Source path. |
| `status` | `IngestionStatus` | Classified ingestion outcome. |
| `pages_extracted` | Non-negative integer; default `0` | Number of extracted page records. |
| `message` | String; default empty | Human-readable outcome message. |
| `error` | Optional string | Error detail when applicable. |

### 6.4 Status values

`IngestionStatus` defines these values:

| Status | Meaning |
|---|---|
| `NEW_DOCUMENT` (`new_document`) | A previously unseen source document was ingested. |
| `UPDATED_DOCUMENT` (`updated_document`) | The same source identity was found with changed content. |
| `SKIPPED_UNCHANGED` (`skipped_unchanged`) | The source is already stored and its relevant hashes are unchanged. |
| `DUPLICATE_CONTENT` (`duplicate_content`) | A different source identity has content already represented in storage. |
| `FAILED` (`failed`) | Extraction or document processing failed and the failure was returned as an ingestion result. |

Storage I/O failures are not silently converted into `FAILED` extraction results. They propagate to the caller because a storage failure may indicate that persistence or recovery needs attention.

## 7. Identity, Hashing, and Text Normalization

Three distinct values serve different purposes:

1. **`document_id`** identifies a source path.
2. **`source_hash`** detects changes to the original file bytes.
3. **`content_hash`** detects changes to the normalized extracted text.

### 7.1 Document identity

The pipeline generates the document identifier deterministically from the UTF-8 representation of the source's relative path, using SHA-256 and retaining the first 16 hexadecimal characters.

This means that the same relative source path maps to the same document identity across repeat runs, while a file at a different path has a different identity even when its content is identical.

### 7.2 Source hash

`source_hash` is calculated from the original file bytes. It provides a byte-level fingerprint of the input file.

### 7.3 Content hash

`content_hash` is calculated from the normalized extracted text encoded as UTF-8. It represents the text that the downstream application will consume, rather than the raw container bytes of the source file.

This distinction matters for line-ending differences and file-format metadata: two source files can have different byte hashes while yielding equivalent normalized text.

### 7.4 Normalization policy

Normalization is deliberately conservative. The implemented pipeline normalizes text line endings and computes the content hash from the normalized text. It does not intentionally rewrite or summarize the document.

The extraction and integrity experiments verified preservation of the tested Markdown content, English and Arabic Unicode text, and meaningful line structure. Normalization is not intended to erase Markdown structure, remove Arabic characters, or collapse the document into a single line.

## 8. Ingestion Classification and Idempotency

The pipeline compares the incoming document identity and content fingerprint with the records already stored.

| Incoming source | Classification |
|---|---|
| New path and new content | `NEW_DOCUMENT` |
| Existing path and unchanged content | `SKIPPED_UNCHANGED` |
| Existing path with changed content | `UPDATED_DOCUMENT` |
| Different path with content already stored | `DUPLICATE_CONTENT` |

The idempotency behavior was exercised with repeat runs against the same fixtures and the real knowledge-base corpus. Reprocessing unchanged files did not add duplicate document or page records.

Duplicate-content classification preserves the distinction between source identity and content identity. A duplicate file can have its own path while sharing the same content hash as another source.

## 9. Persistence Format and Storage Behavior

The processed corpus is persisted as JSON Lines:

```text
data/knowledge_base/processed/
├── documents.jsonl
└── pages.jsonl
```

- `documents.jsonl` contains one serialized `DocumentRecord` per line.
- `pages.jsonl` contains one serialized `PageRecord` per line.
- Each page record references its parent document through `document_id`.

JSONL was selected because it is line-oriented, straightforward to inspect, and can be consumed incrementally with Python without requiring a database service.

### 9.1 Atomic replacement of an individual file

The persistence helper writes to a temporary file in the destination directory, flushes and synchronizes the write, and then uses `os.replace()` to replace the target file. Temporary files are cleaned up when they remain after the operation.

This protects an individual JSONL file from being left half-written by an ordinary write failure during replacement.

### 9.2 Coordinating the two JSONL files

The document and page stores are separate files, so replacing each file atomically does not by itself make the pair a single atomic transaction.

The pipeline therefore uses a compensating rollback strategy when updating the two stores:

1. It captures the previous page-store state.
2. It persists the updated page records.
3. It attempts to persist the updated document records.
4. If document persistence fails, it attempts to restore the previous page-store state.
5. If rollback also fails, it raises a `RuntimeError` indicating that manual storage recovery is required.

This strategy was tested for both restoring an existing page file and removing a newly created page file after a failed operation.

**Known limitation:** this is not a fully transactional two-file database. It cannot guarantee consistency if the process is forcibly terminated or power is lost between file replacements. It also does not provide a multi-writer concurrency lock. A database or a journaled transaction design would be required for stronger crash and concurrency guarantees.

## 10. Command-Line Usage

The ingestion package provides a local CLI that can be invoked without starting the web server or Ollama.

Display the available options:

```powershell
python -m app.ingestion.cli --help
```

Run the default ingestion configuration from the repository root:

```powershell
python -m app.ingestion.cli
```

The CLI was exercised against both controlled fixture files and the configured raw knowledge-base directory. The `--help` command completed successfully. The exact supported flags and defaults are defined by `app/ingestion/cli.py`; use its help output as the authoritative interface when invoking it with custom paths.

## 11. Experiments and Observed Results

The following results are from actual local executions recorded during implementation. Synthetic fixtures and the real repository corpus are distinguished below.

### 11.1 Controlled extraction fixtures

A small controlled set was created for format-level extraction checks:

| File | Source size | Extracted characters | Words | Lines | Observation |
|---|---:|---:|---:|---:|---|
| `sample.txt` | 446 bytes | 436 | 59 | 11 | Extracted text matched the normalized source. |
| `sample.md` | 477 bytes | 465 | 64 | 13 | Source content and Markdown heading structure were preserved. |
| `sample.pdf` | 1,636 bytes | 436 | 59 | 13 | Normalized text matched; the PDF retained two page boundaries. |

These are controlled test documents, not measurements of arbitrary external PDFs. The PDF fixture was text-based; this experiment did not validate OCR or image-only page extraction.

### 11.2 Encoding and Unicode experiment

The encoding experiment used the text `Hello مرحبا 😊`.

Observed results:

- UTF-8 encoded length: **21 bytes**.
- The ASCII letters in the sample use one byte each in UTF-8; the Arabic characters use multiple bytes and the emoji uses four bytes.
- Python's encoded sample lengths were **21 bytes for UTF-8**, **30 bytes for UTF-16**, and **56 bytes for UTF-32**. The UTF-16 and UTF-32 results include the byte-order mark emitted by the generic Python codecs used in the experiment.
- The UTF-8 BOM bytes were `[239, 187, 191]`.
- Decoding UTF-8-with-BOM content using `utf-8-sig` returned the intended text.
- Decoding the sample as ASCII raised `UnicodeDecodeError`, as expected for non-ASCII characters.

This confirmed that Unicode text must be decoded with an appropriate encoding and that UTF-8 BOM handling is relevant to file ingestion.

### 11.3 Discovery against the real local corpus

The raw knowledge-base directory contained six Markdown documents.

| File | Source size (bytes) |
|---|---:|
| `README_03_Local_Runtime_Engineering.md` | 14,492 |
| `README_04_prompt_construction_and_structured_output.md` | 9,608 |
| `README_05_end_to_end_local_ai_assistant.md` | 32,510 |
| `README_06_prompt_regression_testing.md` | 23,258 |
| `README_07_hadota_web_chat.md` | 9,503 |
| `README_08_HADOTA_Local_AI_Assistant.md` | 29,580 |
| **Total** | **118,951** |

Observed discovery results:

- Files discovered: **6**.
- Files skipped: **0**.
- Total source size: **118,951 bytes** (approximately **116.16 KiB**).
- Deterministic ordering check: **passed**.

### 11.4 Extraction against the real local corpus

All six discovered Markdown files were extracted successfully.

| File | Extracted characters |
|---|---:|
| `README_03_Local_Runtime_Engineering.md` | 14,492 |
| `README_04_prompt_construction_and_structured_output.md` | 9,608 |
| `README_05_end_to_end_local_ai_assistant.md` | 32,360 |
| `README_06_prompt_regression_testing.md` | 22,690 |
| `README_07_hadota_web_chat.md` | 9,497 |
| `README_08_HADOTA_Local_AI_Assistant.md` | 29,557 |
| **Total** | **118,204** |

Aggregate observations:

| Metric | Result |
|---|---:|
| Discovered documents | 6 |
| Successfully extracted | 6 |
| Failed extractions | 0 |
| Empty extractions | 0 |
| Logical pages | 6 |
| Extracted characters | 118,204 |
| Source bytes | 118,951 |

Each Markdown document produced one logical page. The extraction timing was observed during local runs but is not treated as a benchmark: the runs were not controlled performance tests and their durations varied.

### 11.5 Content-integrity comparison

The extracted text was compared with the decoded source text using both exact and normalized comparisons.

| Metric | Result |
|---|---:|
| Documents checked | 6 |
| Exact text matches | 5 / 6 (**83.33%**) |
| Normalized text matches | 6 / 6 (**100%**) |

The only exact mismatch was `README_06_prompt_regression_testing.md`:

- The source contained **565 CRLF** line endings.
- The text read through Python's text-mode API contained those line endings normalized to **565 LF** line endings.
- The source decoded text contained **23,255 characters**; extracted text contained **22,690 characters**.
- The normalized comparison matched, so the observed difference was line-ending normalization rather than evidence of lost textual content.

This result demonstrates why a byte-level source hash and a normalized extracted-content hash serve different purposes. The source fingerprint remains tied to the original file bytes, while the content fingerprint reflects the normalized text consumed downstream.

### 11.6 Schema and JSONL round-trip experiment

A schema/persistence experiment created a real-shaped document record and page records, serialized records to JSONL, loaded them again, and compared the restored values.

Observed results:

- Valid document and page records passed Pydantic validation.
- Invalid test inputs were rejected by validation; one negative test group produced four validation errors.
- JSON serialization succeeded.
- JSONL records were read back successfully.
- **2 records** were restored in the controlled round-trip experiment.
- Restored records compared equal to their corresponding originals.

The experiment used a separate file under `data/experiments/`; it was not the production processed-corpus store.

### 11.7 CLI fixture ingestion and idempotency

A separate fixture set was used to exercise the command-line pipeline and persistence behavior.

**Initial ingestion:**

| Metric | Result |
|---|---:|
| Files discovered | 3 |
| `NEW_DOCUMENT` | 3 |
| `UPDATED_DOCUMENT` | 0 |
| `SKIPPED_UNCHANGED` | 0 |
| `DUPLICATE_CONTENT` | 0 |
| `FAILED` | 0 |
| Persisted documents | 3 |
| Persisted pages | 4 |

The four page records came from three files, including a two-page PDF.

**Repeat ingestion without changes:**

- `SKIPPED_UNCHANGED`: **3**.
- New, updated, duplicate, and failed outcomes: **0**.

**Modified source test:**

After changing `sample.txt`, the next run reported:

- `UPDATED_DOCUMENT`: **1**.
- `SKIPPED_UNCHANGED`: **2**.
- Failed files: **0**.

**Duplicate content test:**

A `duplicate.txt` copy with the same bytes as `sample.txt` was added under a different path. The run reported:

- `DUPLICATE_CONTENT`: **1**.
- `SKIPPED_UNCHANGED`: **3**.
- Failed files: **0**.

The final fixture storage contained three document records and four page records, with unique document identifiers and unique `(document_id, page_number)` keys.

### 11.8 Real-corpus ingestion and repeat run

The default CLI was run against the six-file raw knowledge base.

**First run:**

- Discovered: **6**.
- `NEW_DOCUMENT`: **6**.
- `UPDATED_DOCUMENT`: **0**.
- `SKIPPED_UNCHANGED`: **0**.
- `DUPLICATE_CONTENT`: **0**.
- `FAILED`: **0**.

The resulting processed store contained **6 document records** and **6 page records**, totaling **118,204 extracted characters**. All six document IDs were unique, and all page keys were unique.

**Second run without source changes:**

- Discovered: **6**.
- `SKIPPED_UNCHANGED`: **6**.
- All other statuses: **0**.

This confirms repeat-run idempotency for the observed corpus and storage state.

## 12. Failure Handling and Recovery

The pipeline distinguishes extraction/processing failures from persistence failures.

### Extraction and processing failures

Expected extraction and document-processing exceptions are converted into an `IngestionResult` with `FAILED` status and error details. This allows a batch operation to report an individual source failure without misclassifying a storage fault as an extraction problem.

### Persistence failures

JSONL read/write failures are allowed to propagate. When the two-file update sequence has already replaced the page store but document persistence fails, the pipeline attempts to restore the previous page-store state.

If restoration itself fails, the pipeline raises a `RuntimeError` stating that manual storage recovery is required. This is intentionally visible to the caller; silently continuing could leave the document and page stores inconsistent.

### Tested recovery cases

The tests cover:

- Restoring a pre-existing file from its saved state.
- Removing a file that did not exist before the attempted operation.
- Rolling back newly written page data when document persistence fails.
- Restoring existing page data after document persistence fails.
- Raising a clear runtime error when rollback also fails.

## 13. Automated Test Results

The ingestion package and the full existing project suite were tested after implementation.

| Test run | Result |
|---|---:|
| Focused process/rollback tests | **10 passed** in 0.53 s |
| Ingestion test suite | **96 passed** in 2.42 s |
| Full project test suite | **462 passed** in 52.78 s |
| Python compile check | `compileall` completed successfully (exit code `0`) |

The full test suite command was:

```powershell
python -m pytest -v
```

Observed final result:

```text
462 passed in 52.78s
```

The ingestion test suite command was:

```powershell
python -m pytest tests/ingestion/ -v
```

Observed result:

```text
96 passed in 2.42s
```

Compilation check:

```powershell
python -m compileall -q app/ingestion tests/ingestion
```

It completed with exit code `0`.

The full suite includes tests for the existing assistant/runtime/web functionality as well as the newly added ingestion tests. The 462 passing tests establish that the current test suite passed in the tested environment; they do not prove that every possible PDF layout, filesystem failure, or concurrent-write scenario is covered.

## 14. Notebook and Reproducibility Notes

The implementation and observations are documented in:

```text
notebooks/01_document_loading_and_text_extraction.ipynb
```

The notebook contains explanations, experiments, observed outputs, and result interpretations. Markdown explanations and Python code cells are written in English to keep the notebook consistent and portable.

The repository also contains controlled fixtures under `data/experiments/` for repeatable CLI and ingestion checks. Generated fixture storage and processed knowledge-base output are local artifacts and are excluded from version control through `.gitignore`; the source fixtures used for the CLI checks remain available as test inputs.

The source corpus is stored separately from processed output:

- `knowledge base/raw/` contains the six Markdown source documents used in the observed real-corpus run.
- `data/knowledge_base/processed/` contains generated `documents.jsonl` and `pages.jsonl` records.

The processed JSONL files are regenerable outputs, not authoritative source documents.

## 15. Known Limitations

The implementation has explicit boundaries:

1. **No OCR:** scanned PDFs and image-only pages are not converted to text.
2. **PDF layout complexity:** multi-column documents and complex reading order may require specialized extraction and manual review.
3. **Tables:** textual extraction does not guarantee preservation of visual table rows and columns.
4. **One logical page for TXT/Markdown:** page records for these formats represent a logical document page, not physical pages.
5. **Two-file persistence is not a database transaction:** rollback handles tested write failures, but process termination or power loss between replacements can still leave inconsistent state.
6. **No concurrent-writer coordination:** the current pipeline does not provide a multi-process lock around JSONL updates.
7. **Local test corpus:** the measured extraction results apply to the six observed Markdown files and controlled fixtures, not to an arbitrary document collection.
8. **No chunking or retrieval yet:** extracted records are the input to later RAG components; they are not retrieval-ready chunks.

## 16. Scope Boundaries

The following capabilities are deliberately outside this component:

- Text chunking, chunk-size selection, and overlap strategies.
- Embedding generation or embedding-model selection.
- FAISS, ChromaDB, or another vector index.
- Similarity search, metadata filtering, retrieval, or reranking.
- Prompt construction using retrieved passages.
- LLM answer generation and source-grounded response formatting.
- OCR and image understanding for scanned documents.
- Web crawling, HTML extraction, Word document parsing, and spreadsheet ingestion.

These concerns are kept separate so that ingestion can be tested as a deterministic preprocessing layer.

## 17. Final Outcome

The project now has a reusable local ingestion package that discovers supported files, extracts text, preserves document/page associations, validates records, detects unchanged and duplicate content, and persists the results as JSONL.

The implementation was exercised against controlled TXT, Markdown, and PDF fixtures and a real six-document Markdown corpus. The observed real-corpus run extracted **118,204 characters** from six documents with **zero extraction failures**. A repeat run classified all six sources as unchanged. The command-line fixture experiments also verified new-document, update, unchanged, and duplicate-content classifications.

The ingestion-specific test suite passed **96 tests**, and the complete project test suite passed **462 tests** in the recorded environment. The principal remaining limitations are scanned-PDF OCR, complex PDF layout fidelity, and the crash/concurrency guarantees of a two-file JSONL store.

The component provides a validated, reproducible text-and-metadata foundation for the subsequent chunking, embedding, retrieval, and document-question-answering stages.
