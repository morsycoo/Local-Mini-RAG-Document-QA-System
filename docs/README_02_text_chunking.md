# Day 2 — Text Chunking and Document Structure Preservation

## 1. Overview

Day 2 implements and evaluates the text-chunking layer of the **Local Mini RAG Document QA System**.

The objective is to transform the processed page records produced by Day 1 into traceable `ChunkRecord` objects while preserving as much of the source document's structure and provenance as possible.

Two chunking strategies were implemented and compared:

1. **Fixed-window chunking** — a simple size-based baseline with character overlap.
2. **Recursive Markdown chunking** — a structure-aware strategy that uses Markdown blocks and section context when assembling chunks.

The experiment selected `recursive_markdown` as the primary strategy. The fixed-window implementation remains available as a baseline.

Day 2 ends at chunk creation, validation, comparison, and persistence. Embeddings, vector databases, retrieval, reranking, and LLM answer generation are outside this day's scope.

## 2. Goals and Deliverables

The Day 2 work had the following goals:

- Reuse the processed records created by Day 1 rather than rebuilding ingestion.
- Inspect the source corpus and identify its document structure.
- Establish a consistent `ChunkRecord` data contract.
- Implement a fixed-window baseline.
- Parse Markdown structure, including headings and common block types.
- Implement recursive, structure-aware chunking.
- Preserve document IDs, page numbers, source offsets, section paths, and content hashes.
- Compare both strategies using size, coverage, structural-preservation, integrity, and timing measurements.
- Persist generated chunks and the comparison report.
- Reload persisted records and validate them.
- Run focused chunking tests and the full project regression suite.

## 3. Input Corpus and Existing Day 1 Artifacts

Day 2 consumes the processed JSONL files created by Day 1. It does not re-ingest the original files.

### 3.1 Input files

| Input | Path | Purpose |
|---|---|---|
| Processed documents | `data/knowledge_base/processed/documents.jsonl` | Document-level records and source metadata |
| Processed pages | `data/knowledge_base/processed/pages.jsonl` | Page-level text used as the chunking input |

The page records are the direct input to the chunking experiment. Each page is processed independently so that page numbers and page-local character offsets remain meaningful. Document content is not concatenated across pages for chunk generation.

### 3.2 Corpus snapshot

The inspected corpus contained six Markdown documents, each represented by one page record.

| Source file | Characters | Pages |
|---|---:|---:|
| `README_03_Local_Runtime_Engineering.md` | 14,492 | 1 |
| `README_04_prompt_construction_and_structured_output.md` | 9,608 | 1 |
| `README_05_end_to_end_local_ai_assistant.md` | 32,360 | 1 |
| `README_06_prompt_regression_testing.md` | 22,690 | 1 |
| `README_07_hadota_web_chat.md` | 9,497 | 1 |
| `README_08_HADOTA_Local_AI_Assistant.md` | 29,557 | 1 |
| **Total** | **118,204** | **6** |

This is the corpus snapshot used for the persisted comparison report. Results may change when the source corpus changes.

### 3.3 Relevant input record fields

The existing document records contain fields including:

- `document_id`
- `file_name`
- `source_path`
- `file_type`
- `content`
- `source_hash`
- `content_hash`
- `file_size`
- `ingested_at`
- `metadata`

The page records contain:

- `document_id`
- `page_number`
- `content`
- `metadata`

The chunking layer uses the document identifier, page number, and page content to create traceable chunks.

## 4. Repository Components

The Day 2 implementation and tests are organized as follows:

```text
app/
└── chunking/
    ├── __init__.py
    ├── markdown.py
    ├── markdown_parser.py
    ├── models.py
    ├── recursive.py
    └── strategies.py

tests/
└── chunking/
    ├── __init__.py
    ├── test_markdown.py
    ├── test_models.py
    ├── test_recursive.py
    └── test_strategies.py

notebooks/
└── 02_text_chunking.ipynb

data/
└── knowledge_base/
    └── processed/
        ├── documents.jsonl
        ├── pages.jsonl
        └── chunks.jsonl

reports/
├── chunking_comparison.json
└── day_1_ingestion_report.json
```

The tree lists the relevant files observed during the Day 2 repository audit; it is not intended to be a complete inventory of the entire repository.

### 4.1 `app/chunking/models.py`

Defines the `ChunkRecord` Pydantic model used as the output contract for generated chunks.

### 4.2 `app/chunking/strategies.py`

Contains the fixed-window baseline:

```python
fixed_window_chunk(
    content: str,
    chunk_size: int,
    overlap: int,
) -> list[tuple[int, int, str]]
```

The function returns source start and end offsets alongside each chunk's text.

### 4.3 `app/chunking/markdown.py`

Contains the sequential Markdown parser exposed through:

```python
parse_markdown(content)
```

This parser was selected for the recursive Markdown path because it associates parsed blocks with section context.

### 4.4 `app/chunking/markdown_parser.py`

Contains a separate detector-oriented Markdown parsing implementation, including helpers for detecting headings and ordered and unordered lists. It was inspected and compared with the parser in `markdown.py` during implementation.

### 4.5 `app/chunking/recursive.py`

Contains the recursive splitting implementation and its `TextSegment` representation.

The inspected public functions are:

```python
recursive_chunk_text(
    content: str,
    chunk_size: int = 1200,
    min_chunk_size: int = 100,
)

recursive_chunk_markdown(
    content: str,
    chunk_size: int = 1200,
    min_chunk_size: int = 100,
)
```

`TextSegment` carries the segment content, source character offsets, section path, and block type.

## 5. Chunk Record Schema and Provenance

Every generated chunk is represented by a `ChunkRecord` with the following fields:

| Field | Type | Meaning |
|---|---|---|
| `chunk_id` | `str` | Chunk identifier |
| `document_id` | `str` | ID of the source document |
| `page_number` | `int` | Page from which the chunk was produced |
| `chunk_index` | `int` | Position of the chunk within its source page's generated sequence |
| `content` | `str` | Chunk text |
| `char_start` | `int` | Start offset in the original page content |
| `char_end` | `int` | End offset in the original page content |
| `content_hash` | `str` | 64-character hexadecimal content hash |
| `section_path` | `list[str]` | Markdown heading hierarchy associated with the chunk |
| `strategy` | `str` | Strategy used to produce the chunk |
| `config_signature` | `str` | Fingerprint identifying the chunking configuration |
| `chunking_version` | `int` | Version of the chunking implementation/configuration contract |
| `metadata` | `dict[str, Any]` | Additional chunk metadata |

### 5.1 Provenance requirements

The chunk record keeps the relationship between each chunk and its original source:

- `document_id` identifies the source document.
- `page_number` identifies the source page.
- `char_start` and `char_end` locate the chunk in the original page text.
- `section_path` preserves Markdown heading context where available.
- `strategy`, `config_signature`, and `chunking_version` describe how the chunk was generated.
- `content_hash` supports content-integrity validation.

The persistence validation confirmed that the stored offsets and chunk content matched the corresponding source text for all reloaded records.

## 6. Chunking Configuration

The persisted experiment used the following configuration:

| Parameter | Value | Applied to |
|---|---:|---|
| `chunk_size` | 1,200 characters | Both strategies |
| `overlap` | 180 characters | Fixed-window only |
| `min_chunk_size` | 100 characters | Recursive chunking |
| `chunking_version` | 1 | Generated chunk records |

The selected strategy was `recursive_markdown`.

The configuration signature recorded in the experiment report is:

```text
555f1d45d63fb9e67e25d4493aabc970af4bc72c4e04c2c698445f4935d2944f
```

The signature is stored with generated records to identify the configuration used for the experiment. It should be regenerated by the project's configuration-signature logic when the relevant configuration changes; it should not be manually reused for a different setup.

## 7. Fixed-Window Chunking

### 7.1 Purpose

Fixed-window chunking provides a straightforward baseline. It divides page text into character windows and uses overlap between neighboring windows to reduce the chance that information near a boundary is separated without shared context.

### 7.2 Experiment settings

- Maximum configured window size: 1,200 characters.
- Overlap between windows: 180 characters.
- The strategy returns each chunk with source offsets.
- The implementation is tested for normal inputs, boundary conditions, and invalid configurations.

### 7.3 Characteristics

Advantages:

- Simple and inexpensive to execute.
- Produces chunks close to the configured size for most of this corpus.
- Provides a useful baseline for evaluating structure-aware chunking.

Limitations:

- Boundaries are determined by character positions rather than Markdown structure.
- A heading, list, table, paragraph, or code block can be divided between chunks.
- Overlap reduces some boundary effects but does not guarantee structural preservation.

## 8. Markdown Structure Parsing

The Markdown parsing work identifies blocks and heading context before structure-aware chunk assembly.

### 8.1 Structural elements considered

The parser and its tests cover the Markdown constructs used in the Day 2 work, including:

- ATX headings and heading hierarchy.
- Paragraphs.
- Ordered lists.
- Unordered lists.
- Markdown tables.
- Fenced code blocks.
- Horizontal rules.

Code-fence handling is important because Markdown-looking text inside a fenced code block must not be misclassified as document structure. For example, a line beginning with `#` inside a code fence is code content, not a Markdown heading.

### 8.2 Section hierarchy

Headings establish a hierarchy. A parsed block can carry a `section_path` representing the heading ancestry that applies to that block.

For example, a paragraph under `# Machine Learning` and `## Learning Algorithms` can carry:

```python
["Machine Learning", "Learning Algorithms"]
```

A later heading at the same level replaces the previous heading at that level while retaining its parent context.

### 8.3 Parsing and offset integrity

The parser retains source-position information so that parsed blocks and generated segments can be traced back to the original page content. The notebook includes tests and inspections for line tracking, code-fence boundaries, heading detection, block classification, and edge cases.

### 8.4 Parser selection

Two parser modules exist in the package:

- `parse_markdown()` from `app/chunking/markdown.py` — sequential parser selected for the recursive Markdown path.
- `parse_markdown_blocks()` from `app/chunking/markdown_parser.py` — alternative detector-oriented parser inspected during the implementation.

The choice of `parse_markdown()` was based on its section-context behavior in the inspected implementation and the comparative validation performed in the notebook.

## 9. Recursive Structure-Aware Chunking

### 9.1 Purpose

Recursive Markdown chunking aims to keep related document structure together rather than splitting the source only at fixed character positions.

### 9.2 Splitting behavior

The recursive implementation uses structural separators such as paragraph and newline boundaries, followed by sentence-level splitting and a hard fallback when a segment still exceeds the configured size.

The Markdown path first uses parsed structural blocks and section context to assemble `TextSegment` objects. Oversized content may require further splitting; therefore, structural preservation is preferred when possible but is not an absolute guarantee for every block.

### 9.3 Section context and block types

Generated segments carry:

- Text content.
- Source start and end offsets.
- Section path.
- Block type.

This metadata helps preserve context for downstream processing and makes the chunk output more informative than plain text windows.

### 9.4 Minimum-size behavior

The configured recursive minimum chunk size is 100 characters, but it is not a strict guarantee that every output chunk will contain at least 100 characters. Preserving natural Markdown boundaries can produce smaller chunks. The experiment produced 19 recursive chunks below 100 characters, with a minimum size of 22 characters.

These small chunks were retained in the experiment rather than silently merged across structural boundaries. Their effect on future retrieval quality remains to be evaluated.

### 9.5 Overlap behavior

The fixed-window strategy exposes an overlap parameter of 180 characters. The inspected recursive function signatures do not expose an equivalent overlap parameter. Although structure-aware overlap was considered in the design discussion, it should not be treated as an implemented recursive API capability in this version.

## 10. Strategy Comparison

### 10.1 Experimental method

Both strategies were run against the same six page records and the same 118,204 source characters.

The experiment recorded:

- Number of generated chunks.
- Mean, median, minimum, and maximum chunk sizes.
- Chunk-size distribution.
- Covered source characters and source coverage percentage.
- Number of detected structural blocks and the number preserved within a single chunk.
- Structural-preservation percentage.
- Integrity-error count and details.
- Processing-time samples and summary statistics.
- Per-document measurements.

Processing time was sampled five times for each strategy. Timings are specific to this machine, Python environment, corpus, and execution conditions; they are not universal performance guarantees.

### 10.2 Overall results

The following values are taken from the persisted `reports/chunking_comparison.json` file.

| Metric | Fixed-window | Recursive Markdown |
|---|---:|---:|
| Documents | 6 | 6 |
| Pages | 6 | 6 |
| Source characters | 118,204 | 118,204 |
| Generated chunks | 119 | 228 |
| Mean chunk size | 1,164.24 | 518.44 |
| Median chunk size | 1,200 | 457.5 |
| Minimum chunk size | 212 | 22 |
| Maximum chunk size | 1,200 | 1,185 |
| Chunks below 100 characters | 0 | 19 |
| Chunks from 100 to 399 characters | 3 | 76 |
| Chunks from 400 to 799 characters | 2 | 85 |
| Chunks from 800 to 1,200 characters | 114 | 48 |
| Chunks above 1,200 characters | 0 | 0 |
| Covered source characters | 118,204 | 118,204 |
| Source coverage | 100.00% | 100.00% |
| Structural blocks | 740 | 740 |
| Preserved structural blocks | 699 | 731 |
| Structural preservation | 94.46% | 98.78% |
| Integrity errors | 0 | 0 |
| Median processing time | 0.000070 s | 0.011590 s |
| Minimum processing time | 0.000069 s | 0.007432 s |
| Maximum processing time | 0.000073 s | 0.013968 s |

### 10.3 Per-document results

| Source file | Characters | Fixed chunks | Fixed structural preservation | Recursive chunks | Recursive structural preservation |
|---|---:|---:|---:|---:|---:|
| `README_03_Local_Runtime_Engineering.md` | 14,492 | 15 | 97.46% | 32 | 100.00% |
| `README_04_prompt_construction_and_structured_output.md` | 9,608 | 10 | 97.47% | 22 | 100.00% |
| `README_05_end_to_end_local_ai_assistant.md` | 32,360 | 32 | 93.68% | 69 | 98.95% |
| `README_06_prompt_regression_testing.md` | 22,690 | 23 | 98.64% | 48 | 100.00% |
| `README_07_hadota_web_chat.md` | 9,497 | 10 | 77.78% | 15 | 88.89% |
| `README_08_HADOTA_Local_AI_Assistant.md` | 29,557 | 29 | 85.71% | 42 | 96.19% |
| **Total** | **118,204** | **119** | — | **228** | — |

### 10.4 Interpretation

The persisted results show that:

1. Both strategies covered 100% of the source characters.
2. Neither strategy reported an integrity error.
3. Recursive Markdown preserved 731 of 740 structural blocks (98.78%), compared with 699 of 740 (94.46%) for fixed-window chunking.
4. Recursive Markdown generated 228 chunks, compared with 119 fixed-window chunks. This is approximately 91.6% more chunks.
5. Recursive Markdown produced smaller chunks on average, which is consistent with grouping content around structural boundaries rather than filling fixed-size windows.
6. The recorded median recursive processing time was approximately 11.59 ms, compared with approximately 0.07 ms for fixed-window chunking.

The timing difference is small in absolute terms for this six-document corpus, but it is still a measurable trade-off. Performance should be reevaluated on larger corpora and under controlled benchmark conditions before making broader throughput claims.

### 10.5 Meaning and limits of structural preservation

For this experiment, structural preservation measures whether a detected Markdown block remains fully contained within one chunk. It is a document-structure metric, not a direct measure of semantic coherence, answer accuracy, or retrieval quality.

A higher structural-preservation score is evidence that chunk boundaries align better with the parser's Markdown blocks. It does not prove that the resulting chunks are optimal for a particular embedding model or downstream question-answering workload.

## 11. Selected Strategy and Engineering Decision

The selected primary strategy is:

```text
recursive_markdown
```

The fixed-window strategy remains as a baseline for future comparisons.

The decision favors recursive Markdown because it achieved higher structural preservation while maintaining full source coverage and zero recorded integrity errors. Its larger chunk count and higher processing time are accepted for the current corpus.

This is an engineering decision based on the Day 2 measurements. It should be revisited if later retrieval evaluation shows that a different chunk size, overlap policy, or strategy produces better end-to-end results.

## 12. Persistence and Reload Validation

### 12.1 Persisted artifacts

| Artifact | Path | Purpose |
|---|---|---|
| Chunk records | `data/knowledge_base/processed/chunks.jsonl` | Persisted `ChunkRecord` objects |
| Comparison report | `reports/chunking_comparison.json` | Experiment configuration, corpus details, strategy metrics, and per-document results |

The selected recursive strategy generated 228 records. All 228 records were persisted and reloaded successfully.

### 12.2 Persistence implementation

The project already provides generic JSONL persistence through `persist_jsonl()` in `app/ingestion/pipeline.py`. The implementation writes to a temporary file in the destination directory, flushes and synchronizes the data, and then replaces the destination file. This supports atomic replacement of the JSONL output rather than writing partially generated records directly to the final path.

Day 2 reused this existing persistence infrastructure rather than introducing a separate storage mechanism.

### 12.3 Reload validation results

The persistence experiment reported the following checks as passing:

- Chunk JSONL file exists.
- Comparison report exists.
- All chunk records reload successfully.
- Chunk IDs are unique.
- Source offsets and chunk content are valid.
- Content hashes are valid.
- Strategy and configuration metadata are valid.
- Comparison report reloads successfully.
- No validation errors are reported.

| Persistence metric | Result |
|---|---:|
| Generated records | 228 |
| Persisted records | 228 |
| Reloaded records | 228 |
| Validation errors | 0 |
| Chunk JSONL size | 290,032 bytes |
| Comparison report size | 9,103 bytes |

## 13. Notebook Workflow

The Day 2 experiments and implementation notes are documented in:

```text
notebooks/02_text_chunking.ipynb
```

The final notebook audit recorded 126 cells:

- 93 Markdown cells.
- 33 code cells.
- 33 executed code cells.
- 0 code cells without an execution count.

The notebook's main sections are:

1. Objective and Scope.
2. Load Existing Processed Records.
3. Input Validation and Corpus Audit.
4. Understanding Document Structure.
5. Chunking Fundamentals and Configuration.
6. Chunk Record Schema.
7. Fixed-Window Chunking.
8. Markdown Structure Parsing.
9. Recursive Structure-Aware Chunking.
10. Persist and Reload.
11. Tests and Final Validation.

The notebook includes focused investigations and experiments within these sections, including parser edge cases, code-fence handling, heading hierarchy, block classification, recursive chunk generation, strategy comparison, and persistence checks.

### 13.1 Re-running the notebook

Run the notebook from the beginning when reproducing the full workflow. A Jupyter kernel restart clears in-memory variables, even when previous cell outputs remain visible. Cells that depend on values created by earlier experiments must therefore be executed again before running later comparison or persistence cells.

## 14. Test Coverage and Regression Results

The chunking test package contains these modules:

```text
tests/chunking/
├── test_markdown.py
├── test_models.py
├── test_recursive.py
└── test_strategies.py
```

The focused chunking test run completed successfully:

```powershell
python -m pytest tests/chunking -v
```

Result:

```text
53 passed
```

The full project regression run also completed successfully:

```powershell
python -m pytest -q
```

Result:

```text
515 passed
```

These results refer to the test runs completed during Day 2. Run the commands again after subsequent code changes to verify the current repository state.

## 15. Reproduction Commands

Run the following commands from the project root in PowerShell.

### 15.1 Activate the project environment

Use the environment activation method configured for the repository. For a standard Windows virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

If the project uses a different environment path or environment manager, use that configuration instead.

### 15.2 Run focused chunking tests

```powershell
python -m pytest tests/chunking -v
```

### 15.3 Run the full regression suite

```powershell
python -m pytest -q
```

### 15.4 Open the Day 2 notebook

```powershell
jupyter notebook notebooks/02_text_chunking.ipynb
```

Alternatively, open the notebook directly in VS Code and select the project's Python environment as the kernel.

## 16. Known Limitations and Follow-up Work

The following points are known boundaries of the current Day 2 implementation or experiment.

### 16.1 Small recursive chunks

The recursive experiment produced 19 chunks below 100 characters. They were retained to preserve structural boundaries. Their impact on retrieval should be measured later rather than assumed to be harmful or beneficial.

### 16.2 Recursive overlap is not exposed

The inspected recursive function signatures do not include an overlap parameter. The 180-character overlap applies to the fixed-window baseline only. A future recursive overlap policy would require a deliberate implementation and source-offset validation.

### 16.3 Structural preservation is not retrieval evaluation

The experiment evaluates containment of parsed Markdown blocks, not retrieval relevance, recall, ranking quality, context sufficiency, or answer correctness. Those require a separate retrieval evaluation dataset and metrics.

### 16.4 Timing is environment-dependent

The timing samples are small and specific to the current machine and execution environment. They are useful for this experiment's comparison but should not be treated as stable production benchmarks.

### 16.5 Corpus coverage

The inspected Day 2 corpus consisted of six Markdown files represented by six page records. The notebook contains design and inspection work related to file-type routing, but the reported comparison results are specifically for this Markdown corpus. They do not establish equivalent behavior for every PDF, TXT, HTML, or other document type.

### 16.6 CLI and unified chunking pipeline

The Day 2 repository audit did not find a dedicated `app/chunking/cli.py` or `app/chunking/pipeline.py`. These are not represented as completed Day 2 deliverables in this README. They can be introduced in a later integration task if the application workflow requires them.

## 17. Explicitly Out of Scope for Day 2

The following work was intentionally deferred:

- Embedding generation.
- Vector database selection or integration.
- Similarity search.
- Retrieval and context assembly.
- Reranking.
- LLM answer generation.
- End-to-end RAG evaluation.
- Production serving or API integration for chunk retrieval.

These components belong to later stages of the RAG system.

## 18. Final Day 2 Status

| Deliverable | Status |
|---|---|
| Reuse Day 1 processed records | Complete |
| Inspect and validate source corpus | Complete |
| Define and validate `ChunkRecord` | Complete |
| Implement fixed-window baseline | Complete |
| Parse Markdown structure | Complete |
| Implement recursive Markdown chunking | Complete |
| Preserve source provenance and offsets | Complete; validation passed |
| Compare both chunking strategies | Complete |
| Select primary strategy | `recursive_markdown` selected |
| Persist 228 chunk records | Complete |
| Persist comparison report | Complete |
| Reload and validate persisted outputs | Complete; 0 validation errors |
| Focused chunking tests | 53 passed |
| Full project regression | 515 passed |
| Day 2 notebook | Complete; 126 cells, 33 code cells executed |

**Day 2 is complete.** The project has a tested chunking layer, persisted chunk records, a reproducible comparison artifact, and a documented strategy decision. The next RAG stage can consume `chunks.jsonl` as its chunk-level input, subject to the next stage's embedding and retrieval design.
