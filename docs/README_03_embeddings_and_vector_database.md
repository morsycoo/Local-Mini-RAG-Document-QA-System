# Day 3 — Embeddings and Local Vector Database

> **Project:** Local Mini RAG Document Question Answering System  
> **Stage:** Day 3 — Embeddings and Local Vector Database  
> **Status:** Implementation and recorded validation completed; semantic retrieval evaluation is the next stage.

---

## 1. Executive Summary

Day 3 implemented the document-embedding and persistent vector-storage layer of the project. The processed document corpus was embedded using a pinned revision of `intfloat/multilingual-e5-base`. The resulting 228 vectors (768 dimensions each) were validated and indexed in a persistent ChromaDB collection using cosine distance.

The collection was reopened through a new persistent client, and the recorded checks confirmed that it contained 228 records and that its stored IDs matched the source chunk IDs. Re-indexing was also tested for idempotency: the collection remained at 228 records and no duplicate records were created.

The recorded full project test run reported **727 passed, 0 failed, and 23 warnings**. The warnings were documented as ChromaDB deprecation warnings. Code coverage was not measured.

This stage establishes the data and storage foundation for semantic retrieval. It does **not** establish retrieval relevance or answer quality; those require separate retrieval experiments and evaluation.

## 2. Scope and Outcomes

### 2.1 Implemented in Day 3

- Finalized and pinned the embedding model configuration.
- Loaded the processed corpus and its chunk identifiers.
- Generated document embeddings for the existing 228 chunks.
- Saved and validated the embedding matrix and manifest.
- Audited token lengths against the model's 512-token input limit.
- Implemented ChromaDB configuration, storage, metadata mapping, and indexing components.
- Created a persistent ChromaDB collection using cosine distance.
- Indexed the 228 source chunks with their embeddings and metadata.
- Validated persistence, record counts, IDs, sampled records, and indexing idempotency.
- Added automated tests for vector-store configuration, exceptions, ChromaDB storage, metadata mapping, and indexing.
- Produced `reports/3_embedding_report.json`.
- Documented the final results and the proposed Day 4 handoff in the project notebook.

### 2.2 Explicitly Not Established in Day 3

The following were not completed or measured in this stage:

- Retrieval relevance or ranking quality.
- Recall@K, Precision@K, MRR, or NDCG.
- Similarity-threshold optimization.
- Retrieval latency benchmarks.
- Comparative embedding-model benchmarks.
- HNSW parameter benchmarking against alternatives.
- Code coverage.
- Production-scale indexing performance.

A successful indexing pipeline and passing tests must not be interpreted as evidence that retrieved chunks will be relevant to every query.

## 3. Environment

The experiment report records the following environment:

| Component | Recorded value |
|---|---|
| Operating system | Windows |
| Python | 3.13.0 |
| PyTorch | 2.12.0+cu130 |
| Transformers | 5.19.0 |
| Sentence Transformers | 6.3.0 |
| Tokenizers | 0.23.2 |
| ChromaDB | 1.5.9 |
| NumPy | 2.4.5 |
| pytest | 9.0.3 |
| Pydantic | 2.13.4 |
| CUDA availability | `True` |
| CUDA version | 13.0 |
| GPU identified | NVIDIA GeForce RTX 5070 Laptop GPU |

**Device-selection caveat:** CUDA availability and the GPU were recorded, but the device selected during the original embedding-generation run was not independently retained. The report therefore records `selected_embedding_device` as `null`. It would be inaccurate to claim, from this evidence alone, that the original embedding generation definitely ran on the GPU.

The report also leaves embedding batch size and embedding-generation duration as `null`, because those measurements were not available in the retained execution evidence.

## 4. Source Corpus

The processed corpus used by Day 3 contained:

| Metric | Result |
|---|---:|
| Documents | 6 |
| Pages | 6 |
| Chunks | 228 |
| Unique chunk IDs | 228 |
| Empty chunks | 0 |
| Duplicate chunk IDs | 0 |
| Missing parent references | 0 |

### 4.1 Processed corpus artifacts

```text
data/
└── knowledge_base/
    └── processed/
        ├── chunks.jsonl
        ├── documents.jsonl
        ├── pages.jsonl
        └── embeddings/
            ├── embeddings.npz
            └── manifest.json
```

### 4.2 Source hashes recorded in the report

**Chunks SHA-256**

```text
5d8b10ac55184124f0b0ffe7e4fad1bafd340e27d248cb31a61db4ce8d831fa2
```

**Documents SHA-256**

```text
c8f1435055ca3c6c2ff7289e2af197088a86a2e6388515adf5a8f76d50e72264
```

The chunks hash was recorded as matching the source hash in the embedding manifest. The report also records that the corpus has 228 unique chunk IDs, no empty chunks, no duplicate chunk IDs, and no missing parent references.

## 5. Embedding Model and Configuration

### 5.1 Model identity

| Parameter | Value |
|---|---|
| Model ID | `intfloat/multilingual-e5-base` |
| Revision | `ea6fe3baa7f6bfe27fbc99fde19d01b3e2bca610` |
| Embedding dimension | 768 |
| Maximum sequence length | 512 |
| Output data type | `float32` |
| Normalize embeddings | `True` |
| Query prefix | `query: ` |
| Passage prefix | `passage: ` |
| Local files only | `True` |

The model revision is pinned so that the embedding configuration is identifiable and can be reused consistently. The query and passage prefixes are part of the model's expected usage and must not be interchanged.

### 5.2 Embedding artifact

```text
data/knowledge_base/processed/embeddings/
├── embeddings.npz
└── manifest.json
```

Recorded artifact details:

| Metric | Result |
|---|---:|
| Embedding count | 228 |
| Matrix shape | `(228, 768)` |
| Data type | `float32` |
| Zero vectors | 0 |
| Mean L2 norm | 1.0 |
| Minimum L2 norm | 0.9999999403953552 |
| Maximum L2 norm | 1.0000001192092896 |
| All values finite | `True` |
| Chunk-ID alignment | `True` |

**Embedding artifact SHA-256**

```text
1587343fa3e0c2b0bdca75999572684de5d6f1a247f08842c86cefcacfdc773e
```

**Manifest SHA-256**

```text
a17c6e49eda82c7bc759b74b5fd7cf9dd70fbf16e9ef5ec272523d2bc3baa073
```

The embedding report records that the embedding artifact hash matched the manifest and that the embedding matrix passed the recorded shape, dtype, finite-value, zero-vector, normalization, and chunk-ID-alignment checks.

### 5.3 Embedding integrity interpretation

The observed vector norms are effectively 1.0, with only small floating-point deviations. This is consistent with the configured normalization behavior. No zero vectors were found, and the embedding matrix contains one 768-dimensional vector for each of the 228 source chunks.

The retained report does not establish the original embedding-generation batch size, generation duration, or selected execution device. These remain unreported rather than inferred.

## 6. Tokenization Audit

### 6.1 Audit method

The token-length preflight was performed using the `tokenizers` library directly against the local, pinned Hugging Face snapshot's `tokenizer.json`.

The audit used:

- Model: `intfloat/multilingual-e5-base`
- Pinned revision: `ea6fe3baa7f6bfe27fbc99fde19d01b3e2bca610`
- Passage prefix: `passage: `
- Special tokens: included
- Truncation: disabled
- Maximum input length: 512 tokens
- Chunks audited: 228

The direct `tokenizers` approach was used after a Transformers import attempt encountered a Windows paging-file error while loading a CUDA library. The successful audit used the local tokenizer file and did not require rerunning the embedding-generation process.

### 6.2 Results

| Metric | Result |
|---|---:|
| Chunks audited | 228 |
| Minimum token count | 8 |
| Median token count | 109.5 |
| Mean token count | 139.19298245614036 |
| 95th percentile | 326.3 |
| Maximum token count | 452 |
| Model input limit | 512 |
| Overflow count | 0 |
| All chunks within limit | `True` |

### 6.3 Decision

All 228 chunks were within the 512-token limit. The longest observed input was 452 tokens, leaving 60 tokens below the configured limit.

Therefore:

- No chunk required tokenizer-aware subdivision.
- No source chunk was modified for this audit.
- No truncation was applied.
- The existing 228-chunk corpus was retained unchanged.
- The existing document embeddings did not need to be regenerated because of token-length overflow.

## 7. Vector Database Architecture

### 7.1 Storage technology

ChromaDB was used as the local vector database. The implementation uses a persistent client so that the indexed collection remains available across application processes.

**Persistence path**

```text
data/knowledge_base/vector_store/chroma/
```

**Collection name**

```text
hadota_chunks_e5_base_v1
```

### 7.2 Collection configuration

| Parameter | Value |
|---|---|
| Client type | `PersistentClient` |
| Collection | `hadota_chunks_e5_base_v1` |
| Distance metric | `cosine` |
| Embedding dimension | 768 |
| Embedding function | `None` |
| Indexed records | 228 |

The collection is configured with `embedding_function=None` because document embeddings are generated externally using the pinned multilingual E5 configuration. The vector-store layer stores and searches those supplied vectors; it must not silently substitute ChromaDB's default embedding function.

### 7.3 Collection metadata

The recorded collection metadata is:

```python
{
    "distance_metric": "cosine",
    "embedding_model": "intfloat/multilingual-e5-base",
    "embedding_revision": "ea6fe3baa7f6bfe27fbc99fde19d01b3e2bca610",
    "embedding_dimension": 768
}
```

### 7.4 HNSW configuration

The inspected collection reported:

| Parameter | Value |
|---|---:|
| `space` | `cosine` |
| `ef_construction` | 100 |
| `ef_search` | 100 |
| `max_neighbors` | 16 |
| `resize_factor` | 1.2 |
| `sync_threshold` | 1000 |

These are the configured values recorded from the inspected collection. They were not established as optimal through comparative benchmarking.

## 8. Vector Store Implementation

The vector-store implementation was organized under:

```text
app/
└── vector_store/
    ├── __init__.py
    ├── config.py
    ├── exceptions.py
    ├── chroma_store.py
    ├── metadata.py
    └── indexer.py
```

The existing `app/config.py` serves as runtime configuration for the application and Ollama; the vector-store configuration is kept separately under `app/vector_store/config.py`.

### 8.1 Configuration — `config.py`

The vector-store configuration defines the settings required to connect to and manage the persistent ChromaDB collection, including its persistence location and collection identity.

### 8.2 Exceptions — `exceptions.py`

The vector-store exception module provides explicit error types for vector-store operations and validation failures, rather than relying on ambiguous generic failures.

### 8.3 ChromaDB adapter — `chroma_store.py`

The ChromaDB storage component encapsulates persistent-client and collection interactions. The implementation uses the configured cosine collection and explicitly disables ChromaDB-managed embedding generation so externally generated E5 vectors are used.

### 8.4 Metadata mapping — `metadata.py`

The metadata mapper converts the source chunk metadata into the representation accepted by ChromaDB. Its validation is covered by the dedicated metadata-mapping tests.

### 8.5 Indexing — `indexer.py`

The indexer validates the source and embedding inputs before writing batches. The recorded implementation checks include:

- Source and artifact hashes.
- Chunk IDs and ID uniqueness.
- Embedding shape and dimension.
- Embedding dtype.
- Finite embedding values.
- Normalization expectations.
- Alignment between chunk IDs and embeddings.
- Compatibility with the configured embedding model.
- Batch preparation before writes.
- Unexpected IDs.
- Batch-level and final ID/count verification.

The indexer uses a recorded indexing batch size of 64. The successful indexing run processed four batches: 64, 64, 64, and 36 records.

## 9. Indexing Results

### 9.1 Initial indexing

| Metric | Result |
|---|---:|
| Source chunks | 228 |
| Embeddings | 228 |
| Indexed records | 228 |
| Batch size | 64 |
| Number of batches | 4 |
| Batch sizes | 64, 64, 64, 36 |
| Final collection count | 228 |

The successful indexing run inserted the 228 source chunk records into the configured collection.

### 9.2 Persistence validation

The persistence check reopened the collection through a new persistent client.

| Check | Recorded result |
|---|---:|
| Collection count after reopen | 228 |
| Stored IDs | 228 |
| Missing IDs | 0 |
| Unexpected IDs | 0 |
| Sampled records inspected | 3 |
| Sampled documents returned | 3 |
| Sampled metadata records returned | 3 |
| Sampled embedding shape | `(3, 768)` |

The sampled indices were 0, 100, and 227. The recorded inspection found that the sampled documents, metadata, and embeddings matched their corresponding source data.

The three-record inspection is a sample, not a complete content-by-content comparison of all 228 records. Full ID and count integrity checks were also covered by the indexer validation tests.

### 9.3 Idempotency validation

The recorded re-indexing test produced:

| Metric | Before / Result |
|---|---:|
| Count before re-indexing | 228 |
| Records processed | 228 |
| Count after re-indexing | 228 |
| Duplicate records created | 0 |

This demonstrates that the tested re-indexing operation did not increase the collection count or create duplicate records.

## 10. Automated Testing

### 10.1 Full project test run

The retained full-suite output reported:

```text
727 passed
23 warnings
0 failed

Execution time: 55.88 seconds
```

| Metric | Result |
|---|---:|
| Passed | 727 |
| Failed | 0 |
| Warnings | 23 |
| Duration | 55.88 seconds |
| Test success rate | 100% |
| Code coverage | Not measured |

The 100% figure is the test success rate (`727 / 727`), not code coverage.

**Evidence note:** The experiment report explicitly records that these full-suite results came from previously observed pytest output and were not freshly executed by the report-generation script.

### 10.2 Dedicated vector-store tests

The recorded vector-store test breakdown is:

| Test module | Passed |
|---|---:|
| Configuration | 19 |
| Exceptions | 11 |
| ChromaDB Store | 17 |
| Metadata Mapping | 62 |
| Indexer | 37 |
| **Total** | **146** |

The indexer module's recorded result was 37 passing tests. The complete project suite's recorded result was 727 passing tests.

### 10.3 Warnings

The 23 warnings were recorded as ChromaDB deprecation warnings, including:

```text
DeprecationWarning: legacy embedding function config
```

The warning was reported from:

```text
chromadb/api/models/CollectionCommon.py:155
```

The warnings did not cause test failures. They are retained as a dependency-compatibility item for future review.

### 10.4 Runtime smoke tests

The retained test output also reported successful real Ollama smoke tests for:

- Non-streaming generation.
- JSON Schema generation.
- Streaming generation.

These smoke tests are part of the recorded full test-run evidence; they do not constitute a retrieval-quality benchmark.

## 11. Experiment Report

The Day 3 report was created at:

```text
reports/3_embedding_report.json
```

The report-generation check recorded:

| Check | Result |
|---|---|
| Report status | `created_and_verified` |
| JSON parse | Valid |
| Report size | 4,930 bytes |
| Chunk count | 228 |
| Embedding count | 228 |
| ChromaDB count | 228 |
| Tokenization overflow count | 0 |

The report contains environment details, corpus statistics and hashes, tokenization audit results, embedding configuration and validation, vector-database configuration and persistence results, idempotency results, testing results, limitations, and overall validation flags.

The report's overall validation fields record:

| Validation | Result |
|---|---|
| Corpus valid | `True` |
| Tokenization valid | `True` |
| Embedding artifacts valid | `True` |
| Vector database reopened | `True` |
| Vector database IDs match | `True` |
| Retrieval quality evaluated | `False` |

The report records the selected embedding device, embedding batch size, and embedding-generation duration as unknown (`null`) rather than inventing values.

## 12. Final Pipeline Consistency

The recorded artifact counts are consistent across the processed corpus, embedding artifacts, and persistent vector collection:

```text
Source Documents
       |
       v
Document Ingestion
       |
       v
Processed Corpus
  6 Documents
  6 Pages
       |
       v
Text Chunking
  228 Chunks
       |
       v
E5 Embedding Generation
  228 x 768
       |
       v
Embedding Artifact Validation
       |
       v
Metadata Mapping
       |
       v
Persistent ChromaDB Collection
  228 Records
       |
       v
Persistence and Integrity Validation
       |
       v
Automated Testing
  727 Passed
```

| Pipeline stage | Count |
|---|---:|
| Processed documents | 6 |
| Processed pages | 6 |
| Generated chunks | 228 |
| Generated embeddings | 228 |
| Stored ChromaDB records | 228 |

## 13. Known Limitations and Engineering Notes

1. **Small corpus:** The current knowledge base contains six documents and 228 chunks. Its behavior may not generalize to larger or more diverse corpora.
2. **Single embedding configuration:** Only `intfloat/multilingual-e5-base` at the recorded pinned revision was implemented and validated. No comparative model benchmark was completed.
3. **Unrecorded generation metrics:** The original embedding-generation device, batch size, and duration were not independently retained.
4. **Retrieval quality not evaluated:** No Recall@K, Precision@K, MRR, NDCG, query-relevance study, or similarity-threshold optimization was completed.
5. **No latency benchmark:** Retrieval and large-scale indexing latency were not benchmarked.
6. **HNSW tuning not benchmarked:** The recorded HNSW values are configuration evidence, not proof of optimal settings.
7. **No code coverage:** Coverage was not measured.
8. **Dependency warnings:** The full test run reported 23 ChromaDB deprecation warnings.
9. **Sample versus full-content verification:** Three records were sampled for document, metadata, and vector inspection. This sample should not be described as a full content comparison of every stored record.

## 14. Day 3 Acceptance Summary

| Area | Recorded status |
|---|---|
| Python and library environment | Validated |
| CUDA availability and GPU identification | Recorded |
| Embedding model configuration | Pinned |
| Processed corpus | 6 documents, 6 pages, 228 chunks |
| Tokenization audit | 228 audited, 0 overflow |
| Embedding artifact | 228 × 768, float32 |
| Embedding validation | Passed recorded checks |
| ChromaDB persistent collection | Available |
| Indexed records | 228 |
| Persistence after reopen | Verified |
| Source IDs versus stored IDs | Match |
| Re-indexing idempotency | Verified; 0 duplicates |
| Dedicated vector-store tests | 146 passed |
| Recorded full project test run | 727 passed, 0 failed |
| Experiment report | Created and JSON-verified |
| Retrieval quality | Not yet evaluated |

**Day 3 implementation and recorded validation: completed.**  
**Semantic retrieval evaluation: deferred to the next stage.**

## 15. Day 4 Handoff — Semantic Retrieval & Similarity Search

Day 4 should build on the existing corpus, embeddings, and persistent ChromaDB collection. It should not regenerate or reset those artifacts without a separately justified and approved migration.

### 15.1 Existing inputs to reuse

**Query embedding configuration**

```python
{
    "model_id": "intfloat/multilingual-e5-base",
    "revision": "ea6fe3baa7f6bfe27fbc99fde19d01b3e2bca610",
    "embedding_dimension": 768,
    "query_prefix": "query: ",
    "normalize_embeddings": True
}
```

**Persistent database**

```text
data/knowledge_base/vector_store/chroma/
```

**Collection**

```text
hadota_chunks_e5_base_v1
```

**Supporting processed artifacts**

```text
data/knowledge_base/processed/chunks.jsonl
data/knowledge_base/processed/documents.jsonl
data/knowledge_base/processed/embeddings/manifest.json
```

### 15.2 Input contract

The initial retrieval component is expected to receive a non-empty query string, encode it with the same pinned E5 model revision using the `query: ` prefix, and produce a normalized 768-dimensional query vector.

It should connect to the existing persistent collection and return ranked chunks with enough source information for downstream context construction.

Expected logical result fields:

| Field | Purpose |
|---|---|
| `chunk_id` | Unique chunk identifier |
| `document_id` | Original document identifier |
| `text` | Retrieved chunk content |
| `metadata` | Associated chunk metadata |
| `distance` | ChromaDB distance value |

The exact result schema must be aligned with the existing chunk and metadata models during implementation; the table above is the handoff's logical contract, not a claim that a final retrieval result model already exists.

### 15.3 Proposed Day 4 implementation scope

1. **Retrieval architecture:** Create a component that accepts a query, prepares it, generates a query embedding, connects to ChromaDB, performs vector search, and returns ranked chunks.
2. **Query embedding:** Reuse the pinned model revision, apply `query: `, validate the 768-dimensional output, and preserve normalization behavior.
3. **Similarity search:** Support configurable Top-K, cosine-distance search, chunk text, metadata, document identifiers, and distance values.
4. **Result schema:** Define a validated result model aligned with the actual source metadata.
5. **Metadata-aware retrieval:** Preserve source traceability and design filters only around metadata fields that actually exist.
6. **Automated tests:** Cover query validation, prefix handling, embedding dimensions, collection connection, Top-K behavior, empty results, metadata preservation, result ordering, invalid input, dimension mismatch, and database connection failures.
7. **Qualitative experiments:** Run representative queries and inspect retrieved text, ordering, distances, source IDs, and relevance.

### 15.4 Initial Day 4 exclusions

Unless separately approved, the initial retrieval implementation should not:

- Reprocess the document corpus.
- Regenerate all document embeddings.
- Recreate or reset the ChromaDB collection.
- Fine-tune the embedding model.
- Add cross-encoder reranking.
- Add hybrid BM25 and vector retrieval.
- Claim a full retrieval benchmark without a suitable evaluation set.
- Perform production deployment.

These can be considered after the baseline retrieval implementation has been validated.

## 16. Final Deliverable Inventory

| Deliverable | Project path |
|---|---|
| Processed chunks | `data/knowledge_base/processed/chunks.jsonl` |
| Processed documents | `data/knowledge_base/processed/documents.jsonl` |
| Processed pages | `data/knowledge_base/processed/pages.jsonl` |
| Embedding matrix | `data/knowledge_base/processed/embeddings/embeddings.npz` |
| Embedding manifest | `data/knowledge_base/processed/embeddings/manifest.json` |
| Persistent ChromaDB | `data/knowledge_base/vector_store/chroma/` |
| ChromaDB collection | `hadota_chunks_e5_base_v1` |
| Day 3 notebook | `notebooks/03_embeddings_and_vector_database.ipynb` |
| Day 3 experiment report | `reports/3_embedding_report.json` |
| This README | `docs/README_03_embeddings_and_vector_database.md` |

---

## 17. Final Conclusion

Day 3 completed the implementation of the embedding and persistent vector-storage foundation for the Local Mini RAG Document Question Answering System.

The recorded results show:

- Six processed documents and six pages.
- 228 source chunks.
- 228 validated 768-dimensional float32 embeddings.
- Zero tokenization overflows across the 228 audited chunks.
- A persistent ChromaDB collection containing 228 records.
- Successful recorded persistence, ID-integrity, and idempotency checks.
- 146 passing dedicated vector-store tests.
- A recorded full project test run of 727 passed, 0 failed, and 23 warnings.
- A created and JSON-verified experiment report.

The next stage is **Semantic Retrieval & Similarity Search**. It should reuse the existing artifacts and establish retrieval behavior and relevance through dedicated tests and experiments. Day 3's successful indexing and validation are the foundation for that work, not a substitute for retrieval evaluation.
