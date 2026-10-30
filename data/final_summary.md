# Final Summary

## Objective

Study real local LLM generation, decoding behavior, autoregressive generation,
inference timing, and runtime limitations.

## Environment

- Runtime: Ollama
- Local model: qwen3.8:27b
- Hardware: NVIDIA GeForce RTX 5070 Laptop GPU
- Python: 3.13
- PyTorch: 2.12.0+cu130
- CUDA available: True

## Completed Work

- Verified the local model runtime and matching tokenizer.
- Tokenized real prompts and inspected token IDs.
- Generated real text through the Ollama API.
- Studied greedy decoding, sampling, temperature, top-k, and top-p.
- Demonstrated autoregressive context updates and stop conditions.
- Measured aggregate generation latency and token counts.
- Measured TTFT using the first non-empty streamed response chunk.
- Examined prompt evaluation and generation timing metadata.
- Created and validated the benchmark dataset.
- Created and validated the experiment matrix.
- Recorded nine engineering observations.

## Benchmark Dataset

- Total prompts: 24
- Categories: 6
- Prompts per category: four
- Dataset purpose: evaluation and benchmarking, not training

## Engineering Observations

- Greedy Decoding: Greedy decoding was deterministic under the tested configuration.
- Temperature: Lower temperature concentrated probability mass, while higher temperature produced a broader distribution.
- Top-k: Increasing top-k expanded the fixed-size candidate pool.
- Top-p: Increasing top-p allowed a larger probability-mass-based candidate pool.
- Output Length: Increasing max_new_tokens increased total generation latency in the tested runs.
- Input Context: Input-length timing measurements were non-monotonic across the tested single runs.
- TTFT: TTFT varied across individual runs and requires repeated controlled measurements for stronger conclusions.
- KV Cache: The Ollama API did not expose the model's internal KV Cache tensors.
- Runtime Observability: The runtime exposed aggregate generation metadata but not internal logits, hidden states, attention tensors, or KV Cache tensors.

## Runtime Limitations

The tested Ollama API did not expose internal model logits, hidden states,
attention tensors, or KV Cache tensors.

Low-level probability and cache mechanics were therefore demonstrated using
synthetic tensors. These simulations must not be represented as direct
measurements of qwen3.8:27b internal tensors.

Streaming chunks were not guaranteed to correspond one-to-one with tokens.
The recorded intervals are therefore inter-chunk latency, not exact token-level
ITL.

Input-context and TTFT measurements varied between individual runs. The
available single-run comparisons do not support universal performance claims.

## Final Mental Model

Prompt -> Token IDs -> Model Forward Pass -> Logits -> Probabilities
-> Decoding -> Next Token -> Updated Context -> Repeat

Runtime view:

Prompt Processing -> Prefill -> KV Cache, where supported -> Decode
-> Generated Tokens

Model behavior, decoding behavior, and runtime behavior must be analyzed
separately.

## Completion Assessment

Real end-to-end local generation was demonstrated.

The tested runtime did not provide direct access to a real qwen3.8:27b
forward pass and internal logits or KV Cache tensors. Those limitations remain
documented, and synthetic experiments are identified as educational simulations.
