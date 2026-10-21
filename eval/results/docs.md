# Retrieval evaluation: docs

ragout's own docs/ (Markdown, split by heading). Questions are hand-written, grouped by the method each targets; expected strings each occur exactly once in the corpus.

| Setting | Value |
| --- | --- |
| Date | 2026-10-21 |
| Commit | `dcb4065` |
| Questions scored | 42 |
| Cut-off k | 5 |
| Chunking | character, 800 bytes, 0 overlap |
| Embedder | `nomic-embed-text` via Ollama (no task prefixes) |
| LLM methods | `qwen2.5:3b` via Ollama, temperature 0, thinking off |
| Rerankers | `bge-reranker-v2-m3` Q8_0 and `Qwen3-Reranker-4B` Q4_K_M via llama.cpp, top 30 fused candidates |
| Index `noctx` | 27 documents, 211 chunks, built in 3s |
| Index `meta` | 27 documents, 211 chunks, built in 4s |
| Index `llmctx` | 27 documents, 211 chunks, built in 5m30s |

## Overall

| Config | Hit@5 | Recall@5 | MRR | nDCG@5 | Time per question |
| --- | --- | --- | --- | --- | --- |
| BM25, no context | 0.667 | 0.631 | 0.509 | 0.490 | 0 ms |
| Dense, no context | 0.690 | 0.667 | 0.540 | 0.521 | 17 ms |
| Hybrid, no context | 0.667 | 0.631 | 0.546 | 0.503 | 19 ms |
| BM25 + metadata | 0.690 | 0.655 | 0.538 | 0.512 | 0 ms |
| Dense + metadata | 0.738 | 0.708 | 0.589 | 0.559 | 15 ms |
| Hybrid + metadata (default) | 0.738 | 0.702 | 0.608 | 0.561 | 15 ms |
| Default + MultiQuery(3) | 0.714 | 0.679 | 0.621 | 0.563 | 1915 ms |
| Default + HyDE | 0.762 | 0.738 | 0.633 | 0.592 | 2971 ms |
| Default + Decompose(4) | 0.667 | 0.643 | 0.497 | 0.479 | 1348 ms |
| Default + Condense | 0.810 | 0.774 | 0.658 | 0.617 | 115 ms |
| Default + bge-reranker-v2-m3 | 0.881 | 0.857 | 0.657 | 0.638 | 1036 ms |
| Default + Qwen3-Reranker-4B | 0.905 | 0.881 | 0.796 | 0.749 | 9381 ms |
| Default + Neighbors(1) | 0.762 | 0.720 | 0.626 | 0.575 | 50 ms |
| Hybrid + LLM context | 0.738 | 0.720 | 0.620 | 0.577 | 23 ms |
| Default + CRAG (2 hops) | 0.738 | 0.702 | 0.596 | 0.553 | 2659 ms |

## Recall@5 by question category

Each category targets one method; see the question set.

| Config | compound (6) | exact (8) | followup (6) | heading (8) | multichunk (6) | paraphrase (8) |
| --- | --- | --- | --- | --- | --- | --- |
| BM25, no context | 0.500 | 0.875 | 0.500 | 0.875 | 0.583 | 0.375 |
| Dense, no context | 0.667 | 0.750 | 0.333 | 1.000 | 0.667 | 0.500 |
| Hybrid, no context | 0.500 | 0.875 | 0.500 | 1.000 | 0.583 | 0.250 |
| BM25 + metadata | 0.667 | 0.875 | 0.500 | 0.875 | 0.583 | 0.375 |
| Dense + metadata | 0.833 | 0.750 | 0.333 | 1.000 | 0.792 | 0.500 |
| Hybrid + metadata (default) | 0.833 | 0.875 | 0.500 | 1.000 | 0.750 | 0.250 |
| Default + MultiQuery(3) | 0.833 | 0.875 | 0.500 | 0.875 | 0.583 | 0.375 |
| Default + HyDE | 0.917 | 0.875 | 0.500 | 1.000 | 0.750 | 0.375 |
| Default + Decompose(4) | 0.667 | 0.875 | 0.333 | 1.000 | 0.667 | 0.250 |
| Default + Condense | 0.833 | 0.875 | 1.000 | 1.000 | 0.750 | 0.250 |
| Default + bge-reranker-v2-m3 | 0.917 | 0.875 | 0.500 | 1.000 | 0.917 | 0.875 |
| Default + Qwen3-Reranker-4B | 0.917 | 0.875 | 0.833 | 1.000 | 0.750 | 0.875 |
| Default + Neighbors(1) | 0.833 | 0.875 | 0.500 | 1.000 | 0.875 | 0.250 |
| Hybrid + LLM context | 0.917 | 0.875 | 0.500 | 1.000 | 0.792 | 0.250 |
| Default + CRAG (2 hops) | 0.833 | 0.875 | 0.500 | 1.000 | 0.750 | 0.250 |

## CRAG (corrective loop)

The loop runs without its generation step; answers are not scored here.

| Measure | Value |
| --- | --- |
| Abstained on answerable questions (lower is better) | 27 / 42 |
| Abstained on unanswerable questions (higher is better) | 5 / 6 |
| Retrievals per question | 1.54 |
| Evidence chunks per question | 6.2 |
| Recall over all gathered evidence | 0.702 |
| Grader failures | 0 |
