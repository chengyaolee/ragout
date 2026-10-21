# Retrieval evaluation: scifact

BEIR SciFact test split: scientific claims against 5,183 paper abstracts (CC BY-SA 4.0). Relevance is by document; the title is stored as metadata, so it is indexed only when chunk context is on.

| Setting | Value |
| --- | --- |
| Date | 2026-10-21 |
| Commit | `cde35b0` (retrieval code for these configs is unchanged through `d111549`) |
| Questions scored | 300 |
| Cut-off k | 10 |
| Chunking | character, 1000 bytes, 100 overlap |
| Embedder | `nomic-embed-text` via Ollama (no task prefixes) |
| LLM methods | `qwen2.5:3b` via Ollama, temperature 0, thinking off |
| Rerankers | `bge-reranker-v2-m3` Q8_0 and `Qwen3-Reranker-4B` Q4_K_M via llama.cpp, top 30 fused candidates |
| Index `noctx` | 5183 documents, 10094 chunks, built in 4m54s |
| Index `meta` | 5183 documents, 10094 chunks, built in 3m31s |

## Overall

| Config | Hit@10 | Recall@10 | MRR | nDCG@10 | Time per question |
| --- | --- | --- | --- | --- | --- |
| BM25, no context | 0.757 | 0.735 | 0.600 | 0.627 | 6 ms |
| Dense, no context | 0.803 | 0.787 | 0.651 | 0.677 | 36 ms |
| Hybrid, no context | 0.843 | 0.827 | 0.664 | 0.697 | 51 ms |
| BM25 + metadata | 0.777 | 0.754 | 0.609 | 0.638 | 9 ms |
| Dense + metadata | 0.823 | 0.809 | 0.661 | 0.689 | 31 ms |
| Hybrid + metadata (default) | 0.850 | 0.833 | 0.666 | 0.700 | 48 ms |
| Default + MultiQuery(3) | 0.843 | 0.826 | 0.640 | 0.678 | 1765 ms |
| Default + HyDE | 0.843 | 0.827 | 0.669 | 0.701 | 1233 ms |
| Default + bge-reranker-v2-m3 | 0.860 | 0.843 | 0.705 | 0.732 | 1284 ms |
