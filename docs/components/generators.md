# Generators

[Docs](../README.md) · [Generation](../concepts/generation.md) · [Streaming](../concepts/streaming.md)

A generator implements `ragout.Generator`:

```go
GenerateIter(ctx context.Context, query string, candidates []ScoredChunk) (sources []ScoredChunk, tokens iter.Seq2[string, error])
```

`sources` is the chunks actually placed in the prompt, in citation order, and it is returned before `tokens` is pulled. The engine copies it to `Answer.Sources` immediately. The iterator yields token strings. The first error ends the stream.

Both built-in generators build that prompt with a `ContextBudgeter` and then stream HTTP.

## OpenAI

```go
gen := generator.NewOpenAIGenerator(apiKey,
    generator.WithOpenAIModel("gpt-4o-mini"),
    generator.WithOpenAIEndpoint("https://api.openai.com/v1/chat/completions"),
    generator.WithOpenAIBudgeter(budgeter),
    generator.WithOpenAIHTTPClient(client),
)
```

Defaults: model `gpt-4o-mini`, chat-completions endpoint, a budgeter of 4096 tokens, and an HTTP client whose response-header timeout is 30 seconds. The body is a single `user` message with `stream: true`. The SSE parser emits `choices[0].delta.content`.

The budgeter is called as `BuildPrompt("You are an accurate RAG assistant.", query, candidates)`, so that sentence is the first line of the user message.

## Ollama

```go
gen := generator.NewOllamaGenerator("llama3.2", "http://localhost:11434/api/generate",
    generator.WithOllamaBudgeter(budgeter),
    generator.WithOllamaHTTPClient(client),
)
```

An empty endpoint becomes `http://localhost:11434/api/generate`. The budgeter is called with an empty system string, so the prompt starts at "Context Information:". The response is a JSON line stream. Each object contributes its `response` field until `done` is true. The header timeout is 30 seconds.

## Context budgeter

```go
counter, err := generator.NewTiktokenCounter("cl100k_base")
budgeter := generator.NewContextBudgeter(counter, 8192)
```

`NewContextBudgeter(nil, maxTokens)` tries `cl100k_base` and, if the encoding fails to load, uses `NewHeuristicTokenCounter` (about 4 characters per token, at least 1 token for non-empty text). `maxTokens <= 0` becomes 4096.

| Method | Role |
| --- | --- |
| `AssembleContext` | Selects and formats chunks within the budgeter's max. |
| `AssembleContextWithBudget` | Same, with an explicit budget. |
| `BuildPrompt` | Adds the instruction and the question, then calls assemble with whatever tokens remain. |

Selection walks candidates in order, skips any chunk that does not fit, then runs `ReorderLostInTheMiddle`. Citation numbers are assigned after that reorder. The formatted block for each chunk is:

```text
[1] (ID: <chunk id>)
<chunk content>
```

`BuildPrompt` returns the prompt string and the selected chunks. That second return value is what `GenerateIter` hands back as `sources`.

`ReorderLostInTheMiddle` places even positions from the front and odd positions from the back, so the highest-ranked chunk is first and the next is last. Lists of length 0, 1, or 2 are returned unchanged.

## Token counters

`TokenCounter` is `CountTokens(text string) int`.

`NewTiktokenCounter(encodingName)` loads a tiktoken encoding. Use the same encoding name the token chunker uses when you want chunk windows and the budget to agree.

`NewHeuristicTokenCounter` needs no files. It divides `len(text)` by `CharsPerToken` (default 4).

## Next

[Streaming](../concepts/streaming.md) shows `Query`, `QueryStream`, and `QueryIter` on top of `GenerateIter`.
