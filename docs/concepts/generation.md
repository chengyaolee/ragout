# Generation and citations

[Docs](../README.md) · [Hybrid retrieval](hybrid-retrieval.md) · [Streaming](streaming.md)

The generator receives the chunks that survived retrieval and reranking. It does not send all of them. A context budgeter fits them to a token limit, reorders them, and numbers the ones that made it into the prompt.

## The budget

`generator.NewContextBudgeter(counter, maxTokens)` builds the budgeter. `maxTokens` of 0 or less becomes 4096. A nil counter tries a tiktoken counter for `cl100k_base`. If that encoding cannot be loaded, it falls back to a character heuristic of about 4 characters per token.

The OpenAI and Ollama generators each start with that default budgeter (4096 tokens). Replace it with `generator.WithOpenAIBudgeter` or `generator.WithOllamaBudgeter`.

`BuildPrompt` spends tokens on the instruction, the question, and the wrapper text first. Whatever remains is the budget for chunks. Candidates are tried in the order retrieval handed them over. A chunk that does not fit is skipped, and later smaller chunks can still be included. One long chunk does not end the selection.

Each selected chunk is counted with a citation line of the form:

```text
[0] (ID: <chunk id>)
<content>
```

The placeholder index is only for the count. Real numbers are assigned after reordering.

## Order in the prompt

Models attend more reliably to the start and the end of a long context than to the middle. After selection, `ReorderLostInTheMiddle` deals the list from the outside in: the best chunk goes first, the second-best goes last, the third-best goes second, and so on.

Citation numbers follow that display order, not the original retrieval rank. `Answer.Sources[0]` is `[1]` in the prompt, `Sources[1]` is `[2]`, and so on.

The prompt the model sees is:

```text
You are an accurate RAG assistant.

Context Information:
[1] (ID: ...)
...

User Question:
<question>

Answer based strictly on the context above. Cite sources using [1], [2], etc.
```

The OpenAI generator inserts "You are an accurate RAG assistant." The Ollama generator leaves that line empty. In both clients the instruction is part of the single user message the HTTP API sends. There is no separate system-role message.

## What you get back

`GenerateIter` returns the sources (the chunks that were placed in the prompt) before any token is produced. The engine copies those onto `Answer.Sources`. `Answer.Retrieved` is the longer list the generator was given, including chunks the budgeter skipped.

`Answer.Text` is the concatenation of the tokens. It is complete when `Query` returns, when `QueryStream`'s callback has seen every token, or when the `QueryIter` iterator finishes.

## Next

[Streaming](streaming.md) covers the three ways to read those tokens.
