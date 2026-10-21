# Build retrieval workflows

[Docs](../README.md) · [Pipeline](../concepts/pipeline.md) · [Rewrite questions](query-rewrite.md)

`Engine.Query` is one pass: retrieve, then answer from whatever came back. The `flow` package runs retrieval as a workflow instead: a graph of named steps that share one `State`, with edges that can loop. Use it to check the evidence before answering, search again for what is missing, or say "I don't know" rather than guess.

Every step is a plain interface. Any of them can be replaced, removed, or joined by steps of your own.

## Corrective RAG (the ready-made loop)

```go
llm := generator.NewOllamaGenerator("qwen2.5:3b", "")
f := flow.CRAG(engine, llm, gen) // retriever, grader, generator

state, err := f.Run(ctx, "Which services depend on the billing queue, and who owns them?")
fmt.Println(state.Answer.Text) // cited answer, or the abstain message
state.Abstained                 // true when the evidence stayed insufficient
state.Hops                      // retrievals it took
```

```text
retrieve -> grade -+-> generate -> end            (sufficient, or the grader failed)
   ^               +-> refine -> retrieve          (insufficient, hops left)
   |               +-> abstain -> end              (insufficient, hops used up)
```

1. **Retrieve** calls the engine's `Retrieve`, so the query transformer, fusion, reranker, and expander all apply. Evidence is pooled across hops and de-duplicated.
2. **Grade** asks a model which retrieved passages help answer the question, as the CRAG paper grades each document. If none help, the model suggests a different search. Any chat model works, because the reply is read as plain lines. A follow-up is graded together with its conversation.
3. **Refine** makes that search the next query. A search that already ran is not repeated.
4. **Generate** writes the cited answer. **Abstain** returns a fixed message instead.

| Option | Effect |
| --- | --- |
| `flow.MaxHops(n)` | Retrievals per question. Default 2: the first search and one corrective search. |
| `flow.AnswerWhenExhausted()` | When hops run out, answer from what was found instead of abstaining. |
| `flow.AbstainMessage(msg)` | Replaces the default "I couldn't find enough information..." text. |

The grader can be a smaller, cheaper model than the generator. If it fails or replies in a way that cannot be read, the run continues as a plain retrieve-and-answer and the error lands in `State.Errors`. A grader outage degrades the loop instead of breaking it.

Grading relevance passage by passage, rather than asking "is this enough to answer?", is a deliberate choice. On the docs benchmark, a 3B model asked about sufficiency said no to 27 of 42 answerable questions, usually with the answer already retrieved. Asked about relevance, the same model abstained on 6 of 42 and still on 3 of 6 unanswerable questions. If you replace the grader, keep the judgment that simple.

Per run, `flow.WithHistory(turns)` passes the conversation (the first retrieval uses it, so `transform.Condense` can resolve a follow-up), and `flow.WithTokens(fn)` streams the answer as it is generated.

## Change a flow

A flow knows its steps by name. Editing never touches the other steps' wiring.

```go
f := flow.CRAG(engine, grader, gen)

f.Replace(flow.StepGrade, myGrader)                      // swap a component
f.InsertAfter(flow.StepRetrieve, "filter", filterStep)   // add one
f.Remove(flow.StepAbstain)                                // drop one
f.Route(flow.StepGrade, myRouter)                         // change a decision
f.MaxSteps(20)                                            // guard against runaway loops
```

`Clone` copies a flow, so a preset can be adjusted without changing the original. `Steps()` lists the names in order.

## Build your own

```go
f := flow.New().
    Add("retrieve", flow.Retrieve(engine)).
    Add("check", flow.StepFunc(func(ctx context.Context, s *flow.State) error {
        if len(s.Evidence) < 3 {
            s.Values["thin"] = true
        }
        return nil
    })).
    Add("generate", flow.Generate(gen)).
    Add("abstain", flow.Abstain("")).
    Route("check", func(s *flow.State) string {
        if s.Values["thin"] == true {
            return "abstain"
        }
        return "generate"
    }).
    Edge("generate", flow.End)
```

After a step runs, the next one is chosen in this order:
1. **Router:** `Route(from, func(*State) string)` decides from the state.
2. **Edge:** `Edge(from, to)` always goes to the same step.
3. **Order:** otherwise the step added next runs, and after the last step the run ends.

Return `flow.End` to stop.

A step is anything with `Run(ctx, *State) error`. Steps read and write `State`:
- the question and its standalone form;
- `Evidence` and `Searches`;
- `Hops`;
- the grader's `Verdict` and `Missing`;
- `Answer` and `Abstained`.

Anything else your steps need goes in `Values`, which `flow.WithValue` can seed per run. A step that returns an error stops the run. `Run` still returns the state gathered so far.

## What it costs

Each grade is one model call. A corrective hop adds a retrieval and another grade. With the default of 2 hops, a hard question costs up to 2 retrievals and 2 grading calls before the answer. An easy one costs 1 of each. `State.Trace` records each step's duration. Each step is also a `flow.step` span under a `flow.Run` span.

## Measuring it

The [docs evaluation](../../eval/results/docs.md) runs CRAG on answerable and unanswerable questions. It reports how often the loop abstains on each, and what the corrective hop adds to recall. See [Measure retrieval](evaluate.md) to run it on your own data.

## Next

[Measure retrieval](evaluate.md) compares a flow with a plain query.
