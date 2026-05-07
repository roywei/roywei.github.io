# Why Your Agent Can't Use Its SDK: LLM API Design Through the Harness Lens

There's a stretch of code in a reverse-engineered Claude Code source tree where the same word shows up three times in fifteen lines:

![Source comments inside Claude Code's stream reducer calling the SDK's behavior "awkward", "also awkward", and "even more awkwardly", with code that resets text and thinking fields and notes the SDK mutates blocks while it works](assets/claude_code_comments.png)

Three "awkward"s, climbing in intensity. A few hundred lines up there's a fourth comment in the same vein, shorter and more pointed:

> `// use raw stream instead of BetaMessageStream to avoid O(n²) partial JSON parsing`

These sit in the streaming reducer of what is, structurally, Anthropic's own agent harness consuming Anthropic's own API — and they spend their time documenting why the harness can't use Anthropic's own SDK helper. **The model output you see in Claude Code is whatever's left after the harness has worked around three things its own SDK does wrong, plus one thing it does inefficiently.**

This isn't a bug report on the SDK. The SDK is fine for what it was designed for — "print text as it streams in", "give me a `finalMessage` when it's done" — and people ship products on it every day. Agent harnesses are *not* those applications, and the gap between the two is where most of the interesting LLM API design questions of 2025–2026 live.

This post is about that gap: what an agent harness actually needs from its API, where the two big commercial APIs (Anthropic Messages and OpenAI Responses) get it right and wrong, and why their wire-level design choices ripple all the way down to your reducer. There's a [follow-up post](post.html?slug=agent-ir-codesign) that zooms out to the rest of the stack — model template, inference engine, API surface, harness — and asks whether the whole thing is really an IR problem in disguise.

## What the "awkward"ness is hiding

Each comment in that snippet maps to a concrete failure mode that turns the SDK helper from convenient to dangerous when you're building an agent runtime. None of them are obvious from reading the SDK docs.

**1. "the sdk sometimes returns text as part of a content_block_start message, then returns the same text again in a content_block_delta message."** The `MessageStream` accumulator pushes the `content_block_start` object straight into its snapshot, then later appends `text_delta` events to that same snapshot. If the start event already contains text — which it sometimes does, depending on how the stream is chunked — and the next delta repeats that text, naive accumulation produces `"HelloHello"`. The harness's note that "there doesn't seem to be a way to detect when a content_block_delta message duplicates the text" is the key observation: this isn't fixable from outside the SDK by inspecting events. The only safe move is to throw away the starter content (`text: ''`, `thinking: ''`) and let the deltas be canonical. That's exactly what the snippet does.

**2. "use raw stream instead of BetaMessageStream to avoid O(n²) partial JSON parsing."** `BetaMessageStream` calls `partialParse()` on every `input_json_delta` so it can expose an incrementally-parsed `inputJson` value. For a long tool call — say, an editor `apply_diff` with a few KB of arguments — that's a parse pass over `1, 2, 3, ..., n` characters. Quadratic. The harness instead concatenates the partial JSON strings and parses once at `content_block_stop`. Linear, and you get to validate against the schema in one shot rather than fighting with the partial parser's tolerance for malformedness.

**3. "even more awkwardly, the sdk mutates the contents of text blocks as it works. we want the blocks to be immutable, so that we can accumulate state ourselves."** This is the most architectural of the three. SDK blocks are mutated in place as the stream advances. If the harness's UI thread, tool-execution thread, transcript writer, and telemetry sink all hold the same reference, they see different things at different moments — and not consistently. The spread copy (`{ ...part.content_block }`) is the fix, but the principle in the comment is what matters: *the harness wants to accumulate state itself.* That stops the SDK from being a competing source of truth.

You can read each of these as a paper cut. Together they are a leak. The harness's solution is uniform: **raw events are canonical; accumulated state is derived; the runtime owns its own reducer.** The SDK helper's job ends at "consume the SSE stream"; everything past that — accumulation strategy, mutability discipline, stop-condition logic, error recovery — is the runtime's problem and shouldn't be hidden.

## What an agent harness actually does

The word "agent" gets stretched, so let me pin it down. By "agent harness" I mean what Claude Code, Codex, Aider, Cursor's chat agent, and the agent loops in production AI products are: a runtime that

1. Builds a request to the model from a conversation transcript.
2. Streams the response, parsing it incrementally into structured units (text, tool calls, reasoning).
3. Detects when the model wants to invoke a tool, suspends generation, executes the tool (often with permissions and async I/O), and resumes the conversation with the result.
4. Persists the transcript, ships telemetry, supports retry/fallback, and survives malformed model output.

A normal app that calls an LLM cares about (1) and the happy path of (2). An agent harness has to get all four right *and* get them right deterministically across thousands of turns and dozens of edge cases. The SDK convenience helpers were designed for the first kind of caller. That's why they're insufficient for the second.

## The two-state model

The first thing to make explicit when you're designing an agent runtime is **what state lives where**. Most harness bugs are confusion at this boundary.

```
API / server state:
  the conversation as the API sees it
  cached reasoning blocks, response IDs
  previous_response_id (Responses)
  cache_control breakpoints (Anthropic prompt caching)

Agent / runtime state:
  raw event log (canonical source of truth)
  derived snapshots of stream blocks/items
  tool execution status (running, succeeded, failed, validation-error)
  permission checks, retry/fallback machinery
  UI state (what's visible, what's collapsed)
  audit transcript on disk
  telemetry sinks
```

A "stateful API" like OpenAI's Responses with `previous_response_id` shifts some of the *first* set onto the provider — you stop having to resend the full conversation each turn. It does not reduce the *second* set. That's all yours. The harness still has to own its event log, its reducer, its tool-execution machinery, its retries.

The mistake I see most often is treating an SDK helper's accumulated snapshot as the runtime's state. It's not — it's a derived view into a subset of the API state, and it's mutable, and the SDK author owes you no stability guarantees for downstream consumers. Don't mix them up.

With that boundary named, the natural next question is what the major commercial APIs actually offer across it. The two big shapes — Anthropic Messages and OpenAI Responses — implement the same conceptual loop, with very different invariants.

## Two tribes of tool-call lifecycles

Anthropic Messages and OpenAI Responses both implement the same conceptual loop — request, tool call, execute, loop — but with different wire formats and constraints. The differences matter.

### Anthropic: blocks with immediate pairing

The assistant emits a content block:

```json
{ "role": "assistant", "content": [
    { "type": "tool_use", "id": "toolu_...", "name": "Read", "input": {...} }
] }
```

The next message *must* be a user message whose content array starts with the matching `tool_result`:

```json
{ "role": "user", "content": [
    { "type": "tool_result", "tool_use_id": "toolu_...", "content": "..." }
] }
```

Two hard constraints: tool results must immediately follow tool uses (no other turns in between), and they must come **first** in the user content array. There is substantial pairing-repair logic in the harness to enforce this when the transcript mutates — when a tool fails and you need to inject a synthetic error result, when the user cancels mid-execution, when you replay history into a new session.

The benefit of this constraint: history is structurally self-validating. If your messages array typechecks, you can be confident the API will accept it.

### OpenAI Responses: items with call_id correlation

The assistant emits a `function_call` item:

```json
{ "type": "function_call", "call_id": "call_...", "name": "get_weather", "arguments": "{\"location\":\"Paris\"}" }
```

The result comes back as a top-level **input item** in the next request:

```json
{ "type": "function_call_output", "call_id": "call_...", "output": "..." }
```

There's no "must come first" rule. There's no "must immediately follow" rule. Calls and results correlate by `call_id`. You can have multiple turns of conversation between a call and its result, you can interleave with other items, you can reorder. Less repair logic — but you have to track call IDs yourself, and there's no positional invariant to lean on.

### Streaming differences

Anthropic streams tool args as `input_json_delta.partial_json` strings inside `content_block_delta` events, scoped to the block lifecycle (`content_block_start` / `_stop`). Index by block index plus tool_use.id.

OpenAI Responses gives you typed events: `response.output_item.added`, `response.function_call_arguments.delta`, `response.function_call_arguments.done`, `response.output_item.done`. Index by output index plus call_id. The events are more granular and self-describing, and most of the partial-parsing footguns in the Anthropic SDK don't appear here because the events are already split by purpose.

The two designs cost about the same to implement. Anthropic's pays more in pairing repair; OpenAI's pays more in correlation bookkeeping. Neither is wrong — but if you're writing a harness that targets both, you'll want a normalized internal representation, because the wire formats don't compose into a single one cleanly.

## Stateful vs stateless: the false dichotomy

The temptation is to frame this as Responses (stateful) vs Messages (stateless) and pick a winner. Don't. The cleaner mental model is two **orthogonal** axes. The key thing the table forces is that "Responses" is not a column — it sits on both sides of the statefulness axis depending on whether you set `store: true` and pass `previous_response_id`:

```
                  | Stateless                | Stateful (server holds context)
------------------|--------------------------|---------------------------------
Client-only tools | Chat Completions         | Responses + previous_response_id
                  | Messages                 |   (store: true)
                  | Responses (store: false) |
                  |                          |
Server-side tools | Messages + server tools  | Responses + previous_response_id
                  | (web_search, web_fetch,  |   + hosted tools
                  |  code_execution)         |
                  | Responses (store: false) |
                  |   + hosted tools         |
```

Both Anthropic and OpenAI ship server-side tools (Anthropic: `web_search`, `web_fetch`, `code_execution`; OpenAI: `web_search`, `file_search`, `code_interpreter`, `computer_use`, MCP). Only OpenAI Responses ships true server-side state continuation. **Stateless + server tools** (Anthropic Messages, or Responses with `store: false` and hosted tools) is a perfectly reasonable point in the design space — and the misconception worth dispelling is that "stateful" and "server tools" come as a bundle. They don't.

### Stateful pros

- **Lower latency on long conversations.** No resending megabyte-scale history each turn; the server keeps it.
- **Reasoning preservation across turns.** Responses keeps reasoning items server-side, so subsequent turns can attend to the model's prior thinking without you re-shipping it. The win is latency and request payload size — note that prior tokens in the chain are still billed as input either way.
- **Cache friendliness for free.** The provider can implement prefix caching and you don't have to reason about cache breakpoints.
- **Smaller request bodies.** Real wins for clients on metered networks.

### Stateful cons

- **Replay and audit are harder.** The single source of truth lives on the provider's infrastructure, not in your transcript.
- **Multi-host and migration are harder.** A `previous_response_id` is meaningless if you switch models or move providers.
- **Debugging is harder.** "What did the API actually see?" becomes a support ticket instead of a `cat transcript.jsonl`.
- **Failure modes leak.** If the provider expires your conversation or you exceed a server-side TTL, you have to gracefully fall back to resending — which means you needed the transcript anyway.
- **Cost model gets opaque.** When you can't see the conversation as the model sees it, you can't easily reason about why your token bill spiked.

### Stateless pros

- **Transcripts are the source of truth.** Replay, fork, audit, share — all trivially supported.
- **Provider-agnostic.** Your transcript can be sent to vLLM, SGLang, Bedrock, Vertex, Anthropic, all the same.
- **Trivial debugging.** What you sent is what the model saw.
- **Pairs cleanly with prompt caching.** With explicit `cache_control` markers (Anthropic), you can be deliberate about which prefix gets cached and which doesn't.

### Stateless cons

- **Higher per-turn payload.** Especially with long reasoning, images, large tool definitions.
- **Reasoning is dropped between turns.** Unless you store and resend it, which costs tokens.
- **Cache management is the client's problem.** Get the breakpoints wrong and you're paying full price every turn.

For agent harnesses specifically — where audit, replay, and failure recovery matter — I'd argue stateless is the right *floor*, with stateful as an opt-in optimization. The harness should always be able to reconstruct the conversation from its own transcript. Stateful APIs that *expose* the underlying state (let me read it back, let me fork it, let me migrate it) are great. Stateful APIs that *hide* it are a debugging tax.

## What this leaves unsaid

I've been talking about commercial LLM APIs as if they were the whole story. They aren't. The same harness, the same two-state model, the same tool-call lifecycles also have to work against self-hosted models served by vLLM, SGLang, TensorRT-LLM. There the question shifts: should the inference engine just be token-in, token-out — or does the agent era demand something richer?

That's the bigger version of this question, and the answer turns on a deeper observation. The model templates, the engine parsers, the API surfaces, and the harness reducers are all designed by different people with different invariants. The friction between them is most of what makes agent serving unreliable, and most of what could get fixed by codesign.

Maybe what we actually need is an intermediate representation — the same shape compiler people have been solving for forty years. [The next post](post.html?slug=agent-ir-codesign) makes that case.

---

*Sources: [Anthropic tool use](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/implement-tool-use), [Anthropic fine-grained tool streaming](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/fine-grained-tool-streaming), [OpenAI conversation state (Responses)](https://platform.openai.com/docs/guides/conversation-state?api-mode=responses), [OpenAI function calling](https://platform.openai.com/docs/guides/function-calling), Anthropic's [Building Effective Agents](https://www.anthropic.com/research/building-effective-agents), and a study of the [yasasbanukaofficial/claude-code](https://github.com/yasasbanukaofficial/claude-code) reverse-engineered clone (the official Anthropic source is not open). The four "awkward" comments are quoted verbatim from that clone's stream reducer in `src/services/api/claude.ts`.*
