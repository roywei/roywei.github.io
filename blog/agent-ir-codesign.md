# Is This an IR Problem? Codesigning LLM APIs, Inference Engines, and Agent Harnesses

If you stare at the inference engine landscape long enough, an obvious pattern emerges. Every model family ships its own chat template — Hermes, Llama, Qwen, Mistral, DeepSeek, Granite — each with its own conventions for tool calls and reasoning blocks. Every API standard ships its own output shape — Chat Completions `tool_calls` arrays, Anthropic content blocks, OpenAI Responses items. And every serving engine ships hand-written code for each (model, template) pair, plus hand-written formatters for each API.

vLLM today maintains a dozen named tool-call parsers (`hermes`, `mistral`, `llama4_pythonic`, `qwen3_xml`, `deepseekv3`, `granite`, `pythonic`, ...) and a separate set of reasoning parsers for DeepSeek-R1, QwQ, Qwen3, and friends. SGLang has its own set. Each parser is imperative, fragile, and discovered its edge cases in production. New model drops? New parser. New API surface? New formatter on top.

That's an N × M code-gen written by humans. It's also where most of the "my agent silently dropped the tool call" bugs in production come from.

In an [earlier post](post.html?slug=agent-api-design) I argued that the gap between SDK helpers and agent harnesses is a codesign failure between the API, the SDK, and the runtime — Anthropic's own Claude Code source carries comments calling its own SDK *awkward* because the SDK and the harness were built for different audiences. **This post is about the bigger version of the same question.** Not "API vs harness" but the full stack: model template, inference engine, API surface, harness reducer. Each layer is owned by a different team. Each has its own invariants. The friction between them is most of what makes agent serving unreliable — and most of what could get fixed by codesign.

## So where does the inference engine fit?

The clean question — "should the engine just be token-in, token-out?" — has a clean answer: **no, that ship has sailed.** Modern inference engines (vLLM, SGLang, TensorRT-LLM, MLC-LLM) all do at least:

- **Tool-call parsing**: extracting structured `tool_calls` arrays from raw model output, per-template (Hermes, Llama, Mistral, Qwen, DeepSeek). vLLM ships a dozen named parsers; SGLang ships its own.
- **Reasoning parsing**: stripping `<think>...</think>` blocks from DeepSeek-R1, QwQ, and friends and surfacing them separately.
- **Constrained sampling**: forcing output to match a JSON schema or grammar via XGrammar / llguidance / Outlines / SGLang compressed-FSM.
- **Speculative decoding**: draft-and-verify acceleration with EAGLE, Medusa, and tree-traversal mask generation that overlaps with constrained-decoding cost on the CPU side.
- **Hosted-tool execution**: increasingly, the engine itself runs `web_search` or `code_interpreter` semantics — though this is more often the API layer than the engine, the boundary blurs.

The interesting question isn't *whether* the engine should do these things. It's *which interface* it should expose. There are three reasonable layers:

1. **Token in, token out.** The lowest-common-denominator API. Easy to build other things on. Useless for agentic workloads without a heavy harness.
2. **OpenAI-compatible Chat Completions.** The de-facto industry standard. Tool calls are parsed; streaming is supported; constrained outputs available. This is what vLLM and SGLang ship today.
3. **Provider-shaped APIs (Messages, Responses).** Block- or item-based, server tools, prompt caching, possibly state. Far more opinionated; far more useful for harnesses that target those providers.

The right answer is probably "all three, with consistent semantics between them." The engine should expose token-level primitives (for research and unusual deployments), Chat Completions for legacy and the broad ecosystem, and at least one richer API surface for the harness era. vLLM has been moving in that direction — recent releases add Anthropic Messages and OpenAI Responses serving — but matching the wire format is the easy part. The hard parts (hosted tools, server-side state continuation, prompt caching with provider-equivalent semantics, the per-provider affordances harnesses actually rely on) remain uneven across self-hosted stacks. You can adopt the API surface and lose most of what made the API surface useful.

This is one of the most interesting open problems. A self-hostable, provider-shaped API spec — an *agent serving API* — would let inference engines, gateways, and harnesses interop without each provider redefining the universe. The closest active effort is the [Open Responses API](https://www.openresponses.org/), a community-led move (started by OpenAI and others) to formalize the Responses-shaped surface as a vendor-neutral standard; I'll come back to it below. Other related efforts are MCP (a tool-side protocol, not an inference-side one) and the various "OpenAI-compatible" shims (which only cover Chat Completions).

## Codesign in practice

If the engine, the API, and the harness are all doing legitimate work, the question becomes what it looks like when they cooperate. Three concrete examples:

### 1. Tool-call parsers and structural-tag DSLs

When a model emits a tool call, two things have to agree: the **template** the model was trained on (Llama 4's pythonic format, Hermes's XML-ish format, Qwen's chat template) and the **parser** the engine runs to extract structured calls from raw text. If they're misaligned by even a whitespace, your tool call is silently lost or mangled.

XGrammar-2's Structural Tag DSL is the most interesting recent move here. It's a small grammar language with five primitives — `Sequence`, `Tag`, `AnyText`, `TriggeredTags`, `JSONSchema` — that lets a single grammar describe "any text, optionally a `<tool_call>` tagged region containing JSON matching this schema, optionally a `<think>` tagged region, repeating." That's reasoning + tool-call + custom-format expressed as one grammar, and the engine can constrain sampling to it.

Codesign here means: the model's chat template, the inference engine's parser, the constrained-decoding grammar, and the harness's expectation of what events look like all need to line up. When they do, you get strict-mode tool calling with 100% schema validity. When they don't, you get the "model dropped the closing brace" bug that every harness eventually has to handle.

### 2. Speculative decoding × constrained sampling

Speculative decoding generates draft trees of K candidate tokens. To constrain sampling, you need a token mask for *every node* in the draft tree, not just the next single token. Naive integration runs the constrained-decoding parser K times on each draft step, and the cost dominates.

XGrammar-2's `traverse_draft_tree` walks the entire draft in one pass, and — co-developed with TensorRT-LLM — schedules the CPU mask generation to overlap with the GPU verification step. The constrained-decoding cost effectively *hides* inside the speculative-decoding bubble. This only works because the engine, the spec-decode implementation, and the constrained-decoding library all expose compatible APIs.

This is the kind of optimization no single layer can produce. The grammar engine has to expose tree-walking primitives. The serving engine has to schedule them around its forward pass. The harness has to be willing to use strict-mode tool calls in the first place. It's the exact opposite of "token in, token out."

### 3. Fine-grained tool-arg streaming

Anthropic's [fine-grained tool streaming](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/fine-grained-tool-streaming) — now generally available, opted into per-tool by setting `eager_input_streaming: true` on the tool definition — lets the API stream partial tool arguments as `input_json_delta` without buffering until the call is complete. From the harness's side, this changes nothing about correctness — but it changes a lot about UX. You can show "tool args being typed" mid-flight; you can start preparing the tool execution context before the args fully materialize; you can detect malformed prefixes early.

But the tool definition has to opt in, the parser has to accept partial JSON, and the runtime has to know not to act on incomplete args. Three layers, one feature.

## Naming what's going on: an IR problem

What the three codesign examples have in common is *glue*. Tool-call parsers are glue between a chat template and a Chat-Completions-shaped output. Spec-decode tree-mask APIs are glue between a sampler and a grammar. `eager_input_streaming` is glue between a tool definition and a stream encoder. Each piece exists because two adjacent layers had different conventions and somebody wrote a little converter.

Glue is what compiler people decided to stop writing forty years ago. The same N × M → N + M move that gave them intermediate representations applies here:

```
N model frontends → IR → M API backends
```

Each model family contributes one frontend (`hermes → IR`, `pythonic → IR`, `qwen3_xml → IR`). Each API contributes one backend (`IR → chat_completions`, `IR → messages`, `IR → responses`). Frontends and backends decouple. Adding a new model is one frontend, not M parsers; adding a new API is one backend, not N format adapters.

A concrete example. A Qwen 3 model emits this raw stream:

```
some thinking content
<tool_call>
{"name": "get_weather", "arguments": {"location": "SF"}}
</tool_call>
```

The frontend (`qwen3 → IR`) translates that into a sequence of typed IR events:

```
reasoning_delta { text: "some thinking content" }
reasoning_done
tool_call_start { name: "get_weather" }
tool_call_arg_delta { partial: '{"location":' }
tool_call_arg_delta { partial: ' "SF"}' }
tool_call_arg_done { args: { location: "SF" } }
```

A backend (`IR → messages` or `IR → responses`) formats those events for whichever API the harness wants:

```
Messages:   a `thinking` block + a `tool_use` block (id "toolu_…")
Responses:  a `reasoning` item + a `function_call` item (call_id "call_…")
```

The frontend understands Qwen's tags. The backend understands the API shape. Neither knows about the other; they meet in the middle at the IR.

### What the IR would carry

If you took Anthropic's content blocks and OpenAI's Responses items and squinted, you'd see most of it already. A clean IR would carry:

- Span-typed events: `text`, `reasoning`, `tool_call_start`, `tool_call_arg_delta`, `tool_call_done`, `stop`
- Streaming-first lifecycle (open/delta/close, not just final values)
- Pairing semantics (how `tool_call` ↔ `tool_result` correlate)
- Stop-reason taxonomy
- Provenance metadata (which token positions produced which span)

The key invariant: **the IR is what every frontend emits and every backend consumes.** It's also what the constrained-decoding grammar enforces, which closes the loop on the parser-grammar disagreement bugs that show up in production today.

### Three places the IR can live

The IR doesn't have to be a runtime translation step. There are three places it can live, and the tradeoffs are different for each:

```
                          fragility    flexibility    coordination cost
1. Translate at inference  HIGH         HIGH           low (today)
2. Constrain at decode     MEDIUM       MEDIUM         medium (Structural Tag)
3. Train against IR        LOW          LOW            high (Harmony)
```

**1. Translate at inference.** The current vLLM/SGLang state. Per-template parser converts native output to IR / API. Maximum flexibility, minimum coordination cost — but the parser is the failure surface.

**2. Constrain at decode.** XGrammar-2's Structural Tag is this idea. The grammar *is* the IR shape. The model is forced to emit IR-compatible tokens. Parser failures vanish (the engine guaranteed the shape), but the model drifts away from its training distribution, and you still need a grammar per template that maps native tokens onto the IR.

**3. Train against IR.** Harmony is OpenAI's hint at this direction. If the model's *training surface* is the IR, no translation is needed. Cleanest end state — but the coordination cost is one only labs can pay, because you have to retrain or fine-tune to switch IRs.

The mature answer is probably *all three at once*: train models toward an IR, constrain decoding to enforce it, and keep a translation layer for legacy and external models.

### The hard parts compiler IR didn't have to solve

LLVM had it easy compared to this. Five wrinkles specific to agent IRs:

1. **Streaming-first.** Compiler IR is built once and consumed once. Agent IR is built incrementally over hundreds of token deltas with consumers attached at every stage. The "duplicate text in stream events" trap from the [previous post](post.html?slug=agent-api-design) is fundamentally an "incremental SSA with imprecise edits" problem.
2. **Template drift unversioned with parsers.** Llama 3.1 → 3.2 → 3.3 silently moves whitespace; the parser doesn't know to re-test. Compilers had explicit language version specs.
3. **Constrained decoding must agree with the parser.** If the grammar lets through `</tool_call>` but the parser expects `<|tool_call_end|>`, the model can produce IR-invalid output. Compilers don't have this — the codegen is authoritative.
4. **Reasoning is privacy-sensitive.** Some IR spans are not for transport. Compiler IRs don't have HIPAA-like span classes.
5. **Probabilistic source.** The model can emit invalid surface forms even with constrained decoding. The IR layer needs an error-recovery model — closer to a permissive HTML parser than a strict compiler.

### What's already converging

The IR is half-built across several efforts that haven't yet realized they're the same effort:

- **vLLM Unified Parser RFC (#32713)** — merge tool + reasoning parsers into one post-processor with a shared event model. Closest thing to an explicit IR effort in the OSS engine world.
- **XGrammar-2 Structural Tag** — IR-as-grammar. A single Tag DSL describes "any text + optional reasoning + optional tool calls + optional custom format" composably.
- **OpenAI Harmony** — IR-as-training-surface for open-weights GPT models.
- **[Open Responses API](https://www.openresponses.org/)** — a community-led effort, started by OpenAI and others, to formalize the Responses-shaped surface as a vendor-neutral standard. If it ships as a true open spec rather than a wire-level mimic, it becomes the obvious target for self-hosted serving stacks and a natural starting point for the API-side IR.
- **Anthropic content blocks / OpenAI Responses items** — provider-shaped semi-IRs that have already done the hard span-typing work, just not vendor-neutrally.

## Pain points and future directions

Beyond the IR thesis above, the state of the art has gaps. Some are addressable; some are research questions.

**Transcript portability.** There is no shared format for "an agent conversation." Every harness invents one. Anthropic Messages and OpenAI Responses are the closest things, but they're not symmetric and they're tied to commercial APIs. A neutral, provider-agnostic transcript format — the JSON schema for "conversation with tool calls and reasoning" — would let harnesses move between providers without rewriting. The fact that this doesn't exist is a coordination failure.

**Server-tool standardization.** MCP solves the *client-side* tool protocol. It does not solve the server-side hosted-tool protocol. When an Anthropic web search differs from an OpenAI web search differs from a self-hosted retrieval tool, harnesses end up with provider-specific code paths for each. A neutral hosted-tool descriptor — "this tool runs on the provider, here's its event lifecycle, here's how to reference its results" — would help.

**Reasoning-trace privacy.** Reasoning is increasingly the most sensitive part of an agent transcript — it's where the model speculates about user intent, considers approaches it doesn't end up taking, and exposes its internal model. The Claude Code source carries reasoning traces through *five* distinct paths (analytics, customer OTel, beta tracing, local JSONL, API context, bug reports), each with its own enable/disable. The current pattern of "thinking is just another content block" doesn't reflect its sensitivity. A reasoning-specific privacy contract — "this content does not get transported beyond the immediate request" — would be valuable.

**Inference-engine API for agents.** A self-hostable, provider-shaped API that vLLM and SGLang could expose, that Claude Code and Codex and Aider could target, that lets the agent-shaped affordances (block streaming, server tools, prompt caching, optional state) work without a commercial API behind them. The closest active effort is the Open Responses API. If it stabilizes — and if vLLM, SGLang, and the major harnesses all converge on it — most of this gap closes. The risk is the usual one: a standard born from one provider's design ends up underspecified for cases the original didn't motivate.

## The reliability argument

When you ask "how do I make my agent more reliable?", the answer is rarely "use a smarter model." It's almost always "fix the layer that's silently corrupting state."

The SDK bugs from the [previous post](post.html?slug=agent-api-design) — duplicate-text, O(n²) parsing, mutable references — each look minor in isolation. In aggregate they make the difference between an agent that works and an agent that fails one in fifty turns with a phantom symptom you can't reproduce. The harness fixes them by owning the reducer. But the underlying problem is the bigger one this post has been about: the API and the SDK and the model template and the engine parser were each designed to satisfy a slightly different contract, and the harness is left reconciling them.

The way forward is codesign. The model is trained to emit specific tags in a specific order. The inference engine parses those tags into structured events. The API surfaces those events with explicit lifecycle. The SDK exposes the events without hiding the lifecycle. The harness reduces the events into its own state. Each layer is honest about what it's doing, and the cost of being wrong is local rather than spread across all layers.

If we're lucky, an IR catches up to make the contracts machine-checkable. If we're not, we keep writing N × M parsers and watching them silently drop tool calls. The interesting design work in 2026 — for inference engines, for API designers, for SDK authors, for harness builders — is which of those two futures we choose.

---

*Sources: same as the [previous post](post.html?slug=agent-api-design), plus the [XGrammar-2 release blog](https://blog.mlc.ai/2026/05/04/xgrammar-2-fast-customizable-structured-generation), [vLLM's OpenAI-compatible server docs](https://docs.vllm.ai/serving/openai_compatible_server.html), the vLLM Unified Parser RFC (issue #32713 in the vLLM repo), and the parser inventories of vLLM and SGLang. The IR thesis and the "glue" framing are my own; corrections and counter-evidence welcome.*
