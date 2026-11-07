# 0007: One runtime and one drafter interface

- Status: Implemented
- Date: 2026-11-07

## Context

Roadmap S1.4. There are two runtimes. `serve::Engine` (slots, batching, mixed prefill/decode, MTP
speculation, the cross-request prefix store, penalties) runs the server. `session::Session` (one
conversation: chat history and prompt building, prefix reuse within the conversation, context shifting,
DSpark and suffix speculation with an adaptive draft length, per-turn telemetry) runs `peregrine bench`,
`peregrine tune` and several tests. Speculation is implemented twice: depth-1 MTP in `Engine::spec_step`,
k-deep block verification in `Session::generate_with`.

They also differ in two places that change output:

- the sampling key: `Engine` uses `position_key(seed, 0, pos)`, `Session` `position_key(seed, turn, pos)`
  with `turn` incremented per conversation turn;
- `Session` turns off the rotated int8 KV cache for Qwen2 (`Architecture::session_kv_rotation`), `Engine`
  does not.

## Decision

1. **`Engine` is the one runtime.** It gains a generic verification step that all speculation goes
   through, and owns the verify-cost model (`VerifyCost`, `choose_k`, parking), because that model measures
   the engine's own forward passes.
2. **One `Drafter` trait** (`peregrine-core/src/draft.rs`): a drafter proposes (`propose`, with the
   target's position-keyed sampler passed in so a sampling drafter shares the target's noise), and is told
   what was committed (`commit`, with the target's taps for that position) and what was rolled back
   (`rollback`). It declares what it needs from the engine (`needs()`: hidden states, taps, token history).
   MTP, DSpark and suffix speculation implement it; per-slot drafter state lives in the slot.
3. **Speculation stays single-slot** for now, as `Engine` does today: a second user turns it off. The
   batched-speculation case is out of scope until it has its own equality test.
4. **`Session` stays the public conversation type**, rebuilt as a thin layer over a one-slot `Engine`: chat
   history, template, context shifting and prefix reuse within a conversation are conversation concerns and
   stay there; generation, speculation and telemetry come from the engine. Its public API keeps compiling.
5. **No output change in S1.4.** The two differences above become per-slot settings: `SlotCfg` gains the
   sampling `turn` (the server keeps 0, `Session` passes its counter) and a KV-rotation choice (`Session`
   passes `session_kv_rotation()`). Unifying either is a separate, output-changing decision.

## Consequences

- One place to add a speculation method: implement `Drafter`; the scheduling loop does not change.
- Before the move, tests pin today's behaviour where coverage is missing: the same prompt through `Session`
  and a one-slot `Engine` at T=0 and T>0; a golden with taps enabled; the prefix store with a DSpark drafter;
  a second user arriving mid-speculation.
- Each step keeps every golden and every speculation-on/off equality test identical, and steps that touch the
  decode loop carry a decode A/B.
- Revisit when batched speculation is wanted (it needs per-slot drafts in one verification pass).

## Implementation

Roadmap S1.4, in pull requests:

| PR | Step | What it did |
|---|---|---|
| [#53](https://github.com/RantAI-dev/peregrine/pull/53) | — | this decision |
| [#55](https://github.com/RantAI-dev/peregrine/pull/55) | 0 | the behaviour that must not move, pinned; the verify-cost model moved to `draft.rs`, where it belongs |
| [#56](https://github.com/RantAI-dev/peregrine/pull/56) | 1 | the `Drafter` interface, and the two per-slot settings: the sampling `turn` and the KV rotation |
| [#58](https://github.com/RantAI-dev/peregrine/pull/58) | 3 | one verification step in the engine (`Engine::spec_step` became `verify_step`); the MTP block is a drafter |
| [#60](https://github.com/RantAI-dev/peregrine/pull/60) | 4-5 | the suffix index and the DSpark head propose through that same step |
| [#61](https://github.com/RantAI-dev/peregrine/pull/61) | 6 | the session rebuilt as a layer over a one-slot engine |
| [#78](https://github.com/RantAI-dev/peregrine/pull/78) | wrap-up | the last of the old runtime: the benchmark measures the engine, the paths nobody walked are gone, and the docs say one runtime |

Step 2 — the session's own decode loop using the drafter wrappers — was overtaken: the engine's
verification step took the wrappers over, and the session's own loop went away with step 6.

Two things were implemented differently from the decision above, and neither changed an output:

* `needs()` declares the hidden states and the taps, and not the token history. The engine never read
  the history, and the suffix drafter keeps the corpus it searches itself.
* the benchmark (`peregrine bench`) measures the engine now, instead of a prefill and a decode loop of
  its own beside it. The bytes a token moves are the same; the numbers are the engine's.
