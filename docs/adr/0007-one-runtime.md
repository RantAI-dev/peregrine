# 0007: One runtime and one drafter interface

- Status: Accepted
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
