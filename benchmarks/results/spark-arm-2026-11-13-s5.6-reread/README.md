# The re-read on the merged build (S5.6r)

S5.6's disturbed table was measured before S5.4j's join-gap spin window landed. This
directory is the re-read on the merged branch (`origin/s5.6-pool-contended`, both PRs
plus the review fixes): the same disturbance tool (`loadgen`), the same drivers and
cells, ten threads pinned to the ten Cortex-X925 CPUs, quiet and with 1, 2 and 4
disturbers, five rounds per cell.

Three builds, interleaved inside every round (the arm order rotates per round, so a
within-level host drift cannot favour one):

- `head` — `7d46020`, the S5.6 head before S5.4j's gap-stretched window;
- `merged` — `origin/s5.6-pool-contended` (`23d86c8`), both PRs;
- `gap0` — the same branch with `GAP_CAP_US = 0` in
  `crates/peregrine-core/src/pool.rs` (the stretch off), a local build only.

Then the review fix the re-read called for (`fixed`, the same branch with the stretch
gated on the host's run-queue delay) was measured in the same three-plus-one grid.

- Date: 2026-11-13 (the same day as S5.6, later window).
- Host: DGX Spark, aarch64, 20 cores in two clusters (10 × Cortex-X925 at 5–9 and
  15–19, 10 × Cortex-A725). Every process is pinned to the X925 cluster with
  `taskset -c 5-9,15-19`. The host is shared with other tenants all window: the
  1-minute load average is on every raw record (2.6–13.9), and the per-core idle
  share of the ten CPUs over each run's own window beside it. The "quiet" cells are
  quiet *of our disturbers*; the co-tenants never left.
- Cells: Llama 3.2 1B Q4_K_M prefill at 512 and 4 096 tokens (`promptbench`, exact
  mode, `PEREGRINE_EMBED_CACHE_MB=0`), decode (`peregrine bench`, 16-token prompt,
  128 tokens) and gte-modernbert-base F16 on 32 different 128-token texts
  (`reread_embed.py`, exact result cache off).
- Medians of five rounds; spreads are the half-range (°%). Every ratio is
  `after / before` recomputed from the medians by `summarize.py`.

## The three-arm re-read (before the fix)

Peregrine only; tok/s except the embedding cell (texts/s).

| cell | level | head | merged | gap0 | merged/gap0 | merged/head |
|---|---|---|---|---|---|---|
| prefill 512 | 0 | 729.3 | 717.8 | 731.6 | **0.981** | 0.984 |
| prefill 512 | 1 | 598.4 | 596.1 | 611.2 | **0.975** | 0.996 |
| prefill 512 | 2 | 551.0 | 547.0 | 561.6 | **0.974** | 0.993 |
| prefill 512 | 4 | 441.3 | 433.3 | 439.5 | **0.986** | 0.982 |
| prefill 4 096 | 0 | 518.6 | 512.8 | 521.3 | **0.984** | 0.989 |
| prefill 4 096 | 1 | 445.6 | 439.2 | 447.1 | **0.982** | 0.986 |
| prefill 4 096 | 2 | 416.9 | 412.4 | 417.3 | **0.988** | 0.989 |
| prefill 4 096 | 4 | 325.1 | 320.6 | 327.6 | **0.979** | 0.986 |
| decode | 0 | 123.8 | 124.2 | 123.6 | 1.005 | 1.003 |
| decode | 1 | 106.3 | 106.0 | 105.6 | 1.004 | 0.998 |
| decode | 2 | 104.0 | 103.7 | 103.5 | 1.002 | 0.997 |
| decode | 4 | 92.3 | 92.2 | 92.0 | 1.003 | 0.999 |
| embed 32×128 | 0 | 20.0 | 20.3 | 20.3 | 1.002 | 1.013 |
| embed 32×128 | 1 | 17.0 | 17.2 | 17.2 | 1.005 | 1.017 |
| embed 32×128 | 2 | 15.9 | 16.1 | 16.1 | 1.000 | 1.011 |
| embed 32×128 | 4 | 13.1 | 13.3 | 13.2 | 1.005 | 1.012 |

Spreads (half-range) are in `table.txt`: 0.1–2.5 % on the quiet 4 096 prefill cells,
0.4–2.5 % on the disturbed ones, 0.3–4.3 % on decode and 0.2–6.6 % on embed.

**The stretch costs prefill.** `merged` is below `gap0` in every prefill cell at every
level — 1.2–2.6 % — and the deficit is larger than `gap0`'s own half-range in all
eight (at 4 096 it is 0.3–0.5 %). Decode and embed are level (+0.0–0.5 %).

**The merge does not keep S5.6's prefill numbers.** `merged` is 1.0–2.5 % below
`head` on prefill (below 1.0 in all eight cells), while `gap0` — the same build with
the stretch off — is level with `head` (0.996–1.021, nine of eight cells within
spread). So the merge's prefill cost is the stretch, not S5.6's mechanisms.

**The quiet cells are not level.** `merged` is 1.6–2.0 % below `gap0` and 1.1–1.6 %
below `head` on quiet prefill (at 512: 717.8 against 731.6 and 729.3; at 4 096:
512.8 against 521.3 and 518.6). Decode and embed are level quiet. On this host the
stretch costs even when the run queue is empty — the opposite sign from S5.4j's
desktop measurement (+3.3 % there), and the reason the re-read was owed.

## The review fix

`GAP_CAP_US`'s stretch is now taken only while the smoothed run-queue delay is below
`STALL_GATE_PERMILLE` (80): on a host with more runnable threads than the pool's
cores, a longer spin takes a core from the worker the join is waiting for, which is
exactly what S5.6's stall gate parks to avoid. A forced park still wins
(`budgets` returns zero first), and a configured window of zero is still left alone.

The instrumented runs show the gate where it should: the smoothed delay reads 0
permille on the quiet 4 096 cell, 70–133 permille at one disturber (the gate's
threshold) and 366–476 at four.

### After the fix, same grid

| cell | level | head | merged | fixed | gap0 | fixed/gap0 | fixed/merged |
|---|---|---|---|---|---|---|---|
| prefill 512 | 0 | 721.2 | 715.5 | 718.5 | 728.2 | 0.987 | 1.004 |
| prefill 512 | 1 | 605.0 | 596.2 | 601.7 | 606.8 | 0.992 | 1.009 |
| prefill 512 | 2 | 553.2 | 540.0 | 554.9 | 558.0 | 0.995 | 1.027 |
| prefill 512 | 4 | 438.4 | 434.4 | 438.8 | 442.9 | 0.991 | 1.010 |
| prefill 4 096 | 0 | 516.5 | 509.9 | 512.9 | 516.8 | 0.992 | 1.006 |
| prefill 4 096 | 1 | 444.1 | 438.4 | 441.6 | 446.9 | 0.988 | 1.007 |
| prefill 4 096 | 2 | 415.1 | 409.9 | 412.6 | 417.1 | 0.989 | 1.006 |
| prefill 4 096 | 4 | 325.2 | 320.8 | 321.9 | 324.9 | 0.991 | 1.004 |
| decode | 0 | 123.8 | 123.2 | 123.2 | 123.4 | 0.998 | 0.999 |
| decode | 1 | 104.0 | 104.1 | 104.7 | 103.7 | 1.009 | 1.006 |
| decode | 2 | 103.7 | 102.9 | 103.1 | 103.3 | 0.998 | 1.002 |
| decode | 4 | 92.0 | 91.8 | 91.9 | 89.7 | 1.024 | 1.000 |
| embed 32×128 | 0 | 20.9 | 20.3 | 20.2 | 20.0 | 1.014 | 0.998 |
| embed 32×128 | 1 | 17.6 | 17.3 | 17.1 | 17.1 | 0.998 | 0.988 |
| embed 32×128 | 2 | 16.1 | 15.0 | 15.3 | 15.1 | 1.015 | 1.020 |
| embed 32×128 | 4 | 10.4 | 10.8 | 13.1 | 13.2 | 0.999 | 1.220 |

The fix recovers 0.4–2.7 % of the stretch's cost under disturbance (the largest cell
is prefill 512 at two disturbers: 540.0 → 554.9), and `fixed` is within 0.4–1.2 % of
`gap0` in every prefill cell. What is left is the quiet-cell stretch: the run-queue
reading there is 0 permille, so the gate never closes, and the ~1 % that `merged`
loses to `gap0` quiet stays. It is the open point in the design note: this host's
"+noise" is a stretch with no host signal to tell it from the desktop's +3.3 %, and a
cap tied to the measured join wait (not the run queue) is what a next round would
test.

The after window ran later than the before one and the host's load climbed through it
(2.0–13.9 against 2.6–11.4); the arms are still interleaved per round, and every
number above is that window's own medians. The after embedding cell's four-disturber
level is the one place the two windows disagree sharply (head 10.4 against gap0 13.2),
which is why its ratios there are not read as an arm difference.

## Files

- `reread_grid.py` — the prefill/decode grid: three (four) builds inside every round,
  the arm order rotating, the disturber by PID in a `finally`, load and idle share per
  record.
- `reread_embed.py` — the 32-text embedding wave with one server per arm, warmed and
  then measured level by level.
- `summarize.py` — medians, spreads and the arm ratios (`table.txt`).
- `raw/` — one JSONL per cell and phase; `reread-*` is the three-arm re-read,
  `after-*` the same grid with the fix.
