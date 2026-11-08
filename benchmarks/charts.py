#!/usr/bin/env python3
"""Draws the charts in docs/benchmark.md and the README as static SVG files.

The numbers are copied from the result folders under benchmarks/results/ (medians);
each entry names its source. Run `python3 benchmarks/charts.py` after adding a result
and the files under docs/img/ are rewritten. Standard library only.

The charts follow one style: horizontal bars from zero, the value at each bar's tip,
recessive grid lines, Peregrine in blue and llama.cpp in orange (a pair checked for
colour-blind separation in light and dark), and a dark variant chosen by the viewer's
`prefers-color-scheme`. Every chart has a table with the same numbers beside it in
docs/benchmark.md.
"""

from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "docs" / "img"

# Desktop: Ryzen AI 9 HX 370, 12 threads (Peregrine and llama.cpp -fa off from
# benchmarks/results/desktop-zen5-2026-11-08-fa-off; llama.cpp at its faster flash-attention
# setting per row, which is the default for the 1B prompt and the multi-user rows).
# Bonsai 2: the same desktop, against PrismML's fork (benchmarks/results/2026-11-08-bonsai2-prism),
# and the 4-core AVX2 bluefin (benchmarks/results/bluefin-zen2-2026-11-08-bonsai2-prism);
# upstream llama.cpp cannot load PQ2_0. The fork bar is its faster flash-attention setting per row.
# Bluefin: Ryzen 3 PRO 4350G, 4 threads (benchmarks/results/bluefin-zen2-2026-11-08-q5).
# The bluefin llama.cpp bar is the faster flash-attention setting for that row (-fa off for
# single-user decode, llama.cpp's default for prefill and multi-user throughput).
# Each case: (label, peregrine, llama.cpp), tok/s.
DECODE = [
    ("Desktop · Llama 3.2 1B · 512", 62.6, 51.8),
    ("Desktop · Llama 3.2 1B · 4096", 47.0, 33.8),
    ("Desktop · Qwen 2.5 0.5B · 512", 70.0, 75.8),
    ("Desktop · Qwen 2.5 0.5B · 4096", 54.9, 49.1),
    ("Bluefin · Llama 3.2 1B · 512", 46.83, 42.50),
    ("Bluefin · Llama 3.2 1B · 4096", 36.81, 29.25),
    ("Bluefin · Qwen 2.5 0.5B · 512", 83.48, 77.69),
    ("Bluefin · Qwen 2.5 0.5B · 4096", 65.89, 52.20),
    ("Bluefin · SmolLM2 360M · 512", 108.92, 94.45),
    ("Bluefin · SmolLM2 360M · 4096", 63.79, 47.51),
]
PREFILL = [
    ("Desktop · Llama 3.2 1B · 512", 183.8, 262.0),
    ("Desktop · Llama 3.2 1B · 4096", 130.9, 189.0),
    ("Desktop · Qwen 2.5 0.5B · 512", 321.9, 435.2),
    ("Desktop · Qwen 2.5 0.5B · 4096", 215.1, 293.1),
    ("Bluefin · Llama 3.2 1B · 512", 195.67, 188.09),
    ("Bluefin · Llama 3.2 1B · 4096", 144.95, 141.37),
    ("Bluefin · Qwen 2.5 0.5B · 512", 434.78, 303.73),
    ("Bluefin · Qwen 2.5 0.5B · 4096", 277.58, 236.46),
    ("Bluefin · SmolLM2 360M · 512", 443.19, 318.91),
    ("Bluefin · SmolLM2 360M · 4096", 228.45, 217.05),
]
USERS10 = [
    ("Desktop · Llama 3.2 1B · 512", 53.4, 61.7),
    ("Desktop · Qwen 2.5 0.5B · 512", 83.4, 93.4),
    ("Bluefin · Llama 3.2 1B · 512", 55.53, 37.74),
    ("Bluefin · Qwen 2.5 0.5B · 512", 108.41, 68.36),
    ("Bluefin · SmolLM2 360M · 512", 116.76, 67.63),
]
BONSAI = [
    ("Bonsai 2 2B · 512 · decode", 70.64, 32.66),
    ("Bonsai 2 2B · 4096 · decode", 60.64, 30.47),
    ("Bonsai 2 2B · 512 · prompt", 259.25, 108.24),
    ("Bonsai 2 2B · 4096 · prompt", 202.94, 105.19),
    ("Bonsai 2 9B · 512 · decode", 22.97, 9.85),
    ("Bonsai 2 9B · 4096 · decode", 20.36, 9.22),
    ("Bonsai 2 9B · 512 · prompt", 67.24, 26.12),
    ("Bonsai 2 9B · 4096 · prompt", 57.42, 25.53),
    ("Bluefin · Bonsai 2B · 512 · decode", 46.40, 33.63),
    ("Bluefin · Bonsai 2B · 4096 · decode", 42.28, 30.79),
    ("Bluefin · Bonsai 2B · 512 · prompt", 118.67, 56.95),
    ("Bluefin · Bonsai 2B · 4096 · prompt", 109.04, 58.07),
    ("Bluefin · Bonsai 4B · 512 · decode", 24.59, 14.95),
    ("Bluefin · Bonsai 4B · 512 · prompt", 48.05, 18.55),
    # The bluefin 4B at 4096 is absent: the fork's row is not completed (10-minute cap).
]
# The biggest wins, Peregrine ÷ the competitor that runs the same file on the same machine:
# llama.cpp for the Q4_K_M rows, PrismML's fork for the Bonsai 2 rows.
WINS = [
    ("10 users · Bonsai 9B · 512 · Desktop", 21.62 / 7.83),
    ("10 users · Bonsai 2B · 512 · Desktop", 80.59 / 29.96),
    ("Prompt · Bonsai 9B · 512 · Desktop", 67.24 / 26.12),
    ("Prompt · Bonsai 2B · 512 · Desktop", 259.25 / 108.24),
    ("Decode · Bonsai 9B · 512 · Desktop", 22.97 / 9.85),
    ("Decode · Bonsai 2B · 512 · Desktop", 70.64 / 32.66),
    ("10 users · SmolLM2 360M · 4096 · Bluefin", 12.11 / 6.74),
    ("10 users · SmolLM2 360M · 512 · Bluefin", 116.76 / 67.63),
    ("4 users · SmolLM2 360M · 512 · Bluefin", 114.18 / 66.15),
    ("4 users · Qwen 0.5B · 512 · Bluefin", 104.33 / 65.73),
    ("10 users · Qwen 0.5B · 512 · Bluefin", 108.41 / 68.36),
    ("10 users · Llama 1B · 4096 · Bluefin", 7.71 / 5.07),
]
# Prompt processing before and after the November changes, Peregrine against itself.
# Bluefin: the Q5_0 kernels (#103-#105) and the repacked Q4_K kernels (#88). Desktop: the
# prefill chunk default (#92).
SPEEDUPS = [
    ("Bluefin · Qwen 2.5 0.5B (Q5_0 kernels)", 434.78 / 263.19),
    ("Bluefin · Llama 3.2 1B (Q4_K kernels)", 152 / 104),
    ("Desktop · Bonsai 2 2B (prefill chunk)", 1.43),
    ("Bluefin · Qwen 3.5 2B (Q4_K kernels)", 96 / 72),
    ("Desktop · Llama 3.2 1B (prefill chunk)", 1.19),
    ("Desktop · Bonsai 2 9B (prefill chunk)", 1.14),
]

STYLE = """
<style>
  .surface { fill: #fcfcfb; }
  .t1 { fill: #0b0b0b; } .t2 { fill: #52514e; }
  .grid { stroke: #e4e3df; } .ref { stroke: #52514e; }
  .s1 { fill: #2a78d6; } .s2 { fill: #eb6834; }
  .gap { stroke: #fcfcfb; }
  text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
  @media (prefers-color-scheme: dark) {
    .surface { fill: #1a1a19; }
    .t1 { fill: #f0efec; } .t2 { fill: #c3c2b7; }
    .grid { stroke: #383835; } .ref { stroke: #c3c2b7; }
    .s1 { fill: #3987e5; } .s2 { fill: #d95926; }
    .gap { stroke: #1a1a19; }
  }
</style>
"""

W = 760
LABEL_W = 250
RIGHT = 70
BAR_H = 12
PAIR_GAP = 2
GROUP_GAP = 14


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def bar(x0: float, y: float, w: float, h: float, cls: str) -> str:
    """A bar anchored square at the baseline, 4 px rounded at its data end."""
    r = min(4.0, w / 2, h / 2)
    if w <= 0:
        return ""
    return (
        f'<path class="{cls}" d="M{x0:.1f},{y:.1f} H{x0 + w - r:.1f} '
        f"A{r},{r} 0 0 1 {x0 + w:.1f},{y + r:.1f} V{y + h - r:.1f} "
        f'A{r},{r} 0 0 1 {x0 + w - r:.1f},{y + h:.1f} H{x0:.1f} Z"/>'
    )


def nice_max(v: float) -> tuple[float, float]:
    """A clean axis end and tick step covering v."""
    for step in (0.25, 0.5, 1, 2, 5, 10, 20, 25, 50, 100, 200):
        if v / step <= 6:
            return (int(v / step) + 1) * step, step
    return v, v / 5


def fmt(v: float, unit: str) -> str:
    if unit == "x":
        return f"{v:.2f}×"
    if unit == "%":
        return f"+{(v - 1) * 100:.0f}%"
    return f"{v:.0f}" if v >= 100 else f"{v:.1f}"


def svg(title: str, subtitle: str, body: list[str], h: int, legend: bool,
        rival: str = "llama.cpp") -> str:
    head = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{h}" '
        f'viewBox="0 0 {W} {h}" role="img" aria-label="{esc(title)}">',
        STYLE,
        f'<rect class="surface" width="{W}" height="{h}" rx="6"/>',
        f'<text class="t1" x="20" y="30" font-size="16" font-weight="600">{esc(title)}</text>',
        f'<text class="t2" x="20" y="50" font-size="12">{esc(subtitle)}</text>',
    ]
    if legend:
        lx = W - 230
        head += [
            f'<rect class="s1" x="{lx}" y="20" width="10" height="10" rx="2"/>',
            f'<text class="t2" x="{lx + 15}" y="29" font-size="12">Peregrine</text>',
            f'<rect class="s2" x="{lx + 100}" y="20" width="10" height="10" rx="2"/>',
            f'<text class="t2" x="{lx + 115}" y="29" font-size="12">{esc(rival)}</text>',
        ]
    return "\n".join(head + body + ["</svg>\n"])


def axis(top: int, bottom: int, vmax: float, step: float, unit: str) -> list[str]:
    out = []
    plot_w = W - LABEL_W - RIGHT
    v = 0.0
    while v <= vmax + 1e-9:
        x = LABEL_W + v / vmax * plot_w
        out.append(f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}" stroke-width="1"/>')
        label = f"{v:g}×" if unit in ("x", "%") else f"{v:g}"
        out.append(
            f'<text class="t2" x="{x:.1f}" y="{bottom + 16}" font-size="11" text-anchor="middle">{label}</text>'
        )
        v += step
    return out


def paired(name: str, title: str, subtitle: str, rows: list[tuple[str, float, float]], unit_label: str,
           rival: str = "llama.cpp"):
    top = 72
    group_h = 2 * BAR_H + PAIR_GAP
    bottom = top + len(rows) * (group_h + GROUP_GAP) - GROUP_GAP + 8
    vmax, step = nice_max(max(max(p, l) for _, p, l in rows))
    plot_w = W - LABEL_W - RIGHT
    body = axis(top - 6, bottom, vmax, step, "")
    y = top
    for label, p, l in rows:
        body.append(
            f'<text class="t1" x="{LABEL_W - 12}" y="{y + group_h / 2 + 4:.1f}" font-size="12" text-anchor="end">{esc(label)}</text>'
        )
        for i, (v, cls) in enumerate(((p, "s1"), (l, "s2"))):
            by = y + i * (BAR_H + PAIR_GAP)
            w = v / vmax * plot_w
            body.append(bar(LABEL_W, by, w, BAR_H, cls))
            body.append(
                f'<text class="t2" x="{LABEL_W + w + 6:.1f}" y="{by + BAR_H - 2}" font-size="11">{fmt(v, "")}</text>'
            )
        y += group_h + GROUP_GAP
    body.append(
        f'<text class="t2" x="{LABEL_W + plot_w / 2:.1f}" y="{bottom + 34}" font-size="11" text-anchor="middle">{esc(unit_label)}</text>'
    )
    (OUT / f"{name}.svg").write_text(svg(title, subtitle, body, bottom + 48, True, rival))


def single(name: str, title: str, subtitle: str, rows: list[tuple[str, float]], unit: str, axis_label: str):
    top = 72
    bottom = top + len(rows) * (BAR_H + 2 + GROUP_GAP) - GROUP_GAP + 8
    vmax, step = nice_max(max(v for _, v in rows))
    plot_w = W - LABEL_W - RIGHT
    body = axis(top - 6, bottom, vmax, step, unit)
    # The 1.0× line: level with llama.cpp (or with Peregrine's own earlier speed).
    x1 = LABEL_W + 1.0 / vmax * plot_w
    body.append(
        f'<line class="ref" x1="{x1:.1f}" y1="{top - 10}" x2="{x1:.1f}" y2="{bottom}" stroke-width="1.5" stroke-dasharray="4 3"/>'
    )
    y = top
    for label, v in rows:
        w = v / vmax * plot_w
        body.append(
            f'<text class="t1" x="{LABEL_W - 12}" y="{y + BAR_H - 1}" font-size="12" text-anchor="end">{esc(label)}</text>'
        )
        body.append(bar(LABEL_W, y, w, BAR_H + 2, "s1"))
        body.append(
            f'<text class="t1" x="{LABEL_W + w + 6:.1f}" y="{y + BAR_H}" font-size="12" font-weight="600">{fmt(v, unit)}</text>'
        )
        y += BAR_H + 2 + GROUP_GAP
    body.append(
        f'<text class="t2" x="{LABEL_W + plot_w / 2:.1f}" y="{bottom + 34}" font-size="11" text-anchor="middle">{esc(axis_label)}</text>'
    )
    (OUT / f"{name}.svg").write_text(svg(title, subtitle, body, bottom + 48, False))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    single(
        "bench-wins",
        "Where Peregrine is fastest",
        "Peregrine ÷ the competitor on the same machine and file (llama.cpp, or PrismML's fork for "
        "Bonsai 2); the dashed line is level",
        WINS,
        "x",
        "times the competitor's speed (tok/s)",
    )
    paired(
        "bench-bonsai",
        "Bonsai 2 against PrismML's llama.cpp fork",
        "tok/s, higher is better; upstream llama.cpp cannot load these PQ2_0 files, so the fork is the "
        "competitor",
        BONSAI,
        "tokens per second",
        rival="PrismML fork",
    )
    paired(
        "bench-decode",
        "Single-user decode",
        "tok/s, higher is better; context length after the model name",
        DECODE,
        "tokens per second",
    )
    paired(
        "bench-prefill",
        "Prompt processing",
        "tok/s, higher is better; context length after the model name",
        PREFILL,
        "tokens per second",
    )
    paired(
        "bench-users10",
        "Total throughput, 10 users at once",
        "tok/s summed over all users, 512-token context, higher is better",
        USERS10,
        "tokens per second",
    )
    single(
        "bench-speedups",
        "Prompt processing: the November changes",
        "Peregrine after ÷ before, same machine and file; the dashed line is no change",
        SPEEDUPS,
        "%",
        "times the earlier speed",
    )


if __name__ == "__main__":
    main()
