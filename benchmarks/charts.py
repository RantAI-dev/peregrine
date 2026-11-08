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

# Desktop: Ryzen AI 9 HX 370, 12 threads (benchmarks/results/desktop-zen5-2026-11-07).
# Bluefin: Ryzen 3 PRO 4350G, 4 threads (benchmarks/results/bluefin-zen2-2026-11-08).
# Each case: (label, peregrine, llama.cpp), tok/s.
DECODE = [
    ("Desktop · Llama 3.2 1B · 512", 61.4, 49.4),
    ("Desktop · Llama 3.2 1B · 4096", 45.5, 31.9),
    ("Desktop · Qwen 2.5 0.5B · 512", 66.5, 70.8),
    ("Desktop · Qwen 2.5 0.5B · 4096", 53.0, 46.1),
    ("Bluefin · Llama 3.2 1B · 512", 45.4, 39.5),
    ("Bluefin · Qwen 2.5 0.5B · 512", 52.8, 74.9),
    ("Bluefin · Qwen 2.5 0.5B · 4096", 45.1, 41.0),
]
PREFILL = [
    ("Desktop · Llama 3.2 1B · 512", 147.0, 252.6),
    ("Desktop · Llama 3.2 1B · 4096", 116.8, 187.9),
    ("Desktop · Qwen 2.5 0.5B · 512", 271.1, 419.4),
    ("Desktop · Qwen 2.5 0.5B · 4096", 200.7, 269.9),
    ("Bluefin · Llama 3.2 1B · 512", 196.3, 188.3),
    ("Bluefin · Llama 3.2 1B · 4096", 143.7, 141.2),
    ("Bluefin · Qwen 2.5 0.5B · 512", 260.0, 302.9),
    ("Bluefin · Qwen 2.5 0.5B · 4096", 195.2, 236.0),
]
USERS10 = [
    ("Desktop · Llama 3.2 1B · 512", 52.0, 62.5),
    ("Desktop · Qwen 2.5 0.5B · 512", 80.0, 93.8),
    ("Bluefin · Llama 3.2 1B · 512", 55.5, 37.4),
    ("Bluefin · Qwen 2.5 0.5B · 512", 71.3, 68.7),
]
# The biggest wins, Peregrine ÷ llama.cpp on the same machine and file.
WINS = [
    ("10 users · Llama 1B · 512 · Bluefin", 55.5 / 37.4),
    ("Decode · Llama 1B · 4096 · Desktop", 45.5 / 31.9),
    ("4 users · Llama 1B · 512 · Bluefin", 53.5 / 39.5),
    ("Decode · Llama 1B · 512 · Desktop", 61.4 / 49.4),
    ("Decode · Llama 1B · 512 · Bluefin", 45.4 / 39.5),
    ("Decode · Qwen 0.5B · 4096 · Desktop", 53.0 / 46.1),
    ("Decode · Qwen 0.5B · 4096 · Bluefin", 45.1 / 41.0),
    ("Prompt · Llama 1B · 512 · Bluefin", 196.3 / 188.3),
]
# Prompt processing before and after the November changes, Peregrine against itself.
# Bluefin: the repacked Q4_K kernels (#88). Desktop: the prefill chunk default (#92).
SPEEDUPS = [
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


def svg(title: str, subtitle: str, body: list[str], h: int, legend: bool) -> str:
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
            f'<text class="t2" x="{lx + 115}" y="29" font-size="12">llama.cpp</text>',
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


def paired(name: str, title: str, subtitle: str, rows: list[tuple[str, float, float]], unit_label: str):
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
    (OUT / f"{name}.svg").write_text(svg(title, subtitle, body, bottom + 48, True))


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
        "Where Peregrine is fastest against llama.cpp",
        "Peregrine ÷ llama.cpp v0.9.0, same machine, same Q4_K_M file; the dashed line is level",
        WINS,
        "x",
        "times llama.cpp's speed (tok/s)",
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
