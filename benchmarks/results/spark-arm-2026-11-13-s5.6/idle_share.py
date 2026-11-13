#!/usr/bin/env python3
"""Per-core idle share of the Cortex-X925 cluster over a measurement window.

Reads two /proc/stat snapshots and prints, for CPUs 5-9 and 15-19, the idle share
of the delta: `idle = (idle + iowait) / total`. A window with an unexpected idle
share (a co-tenant's job arriving or leaving) is not a standing.

  python3 idle_share.py before.txt after.txt
  python3 idle_share.py --snapshot > before.txt

The engine and the disturbance threads are pinned to exactly those ten CPUs, so
this is the share the measurement actually saw.
"""

import sys

CPUS = [5, 6, 7, 8, 9, 15, 16, 17, 18, 19]


def snapshot(path=None):
    lines = open(path).read().splitlines() if path else open("/proc/stat").read().splitlines()
    out = {}
    for line in lines:
        parts = line.split()
        if not parts or not parts[0].startswith("cpu"):
            continue
        name = parts[0]
        if name == "cpu":
            continue
        cpu = int(name[3:])
        vals = [int(v) for v in parts[1:]]
        # user nice system idle iowait irq softirq steal guest guest_nice
        idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
        total = sum(vals[:8])
        out[cpu] = (idle, total)
    return out


def main():
    if "--snapshot" in sys.argv:
        for cpu in CPUS:
            i, t = snapshot().get(cpu, (0, 0))
            print(cpu, i, t)
        return
    def read(path):
        return {int(l.split()[0]): (int(l.split()[1]), int(l.split()[2])) for l in open(path)}

    a = read(sys.argv[1])
    b = snapshot() if sys.argv[2] == "-" else read(sys.argv[2])
    vals = []
    for cpu in CPUS:
        di = b[cpu][0] - a[cpu][0]
        dt = b[cpu][1] - a[cpu][1]
        if dt > 0:
            vals.append(di / dt)
    print(f"idle_share {sum(vals) / len(vals):.3f} min {min(vals):.3f} max {max(vals):.3f}")


if __name__ == "__main__":
    main()
