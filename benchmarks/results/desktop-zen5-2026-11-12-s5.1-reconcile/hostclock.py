#!/usr/bin/env python3
"""How fast do this guest's clocks run? A remote HTTP Date header (1 s resolution) is the
only clock here that is not inside the VM. A 40 s burn loop is timed by the host, by
CLOCK_MONOTONIC, by CLOCK_MONOTONIC_RAW, by the child's CPU time and by the TSC."""
import resource, subprocess, sys, time, urllib.request, email.utils

def host(now=time.time()):
    r = urllib.request.urlopen("https://example.com", timeout=15)
    return email.utils.parsedate_to_datetime(r.headers["Date"]).timestamp()

secs = float(sys.argv[1]) if len(sys.argv) > 1 else 40.0
h0 = host(); r0 = resource.getrusage(resource.RUSAGE_CHILDREN); m0 = time.monotonic()
out = subprocess.run(["taskset", "-c", "4", "./clockprobe", str(secs)],
                     capture_output=True, text=True)
m1 = time.monotonic(); h1 = host(); r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
cpu = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
print(out.stdout.strip())
print(f"host Date   {h1 - h0:.0f} s (+-2, 1 s resolution)")
print(f"MONOTONIC   {m1 - m0:.2f} s")
print(f"CPU time    {cpu:.2f} s   (one thread, so this is the real elapsed time)")
print(f"MONOTONIC / host = {(m1 - m0) / (h1 - h0):.3f}   CPU / host = {cpu / (h1 - h0):.3f}")
