#!/bin/bash
# Bracket a fixed-MAC-count byte-dot run with two remote Date headers: the host clock is
# the only one outside the VM. 1 s resolution, so the window is 20-40 s.
set -u
hostdate() { curl -s -I -m 10 https://example.com | awk -F': ' 'tolower($1)=="date"{print $2}' | tr -d '\r'; }
h0=$(hostdate); out=$(taskset -c 4 ./roof_abs "${1:-2000000000}"); h1=$(hostdate)
echo "$out"
python3 - "$h0" "$h1" "$out" <<'PY'
import sys, re, email.utils
a = email.utils.parsedate_to_datetime(sys.argv[1]); b = email.utils.parsedate_to_datetime(sys.argv[2])
host = (b - a).total_seconds()
macs = float(re.search(r"MACs ([\d.e+]+)", sys.argv[3]).group(1))
print(f"host Date window {host:.0f} s (+-2)  ->  {macs/host/1e9:.1f} GMAC/s in real seconds")
PY
