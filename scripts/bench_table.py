#!/usr/bin/env python3
"""
Print a combined theoretical-vs-measured table for the PFR prove phase.

Usage:
    cargo bench --bench theoretical 2>&1 | tee /tmp/theo.txt
    cargo bench --bench comparison 2>&1
    python3 scripts/bench_table.py /tmp/theo.txt
"""

import json, os, re, sys

SIZES = [(64,64),(64,256),(128,128),(128,512),(256,256),(256,1024)]
BASE = os.path.join(os.path.dirname(__file__), "..", "target", "criterion")

def load_criterion_ms(group, label):
    """
    Try criterion path layouts in order of preference (newest first):
      1. this_prove/n=64,m=64        (BenchmarkGroup, current layout)
      2. this_prove_n=64,m=64/prove  (per-size group with inner name "prove")
      3. this_prove_n=64,m=64        (old flat layout)
    """
    dir_name = group.replace("/", "_")
    candidates = [
        os.path.join(BASE, f"{dir_name}/{label}/new/estimates.json"),
        os.path.join(BASE, f"{dir_name}_{label}/prove/new/estimates.json"),
        os.path.join(BASE, f"{dir_name}_{label}/new/estimates.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            d = json.load(open(path))
            return d["mean"]["point_estimate"] / 1e6
    return None

def parse_theoretical(path):
    """Extract theor(ms) values from the theoretical bench stdout."""
    theor = {}
    with open(path) as f:
        for line in f:
            # matches table rows like: ║ n=  64, m=  64 ║  2.16  ...  ║  11.62  ...  ║  3.16  ║
            m = re.search(r'n=\s*(\d+),\s*m=\s*(\d+).*?([\d]+\.[\d]+)\s.*?([\d]+\.[\d]+)\s.*?([\d]+\.[\d]+)', line)
            if m:
                n, m2 = int(m.group(1)), int(m.group(2))
                sc, pr, vr = float(m.group(3)), float(m.group(4)), float(m.group(5))
                theor[(n, m2)] = (sc, pr, vr)
    return theor

if len(sys.argv) < 2:
    print("Usage: bench_table.py <theoretical_stdout.txt>")
    sys.exit(1)

theor = parse_theoretical(sys.argv[1])

print()
print(f"{'(n,m)':>14}  {'theor_prove':>12}  {'meas_prove':>11}  {'ratio':>6}  {'theirs_prove':>13}")
print("-" * 65)
for n, m in SIZES:
    key   = (n, m)
    label = f"n={n},m={m}"
    _, pr_t, _ = theor.get(key, (None, None, None))
    pr_m = load_criterion_ms("this/prove", label)
    th_m = load_criterion_ms("theirs/prove", label)

    pr_t_s  = f"{pr_t:.2f} ms"  if pr_t is not None else "N/A"
    pr_m_s  = f"{pr_m:.2f} ms"  if pr_m is not None else "N/A"
    th_m_s  = f"{th_m:.2f} ms"  if th_m is not None else "N/A"
    ratio_s = f"{pr_m/pr_t:.2f}x" if (pr_m and pr_t) else "N/A"

    print(f"({n:3},{m:4})      {pr_t_s:>12}  {pr_m_s:>11}  {ratio_s:>6}  {th_m_s:>13}")
