#!/usr/bin/env python3
"""Performance regression gate.

Reads the JSON line printed by moldcool_bench (stdin or file), compares the throughput numbers
to bench/baseline.json for the named profile, and exits non-zero if any metric fell below
baseline * min_fraction. Usage:

    ./moldcool_bench | python bench/check_perf.py --profile ci --min-fraction 0.6
    python bench/check_perf.py --profile local --update < bench_output.json   # rewrite baseline
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BASELINE = os.path.join(HERE, "baseline.json")
METRICS = ["explicit_serial_mlups", "explicit_omp_mlups"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="local")
    ap.add_argument("--min-fraction", type=float, default=0.75)
    ap.add_argument("--update", action="store_true")
    ap.add_argument("input", nargs="?", default="-")
    a = ap.parse_args()
    raw = sys.stdin.read() if a.input == "-" else open(a.input).read()
    line = [l for l in raw.splitlines() if l.strip().startswith("{")][-1]
    now = json.loads(line)
    base_all = json.load(open(BASELINE)) if os.path.exists(BASELINE) else {}
    if a.update:
        base_all[a.profile] = now
        json.dump(base_all, open(BASELINE, "w"), indent=2)
        print(f"baseline[{a.profile}] updated: {now}")
        return 0
    base = base_all.get(a.profile)
    if base is None:
        print(f"no baseline for profile '{a.profile}', recording it")
        base_all[a.profile] = now
        json.dump(base_all, open(BASELINE, "w"), indent=2)
        return 0
    ok = True
    for m in METRICS:
        frac = now[m] / base[m]
        status = "ok" if frac >= a.min_fraction else "REGRESSION"
        if frac < a.min_fraction:
            ok = False
        print(f"{m:26s} now {now[m]:9.2f}  baseline {base[m]:9.2f}  ratio {frac:5.2f}  {status}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
