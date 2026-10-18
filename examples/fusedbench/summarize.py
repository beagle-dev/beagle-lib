#!/usr/bin/env python3
"""summarize.py: fusedbench CSVs -> summary.md (fused-handoff/PLAN-fused-uppartials.md sections 7.1 and 9).

usage: python3 -I summarize.py [-o summary.md] CSV_OR_DIRECTORY...

For each CSV (one fusedbench process) and configuration, the ratios A/B and A2/B of T_up_min_us and T_eval_min_us
are taken within the process (> 1: fused faster), and only between rows in the same completion mode (fusedbench
--mode; process_mode column; rows whose mode changed during their timed blocks, "changed", are skipped). Each regime then lists the median and range of its per-process ratios, fast and slow
modes in separate tables, followed by the crossovers C1-C4 (where the median T_up ratio A/B crosses 1, interpolated
in log x). Failed checks (ok = 0) are listed first. The output defaults to summary.md next to the first CSV.
"""
import csv
import math
import os
import statistics
import sys

KEY = ("backend", "device", "precision", "states", "patterns", "categories", "taxa", "tree", "update", "scaling",
       "scalers")


def read_rows(path):
    with open(path) as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


def csv_paths(args):
    paths = []
    for a in args:
        if os.path.isdir(a):
            paths += sorted(os.path.join(a, n) for n in os.listdir(a) if n.endswith(".csv"))
        else:
            paths.append(a)
    return paths


def collect(paths):
    """{(mode, key): {"A/B": {"T_up": [...], "T_eval": [...]}, "A2/B": ..., "B_up": [...]}}, failures"""
    ratios, failures = {}, []
    for path in paths:
        configs = {}
        for r in read_rows(path):
            if r["ok"] != "1":
                failures.append((os.path.basename(path), r))
            configs.setdefault(tuple(r[k] for k in KEY), {})[r["impl"]] = r
        for key, impls in configs.items():
            b = impls.get("B")
            if b is None:
                continue
            entry = ratios.setdefault((b["process_mode"], key), {"A/B": {"T_up": [], "T_eval": []},
                                                                 "A2/B": {"T_up": [], "T_eval": []}, "B_up": []})
            entry["B_up"].append(float(b["T_up_min_us"]))
            for base in ("A", "A2"):
                a = impls.get(base)
                if a is None or a["process_mode"] != b["process_mode"]:
                    continue
                for t in ("T_up", "T_eval"):
                    entry[base + "/B"][t].append(float(a[t + "_min_us"]) / float(b[t + "_min_us"]))
    return ratios, failures


def cell(values):
    if not values:
        return "-"
    m = statistics.median(values)
    return f"{m:.2f}" if len(values) == 1 else f"{m:.2f} ({min(values):.2f}-{max(values):.2f})"


def crossing(points):
    """x where the ratio crosses 1, interpolated in log x between the first pair of points on either side"""
    for (x0, r0), (x1, r1) in zip(points, points[1:]):
        if (r0 - 1) * (r1 - 1) < 0:
            t = (1 - r0) / (r1 - r0)
            return math.exp(math.log(x0) + t * (math.log(x1) - math.log(x0)))
    return None


CROSSOVERS = [
    ("C1", "4-state balanced full, none/read, P 1000: T_up A/B against taxa", "taxa",
     lambda k: k["states"] == "4" and k["tree"] == "balanced" and k["update"] == "full" and
     k["scaling"] in ("none", "read") and k["patterns"] == "1000"),
    ("C2", "4 states, n 1000, balanced full: T_up A/B against patterns", "patterns",
     lambda k: k["states"] == "4" and k["taxa"] == "1000" and k["tree"] == "balanced" and k["update"] == "full"),
    ("C3", "codon balanced full, none/read: T_up A/B against patterns", "patterns",
     lambda k: k["states"] == "61" and k["tree"] == "balanced" and k["update"] == "full" and
     k["scaling"] in ("none", "read")),
    ("C4", "codon balanced full, write: T_up A/B against patterns", "patterns",
     lambda k: k["states"] == "61" and k["tree"] == "balanced" and k["update"] == "full" and k["scaling"] == "write"),
]


def main(argv):
    out = None
    if len(argv) > 2 and argv[1] == "-o":
        out, argv = argv[2], argv[:1] + argv[3:]
    paths = csv_paths(argv[1:])
    if not paths:
        sys.exit(__doc__)
    out = out or os.path.join(os.path.dirname(paths[0]), "summary.md")
    ratios, failures = collect(paths)
    lines = ["# fusedbench summary", "", f"{len(paths)} CSV files: " + ", ".join(os.path.basename(p) for p in paths),
             "", "Ratios of minimum times within one process, A/B and A2/B (> 1: fused faster); median (range) over "
             "processes.", ""]
    if failures:
        lines += ["## Failed checks", "", "| file | impl | configuration | logL | logL_ref | max_ulp_vs_A |",
                  "|---|---|---|---|---|---|"]
        for name, r in failures:
            conf = " ".join(f"{k}={r[k]}" for k in KEY[3:])
            lines.append(f"| {name} | {r['impl']} | {conf} | {r['logL']} | {r['logL_ref']} | {r['max_ulp_vs_A']} |")
        lines.append("")
    for mode in ("fast", "slow"):
        keys = sorted((k for m, k in ratios if m == mode), key=lambda k: tuple(int(x) if x.isdigit() else x for x in k))
        if not keys:
            continue
        lines += [f"## {mode}-mode processes", "",
                  "| states | P | C | n | tree | update | scaling | T_up A/B | T_up A2/B | T_eval A/B | T_eval A2/B "
                  "| B T_up µs | processes |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for k in keys:
            e = ratios[(mode, k)]
            d = dict(zip(KEY, k))
            lines.append(f"| {d['states']} | {d['patterns']} | {d['categories']} | {d['taxa']} | {d['tree']} | "
                         f"{d['update']} | {d['scaling']} | {cell(e['A/B']['T_up'])} | {cell(e['A2/B']['T_up'])} | "
                         f"{cell(e['A/B']['T_eval'])} | {cell(e['A2/B']['T_eval'])} | {min(e['B_up']):.1f} | "
                         f"{len(e['B_up'])} |")
        lines.append("")
        lines += [f"### Crossovers ({mode} mode)", ""]
        for name, title, axis, match in CROSSOVERS:
            series = {}
            for m, k in ratios:
                d = dict(zip(KEY, k))
                if m != mode or not match(d) or not ratios[(m, k)]["A/B"]["T_up"]:
                    continue
                label = f"{d['scaling']}" + (f", n {d['taxa']}" if axis == "patterns" and d["states"] == "61" else "")
                series.setdefault(label, []).append((int(d[axis]), statistics.median(ratios[(m, k)]["A/B"]["T_up"])))
            if not series:
                lines.append(f"- **{name}** ({title}): no data")
                continue
            for label, points in sorted(series.items()):
                points.sort()
                x = crossing(points)
                where = f"crosses 1 at {axis} ≈ {x:.0f}" if x else "no crossing in range"
                pts = ", ".join(f"{px}: {pr:.2f}" for px, pr in points)
                lines.append(f"- **{name}** ({title}; {label}): {pts}; {where}")
        lines.append("")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print(f"summarize.py: wrote {out}")


if __name__ == "__main__":
    main(sys.argv)
