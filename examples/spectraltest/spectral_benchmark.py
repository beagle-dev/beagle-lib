#!/usr/bin/env python3
"""Standard vs spectral BEAGLE CPU implementations: time both over a grid of state counts, pattern counts and
rate categories, and find where one becomes faster than the other.

Subcommands
  run      time every configuration of the grid with spectralbench, the two implementations back to back, and
           append one CSV row per implementation and configuration. Configurations already in the file are
           skipped, so an interrupted run can be resumed.
  analyze  read one or more such CSVs and write a markdown report: the pattern count at which the standard
           implementation becomes faster (the crossover), speedup tables, phase times and a log likelihood
           check; also a CSV of crossovers. With --plots DIR, draws figures with plot_spectral_benchmark.R (base
           R, no packages). A likelihood-only run (--nogradient) can extend a full run to more patterns.

A configuration is one evaluation after a change of the substitution model (see spectralbench.cpp):
  likelihood = setEigenDecomposition + updateTransitionMatrices + post-order traversal + root log likelihood
  gradient   = likelihood + TOP pre-order traversal + adjoint gradient

Example, from a CMake build directory of BEAGLE configured with BUILD_SPECTRAL and BUILD_SSE, after
`make spectralbench`. On macOS, stay on AC power with Low Power Mode off and the lid open: caffeinate prevents
idle sleep, not sleep when the lid closes.
  caffeinate -i python3 ../examples/spectraltest/spectral_benchmark.py run \\
      --bench examples/spectralbench --libdir libhmsbeagle:libhmsbeagle/CPU --out bench.csv
  python3 ../examples/spectraltest/spectral_benchmark.py analyze bench.csv --report report.md --plots figures
"""

import argparse
import csv
import math
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict

DEFAULT_STATES = "4,5,6,8,10,12,16,20,24,28,32,40,48,56,61,64,80,96,112,128"
DEFAULT_PATTERNS = "1,2,4,8,16,32,64,128,256,512,1024"
DEFAULT_CATEGORIES = "1,4"
KEY_FIELDS = ("impl", "vector", "states", "patterns", "categories", "tips", "model", "tipdata")
PHASES = ("eigen_ms", "matrices_ms", "post_ms", "root_ms", "pre_ms", "adjoint_ms")
METRICS = ("likelihood_ms", "gradient_ms", "matrices_ms", "post_ms", "pre_ms", "adjoint_ms")


def int_list(text):
    return [int(x) for x in text.split(",") if x.strip()]


def machine_description():
    lines = ["platform: " + platform.platform()]
    if sys.platform == "darwin":
        for command, label in ((["sysctl", "-n", "machdep.cpu.brand_string"], "cpu"),
                               (["pmset", "-g"], None)):
            try:
                out = subprocess.run(command, capture_output=True, text=True, timeout=10).stdout
            except (OSError, subprocess.SubprocessError):
                continue
            if label:
                lines.append(label + ": " + out.strip())
            else:
                for line in out.splitlines():
                    if "lowpowermode" in line:
                        lines.append("low power mode: " + line.split()[-1])
    else:
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        lines.append("cpu: " + line.split(":", 1)[1].strip())
                        break
        except OSError:
            pass
    return lines


def read_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def key_of(row):
    return tuple(str(row[k]) for k in KEY_FIELDS)


# ---------------------------------------------------------------------------------------------------------------- run

def run(args):
    bench = os.path.abspath(args.bench)
    env = dict(os.environ)
    if args.libdir:
        dirs = [os.path.abspath(d) for d in args.libdir.split(os.pathsep) if d]
        for variable in ("DYLD_LIBRARY_PATH", "LD_LIBRARY_PATH"):
            env[variable] = os.pathsep.join(dirs + ([env[variable]] if env.get(variable) else []))

    header = subprocess.run([bench, "--header"], capture_output=True, text=True, env=env, check=True).stdout.strip()
    done = {key_of(r) for r in read_rows(args.out)}
    if not os.path.exists(args.out):
        with open(args.out, "w") as f:
            f.write(header + "\n")
    with open(args.out + ".meta.txt", "a") as f:
        f.write("# " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
        f.write("command: " + " ".join(sys.argv) + "\n")
        for line in machine_description():
            f.write(line + "\n")

    model = "complex" if args.complex else "reversible"
    tipdata = "partials" if args.tippartials else "states"
    impls = [x for x in args.impls.split(",") if x]
    configs = []
    for k in int_list(args.categories):
        for s in int_list(args.states):
            for c in int_list(args.patterns):
                if s * s * c * k <= args.max_work:
                    configs.append((s, c, k))
    total = len(configs) * len(impls)
    count = 0
    start = time.time()
    for s, c, k in configs:
        for impl in impls:
            count += 1
            key = (impl, args.vector, str(s), str(c), str(k), str(args.tips), model, tipdata)
            if key in done:
                continue
            command = [bench, "--impl", impl, "--vector", args.vector, "--states", str(s), "--patterns", str(c),
                       "--categories", str(k), "--tips", str(args.tips), "--budget", str(args.budget),
                       "--minreps", str(args.minreps), "--seed", str(args.seed)]
            if args.complex:
                command.append("--complex")
            if args.tippartials:
                command.append("--tippartials")
            if args.nogradient:
                command.append("--nogradient")
            result = subprocess.run(command, capture_output=True, text=True, env=env)
            line = result.stdout.strip()
            if result.returncode != 0 or not line:
                print(f"[{count}/{total}] {impl} S={s} C={c} K={k}: failed ({result.stderr.strip()})", file=sys.stderr)
                continue
            with open(args.out, "a") as f:
                f.write(line + "\n")
            row = dict(zip(header.split(","), line.split(",")))
            gradient = "" if args.nogradient else f", gradient {float(row['gradient_ms']):10.4f} ms"
            print(f"[{count}/{total}] {row['implementation']:32s} S={s:3d} C={c:5d} K={k}: likelihood "
                  f"{float(row['likelihood_ms']):10.4f} ms{gradient} ({time.time() - start:.0f} s)",
                  file=sys.stderr, flush=True)


# ------------------------------------------------------------------------------------------------------------ analyze

def crossover(points):
    """points: sorted [(patterns, speedup)] with speedup = standard / spectral time. Returns (value, text): the
    pattern count at which the speedup first falls below 1 (log-linear interpolation), 0 if the standard
    implementation is faster already at the smallest count, inf if the spectral one stays faster."""
    if not points:
        return float("nan"), "n/a"
    if points[0][1] < 1.0:
        return 0.0, f"< {points[0][0]}"
    for (c0, s0), (c1, s1) in zip(points, points[1:]):
        if s1 < 1.0:
            f = math.log(s0) / (math.log(s0) - math.log(s1))
            value = math.exp(math.log(c0) + f * (math.log(c1) - math.log(c0)))
            return value, f"{value:.0f}"
    return float("inf"), f"> {points[-1][0]}"


def fmt_speedup(x):
    if x != x:
        return ""
    return f"{x:.1f}" if x >= 10 else (f"{x:.2f}" if x >= 1 else f"{x:.2f}")


def has_gradient_data(row):
    return float(row["pre_ms"]) > 0 or float(row["adjoint_ms"]) > 0


def analyze(args):
    # several CSVs may be combined, e.g. a full run and a likelihood-only (--nogradient) extension; for a
    # configuration in both, the row with gradient times is kept (its likelihood times are equally valid)
    latest = {}
    for path in args.csv:
        for r in read_rows(path):
            k = key_of(r)
            if k not in latest or has_gradient_data(r) or not has_gradient_data(latest[k]):
                latest[k] = r
    rows = list(latest.values())
    groups = defaultdict(list)
    for r in rows:
        groups[(r["model"], r["tipdata"], r["tips"], r["vector"])].append(r)

    out = []
    crossovers = []
    out.append("# Standard vs spectral BEAGLE CPU implementations\n")
    for path in args.csv:
        meta = path + ".meta.txt"
        if os.path.exists(meta):
            out.append("Run details (from " + os.path.basename(meta) + "):\n\n```")
            with open(meta) as f:
                out.append(f.read().rstrip())
            out.append("```\n")
    out.append("One evaluation after a change of the substitution model: *likelihood* = setEigenDecomposition + "
               "updateTransitionMatrices + post-order traversal + root log likelihood; *gradient* = likelihood + "
               "TOP pre-order traversal + adjoint gradient. Times are medians over repeated evaluations. "
               "Speedup = standard time / spectral time (above 1: spectral is faster). The crossover is the "
               "pattern count at which the standard implementation becomes faster, interpolated on a log scale.\n")

    for (model, tipdata, tips, vector), grs in sorted(groups.items()):
        table = {}  # (impl, S, C, K) -> row
        names = defaultdict(set)
        for r in grs:
            table[(r["impl"], int(r["states"]), int(r["patterns"]), int(r["categories"]))] = r
            names[r["impl"]].add(r["implementation"])
        states = sorted({s for (_, s, _, _) in table})
        patterns = sorted({c for (_, _, c, _) in table})
        categories = sorted({k for (_, _, _, k) in table})
        # a run with --nogradient has no pre-order or adjoint times
        has_gradient = any(has_gradient_data(r) for r in grs)
        headline = ("likelihood_ms", "gradient_ms") if has_gradient else ("likelihood_ms",)
        out.append(f"## {model} model, tips as {tipdata}, {tips} tips, vector {vector}\n")
        out.append("Implementations: standard = " + ", ".join(sorted(names["standard"])) +
                   "; spectral = " + ", ".join(sorted(names["spectral"])) + ".\n")

        def speedup(metric, s, c, k):
            a, b = table.get(("standard", s, c, k)), table.get(("spectral", s, c, k))
            if not a or not b:
                return float("nan")
            if metric in ("gradient_ms", "pre_ms", "adjoint_ms") and not (has_gradient_data(a) and
                                                                         has_gradient_data(b)):
                return float("nan")
            ta, tb = float(a[metric]), float(b[metric])
            return ta / tb if tb > 0 else float("nan")

        # crossover table
        out.append("### Crossover pattern count (spectral is faster below it)\n")
        head = "| states | " + " | ".join(f"{m.replace('_ms', '')} K={k}" for m in headline
                                         for k in categories) + " |"
        out.append(head)
        out.append("|" + "---|" * (1 + len(headline) * len(categories)))
        for s in states:
            cells = []
            for m in headline:
                for k in categories:
                    points = [(c, speedup(m, s, c, k)) for c in patterns]
                    points = [(c, x) for c, x in points if x == x]
                    value, text = crossover(points)
                    cells.append(text)
                    crossovers.append({"model": model, "tipdata": tipdata, "tips": tips, "vector": vector,
                                       "metric": m.replace("_ms", ""), "categories": k, "states": s,
                                       "crossover_patterns": value})
            out.append(f"| {s} | " + " | ".join(cells) + " |")
        out.append("")

        # power law through the interpolated crossovers: patterns* = a * states^b
        out.append("Power-law fit through the interpolated crossovers, patterns* ≈ a × states^b (the censored "
                   "entries of the table above are left out). This is only a rough summary: where the crossover "
                   "levels off at large state counts, the fit overstates it, so read the table:\n")
        out.append("| metric | K | a | b | states used | R² |")
        out.append("|---|---|---|---|---|---|")
        for m in (x.replace("_ms", "") for x in headline):
            for k in categories:
                pts = [(math.log(c["states"]), math.log(c["crossover_patterns"])) for c in crossovers
                       if c["model"] == model and c["tipdata"] == tipdata and c["tips"] == tips and
                       c["vector"] == vector and c["metric"] == m and c["categories"] == k and
                       0 < c["crossover_patterns"] < float("inf")]
                if len(pts) < 2:
                    out.append(f"| {m} | {k} | | | {len(pts)} | |")
                    continue
                n = len(pts)
                mx = sum(x for x, _ in pts) / n
                my = sum(y for _, y in pts) / n
                sxx = sum((x - mx) ** 2 for x, _ in pts)
                sxy = sum((x - mx) * (y - my) for x, y in pts)
                b = sxy / sxx if sxx > 0 else float("nan")
                a = math.exp(my - b * mx)
                ss = sum((y - my) ** 2 for _, y in pts)
                res = sum((y - (my + b * (x - mx))) ** 2 for x, y in pts)
                r2 = 1 - res / ss if ss > 0 else float("nan")
                out.append(f"| {m} | {k} | {a:.3g} | {b:.2f} | {n} | {r2:.2f} |")
        out.append("")

        # speedup grids
        for m in (("likelihood_ms", "gradient_ms", "post_ms", "pre_ms", "adjoint_ms") if has_gradient
                  else ("likelihood_ms", "matrices_ms", "post_ms")):
            for k in categories:
                out.append(f"### Speedup, {m.replace('_ms', '')}, K = {k}\n")
                out.append("| states \\ patterns | " + " | ".join(str(c) for c in patterns) + " |")
                out.append("|" + "---|" * (1 + len(patterns)))
                for s in states:
                    out.append(f"| {s} | " + " | ".join(fmt_speedup(speedup(m, s, c, k)) for c in patterns) + " |")
                out.append("")

        # phase times at one pattern and at the largest common pattern count
        for c in sorted({patterns[0], 64} & set(patterns)):
            for k in categories:
                out.append(f"### Phase times (ms), {c} pattern{'s' if c > 1 else ''}, K = {k}\n")
                out.append("| states | impl | " + " | ".join(p.replace("_ms", "") for p in PHASES) +
                           " | likelihood | gradient |")
                out.append("|" + "---|" * (4 + len(PHASES)))
                for s in states:
                    for impl in ("standard", "spectral"):
                        r = table.get((impl, s, c, k))
                        if r and (has_gradient_data(r) or not has_gradient):
                            out.append(f"| {s} | {impl} | " + " | ".join(f"{float(r[p]):.4g}" for p in PHASES) +
                                       f" | {float(r['likelihood_ms']):.4g} | {float(r['gradient_ms']):.4g} |")
                out.append("")

        # log likelihood agreement
        worst, where = 0.0, None
        for (impl, s, c, k), r in table.items():
            if impl != "standard" or ("spectral", s, c, k) not in table:
                continue
            a, b = float(r["lnL"]), float(table[("spectral", s, c, k)]["lnL"])
            d = abs(a - b) / max(abs(a), 1e-300)
            if d > worst:
                worst, where = d, (s, c, k)
        out.append(f"Log likelihoods of the two implementations agree to a relative {worst:.1e}" +
                   (f" (worst at S={where[0]}, C={where[1]}, K={where[2]})." if where else ".") + "\n")

    with open(args.report, "w") as f:
        f.write("\n".join(out) + "\n")
    crossover_csv = os.path.splitext(args.report)[0] + "_crossovers.csv"
    with open(crossover_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(crossovers[0].keys()) if crossovers else ["model"])
        w.writeheader()
        for c in crossovers:
            w.writerow(c)
    print(f"wrote {args.report} and {crossover_csv}", file=sys.stderr)

    if args.plots:
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plot_spectral_benchmark.R")
        os.makedirs(args.plots, exist_ok=True)
        subprocess.run(["Rscript", script] + args.csv + [args.plots], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="time the grid")
    p.add_argument("--bench", default="examples/spectralbench", help="the spectralbench executable")
    p.add_argument("--libdir", default="", help="directories with libhmsbeagle and its plugins, joined by "
                                                "os.pathsep (for a build tree: libhmsbeagle:libhmsbeagle/CPU)")
    p.add_argument("--out", default="spectral_benchmark.csv")
    p.add_argument("--states", default=DEFAULT_STATES)
    p.add_argument("--patterns", default=DEFAULT_PATTERNS)
    p.add_argument("--categories", default=DEFAULT_CATEGORIES)
    p.add_argument("--impls", default="standard,spectral")
    p.add_argument("--vector", default="sse", choices=("sse", "none"))
    p.add_argument("--tips", type=int, default=64)
    p.add_argument("--budget", type=float, default=0.5, help="seconds of timed evaluations per configuration")
    p.add_argument("--minreps", type=int, default=3)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-work", type=float, default=2 ** 24, dest="max_work",
                   help="skip configurations with states^2 * patterns * categories above this")
    p.add_argument("--complex", action="store_true", help="asymmetric circulant model (complex eigenvalues)")
    p.add_argument("--tippartials", action="store_true", help="tips as partials instead of states")
    p.add_argument("--nogradient", action="store_true", help="time the likelihood only")
    p.set_defaults(func=run)

    a = sub.add_parser("analyze", help="report on a CSV written by run")
    a.add_argument("csv", nargs="+", help="one or more CSVs written by run")
    a.add_argument("--report", default="spectral_benchmark.md")
    a.add_argument("--plots", default="", help="directory for figures (needs Rscript)")
    a.set_defaults(func=analyze)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
