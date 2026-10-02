"""TODO.md plan step D1: a TinyGPU run of synthetictest or hmctest against its references (d1_refs.sh). Reads files only.
    d1_compare.py synthetictest <GPU stdout> <CPU SP stdout> <CPU DP stdout> [--double]
    d1_compare.py hmctest <GPU stdout> <CPU SP stdout> <other-GPU stdout>
synthetictest (with --sitelikes): every site value, and p0/p1, within this run's own tolerance of the DP value, per printed
family: max(10 * max|SP-DP|, 1e-6 * max|DP|) + 2e-5 (the CPU single-precision error, ten times; 2e-5 is two roundings of
%.5f). logL, d1 and d2, which kernelSumSites* reduce in float in blocks of 128, within 8 * 2^-24 * 9 * sum|site values|
+ 2e-5 of synthetictest's own double sums of the same GPU site values (sumLogL, ...; pattern weights are at most 9). Those
sums are not compared with DP: the GPU's per-site float error, invisible at 5 decimals, adds up over 10000 weighted sites
(0.06 in d2 on the Mac's OpenCL GPU). hmctest: every number after 'Impl Desc' within 1e-4 relative + 1e-6 absolute of the
CPU SP run (hmctest --tinygpu), except the cross-product line ('now:'), compared with the Mac's OpenCL GPU instead: the
GPU kernel skips a pattern with a missing-state tip (kernels4Derivatives.cu:528), the CPU counts it (BeagleCPUImpl.hpp:
2588-2615). Both: the three files must print the same lines with the numbers blanked out.
--double (TODO.md plan step A7: a GPU run in double precision): every value within 1e-9 * max|DP| + 2e-5 of the DP reference,
the print's rounding only, and the GPU's sums at double precision's rounding (2^-53, not 2^-24)."""
import math, re, sys

NUM = re.compile(r"-?(?:nan|inf|\d+\.?\d*(?:e[-+]?\d+)?)", re.I)

def body(path, start):   # the program's output after its resource header, without timings
    ls = open(path, errors="replace").read().splitlines()
    i = next((k for k, l in enumerate(ls) if l.lstrip().startswith(start)), None)
    if i is None: sys.exit(f"FAIL: {path}: no '{start}' line (no instance)")
    return [l for l in ls[i + 1:] if not l.startswith("best run:")]

def same_structure(files):
    skel = [[NUM.sub("#", l) for l in b] for b in files]
    for k, (g, *refs) in enumerate(zip(*skel)):
        if any(r != g for r in refs): sys.exit(f"FAIL: the outputs differ in structure at line {k + 1}: {g[:120]}")
    if len({len(s) for s in skel}) != 1: sys.exit(f"FAIL: the outputs have different line counts {[len(s) for s in skel]}")

SUMS = {"logL": ("sumLogL", "site likelihoods"), "d1": ("sumFirstDerivs", "site first derivs"),
        "d2": ("sumSecondDerivs", "site second derivs")}

def families(lines):   # {family: [values]}; a '--sitelikes' line is one family, 'name = value' pairs are the rest
    fam = {}
    for l in lines:
        m = re.match(r"(site [a-z ]+) = (.*)", l)
        pairs = [(m.group(1), v) for v in NUM.findall(m.group(2))] if m else re.findall(rf"(\w+) = ({NUM.pattern})", l, re.I)
        for k, v in pairs: fam.setdefault(k, []).append(float(v))
    return fam

def synthetictest(gpu, sp, dp, double=False):
    same_structure([gpu, sp, dp])
    g, s, d = families(gpu), families(sp), families(dp)
    ok = True
    for f in g:
        if f.startswith("sum"): continue   # synthetictest's own double sums: the references for logL, d1 and d2
        if f in SUMS:   # kernelSumSites* reduce in float: against the double sum of the same GPU site values
            ref, what = g[SUMS[f][0]], SUMS[f][0]
            tol = 8 * (2**-53 if double else 2**-24) * 9 * sum(map(abs, g[SUMS[f][1]])) + 2e-5
        else:
            if not all(map(math.isfinite, s[f] + d[f])):   # a non-finite reference would make the tolerance infinite
                ok = False; print(f"  {f:18} a reference value is not finite: OVER"); continue
            dsp = max(abs(a - b) for a, b in zip(s[f], d[f]))
            ref, what = d[f], f"DP (|SP-DP| {dsp:.3g})"
            tol = (1e-9 * max(map(abs, d[f])) if double else max(10 * dsp, 1e-6 * max(map(abs, d[f])))) + 2e-5
        err = max(abs(a - b) if math.isfinite(a) else math.inf for a, b in zip(g[f], ref))
        ok &= err <= tol
        print(f"  {f:18} n={len(g[f]):<6} |GPU - {what}| {err:.3g}  tol {tol:.3g}  {'ok' if err <= tol else 'OVER'}")
    return ok

def hmctest(gpu, sp, ref):
    same_structure([gpu, sp, ref])
    ok, n = True, 0
    for lg, ls, lr in zip(gpu, sp, ref):
        for a, b in zip(map(float, NUM.findall(lg)), map(float, NUM.findall(lr if lg.startswith("now:") else ls))):
            n += 1
            if not abs(a - b) <= 1e-4 * abs(b) + 1e-6:
                ok = False; print(f"  over tolerance: gpu {a!r} reference {b!r} in: {lg[:100]}")
    print(f"  {n} numbers compared")
    return ok and n > 0

if __name__ == "__main__":
    double = "--double" in sys.argv
    prog, files = sys.argv[1], [a for a in sys.argv[2:] if a != "--double"][:3]
    ok = {"synthetictest": lambda: synthetictest(*[body(p, "Flags:") for p in files], double=double),
          "hmctest": lambda: hmctest(*[body(p, "Impl Desc") for p in files])}[prog]()
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
