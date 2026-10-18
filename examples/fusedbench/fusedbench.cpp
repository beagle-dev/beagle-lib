/*
 * Copyright 2026 Phylogenetic Likelihood Working Group
 * This file is part of BEAGLE.
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 *
 * fusedbench: benchmark and correctness harness for the fused updatePartials TEST implementation
 * (BEAGLE_FLAG_PARALLELOPS_FUSED; fused-handoff/PLAN-fused-uppartials.md sections 7 and 8). In one process and on
 * identical std::mt19937 data it creates the instances
 *   REF  CPU, double precision;
 *   A    the standard GPU implementation, default flags;
 *   A2   (4 states) A's flags with BEAGLE_FLAG_PARALLELOPS_STREAMS preferred: the per-operation path;
 *   B    A's flags with BEAGLE_FLAG_PARALLELOPS_FUSED required (--b-require none: a second standard instance).
 * It checks B against A (the partials of every internal buffer, site and root log likelihoods: bitwise, or at most
 * --ulp units in the last place) and every instance against REF (site and root log likelihoods), after poisoning
 * every internal partials buffer with NaN. Bench mode then times updatePartials (T_up: K back-to-back calls and one
 * root log likelihood, minus t_sync, per call) and whole evaluations (T_eval) in R blocks ordered A, A2, B, B, A2, A,
 * and writes one CSV row per timed instance. Ratios are meaningful only within one process (on the M1 Max,
 * successive OpenCL processes alternate between fast and about 14x slower completion).
 *
 * Usage:
 *   fusedbench [global options] [configuration options]       one configuration
 *   fusedbench [global options] --configs FILE                one configuration per line of FILE, in turn
 *   fusedbench [global options] --check [--quick] [--states LIST] [--categories LIST] [--stop-on-fail]
 *
 * Exit status: 0 every check passed; 1 a check failed; 2 usage; 77 B could not be created (bench mode), or B was
 * unavailable for every configuration (--check). The last stdout line is PASS or FAIL: <first failure>.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "libhmsbeagle/beagle.h"

namespace {

const char* const kUsage =
    "usage: fusedbench [global options] [configuration options]\n"
    "       fusedbench [global options] --configs FILE\n"
    "       fusedbench [global options] --check [--quick] [--states LIST] [--categories LIST] [--stop-on-fail]\n"
    "global:        --framework opencl|tinygpu|cpu --rsrc R --double --impls LIST (default REF,A,A2,B)\n"
    "               --reps R --k N --seed s --csv FILE --machine NAME --diag --b-require fused|none --ulp N\n"
    "               --mode fast|slow|any (bench: every GPU instance in this completion mode; any: B's)\n"
    "configuration: --states S --patterns P --categories C --taxa n --tree balanced|pectinate\n"
    "               --update full|branch-deep|branch-random --scaling none|read|write [--logscalers]\n"
    "               --tips states|partials|mixed\n";

[[noreturn]] void usage(const std::string& why) {
    fprintf(stderr, "fusedbench: %s\n%s", why.c_str(), kUsage);
    exit(2);
}

int toInt(const std::string& s) {
    char* end = nullptr;
    const long v = strtol(s.c_str(), &end, 10);
    if (s.empty() || *end != '\0') usage("not an integer: " + s);
    return (int) v;
}

std::vector<int> intList(const std::string& s) {
    std::vector<int> v;
    std::stringstream ss(s);
    for (std::string item; std::getline(ss, item, ',');) v.push_back(toInt(item));
    return v;
}

struct Config {
    int states = 4, patterns = 1000, categories = 4, taxa = 10;
    std::string tree = "balanced", update = "full", scaling = "none", tips = "states";
    bool logScalers = false;
};

std::string describe(const Config& c) {
    char b[256];
    snprintf(b, sizeof(b), "--states %d --patterns %d --categories %d --taxa %d --tree %s --update %s --scaling %s%s"
             " --tips %s", c.states, c.patterns, c.categories, c.taxa, c.tree.c_str(), c.update.c_str(),
             c.scaling.c_str(), c.logScalers ? " --logscalers" : "", c.tips.c_str());
    return b;
}

// one configuration option at args[i], advancing i past its value; false if args[i] is not one
bool configOption(Config& c, const std::vector<std::string>& args, size_t& i) {
    const std::string& a = args[i];
    auto value = [&]() -> std::string { if (i + 1 >= args.size()) usage("missing value after " + a); return args[++i]; };
    if (a == "--states") c.states = toInt(value());
    else if (a == "--patterns") c.patterns = toInt(value());
    else if (a == "--categories") c.categories = toInt(value());
    else if (a == "--taxa") c.taxa = toInt(value());
    else if (a == "--tree") c.tree = value();
    else if (a == "--update") c.update = value();
    else if (a == "--scaling") c.scaling = value();
    else if (a == "--tips") c.tips = value();
    else if (a == "--logscalers") c.logScalers = true;
    else return false;
    return true;
}

void validate(const Config& c) {
    auto in = [](const std::string& x, std::initializer_list<const char*> l) {
        for (const char* y : l) if (x == y) return true;
        return false;
    };
    if (c.states < 2 || c.patterns < 1 || c.categories < 1 || c.taxa < 3) usage("bad configuration: " + describe(c));
    if (!in(c.tree, {"balanced", "pectinate"})) usage("--tree balanced|pectinate (random only in --check)");
    if (!in(c.update, {"full", "branch-deep", "branch-random"})) usage("--update full|branch-deep|branch-random");
    if (!in(c.scaling, {"none", "read", "write"})) usage("--scaling none|read|write");
    if (!in(c.tips, {"states", "partials", "mixed"})) usage("--tips states|partials|mixed");
}

struct Options {
    std::string framework = "opencl", csv, configs, machine = "local", bRequire = "fused", mode = "any";
    std::string statesList, categoriesList, args;
    std::set<std::string> impls = {"REF", "A", "A2", "B"};
    int rsrc = -1, reps = 6, k = 0, seed = 1, ulp = 0;
    bool dbl = false, check = false, quick = false, stopOnFail = false, diag = false;
    Config cfg;
};

Options gOpt;
int gRsrc = -1;            // the GPU resource (0 for --framework cpu); -1: none found
long gCalls = 0, gBCalls = 0;
int gScriptError = 0;      // the first BEAGLE error of the instance script running now

void note(int rc) { if (rc < 0 && gScriptError == 0) gScriptError = rc; }

struct Status {
    int failures = 0, ran = 0, unavailable = 0;
    std::string first;
    bool stop = false;     // --stop-on-fail after a failure
    void fail(const std::string& what) {
        ++failures;
        if (first.empty()) first = what;
        printf("FAILED %s\n", what.c_str());
        fflush(stdout);
        if (gOpt.stopOnFail) stop = true;
    }
} gStatus;

int paddedStates(int s) {  // BeagleGPUImpl::createInstance's rule
    for (int p : {4, 16, 32, 48, 64, 80, 128, 192, 256}) if (s <= p) return p;
    return s + s % 16;
}

using Clock = std::chrono::steady_clock;

double us(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::micro>(b - a).count(); }

// ---- model, trees and data ----

// Symmetric eigen decomposition by cyclic Jacobi rotations: a = v diag(d) v^T (spectralbench)
void jacobi(std::vector<double>& a, int n, std::vector<double>& v, std::vector<double>& d) {
    v.assign(n * n, 0.0);
    for (int i = 0; i < n; ++i) v[i * n + i] = 1.0;
    for (int sweep = 0; sweep < 100; ++sweep) {
        double off = 0.0;
        for (int p = 0; p < n; ++p) for (int q = p + 1; q < n; ++q) off += a[p * n + q] * a[p * n + q];
        if (off < 1e-30) break;
        for (int p = 0; p < n; ++p) for (int q = p + 1; q < n; ++q) {
            if (std::fabs(a[p * n + q]) < 1e-300) continue;
            const double theta = (a[q * n + q] - a[p * n + p]) / (2.0 * a[p * n + q]);
            const double t = (theta >= 0 ? 1.0 : -1.0) / (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
            const double c = 1.0 / std::sqrt(t * t + 1.0), s = t * c;
            for (int k = 0; k < n; ++k) {
                const double akp = a[k * n + p], akq = a[k * n + q];
                a[k * n + p] = c * akp - s * akq; a[k * n + q] = s * akp + c * akq;
            }
            for (int k = 0; k < n; ++k) {
                const double apk = a[p * n + k], aqk = a[q * n + k];
                a[p * n + k] = c * apk - s * aqk; a[q * n + k] = s * apk + c * aqk;
            }
            for (int k = 0; k < n; ++k) {
                const double vkp = v[k * n + p], vkq = v[k * n + q];
                v[k * n + p] = c * vkp - s * vkq; v[k * n + q] = s * vkp + c * vkq;
            }
        }
    }
    d.resize(n);
    for (int i = 0; i < n; ++i) d[i] = a[i * n + i];
}

// Random reversible rate matrix Q = R diag(pi), one substitution per unit time (spectralbench)
void randomReversible(std::mt19937& rng, int n, std::vector<double>& pi, std::vector<double>& evec,
                      std::vector<double>& ivec, std::vector<double>& eval) {
    std::uniform_real_distribution<double> u(0.5, 1.5);
    pi.resize(n);
    double sum = 0.0;
    for (auto& p : pi) { p = u(rng); sum += p; }
    for (auto& p : pi) p /= sum;
    std::vector<double> r(n * n, 0.0), q(n * n, 0.0);
    for (int i = 0; i < n; ++i) for (int j = i + 1; j < n; ++j) r[i * n + j] = r[j * n + i] = u(rng);
    double rate = 0.0;
    for (int i = 0; i < n; ++i) {
        double row = 0.0;
        for (int j = 0; j < n; ++j) if (i != j) { q[i * n + j] = r[i * n + j] * pi[j]; row += q[i * n + j]; }
        q[i * n + i] = -row;
        rate += pi[i] * row;
    }
    for (auto& x : q) x /= rate;
    std::vector<double> b(n * n), uvec, d;
    for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) b[i * n + j] = std::sqrt(pi[i]) * q[i * n + j] / std::sqrt(pi[j]);
    jacobi(b, n, uvec, d);
    evec.resize(n * n); ivec.resize(n * n);
    for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) {
        evec[i * n + j] = uvec[i * n + j] / std::sqrt(pi[i]);
        ivec[i * n + j] = uvec[j * n + i] * std::sqrt(pi[j]);
    }
    eval = d;
}

struct Op { int dest, c1, m1, c2, m2; };   // children and their transition matrices; op i writes scale buffer i

// synthetictest's default layout: op i = (2i, 2i+1) -> n+i, root 2n-2
std::vector<Op> balancedOps(int n) {
    std::vector<Op> ops;
    for (int i = 0; i < n - 1; ++i) ops.push_back({n + i, 2 * i, 2 * i, 2 * i + 1, 2 * i + 1});
    return ops;
}

// op 0 = (0, 1) -> n; op k = (n+k-1, k+1) -> n+k
std::vector<Op> pectinateOps(int n) {
    std::vector<Op> ops = {{n, 0, 0, 1, 1}};
    for (int k = 1; k < n - 1; ++k) ops.push_back({n + k, n + k - 1, n + k - 1, k + 1, k + 1});
    return ops;
}

// spectralbench's random tree: two random active nodes joined at each step, the root last
std::vector<Op> randomOps(std::mt19937& rng, int n) {
    std::vector<int> active(n);
    std::iota(active.begin(), active.end(), 0);
    std::vector<Op> ops;
    for (int next = n; next < 2 * n - 1; ++next) {
        int j[2];
        for (int& x : j) {
            const int a = std::uniform_int_distribution<int>(0, (int) active.size() - 1)(rng);
            x = active[a];
            active.erase(active.begin() + a);
        }
        ops.push_back({next, j[0], j[0], j[1], j[1]});
        active.push_back(next);
    }
    return ops;
}

// stress (k): ops alternating states-states (a cherry), partials-partials (chain + cherry) and states-partials
// (chain + tip, the partials child first); n - 2 a multiple of 3 (1001 tips: 1000 ops)
std::vector<Op> chainOps(int n) {
    std::vector<Op> ops = {{n, 0, 0, 1, 1}};
    int chain = n;
    for (int t = 2; t + 2 < n; t += 3) {
        const int d = n + (int) ops.size();
        ops.push_back({d, t, t, t + 1, t + 1});
        ops.push_back({d + 1, chain, chain, d, d});
        ops.push_back({d + 2, d + 1, d + 1, t + 2, t + 2});
        chain = d + 2;
    }
    return ops;
}

// stress (h), n = 5: buffer 5 is read by op 1 and overwritten by op 2 of the same call; matrix 7 is its second edge
std::vector<Op> reuseOps() { return {{5, 0, 0, 1, 1}, {6, 5, 5, 2, 2}, {5, 3, 3, 4, 4}, {8, 6, 6, 5, 7}}; }

struct Problem {
    Config cfg;
    int n = 0;
    std::vector<Op> ops;                     // the full post-order list
    std::vector<int> parent;                 // per node, -1 at the root (not for the reuse tree)
    std::vector<double> length;              // per transition matrix (the edge above node i)
    std::vector<std::vector<int>> states;    // per tip; stateCount codes a gap
    std::vector<char> tipStates;             // tip t set by beagleSetTipStates, else by one-hot partials
    std::vector<double> pi, evec, ivec, eval, rates, catWeights, patternWeights;
    int nodes() const { return 2 * n - 1; }
    int root() const { return ops.back().dest; }
    int cum() const { return n - 1; }        // n scale buffers: op i writes i, the cumulative buffer is n-1
    int stateTips() const { return (int) std::count(tipStates.begin(), tipStates.end(), 1); }
    std::vector<int> internal() const {      // destinations, each once, in list order
        std::vector<int> v;
        for (const Op& o : ops) if (std::find(v.begin(), v.end(), o.dest) == v.end()) v.push_back(o.dest);
        return v;
    }
};

Problem makeProblem(const Config& c, int seed) {
    Problem p;
    p.cfg = c;
    p.n = (c.tree == "reuse") ? 5 : c.taxa;
    std::mt19937 rng(seed);
    if (c.tree == "balanced") p.ops = balancedOps(p.n);
    else if (c.tree == "pectinate") p.ops = pectinateOps(p.n);
    else if (c.tree == "random") p.ops = randomOps(rng, p.n);
    else if (c.tree == "chain") p.ops = chainOps(p.n);
    else p.ops = reuseOps();
    p.parent.assign(p.nodes(), -1);
    if (c.tree != "reuse") for (const Op& o : p.ops) p.parent[o.c1] = p.parent[o.c2] = o.dest;
    std::uniform_real_distribution<double> len(0.05, 0.3);
    p.length.resize(p.nodes() - 1);
    for (double& l : p.length) l = len(rng);
    const int S = c.states, P = c.patterns, C = c.categories;
    randomReversible(rng, S, p.pi, p.evec, p.ivec, p.eval);
    for (int k = 0; k < C; ++k) p.rates.push_back(C == 1 ? 1.0 : 0.25 + 1.5 * k / (C - 1));   // mean 1
    p.catWeights.assign(C, 1.0 / C);
    p.patternWeights.assign(P, 1.0);
    std::uniform_int_distribution<int> state(0, S - 1);
    std::bernoulli_distribution gap(0.05);
    p.states.assign(p.n, std::vector<int>(P));
    for (auto& tip : p.states) for (int& s : tip) { s = state(rng); if (gap(rng)) s = S; }
    p.tipStates.resize(p.n);
    for (int t = 0; t < p.n; ++t) p.tipStates[t] = (c.tips == "states") || (c.tips == "mixed" && t % 3 != 0);
    return p;
}

// the ops whose destination is a proper ancestor of node v, in list order
std::vector<int> pathOps(const Problem& p, int v) {
    std::vector<char> ancestor(p.nodes(), 0);
    for (int u = p.parent[v]; u >= 0; u = p.parent[u]) ancestor[u] = 1;
    std::vector<int> path;
    for (int i = 0; i < (int) p.ops.size(); ++i) if (ancestor[p.ops[i].dest]) path.push_back(i);
    return path;
}

int deepestTip(const Problem& p) {
    int best = 0;
    size_t depth = 0;
    for (int t = 0; t < p.n; ++t) {
        const size_t d = pathOps(p, t).size();
        if (d > depth) { depth = d; best = t; }
    }
    return best;
}

// branch-random: one sequence of branches (non-root nodes) per configuration, from the seed alone, replayed
// identically for every instance and block
std::vector<int> branchSequence(int n, int seed) {
    std::mt19937 rng(seed + 7919);
    std::uniform_int_distribution<int> node(0, 2 * n - 3);
    std::vector<int> seq(64);
    for (int& v : seq) v = node(rng);
    return seq;
}

double altLength(double l) { return 1.5 * l; }

// built-in checks of the path and sequence rules (every mode)
void selfTest() {
    auto deep = [](const char* tree, int n) {
        Config c;
        c.tree = tree; c.taxa = n; c.states = 4; c.patterns = 1;
        const Problem p = makeProblem(c, 1);
        return pathOps(p, deepestTip(p)).size();
    };
    const bool ok = deep("balanced", 10) == 4 && deep("balanced", 1000) == 10 && deep("pectinate", 10) == 9 &&
                    deep("pectinate", 1000) == 999 && branchSequence(1000, 5) == branchSequence(1000, 5) &&
                    chainOps(1001).size() == 1000;
    if (!ok) { printf("FAIL: fusedbench self-test (tree paths or branch sequence)\n"); exit(1); }
}

// ---- instances ----

struct Inst {
    std::string name;      // REF, A, A2, B (or a stress case's own)
    int id = -1;
    std::string implName;
    long flags = 0;
    int partialsCount = 0;
};

long frameworkFlag() {
    if (gOpt.framework == "tinygpu") return BEAGLE_FLAG_FRAMEWORK_TINYGPU;
    if (gOpt.framework == "cpu") return BEAGLE_FLAG_FRAMEWORK_CPU;
    return BEAGLE_FLAG_FRAMEWORK_OPENCL;
}

void flagsFor(const std::string& name, bool logScalers, int* resource, long* pref, long* req) {
    *pref = BEAGLE_FLAG_SCALING_MANUAL | (logScalers ? BEAGLE_FLAG_SCALERS_LOG : BEAGLE_FLAG_SCALERS_RAW);
    if (name == "REF") {
        *resource = 0;
        *req = BEAGLE_FLAG_FRAMEWORK_CPU | BEAGLE_FLAG_PROCESSOR_CPU | BEAGLE_FLAG_PRECISION_DOUBLE;
        return;
    }
    *resource = gRsrc;
    *req = frameworkFlag() | (gOpt.dbl ? BEAGLE_FLAG_PRECISION_DOUBLE : BEAGLE_FLAG_PRECISION_SINGLE);
    if (name == "A2") *pref |= BEAGLE_FLAG_PARALLELOPS_STREAMS;   // preferred: TinyGPU's resource lacks STREAMS
    if (name == "B" && gOpt.bRequire == "fused") *req |= BEAGLE_FLAG_PARALLELOPS_FUSED;
}

void setMatrices(int id, const std::vector<int>& idx, const std::vector<double>& len) {
    note(beagleUpdateTransitionMatrices(id, 0, idx.data(), NULL, NULL, len.data(), (int) idx.size()));
}

void load(const Problem& p, int id) {
    const int S = p.cfg.states, P = p.cfg.patterns;
    beagleSetEigenDecomposition(id, 0, p.evec.data(), p.ivec.data(), p.eval.data());
    beagleSetStateFrequencies(id, 0, p.pi.data());
    beagleSetCategoryRates(id, p.rates.data());
    beagleSetCategoryWeights(id, 0, p.catWeights.data());
    beagleSetPatternWeights(id, p.patternWeights.data());
    std::vector<double> partials((size_t) S * P);
    for (int t = 0; t < p.n; ++t) {
        if (p.tipStates[t]) { beagleSetTipStates(id, t, p.states[t].data()); continue; }
        for (int i = 0; i < P; ++i) for (int s = 0; s < S; ++s)
            partials[(size_t) i * S + s] = (p.states[t][i] == S || p.states[t][i] == s) ? 1.0 : 0.0;
        beagleSetTipPartials(id, t, partials.data());
    }
    std::vector<int> all(p.length.size());
    std::iota(all.begin(), all.end(), 0);
    setMatrices(id, all, p.length);
}

// whether instances get partialsBufferCount 2n-1: BeagleGPUImpl::setPartials refuses indices >= partialsBufferCount
// (the CPU checks partials + compact buffers), so poisoning internal buffers needs it. That is one extra buffer per
// compact tip; not when it adds more than 512 MB (codon at n = 1000 and P >= 500)
bool poisonable(const Problem& p) {
    const double extra = (double) p.stateTips() * p.cfg.patterns * paddedStates(p.cfg.states) * p.cfg.categories;
    return extra * (gOpt.dbl ? 8 : 4) <= 512.0 * (1 << 20);
}

// an instance of p with the given flags, loaded; its number, or a negative BEAGLE error code
int createRaw(const Problem& p, int resource, long pref, long req, Inst& in) {
    if (resource < 0) return BEAGLE_ERROR_NO_RESOURCE;
    const int st = p.stateTips();
    in.partialsCount = poisonable(p) ? p.nodes() : (p.n - st) + (p.nodes() - p.n);
    BeagleInstanceDetails d;
    const int id = beagleCreateInstance(p.n, in.partialsCount, st, p.cfg.states, p.cfg.patterns, 1, p.nodes() - 1,
                                        p.cfg.categories, p.n, &resource, 1, pref, req, &d);
    if (id < 0) return id;
    in.id = id;
    in.implName = d.implName ? d.implName : "";
    in.flags = d.flags;
    load(p, id);
    return id;
}

int create(const Problem& p, Inst& in) {
    int resource;
    long pref, req;
    flagsFor(in.name, p.cfg.logScalers, &resource, &pref, &req);
    return createRaw(p, resource, pref, req, in);
}

bool endsWith(const std::string& s, const std::string& t) {
    return s.size() >= t.size() && s.compare(s.size() - t.size(), t.size(), t) == 0;
}

struct Set {               // one configuration's instances, finalized with it
    std::vector<Inst> insts;
    bool bUnavailable = false;
    std::string error;
    Set() = default;
    Set(const Set&) = delete;
    Set& operator=(const Set&) = delete;
    ~Set() { for (const Inst& i : insts) beagleFinalizeInstance(i.id); }
    const Inst* find(const char* name) const {
        for (const Inst& i : insts) if (i.name == name) return &i;
        return nullptr;
    }
};

// creates names (B first: without it the configuration is skipped) and asserts their implementations
void createSet(Set& s, const Problem& p, const std::vector<std::string>& names) {
    std::vector<std::string> order = names;
    std::stable_partition(order.begin(), order.end(), [](const std::string& n) { return n == "B"; });
    for (const std::string& name : order) {
        Inst in;
        in.name = name;
        const int rc = create(p, in);
        if (rc < 0) {
            if (name == "B") s.bUnavailable = true;
            else s.error = name + " could not be created (error " + std::to_string(rc) + ")";
            return;
        }
        s.insts.push_back(in);
    }
    std::sort(s.insts.begin(), s.insts.end(), [&](const Inst& x, const Inst& y) {
        return std::find(names.begin(), names.end(), x.name) < std::find(names.begin(), names.end(), y.name);
    });
    const Inst* a = s.find("A");
    const Inst* a2 = s.find("A2");
    const Inst* b = s.find("B");
    const bool fused = gOpt.bRequire == "fused";
    if (a && (endsWith(a->implName, "-Fused") || (a->flags & BEAGLE_FLAG_PARALLELOPS_STREAMS)))
        s.error = "A is " + a->implName + " with flags " + std::to_string(a->flags) + ": not the standard default path";
    else if (a2 && (!(a2->flags & BEAGLE_FLAG_PARALLELOPS_STREAMS) || (a && a2->implName != a->implName)))
        s.error = "A2 is " + a2->implName + " without BEAGLE_FLAG_PARALLELOPS_STREAMS or not A's implementation";
    else if (b && fused && (!endsWith(b->implName, "-Fused") || !(b->flags & BEAGLE_FLAG_PARALLELOPS_FUSED)))
        s.error = "B is " + b->implName + ": not a fused implementation";
    else if (b && !fused && (endsWith(b->implName, "-Fused") || (a && b->implName != a->implName)))
        s.error = "B is " + b->implName + ": --b-require none wants A's implementation";
}

std::vector<std::string> namesFor(const Config& c) {
    std::vector<std::string> v;
    for (const char* n : {"REF", "A", "A2", "B"}) {
        if (!gOpt.impls.count(n)) continue;
        if (!strcmp(n, "A2") && (c.states > 4 || gOpt.framework == "cpu")) continue;
        if (!strcmp(n, "REF") && c.states > 4 && c.taxa >= 1000 && c.patterns >= 250) continue;   // CPU cost
        v.push_back(n);
    }
    return v;
}

// ---- calls ----

// one BeagleOperation per op of which; pattern[j % size] gives op j's scaling: n none, r read, w write (its own
// scale buffer, the op's index in the full list)
std::vector<int> encode(const std::vector<Op>& ops, const std::vector<int>& which, const std::string& pattern) {
    std::vector<int> a;
    for (size_t j = 0; j < which.size(); ++j) {
        const Op& o = ops[which[j]];
        const char m = pattern[j % pattern.size()];
        a.insert(a.end(), {o.dest, m == 'w' ? which[j] : BEAGLE_OP_NONE, m == 'r' ? which[j] : BEAGLE_OP_NONE,
                           o.c1, o.m1, o.c2, o.m2});
    }
    return a;
}

std::vector<int> written(const std::vector<int>& which, const std::string& pattern) {
    std::vector<int> w;
    for (size_t j = 0; j < which.size(); ++j) if (pattern[j % pattern.size()] == 'w') w.push_back(which[j]);
    return w;
}

std::vector<int> allOps(const Problem& p) {
    std::vector<int> v(p.ops.size());
    std::iota(v.begin(), v.end(), 0);
    return v;
}

int update(const Inst& in, const int* ops, int count, int cum) {
    if (gOpt.diag) fprintf(stderr, "#call %ld begin %s\n", gCalls, in.name.c_str());
    const int rc = beagleUpdatePartials(in.id, reinterpret_cast<const BeagleOperation*>(ops), count, cum);
    if (gOpt.diag) fprintf(stderr, "#call %ld end %s\n", gCalls, in.name.c_str());
    ++gCalls;
    if (in.name == "B") ++gBCalls;
    return rc;
}

void update(const Inst& in, const std::vector<int>& a, int cum) { note(update(in, a.data(), (int) a.size() / 7, cum)); }

void reset(const Inst& in, int cum) { note(beagleResetScaleFactors(in.id, cum)); }

void accumulate(const Inst& in, const std::vector<int>& idx, int cum) {
    note(beagleAccumulateScaleFactors(in.id, idx.data(), (int) idx.size(), cum));
}

void removeFactors(const Inst& in, const std::vector<int>& idx, int cum) {
    note(beagleRemoveScaleFactors(in.id, idx.data(), (int) idx.size(), cum));
}

double rootLogL(const Inst& in, const Problem& p, int cum) {
    int root = p.root(), w = 0, f = 0;
    double l = NAN;
    beagleCalculateRootLogLikelihoods(in.id, &root, &w, &f, &cum, 1, &l);   // non-finite: compared, not an error
    return l;
}

// the minimum of 20 root log likelihood calls on an idle queue
double syncTime(const Inst& in, const Problem& p, int cum) {
    rootLogL(in, p, cum);
    double best = 1e300;
    for (int r = 0; r < 20; ++r) {
        const auto t0 = Clock::now();
        rootLogL(in, p, cum);
        best = std::min(best, us(t0, Clock::now()));
    }
    return best;
}

// Apple's OpenCL on the M1 Max fixes an instance's completion latency when its context is created, alternating
// system-wide between fast (t_sync < 1 ms) and slow (5-8 ms) with each context (so successive single-instance
// processes alternate, and instances created one after another alternate within a process). Each GPU instance is
// recreated, a new context, until its mode is the target (any: B's, the first created); after 4 attempts it
// keeps what it has.
void matchModes(Set& s, const Problem& p, std::string target) {
    for (const char* name : {"B", "A", "A2"}) {
        for (size_t i = 0; i < s.insts.size(); ++i) {
            Inst& in = s.insts[i];
            if (in.name != name) continue;
            for (int attempt = 0; ; ++attempt) {
                const std::string mode = syncTime(in, p, BEAGLE_OP_NONE) < 1000.0 ? "fast" : "slow";
                if (target == "any") target = mode;
                if (mode == target) break;
                if (attempt == 4) { printf("  (%s stays in %s mode after 4 recreations)\n", in.name.c_str(), mode.c_str()); break; }
                Inst fresh;   // created before the old one is finalized, which would hand its context's mode back
                fresh.name = in.name;
                if (create(p, fresh) < 0) { s.error = in.name + " could not be recreated"; return; }
                beagleFinalizeInstance(in.id);
                in = fresh;
            }
        }
    }
}

// createSet, then matchModes unless B is unavailable; target any (bench mode's default) means B's mode
void openSet(Set& s, const Problem& p, const std::vector<std::string>& names, const std::string& target) {
    createSet(s, p, names);
    if (!s.bUnavailable && s.error.empty()) matchModes(s, p, target);
}

// --check's mode: fast unless --mode says otherwise (checks sync hundreds of times per configuration)
std::string checkMode() { return gOpt.mode == "any" ? "fast" : gOpt.mode; }

// a full evaluation's ops with scaling s: none; read (after a write warm-up); write (factors accumulated by
// beagleAccumulateScaleFactors); writecum (accumulated by updatePartials); writeread (ops alternately write and read,
// after a write warm-up)
void evalFull(const Inst& in, const Problem& p, const std::string& s) {
    const std::vector<int> all = allOps(p);
    const int cum = p.cum();
    if (s == "none") { update(in, encode(p.ops, all, "n"), BEAGLE_OP_NONE); return; }
    reset(in, cum);
    if (s == "writecum") { update(in, encode(p.ops, all, "w"), cum); return; }
    update(in, encode(p.ops, all, "w"), BEAGLE_OP_NONE);
    accumulate(in, all, cum);
    if (s == "read") update(in, encode(p.ops, all, "r"), BEAGLE_OP_NONE);
    else if (s == "writeread") { removeFactors(in, written(all, "wr"), cum); update(in, encode(p.ops, all, "wr"), cum); }
}

// after evalFull: matrix v gets length len, then the ops above it with scaling s
void evalBranch(const Inst& in, const Problem& p, int v, double len, const std::string& s) {
    setMatrices(in.id, {v}, {len});
    const std::vector<int> path = pathOps(p, v);
    const int cum = p.cum();
    if (s == "none" || s == "read") { update(in, encode(p.ops, path, s == "none" ? "n" : "r"), BEAGLE_OP_NONE); return; }
    const std::string pattern = (s == "writeread") ? "wr" : "w";
    removeFactors(in, written(path, pattern), cum);
    if (s == "write") { update(in, encode(p.ops, path, "w"), BEAGLE_OP_NONE); accumulate(in, path, cum); }
    else update(in, encode(p.ops, path, pattern), cum);
}

int rootCum(const Problem& p, const std::string& s) { return s == "none" ? BEAGLE_OP_NONE : p.cum(); }

// ---- checks ----

void poison(const Inst& in, const Problem& p) {
    if (!poisonable(p)) return;
    const std::vector<double> nan((size_t) p.cfg.patterns * p.cfg.states * p.cfg.categories, NAN);
    for (int b : p.internal()) note(beagleSetPartials(in.id, b, nan.data()));
}

struct Outcome { double logL = NAN; std::vector<double> site; };

Outcome collect(const Inst& in, const Problem& p, int cum) {
    Outcome o;
    o.logL = rootLogL(in, p, cum);
    o.site.resize(p.cfg.patterns);
    beagleGetSiteLogLikelihoods(in.id, o.site.data());
    return o;
}

// distance in units in the last place of the GPU precision; 0 when bitwise equal (NaN equals NaN)
uint64_t ulps(double x, double y) {
    if (std::isnan(x) || std::isnan(y)) return (std::isnan(x) && std::isnan(y)) ? 0 : UINT64_MAX;
    int64_t i, j;
    if (gOpt.dbl) {
        memcpy(&i, &x, 8); memcpy(&j, &y, 8);
        if (i < 0) i = INT64_MIN - i;
        if (j < 0) j = INT64_MIN - j;
        return (i > j) ? (uint64_t) i - (uint64_t) j : (uint64_t) j - (uint64_t) i;
    }
    const float fx = (float) x, fy = (float) y;
    int32_t a, b;
    memcpy(&a, &fx, 4); memcpy(&b, &fy, 4);
    i = (a < 0) ? (int64_t) INT32_MIN - a : a;
    j = (b < 0) ? (int64_t) INT32_MIN - b : b;
    return (uint64_t) std::llabs(i - j);
}

uint64_t maxUlps(const std::vector<double>& x, const std::vector<double>& y) {
    uint64_t m = 0;
    for (size_t i = 0; i < x.size(); ++i) m = std::max(m, ulps(x[i], y[i]));
    return m;
}

// B against A on every internal buffer (blocking reads: never inside a timed block)
uint64_t partialsUlps(const Inst& a, const Inst& b, const Problem& p) {
    std::vector<double> x((size_t) p.cfg.patterns * p.cfg.states * p.cfg.categories), y(x.size());
    uint64_t m = 0;
    for (int buf : p.internal()) {
        beagleGetPartials(a.id, buf, BEAGLE_OP_NONE, x.data());
        beagleGetPartials(b.id, buf, BEAGLE_OP_NONE, y.data());
        m = std::max(m, maxUlps(x, y));
    }
    return m;
}

// within tinygputest's dpLogLOk (double precision: 1e-9 relative) or, in single precision,
// 1e-5 * sum_i w_i |siteLogL_i| + 1e-3 (scale = that sum); non-finite values must agree
bool nearRef(double x, double r, double scale) {
    if (!std::isfinite(x) || !std::isfinite(r)) return (std::isnan(x) && std::isnan(r)) || x == r;
    const double tol = gOpt.dbl ? 1e-9 * std::max(1.0, std::fabs(r)) : 1e-5 * scale + 1e-3;
    return std::fabs(x - r) <= tol;
}

std::string vsRef(const Outcome& x, const Outcome& r) {
    double scale = 0.0;
    for (size_t i = 0; i < r.site.size(); ++i) scale += std::fabs(r.site[i]);   // pattern weights are 1
    if (!nearRef(x.logL, r.logL, scale)) {
        char b[160];
        snprintf(b, sizeof(b), "logL %.10g, REF %.10g", x.logL, r.logL);
        return b;
    }
    for (size_t i = 0; i < r.site.size(); ++i) {
        if (!nearRef(x.site[i], r.site[i], std::fabs(r.site[i]))) {
            char b[160];
            snprintf(b, sizeof(b), "site %zu logL %.10g, REF %.10g", i, x.site[i], r.site[i]);
            return b;
        }
    }
    return "";
}

struct Verdict { bool ok = true; double dRef = NAN, logL = NAN, logLRef = NAN; uint64_t ulp = 0; };

std::string ulpText(uint64_t u) { return u == UINT64_MAX ? "NaN mismatch" : std::to_string(u) + " ulp"; }

// poisons every instance's internal partials with NaN, runs script on each, and compares B with A (partials of every
// internal buffer, site and root log likelihoods) and every instance with REF (refUsable: not when reads divide by
// stored log factors, section 2.1 D11); "" or the first problem. outs and verdicts are per instance.
std::string subCheck(const Set& s, const Problem& p, const std::function<void(const Inst&)>& script, int cum,
                     bool refUsable, std::vector<Outcome>* outs = nullptr, std::vector<Verdict>* verdicts = nullptr) {
    std::vector<Outcome> o(s.insts.size());
    for (size_t i = 0; i < s.insts.size(); ++i) {
        gScriptError = 0;
        poison(s.insts[i], p);
        script(s.insts[i]);
        if (gScriptError) return s.insts[i].name + ": BEAGLE error " + std::to_string(gScriptError);
        o[i] = collect(s.insts[i], p, cum);
    }
    std::vector<Verdict> v(s.insts.size());
    std::string why;
    size_t ref = s.insts.size(), a = ref;
    for (size_t i = 0; i < s.insts.size(); ++i) {
        if (s.insts[i].name == "REF") ref = i;
        if (s.insts[i].name == "A") a = i;
    }
    for (size_t i = 0; i < s.insts.size(); ++i) {
        const Inst& in = s.insts[i];
        v[i].logL = o[i].logL;
        if (in.name == "REF") continue;
        if (ref < s.insts.size() && refUsable) {
            v[i].logLRef = o[ref].logL;
            v[i].dRef = o[i].logL - o[ref].logL;
            const std::string w = vsRef(o[i], o[ref]);
            if (!w.empty()) { v[i].ok = false; if (why.empty()) why = in.name + " vs REF: " + w; }
        }
        if (in.name == "B" && a < s.insts.size()) {
            v[i].ulp = std::max({ulps(o[i].logL, o[a].logL), maxUlps(o[i].site, o[a].site),
                                 partialsUlps(s.insts[a], in, p)});
            if (v[i].ulp > (uint64_t) gOpt.ulp) { v[i].ok = false; if (why.empty()) why = "B vs A: " + ulpText(v[i].ulp); }
        }
    }
    if (outs) *outs = o;
    if (verdicts) *verdicts = v;
    return why;
}

// a stress case on fresh instances (REF, A, B): script, then compare, then post (optional, "" or a problem); false if
// B is unavailable
bool scripted(const std::string& label, const Problem& p, const std::function<void(const Inst&)>& script, int cum, std::vector<Outcome>* outs = nullptr,
              const std::function<std::string(const Set&)>& post = nullptr) {
    const auto t0 = Clock::now();
    std::vector<std::string> names;
    for (const char* n : {"REF", "A", "B"}) if (gOpt.impls.count(n)) names.push_back(n);
    Set s;
    openSet(s, p, names, checkMode());
    if (s.bUnavailable) { printf("%s: B unavailable, skipped\n", label.c_str()); ++gStatus.unavailable; return false; }
    ++gStatus.ran;
    if (!s.error.empty()) { gStatus.fail(label + ": " + s.error); return true; }
    std::string why = subCheck(s, p, script, cum, true, outs);
    if (why.empty() && post) why = post(s);
    if (why.empty()) printf("%s: ok (%.1f s)\n", label.c_str(), us(t0, Clock::now()) * 1e-6);
    else gStatus.fail(label + ": " + why);
    fflush(stdout);
    return true;
}

// ---- --check ----

// PLAN section 8 item 2: S x C x P in full, with tree, taxa, tips and scalers rotating; the 4-state configurations at
// C in {2, 4, 8} first (the first 3-D thread blocks on NV)
std::vector<Config> checkMatrix() {
    const bool q = gOpt.quick;
    const std::vector<int> states = gOpt.statesList.empty() ? (q ? std::vector<int>{4, 61} : std::vector<int>{4, 61, 40, 20, 16})
                                                           : intList(gOpt.statesList);
    const std::vector<int> cats = intList(gOpt.categoriesList);
    auto categories = [&](int S) {
        if (!cats.empty()) return cats;
        if (S == 4) return q ? std::vector<int>{2, 4, 8, 1} : std::vector<int>{2, 4, 8, 1, 3};
        return q ? std::vector<int>{1, 4} : std::vector<int>{1, 2, 3, 4};
    };
    auto patterns = [&](int S) {
        if (S == 4) return q ? std::vector<int>{3, 65} : std::vector<int>{1, 3, 63, 64, 65, 1001};
        return q ? std::vector<int>{5, 9} : std::vector<int>{1, 3, 4, 5, 7, 9, 1001};
    };
    const std::vector<int> taxa = q ? std::vector<int>{3, 5, 17, 64} : std::vector<int>{3, 4, 5, 17, 64};
    const char* trees[] = {"balanced", "pectinate", "random"};
    const char* tips[] = {"states", "partials", "mixed"};
    std::vector<Config> m;
    int k = 0;
    auto add = [&](int S, int C) {
        for (int P : patterns(S)) {
            Config c;
            c.states = S; c.categories = C; c.patterns = P;
            c.tree = trees[k % 3];
            c.taxa = taxa[k % taxa.size()];
            c.tips = tips[(k / 3) % 3];
            c.logScalers = (k % 7 == 3);
            m.push_back(c);
            ++k;
        }
    };
    auto first = [](int S, int C) { return S == 4 && (C == 2 || C == 4 || C == 8); };
    for (int S : states) for (int C : categories(S)) if (first(S, C)) add(S, C);
    for (int S : states) for (int C : categories(S)) if (!first(S, C)) add(S, C);
    return m;
}

void checkConfig(const Config& c, int index) {
    const Problem p = makeProblem(c, gOpt.seed);
    char d[160];   // every scaling mode and update runs on these instances
    snprintf(d, sizeof(d), "S %d, P %d, C %d, n %d, %s tree, %s tips%s", c.states, c.patterns, c.categories, c.taxa,
             c.tree.c_str(), c.tips.c_str(), c.logScalers ? ", log scalers" : "");
    const std::string label = "check " + std::to_string(index) + " (" + d + ")";
    const auto t0 = Clock::now();
    Set s;
    openSet(s, p, namesFor(c), checkMode());
    if (s.bUnavailable) { printf("%s: B unavailable, skipped\n", label.c_str()); ++gStatus.unavailable; return; }
    ++gStatus.ran;
    if (!s.error.empty()) { gStatus.fail(label + ": " + s.error); return; }
    const int deep = deepestTip(p), random = branchSequence(p.n, gOpt.seed + index)[0];
    int passed = 0, total = 0;
    for (const char* sc : {"none", "read", "write", "writecum", "writeread"}) {
        if (!strcmp(sc, "none") && !gOpt.dbl && c.taxa > 17) continue;    // single precision underflows
        for (const char* up : {"full", "branch-deep", "branch-random"}) {
            const int v = strcmp(up, "branch-deep") ? random : deep;
            auto script = [&](const Inst& in) {
                evalFull(in, p, sc);
                if (strcmp(up, "full")) evalBranch(in, p, v, altLength(p.length[v]), sc);
            };
            const bool refUsable = !(c.logScalers && (!strcmp(sc, "read") || !strcmp(sc, "writeread")));
            const std::string why = subCheck(s, p, script, rootCum(p, sc), refUsable);
            ++total;
            if (why.empty()) ++passed;
            else { gStatus.fail(label + " " + sc + " " + up + ": " + why); if (gStatus.stop) return; }
        }
    }
    printf("%s: %d/%d ok (%.1f s)\n", label.c_str(), passed, total, us(t0, Clock::now()) * 1e-6);
    fflush(stdout);
}

Config stressConfig(int S, int C, const char* tree, int taxa, int P4, int PX) {
    Config c;
    c.states = S; c.categories = C; c.tree = tree; c.taxa = taxa; c.patterns = (S == 4) ? P4 : PX;
    return c;
}

// PLAN section 8 item 3, cases (a)-(i) and (k) at one state and category count
void stressCases(int S, int C) {
    const std::string at = " (S " + std::to_string(S) + ", C " + std::to_string(C) + ")";
    const int seed = gOpt.seed;
    auto stop = [] { return gStatus.stop; };
    {   // (a) 64 back-to-back calls alternating balanced and pectinate lists over the same buffers; then 600 1000-op
        // calls (an arena wrap) reading the factors of a write warm-up, so that the 4-state baseline uses multigrid;
        // REF computes only the last list. --quick: 8 alternating calls, and the 600 calls only at 4 states
        const Problem p = makeProblem(stressConfig(S, C, "balanced", 1001, 65, 5), seed);
        const std::vector<int> all = allOps(p);
        const std::vector<int> bal = encode(p.ops, all, "w"), pec = encode(pectinateOps(p.n), all, "w");
        const std::vector<int> read = encode(p.ops, all, "r");
        const int cum = p.cum(), alternating = gOpt.quick ? 8 : 64;
        scripted("stress (a) " + std::to_string(alternating) + " alternating calls" + at, p, [&](const Inst& in) {
            for (int k = (in.name == "REF") ? alternating - 1 : 0; k < alternating; ++k)
                update(in, (k % 2) ? pec : bal, BEAGLE_OP_NONE);
            reset(in, cum);
            accumulate(in, all, cum);
        }, cum);
        if (stop()) return;
        if (!gOpt.quick || S == 4) {
            scripted("stress (a) 600 calls of 1000 ops" + at, p, [&](const Inst& in) {
                reset(in, cum);
                update(in, bal, BEAGLE_OP_NONE);
                accumulate(in, all, cum);
                for (int k = (in.name == "REF") ? 599 : 0; k < 600; ++k) update(in, read, BEAGLE_OP_NONE);
            }, cum);
            if (stop()) return;
        }
    }
    {   // (b) op 0 writes scale buffer 0; the last op of the same call reads it
        const Problem p = makeProblem(stressConfig(S, C, "balanced", 17, 65, 9), seed);
        const std::vector<int> all = allOps(p);
        std::vector<int> a = encode(p.ops, all, "w");
        const size_t last = all.size() - 1;
        a[last * 7 + 1] = BEAGLE_OP_NONE;
        a[last * 7 + 2] = 0;
        std::vector<int> acc(all.begin(), all.end() - 1);
        acc.push_back(0);
        const int cum = p.cum();
        scripted("stress (b) same-call write then read" + at, p, [&](const Inst& in) {
            reset(in, cum);
            update(in, a, BEAGLE_OP_NONE);
            accumulate(in, acc, cum);
        }, cum);
        if (stop()) return;
    }
    {   // (c) an op list longer than the capacity max(64, partials + compact buffers): the full list repeated
        const Problem p = makeProblem(stressConfig(S, C, "balanced", 64, 65, 9), seed);
        const std::vector<int> all = allOps(p);
        const int cap = std::max(64, p.nodes() + p.stateTips()), reps = cap / (int) all.size() + 1;   // poisonable()
        std::vector<int> a;
        for (int r = 0; r < reps; ++r) { const std::vector<int> e = encode(p.ops, all, "w"); a.insert(a.end(), e.begin(), e.end()); }
        const int cum = p.cum();
        scripted("stress (c) " + std::to_string(a.size() / 7) + " ops in one call" + at, p, [&](const Inst& in) {
            update(in, a, BEAGLE_OP_NONE);
            reset(in, cum);
            accumulate(in, all, cum);
        }, cum);
        if (stop()) return;
    }
    {   // (d) beagleEnsureBufferCounts growth between calls (the pools move)
        Config c = stressConfig(S, C, "balanced", 17, 65, 9);
        c.tips = "mixed";
        const Problem p = makeProblem(c, seed);
        const int deep = deepestTip(p);
        scripted("stress (d) buffer growth between calls" + at, p, [&](const Inst& in) {
            evalFull(in, p, "writecum");
            note(beagleEnsureBufferCounts(in.id, in.partialsCount + 64, p.nodes() - 1 + 8, p.n + 8));
            evalBranch(in, p, deep, altLength(p.length[deep]), "writecum");
        }, p.cum());
        if (stop()) return;
    }
    if (gOpt.framework != "cpu") {   // (e) a degree-2 op: BEAGLE_ERROR_NO_IMPLEMENTATION with the poison intact (the
        // CPU computes degree-2 ops); a count of 0 is a no-op
        const std::string label = "stress (e) degree-2 op and empty call" + at;
        const Problem p = makeProblem(stressConfig(S, C, "balanced", 5, 65, 9), seed);
        std::vector<int> a = encode(p.ops, allOps(p), "n");
        a[2 * 7 + 5] = a[2 * 7 + 6] = BEAGLE_OP_NONE;
        std::vector<std::string> names;
        for (const char* n : {"A", "B"}) if (gOpt.impls.count(n)) names.push_back(n);
        Set s;
        openSet(s, p, names, checkMode());
        if (s.bUnavailable) { printf("%s: B unavailable, skipped\n", label.c_str()); ++gStatus.unavailable; }
        else {
            ++gStatus.ran;
            auto poisoned = [&](const Inst& in) {
                std::vector<double> x((size_t) p.cfg.patterns * S * C);
                for (int b : p.internal()) {
                    beagleGetPartials(in.id, b, BEAGLE_OP_NONE, x.data());
                    for (double v : x) if (!std::isnan(v)) return false;
                }
                return true;
            };
            std::string why = s.error;
            for (const Inst& in : s.insts) {
                if (!why.empty()) break;
                gScriptError = 0;
                poison(in, p);
                const int rc = update(in, a.data(), (int) a.size() / 7, BEAGLE_OP_NONE);
                if (gScriptError) why = in.name + ": poisoning failed (" + std::to_string(gScriptError) + ")";
                else if (rc != BEAGLE_ERROR_NO_IMPLEMENTATION) why = in.name + " returned " + std::to_string(rc) + " for a degree-2 op";
                else if (!poisoned(in)) why = in.name + " wrote partials in a refused call";
                else if (update(in, a.data(), 0, BEAGLE_OP_NONE) != BEAGLE_SUCCESS) why = in.name + ": a call of 0 ops failed";
                else if (!poisoned(in)) why = in.name + " wrote partials in a call of 0 ops";
            }
            if (why.empty()) printf("%s: ok\n", label.c_str());
            else gStatus.fail(label + ": " + why);
        }
        if (stop()) return;
    }
    {   // (f) every internal buffer NaN-poisoned, a random tree with mixed tips: B's partials must all be finite
        Config c = stressConfig(S, C, "random", 17, 63, 7);
        c.tips = "mixed";
        const Problem p = makeProblem(c, seed);
        scripted("stress (f) NaN poison" + at, p, [&](const Inst& in) { evalFull(in, p, "writecum"); }, p.cum(),
                 nullptr, [&](const Set& s) -> std::string {
            const Inst* b = s.find("B");
            if (!b) return "";
            std::vector<double> x((size_t) p.cfg.patterns * S * C);
            for (int buf : p.internal()) {
                beagleGetPartials(b->id, buf, BEAGLE_OP_NONE, x.data());
                for (double v : x) if (!std::isfinite(v)) return "B left a non-finite partial in buffer " + std::to_string(buf);
            }
            return "";
        });
        if (stop()) return;
    }
    {   // (g) a call without writes (grid.y = C) followed directly by a call with writes (grid.y = 1)
        const Problem p = makeProblem(stressConfig(S, C, "balanced", 17, 64, 9), seed);
        const std::vector<int> all = allOps(p);
        const int cum = p.cum();
        scripted("stress (g) no-write call then write call" + at, p, [&](const Inst& in) {
            reset(in, cum);
            update(in, encode(p.ops, all, "n"), BEAGLE_OP_NONE);
            update(in, encode(p.ops, all, "w"), cum);
        }, cum);
        if (stop()) return;
    }
    {   // (h) write after read of a temporary buffer inside one call
        const Problem p = makeProblem(stressConfig(S, C, "reuse", 5, 65, 9), seed);
        scripted("stress (h) buffer reuse in one call" + at, p, [&](const Inst& in) { evalFull(in, p, "writecum"); },
                 p.cum());
        if (stop()) return;
    }
    {   // (i) compact-tip invariance: states tips and the same data as partials tips give the same logL
        Config c = stressConfig(S, C, "balanced", 17, 65, 9);
        std::vector<Outcome> st, pa;
        const Problem ps = makeProblem(c, seed);
        c.tips = "partials";
        const Problem pp = makeProblem(c, seed);
        auto script = [](const Problem& p) { return [&p](const Inst& in) { evalFull(in, p, "writecum"); }; };
        if (scripted("stress (i) states tips" + at, ps, script(ps), ps.cum(), &st) &&
            scripted("stress (i) partials tips" + at, pp, script(pp), pp.cum(), &pa) && !st.empty() && !pa.empty()) {
            double scale = 0.0;
            for (double x : st.back().site) scale += std::fabs(x);
            if (!nearRef(pa.back().logL, st.back().logL, scale))
                gStatus.fail("stress (i) compact-tip invariance" + at + ": B logL " + std::to_string(st.back().logL) +
                             " with states tips, " + std::to_string(pa.back().logL) + " with partials tips");
            else printf("stress (i) compact-tip invariance%s: ok\n", at.c_str());
        }
        if (stop()) return;
    }
    {   // (k) a 1000-op chain alternating PP, SP and SS, writes in runs of 2 and 3 between reads, repeated back to
        // back; the last repetition also accumulates (one cycle: repeated cycles drift in single precision)
        const Problem p = makeProblem(stressConfig(S, C, "chain", 1001, 65, 9), seed);
        const std::vector<int> all = allOps(p);
        const std::string pattern = "wwrwwwr";
        const std::vector<int> a = encode(p.ops, all, pattern), w = written(all, pattern);
        const int cum = p.cum(), reps = gOpt.quick ? 10 : 100;
        scripted("stress (k) 1000-op chain x" + std::to_string(reps) + at, p, [&](const Inst& in) {
            reset(in, cum);
            update(in, encode(p.ops, all, "w"), BEAGLE_OP_NONE);
            accumulate(in, all, cum);
            for (int r = (in.name == "REF") ? reps - 1 : 0; r < reps - 1; ++r) update(in, a, BEAGLE_OP_NONE);
            removeFactors(in, w, cum);
            update(in, a, cum);
        }, cum);
    }
}

// (j): selection of the fused implementation by BEAGLE_FLAG_PARALLELOPS_FUSED
void stressSelection() {
    const std::string label = "stress (j) selection";
    if (gOpt.bRequire != "fused") { printf("%s: skipped (--b-require none)\n", label.c_str()); return; }
    const Problem p = makeProblem(stressConfig(4, 4, "balanced", 5, 9, 9), gOpt.seed);
    int resource;
    long pref, req;
    flagsFor("A", false, &resource, &pref, &req);
    Inst b;
    b.name = "B";
    if (create(p, b) < 0) { printf("%s: B unavailable, skipped\n", label.c_str()); ++gStatus.unavailable; return; }
    beagleFinalizeInstance(b.id);
    ++gStatus.ran;
    std::string why;
    Inst x;
    if (createRaw(p, resource, pref, req, x) < 0 || endsWith(x.implName, "-Fused"))
        why = "an ordinary request got " + x.implName;
    if (x.id >= 0) beagleFinalizeInstance(x.id);
    Inst y;
    if (why.empty() && (createRaw(p, resource, pref | BEAGLE_FLAG_SCALING_AUTO, req | BEAGLE_FLAG_PARALLELOPS_FUSED, y) < 0 ||
                        !endsWith(y.implName, "-Fused") || !(y.flags & BEAGLE_FLAG_SCALING_MANUAL) ||
                        (y.flags & BEAGLE_FLAG_SCALING_AUTO)))
        why = "FUSED required with AUTO preferred got " + y.implName + " flags " + std::to_string(y.flags);
    if (y.id >= 0) beagleFinalizeInstance(y.id);
    Inst z;
    if (why.empty() && createRaw(p, resource, pref, req | BEAGLE_FLAG_PARALLELOPS_FUSED | BEAGLE_FLAG_SCALING_AUTO, z) >= 0) {
        why = "FUSED and AUTO required gave " + z.implName;
        beagleFinalizeInstance(z.id);
    }
    Inst w;
    int rc = 0;
    if (why.empty() && (rc = createRaw(p, 0, pref, req | BEAGLE_FLAG_PARALLELOPS_FUSED, w)) != BEAGLE_ERROR_NO_RESOURCE) {
        why = "FUSED required on resource 0 returned " + std::to_string(rc);
        if (w.id >= 0) beagleFinalizeInstance(w.id);
    }
    const BeagleResourceList* rl = beagleGetResourceList();
    for (int i = 0; why.empty() && i < rl->length; ++i) {
        const long f = rl->list[i].supportFlags;
        if ((f & BEAGLE_FLAG_PARALLELOPS_FUSED) && !(f & BEAGLE_FLAG_PROCESSOR_GPU))
            why = std::string("resource ") + std::to_string(i) + " (" + rl->list[i].name + ") offers FUSED but is no GPU";
    }
    if (why.empty()) printf("%s: ok (the plugin's refusal of OpenCL CPU devices is untested here)\n", label.c_str());
    else gStatus.fail(label + ": " + why);
}

int runCheck() {
    const std::vector<Config> m = checkMatrix();
    for (const Config& c : m) {
        if (paddedStates(c.states) >= 80 && (gOpt.impls.count("A") || gOpt.impls.count("A2"))) {
            printf("FAIL: write + cumulative at padded S >= 80 has no base path (it exits); use --impls REF,B\n");
            return 1;
        }
    }
    printf("fusedbench --check: %zu configurations, then the stress cases\n", m.size());
    for (size_t i = 0; i < m.size() && !gStatus.stop; ++i) checkConfig(m[i], (int) i);
    std::vector<int> states = gOpt.statesList.empty() ? std::vector<int>{4, 61} : intList(gOpt.statesList);
    const std::vector<int> cats = intList(gOpt.categoriesList);
    const int C = (cats.empty() || std::find(cats.begin(), cats.end(), 4) != cats.end()) ? 4 : cats[0];
    for (int S : states) if (!gStatus.stop) stressCases(S, C);
    if (!gStatus.stop && std::find(states.begin(), states.end(), 4) != states.end()) stressSelection();
    printf("B updatePartials calls: %ld\n", gBCalls);
    if (gStatus.failures) { printf("FAIL: %s\n", gStatus.first.c_str()); return 1; }
    if (gStatus.ran == 0) { printf("FAIL: B unavailable for every configuration\n"); return 77; }
    printf("PASS\n");
    return 0;
}

// ---- bench mode ----

double median(std::vector<double> x) {
    std::sort(x.begin(), x.end());
    const size_t n = x.size();
    return (n % 2 == 1) ? x[n / 2] : 0.5 * (x[n / 2 - 1] + x[n / 2]);
}

struct BenchPlan {         // a configuration's timed calls, encoded once
    bool full = true, write = false;
    int rootCum = BEAGLE_OP_NONE;
    std::vector<std::vector<int>> calls;     // T_up call k: calls[k % size]; evaluation j's ops: calls[j % size]
    std::vector<std::vector<int>> removed;   // branch write: the scale buffers evaluation j recomputes
    std::vector<int> branches, matrices;     // the changed matrix of branch evaluation j; every matrix
    std::vector<double> lengths[2];          // full evaluations: every length, the first edge alternating
    double pathMean = 0, pathMax = 0;
};

BenchPlan benchPlan(const Problem& p) {
    BenchPlan b;
    const Config& c = p.cfg;
    const std::string mode = (c.scaling == "none") ? "n" : (c.scaling == "read") ? "r" : "w";
    b.write = (c.scaling == "write");
    b.rootCum = rootCum(p, c.scaling);
    b.full = (c.update == "full");
    if (b.full) {
        b.calls.push_back(encode(p.ops, allOps(p), mode));
        b.matrices.resize(p.length.size());
        std::iota(b.matrices.begin(), b.matrices.end(), 0);
        b.lengths[0] = b.lengths[1] = p.length;
        b.lengths[0][0] = altLength(p.length[0]);
        b.pathMean = b.pathMax = (double) p.ops.size();
        return b;
    }
    b.branches = (c.update == "branch-deep") ? std::vector<int>{deepestTip(p)} : branchSequence(p.n, gOpt.seed);
    for (int v : b.branches) {
        const std::vector<int> path = pathOps(p, v);
        b.calls.push_back(encode(p.ops, path, mode));
        b.removed.push_back(path);
        b.pathMean += (double) path.size() / b.branches.size();
        b.pathMax = std::max(b.pathMax, (double) path.size());
    }
    return b;
}

// evaluation j: the changed matrices, scale maintenance, updatePartials, root log likelihood
void evaluation(const Inst& in, const Problem& p, const BenchPlan& b, int j) {
    const int cum = p.cum();
    if (b.full) {
        setMatrices(in.id, b.matrices, b.lengths[j % 2]);
        if (b.write) reset(in, cum);
    } else {
        const size_t e = j % b.branches.size();
        const int v = b.branches[e];
        setMatrices(in.id, {v}, {(j % 2) ? p.length[v] : altLength(p.length[v])});
        if (b.write) removeFactors(in, b.removed[e], cum);
    }
    update(in, b.calls[j % b.calls.size()], b.write ? cum : BEAGLE_OP_NONE);
    rootLogL(in, p, b.rootCum);
}

double upBlock(const Inst& in, const Problem& p, const BenchPlan& b, int K, double* host) {
    const auto t0 = Clock::now();
    for (int k = 0; k < K; ++k) update(in, b.calls[k % b.calls.size()], BEAGLE_OP_NONE);
    const auto t1 = Clock::now();
    rootLogL(in, p, b.rootCum);
    const auto t2 = Clock::now();
    if (host) *host = us(t0, t1);
    return us(t0, t2);
}

double evalBlock(const Inst& in, const Problem& p, const BenchPlan& b, int E) {
    const auto t0 = Clock::now();
    for (int j = 0; j < E; ++j) evaluation(in, p, b, j);
    return us(t0, Clock::now());
}

// the smallest power of 2 from first whose block takes at least target microseconds, at most 16384
int calibrate(const std::function<double(int)>& block, double target, int first) {
    for (int n = first; ; n *= 2) if (n >= 16384 || block(n) >= target) return n;
}

struct Csv {
    FILE* f = nullptr;
    std::set<std::string> named;
    void names(const Set& s) {
        if (!f) return;
        for (const Inst& in : s.insts) {
            if (named.insert(in.name).second) fprintf(f, "# implName %s: %s\n", in.name.c_str(), in.implName.c_str());
        }
    }
    void open(const Set& s) {
        if (gOpt.csv.empty() || f) return;
        f = fopen(gOpt.csv.c_str(), "w");
        if (!f) { fprintf(stderr, "fusedbench: cannot write %s\n", gOpt.csv.c_str()); exit(2); }
        const char* dyld = getenv("DYLD_LIBRARY_PATH");
        fprintf(f, "# fusedbench%s\n# DYLD_LIBRARY_PATH=%s\n", gOpt.args.c_str(), dyld ? dyld : "");
        names(s);
        fprintf(f, "machine,backend,device,precision,states,padded_states,patterns,categories,taxa,tree,update,scaling,"
                   "scalers,path_ops_mean,path_ops_max,impl,implName,K,E,reps,T_up_min_us,T_up_med_us,T_host_us,"
                   "T_eval_min_us,T_eval_med_us,t_sync_us,process_mode,logL,logL_ref,dlogL,max_ulp_vs_A,ok\n");
    }
    ~Csv() { if (f) fclose(f); }
};

std::string deviceName() {
    const BeagleResourceList* rl = beagleGetResourceList();
    std::string d = (gRsrc >= 0 && gRsrc < rl->length) ? rl->list[gRsrc].name : "none";
    std::replace(d.begin(), d.end(), ',', ';');
    return d;
}

// one configuration: the instances, the checked warm-up evaluation, then the timed blocks; 77 if B is unavailable
int benchConfig(const Config& c, int index, Csv& csv) {
    const Problem p = makeProblem(c, gOpt.seed);
    printf("config %d: %s%s\n", index, describe(c).c_str(), poisonable(p) ? "" : " (internal buffers not poisoned)");
    Set s;
    openSet(s, p, namesFor(c), gOpt.mode);
    if (s.bUnavailable) { printf("  B unavailable\n"); return 77; }
    const std::string label = "config " + std::to_string(index) + " (" + describe(c) + ")";
    if (!s.error.empty()) { gStatus.fail(label + ": " + s.error); return 1; }
    const BenchPlan b = benchPlan(p);
    std::vector<double> tSync(s.insts.size(), 0.0);
    csv.open(s);
    csv.names(s);
    const std::string sc = (c.scaling == "write") ? "writecum" : c.scaling;
    std::vector<Verdict> v;
    const std::string why = subCheck(s, p, [&](const Inst& in) {
        evalFull(in, p, sc);
        if (!b.full) evalBranch(in, p, b.branches[0], altLength(p.length[b.branches[0]]), sc);
    }, b.rootCum, !(c.logScalers && c.scaling == "read"), nullptr, &v);
    if (!why.empty()) gStatus.fail(label + ": " + why);
    if (gStatus.stop || v.empty()) return 1;   // v is empty when an instance's calls failed

    std::vector<size_t> timed;
    for (size_t i = 0; i < s.insts.size(); ++i) if (s.insts[i].name != "REF") timed.push_back(i);
    std::vector<int> K(s.insts.size()), E(s.insts.size());
    std::vector<char> fast(s.insts.size());
    for (size_t i : timed) {
        const Inst& in = s.insts[i];
        tSync[i] = syncTime(in, p, b.rootCum);
        fast[i] = tSync[i] < 1000.0;
        const double target = fast[i] ? 20000.0 : 200000.0;   // slow syncs vary by 5-8 ms
        upBlock(in, p, b, 1, nullptr);                          // untimed: first-touch costs
        evalBlock(in, p, b, 1);
        K[i] = gOpt.k ? gOpt.k : calibrate([&](int n) { return upBlock(in, p, b, n, nullptr); }, target, 1);
        E[i] = gOpt.k ? std::max(3, gOpt.k) : calibrate([&](int n) { return evalBlock(in, p, b, n); }, target, 4);
    }
    for (size_t i : timed) {
        if (fast[i] != fast[timed[0]])
            printf("  (completion modes differ: %s %s, %s %s; summarize.py compares only rows in one mode)\n",
                   s.insts[timed[0]].name.c_str(), fast[timed[0]] ? "fast" : "slow", s.insts[i].name.c_str(),
                   fast[i] ? "fast" : "slow");
    }
    std::vector<size_t> order;   // A, A2, B, B, A2, A
    for (const char* n : {"A", "A2", "B", "B", "A2", "A"})
        for (size_t i : timed) if (s.insts[i].name == n) order.push_back(i);
    std::vector<std::vector<double>> up(s.insts.size()), host(s.insts.size()), ev(s.insts.size());
    for (int r = 0; r < gOpt.reps; ++r) {
        for (size_t i : order) {
            const Inst& in = s.insts[i];
            double h = 0;
            up[i].push_back((upBlock(in, p, b, K[i], &h) - tSync[i]) / K[i]);
            host[i].push_back(h / K[i]);
            ev[i].push_back(evalBlock(in, p, b, E[i]) / E[i]);
        }
    }
    for (size_t i : timed) {
        const Inst& in = s.insts[i];
        const double upMin = *std::min_element(up[i].begin(), up[i].end());
        const double evMin = *std::min_element(ev[i].begin(), ev[i].end());
        const double hostMin = *std::min_element(host[i].begin(), host[i].end());
        printf("  %-3s %-24s T_up %10.2f us (med %10.2f)  T_host %9.2f  T_eval %10.2f us (med %10.2f)  K %5d E %4d"
               "  t_sync %8.1f %s  logL %.6f  dREF %.3g  ulpA %s  %s\n",
               in.name.c_str(), in.implName.c_str(), upMin, median(up[i]), hostMin, evMin, median(ev[i]), K[i], E[i],
               tSync[i], fast[i] ? "fast" : "slow", v[i].logL, v[i].dRef, ulpText(v[i].ulp).c_str(),
               v[i].ok ? "ok" : "CHECK FAILED");
        if (csv.f) {
            fprintf(csv.f, "%s,%s,%s,%s,%d,%d,%d,%d,%d,%s,%s,%s,%s,%.4g,%.4g,%s,%s,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.2f,"
                           "%s,%.10g,%.10g,%.4g,%lld,%d\n",
                    gOpt.machine.c_str(), gOpt.framework.c_str(), deviceName().c_str(), gOpt.dbl ? "double" : "single",
                    c.states, paddedStates(c.states), c.patterns, c.categories, c.taxa, c.tree.c_str(),
                    c.update.c_str(), c.scaling.c_str(), c.logScalers ? "log" : "raw", b.pathMean, b.pathMax,
                    in.name.c_str(), in.implName.c_str(), K[i], E[i], gOpt.reps, upMin, median(up[i]), hostMin, evMin,
                    median(ev[i]), tSync[i], fast[i] ? "fast" : "slow", v[i].logL, v[i].logLRef, v[i].dRef,
                    v[i].ulp == UINT64_MAX ? -1LL : (long long) v[i].ulp, v[i].ok ? 1 : 0);   // -1: NaN mismatch
            fflush(csv.f);
        }
    }
    fflush(stdout);
    return 0;
}

std::vector<Config> readConfigs(const std::string& path) {
    std::ifstream in(path);
    if (!in) usage("cannot read " + path);
    std::vector<Config> v;
    int lineNo = 0;
    for (std::string line; std::getline(in, line);) {
        ++lineNo;
        line = line.substr(0, line.find('#'));
        std::istringstream ss(line);
        std::vector<std::string> args;
        for (std::string t; ss >> t;) args.push_back(t);
        if (args.empty()) continue;
        Config c;
        for (size_t i = 0; i < args.size(); ++i)
            if (!configOption(c, args, i)) usage(path + ":" + std::to_string(lineNo) + ": not a configuration option: " + args[i]);
        v.push_back(c);
    }
    return v;
}

int resolveResource() {
    if (gOpt.rsrc >= 0) return gOpt.rsrc;
    if (gOpt.framework == "cpu") return 0;
    const BeagleResourceList* rl = beagleGetResourceList();
    for (int i = 0; i < rl->length; ++i) {
        const long f = rl->list[i].supportFlags;
        if ((f & frameworkFlag()) && (f & BEAGLE_FLAG_PROCESSOR_GPU)) return i;
    }
    return -1;
}

} // namespace

int main(int argc, char** argv) {
    const std::vector<std::string> args(argv + 1, argv + argc);
    std::string statesArg, categoriesArg;
    for (const std::string& a : args) gOpt.args += " " + a;
    for (size_t i = 0; i < args.size(); ++i) {
        const std::string& a = args[i];
        auto value = [&]() -> std::string { if (i + 1 >= args.size()) usage("missing value after " + a); return args[++i]; };
        if (a == "--states") statesArg = value();
        else if (a == "--categories") categoriesArg = value();
        else if (configOption(gOpt.cfg, args, i)) {}
        else if (a == "--framework") gOpt.framework = value();
        else if (a == "--rsrc") gOpt.rsrc = toInt(value());
        else if (a == "--double") gOpt.dbl = true;
        else if (a == "--impls") {
            gOpt.impls.clear();
            std::stringstream ss(value());
            for (std::string n; std::getline(ss, n, ',');) {
                if (n != "REF" && n != "A" && n != "A2" && n != "B") usage("--impls takes REF, A, A2 and B");
                gOpt.impls.insert(n);
            }
        }
        else if (a == "--reps") gOpt.reps = toInt(value());
        else if (a == "--k") gOpt.k = toInt(value());
        else if (a == "--seed") gOpt.seed = toInt(value());
        else if (a == "--csv") gOpt.csv = value();
        else if (a == "--configs") gOpt.configs = value();
        else if (a == "--machine") gOpt.machine = value();
        else if (a == "--b-require") gOpt.bRequire = value();
        else if (a == "--ulp") gOpt.ulp = toInt(value());
        else if (a == "--mode") gOpt.mode = value();
        else if (a == "--check") gOpt.check = true;
        else if (a == "--quick") gOpt.quick = true;
        else if (a == "--stop-on-fail") gOpt.stopOnFail = true;
        else if (a == "--diag") gOpt.diag = true;
        else usage("unknown option " + a);
    }
    if (gOpt.framework != "opencl" && gOpt.framework != "tinygpu" && gOpt.framework != "cpu")
        usage("--framework opencl|tinygpu|cpu");
    if (gOpt.bRequire != "fused" && gOpt.bRequire != "none") usage("--b-require fused|none");
    if (gOpt.mode != "any" && gOpt.mode != "fast" && gOpt.mode != "slow") usage("--mode fast|slow|any");
    if (gOpt.reps < 1 || gOpt.k < 0 || gOpt.ulp < 0) usage("--reps, --k and --ulp must be positive");
    if (gOpt.check) {
        gOpt.statesList = statesArg;
        gOpt.categoriesList = categoriesArg;
    } else {
        if (!statesArg.empty()) gOpt.cfg.states = toInt(statesArg);
        if (!categoriesArg.empty()) gOpt.cfg.categories = toInt(categoriesArg);
    }
    selfTest();
    gRsrc = resolveResource();
    if (gOpt.check) return runCheck();

    const std::vector<Config> configs = gOpt.configs.empty() ? std::vector<Config>{gOpt.cfg} : readConfigs(gOpt.configs);
    for (const Config& c : configs) {
        validate(c);
        if (c.scaling == "write" && paddedStates(c.states) >= 80 && (gOpt.impls.count("A") || gOpt.impls.count("A2"))) {
            printf("FAIL: write + cumulative at padded S >= 80 has no base path (it exits): %s\n", describe(c).c_str());
            return 1;
        }
    }
    Csv csv;
    int rc = 0;
    for (size_t i = 0; i < configs.size() && !gStatus.stop; ++i) {
        if (benchConfig(configs[i], (int) i, csv) == 77) { rc = 77; break; }
    }
    printf("B updatePartials calls: %ld\n", gBCalls);
    if (gStatus.failures) { printf("FAIL: %s\n", gStatus.first.c_str()); return 1; }
    if (rc == 77) { printf("FAIL: B unavailable (implementation selected by BEAGLE_FLAG_PARALLELOPS_FUSED)\n"); return 77; }
    printf("PASS\n");
    return 0;
}
