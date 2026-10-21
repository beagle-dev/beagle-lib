#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <sys/resource.h>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#define HRATEBENCH_HAVE_BLAS 1
#endif

#include "libhmsbeagle/beagle.h"
#include "hratereference.h"

using Clock = std::chrono::steady_clock;

struct Options {
    std::string impl = "hierarchical";
    std::string vector = "none";
    std::string precision = "double";
    int states = 61;
    int patterns = 10000;
    int categories = 4;
    int tips = 8;
    std::string tipdata = "partials";
    std::string statetree = "random";
    double lambda = 2.0;
    int lengths = 4;
    unsigned long long seed = 20261013ULL;
    int threads = 1;
    std::string threadmode;
    std::string scaling = "none";
    int warmup = 3;
    int minreps = 20;
    double budget = 2.0;
    bool check = false;
    std::string repsOut;
    std::string load;
    bool rebuild = false;
    bool model = false;
    std::string kernel;
    int group = 0;
    std::string order;
    int tile = 0;
};

struct Problem {
    StateTree tree;
    std::vector<double> rates, weights, pi;
    std::vector<std::vector<int>> tipStates;
    std::vector<BeagleOperation> ops;
    std::vector<double> distinctLengths;
    std::vector<int> lengthOf;
    int nodes = 0;
    int root = 0;
    int slots = 0;
};

struct Timings {
    std::vector<double> set, up, scale, root;
    double firstMs = 0.0;
    double lnL = 0.0;
};

static double since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

static double quantile(std::vector<double> v, double q) {
    if (v.empty())
        return NAN;
    std::sort(v.begin(), v.end());
    double pos = q * (v.size() - 1);
    size_t i = (size_t) pos;
    double f = pos - i;
    if (i + 1 < v.size())
        return v[i] * (1.0 - f) + v[i + 1] * f;
    return v[i];
}

static double minimum(const std::vector<double>& v) {
    return v.empty() ? NAN : *std::min_element(v.begin(), v.end());
}

static std::string num(double x, const char* fmt = "%.6g") {
    if (std::isnan(x))
        return "";
    char buf[64];
    snprintf(buf, sizeof(buf), fmt, x);
    return buf;
}

static StateTree makeStateTree(const Options& o, std::mt19937_64& rng) {
    if (o.statetree == "star") {
        std::vector<double> pi(o.states, 1.0 / o.states);
        return makeStarTree(o.states, pi);
    }
    if (o.statetree == "balanced" || o.statetree == "caterpillar")
        return makeShapedTree(o.states, o.statetree == "caterpillar", rng, o.lambda);
    return makeRandomTree(o.states, rng, o.lambda);
}

static Problem makeProblem(const Options& o) {
    std::mt19937_64 rng(o.seed);
    Problem pr;
    const int S = o.states, C = o.categories, N = o.tips, K = o.patterns;
    pr.tree = makeStateTree(o, rng);
    pr.pi.assign(pr.tree.pi.begin(), pr.tree.pi.begin() + S);

    std::gamma_distribution<double> gamma(0.5, 2.0);
    double mean = 0.0;
    for (int l = 0; l < C; ++l) {
        pr.rates.push_back(C == 1 ? 1.0 : gamma(rng) + 0.01);
        mean += pr.rates.back() / C;
    }
    for (double& r : pr.rates)
        r /= mean;
    std::uniform_real_distribution<double> weight(0.2, 1.0);
    double total = 0.0;
    for (int l = 0; l < C; ++l)
        total += pr.weights.emplace_back(weight(rng));
    for (double& w : pr.weights)
        w /= total;

    std::uniform_int_distribution<int> state(0, S - 1);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::vector<int> ancestor(K);
    for (int k = 0; k < K; ++k)
        ancestor[k] = state(rng);
    pr.tipStates.assign(N, std::vector<int>(K));
    for (int t = 0; t < N; ++t)
        for (int k = 0; k < K; ++k) {
            double u = unit(rng);
            pr.tipStates[t][k] = u < 0.05 ? S : (u < 0.6 ? ancestor[k] : state(rng));
        }

    pr.nodes = 2 * N - 1;
    pr.root = pr.nodes - 1;
    pr.slots = pr.nodes - 1;
    std::vector<int> pool;
    for (int t = 0; t < N; ++t)
        pool.push_back(t);
    for (int m = N; m < pr.nodes; ++m) {
        int i = (int) (rng() % pool.size());
        int a = pool[i];
        pool.erase(pool.begin() + i);
        int j = (int) (rng() % pool.size());
        int b = pool[j];
        pool.erase(pool.begin() + j);
        pr.ops.push_back({ m, BEAGLE_OP_NONE, BEAGLE_OP_NONE, a, a, b, b });
        pool.push_back(m);
    }
    std::uniform_real_distribution<double> length(0.02, 0.6);
    int D = std::max(1, o.lengths);
    for (int d = 0; d < D; ++d)
        pr.distinctLengths.push_back(length(rng));
    for (int n = 0; n < pr.slots; ++n)
        pr.lengthOf.push_back(n % D);
    return pr;
}

static bool loadProblem(Options& o, Problem& pr) {
    FILE* f = fopen(o.load.c_str(), "r");
    if (!f)
        return false;
    char tag[64];
    int version = 0, S = 0, N = 0, K = 0;
    if (fscanf(f, "%63s %d %d %d %d", tag, &version, &S, &N, &K) != 5 || std::string(tag) != "hratebench-input" ||
        version != 1 || S < 2 || N < 2 || K < 1) {
        fclose(f);
        return false;
    }
    StateTree& t = pr.tree;
    t.S = S;
    t.parent.assign(2 * S - 1, -1);
    t.left.assign(2 * S - 1, -1);
    t.right.assign(2 * S - 1, -1);
    t.lambda.assign(2 * S - 1, 0.0);
    for (int i = 0; i < S - 1; ++i) {
        int m, a, b;
        double lam;
        if (fscanf(f, "%d %d %d %lf", &m, &a, &b, &lam) != 4 || m < S || m > 2 * S - 2) {
            fclose(f);
            return false;
        }
        join(t, m, a, b);
        t.lambda[m] = lam;
    }
    std::vector<double> tipPi(S);
    for (int x = 0; x < S; ++x)
        if (fscanf(f, "%lf", &tipPi[x]) != 1) {
            fclose(f);
            return false;
        }
    finishTree(t, tipPi);
    pr.pi = tipPi;
    pr.nodes = 2 * N - 1;
    pr.root = pr.nodes - 1;
    pr.slots = pr.nodes - 1;
    pr.ops.clear();
    for (int i = 0; i < N - 1; ++i) {
        int m, a, b;
        if (fscanf(f, "%d %d %d", &m, &a, &b) != 3) {
            fclose(f);
            return false;
        }
        pr.ops.push_back({ m, BEAGLE_OP_NONE, BEAGLE_OP_NONE, a, a, b, b });
    }
    pr.distinctLengths.assign(pr.slots, 0.0);
    pr.lengthOf.assign(pr.slots, 0);
    for (int n = 0; n < pr.slots; ++n) {
        if (fscanf(f, "%lf", &pr.distinctLengths[n]) != 1) {
            fclose(f);
            return false;
        }
        pr.lengthOf[n] = n;
    }
    pr.tipStates.assign(N, std::vector<int>(K));
    for (int x = 0; x < N; ++x)
        for (int k = 0; k < K; ++k)
            if (fscanf(f, "%d", &pr.tipStates[x][k]) != 1) {
                fclose(f);
                return false;
            }
    fclose(f);
    pr.rates.assign(1, 1.0);
    pr.weights.assign(1, 1.0);
    o.states = S;
    o.tips = N;
    o.patterns = K;
    o.categories = 1;
    o.statetree = "loaded";
    return true;
}

static void analyticEigen(const StateTree& t, std::vector<double>& evec, std::vector<double>& ievc,
                          std::vector<double>& eval) {
    const int S = t.S, N = t.nodeCount();
    evec.assign((size_t) S * S, 0.0);
    ievc.assign((size_t) S * S, 0.0);
    eval.assign(S, 0.0);
    std::vector<double> Lambda(N, 0.0);
    std::vector<int> post;
    postOrder(t, t.root(), post);
    for (int i = (int) post.size() - 1; i >= 0; --i) {
        int m = post[i];
        Lambda[m] = t.lambda[m] + (t.parent[m] >= 0 ? Lambda[t.parent[m]] : 0.0);
    }
    std::vector<std::vector<int>> tips(N);
    for (int x = 0; x < S; ++x)
        tips[x] = { x };
    for (int m : post) {
        tips[m] = tips[t.left[m]];
        tips[m].insert(tips[m].end(), tips[t.right[m]].begin(), tips[t.right[m]].end());
    }
    for (int x = 0; x < S; ++x) {
        evec[(size_t) x * S] = 1.0;
        ievc[x] = t.pi[x];
    }
    int column = 1;
    std::vector<double> v(S);
    for (int m : post) {
        int a = t.left[m], b = t.right[m];
        std::fill(v.begin(), v.end(), 0.0);
        for (int x : tips[a])
            v[x] = 1.0 / t.pi[a];
        for (int x : tips[b])
            v[x] = -1.0 / t.pi[b];
        double norm = 0.0;
        for (int x = 0; x < S; ++x)
            norm += t.pi[x] * v[x] * v[x];
        for (int x = 0; x < S; ++x) {
            evec[(size_t) x * S + column] = v[x];
            ievc[(size_t) column * S + x] = t.pi[x] * v[x] / norm;
        }
        eval[column] = -Lambda[m];
        ++column;
    }
}

static void denseP(const StateTree& tree, const std::vector<double>& rates, double t, std::vector<double>& out) {
    const int S = tree.S, N = tree.nodeCount(), C = (int) rates.size();
    std::vector<int> post;
    postOrder(tree, tree.root(), post);
    std::vector<int> dfs;
    std::vector<int> lo(N), hi(N);
    std::vector<std::pair<int, bool>> stack{ { tree.root(), false } };
    while (!stack.empty()) {
        auto [m, done] = stack.back();
        stack.pop_back();
        if (m < S) {
            lo[m] = (int) dfs.size();
            dfs.push_back(m);
            hi[m] = (int) dfs.size();
            continue;
        }
        if (done) {
            lo[m] = lo[tree.left[m]];
            hi[m] = hi[tree.right[m]];
            continue;
        }
        stack.push_back({ m, true });
        stack.push_back({ tree.right[m], false });
        stack.push_back({ tree.left[m], false });
    }
    out.assign((size_t) C * S * S, 0.0);
    std::vector<double> prZ0, H, Cn(N);
    for (int l = 0; l < C; ++l) {
        nodeScalars(tree, t * rates[l], prZ0, H);
        Cn[tree.root()] = H[tree.root()];
        for (int i = (int) post.size() - 1; i >= 0; --i) {
            int m = post[i];
            Cn[tree.left[m]] = H[tree.left[m]] + Cn[m];
            Cn[tree.right[m]] = H[tree.right[m]] + Cn[m];
        }
        double* P = out.data() + (size_t) l * S * S;
        for (int m : post) {
            int a = tree.left[m], b = tree.right[m];
            double c = Cn[m];
            for (int i = lo[a]; i < hi[a]; ++i)
                for (int j = lo[b]; j < hi[b]; ++j) {
                    int x = dfs[i], y = dfs[j];
                    P[(size_t) x * S + y] = tree.pi[y] * c;
                    P[(size_t) y * S + x] = tree.pi[x] * c;
                }
        }
        for (int x = 0; x < S; ++x)
            P[(size_t) x * S + x] = tree.pi[x] * Cn[x];
    }
}

static long scalingFlags(const std::string& s) {
    if (s == "manual")
        return BEAGLE_FLAG_SCALING_MANUAL | BEAGLE_FLAG_SCALERS_LOG;
    if (s == "always")
        return BEAGLE_FLAG_SCALING_ALWAYS;
    return 0;
}

static uint64_t fnv(uint64_t h, const void* data, size_t n) {
    const unsigned char* p = (const unsigned char*) data;
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static int expectedPartitions(const Options& o) {
    if (o.threadmode != "auto")
        return 1;
    int minPat = o.states <= 4 ? 768 : 2;
    if (o.patterns < minPat)
        return 1;
    return std::max(1, std::min(o.patterns / (minPat / 2), o.threads));
}

struct BeagleRun {
    int instance = -1;
    std::string implName;
    int failed = 0;
    std::string failedCall;
};

static bool call(BeagleRun& r, const char* name, int code) {
    if (code < 0 && r.failed == 0) {
        r.failed = code;
        r.failedCall = name;
    }
    return code >= 0;
}

static void setupBeagle(const Options& o, const Problem& pr, bool hierarchical, bool oracle, BeagleRun& run,
                        std::vector<std::vector<double>>& payloads) {
    const int S = o.states, C = o.categories, N = o.tips, K = o.patterns;
    const bool single = !oracle && o.precision == "single";
    long prec = single ? BEAGLE_FLAG_PRECISION_SINGLE : BEAGLE_FLAG_PRECISION_DOUBLE;
    long req = BEAGLE_FLAG_PROCESSOR_CPU | prec;
    if (hierarchical)
        req |= BEAGLE_FLAG_HIERARCHICAL_REPRESENTATION;
    else
        req |= (!oracle && o.vector == "sse") ? BEAGLE_FLAG_VECTOR_SSE : BEAGLE_FLAG_VECTOR_NONE;
    std::string threadmode = oracle ? "none" : o.threadmode;
    std::string scaling = oracle ? "none" : o.scaling;
    long prefs = prec | scalingFlags(scaling) |
                 (threadmode == "auto" ? BEAGLE_FLAG_THREADING_CPP : BEAGLE_FLAG_THREADING_NONE);
    if (!oracle && o.impl == "eigen")
        prefs |= BEAGLE_FLAG_EIGEN_REAL;
    const bool states = o.tipdata == "states";
    const int internal = N - 1;
    int scaleBuffers = scaling == "none" ? 0 : internal + 1;
    BeagleInstanceDetails det;
    run.instance = beagleCreateInstance(N, states ? internal : pr.nodes, states ? N : 0, S, K, 1, pr.slots, C,
                                        scaleBuffers, nullptr, 0, prefs, req, &det);
    if (run.instance < 0) {
        call(run, "beagleCreateInstance", run.instance);
        return;
    }
    run.implName = det.implName;
    for (int t = 0; t < N; ++t) {
        if (states) {
            if (!call(run, "beagleSetTipStates", beagleSetTipStates(run.instance, t, pr.tipStates[t].data())))
                return;
        } else {
            std::vector<double> p = partialsFromStates(pr.tipStates[t], S);
            if (!call(run, "beagleSetTipPartials", beagleSetTipPartials(run.instance, t, p.data())))
                return;
        }
    }
    std::vector<double> patternWeights(K, 1.0);
    if (!call(run, "beagleSetStateFrequencies", beagleSetStateFrequencies(run.instance, 0, pr.pi.data())) ||
        !call(run, "beagleSetCategoryRates", beagleSetCategoryRates(run.instance, pr.rates.data())) ||
        !call(run, "beagleSetCategoryWeights", beagleSetCategoryWeights(run.instance, 0, pr.weights.data())) ||
        !call(run, "beagleSetPatternWeights", beagleSetPatternWeights(run.instance, patternWeights.data())))
        return;
    if (threadmode == "auto" &&
        !call(run, "beagleSetCPUThreadCount", beagleSetCPUThreadCount(run.instance, o.threads)))
        return;
    if (!oracle && o.impl == "eigen") {
        std::vector<double> evec, ievc, eval;
        analyticEigen(pr.tree, evec, ievc, eval);
        if (!call(run, "beagleSetEigenDecomposition",
                  beagleSetEigenDecomposition(run.instance, 0, evec.data(), ievc.data(), eval.data())))
            return;
    }
    if (hierarchical && o.model) {
        ModelArrays m = modelArrays(pr.tree);
        if (!call(run, "beagleSetEigenDecomposition",
                  beagleSetEigenDecomposition(run.instance, 0, m.vectors.data(), m.inverse.data(), m.values.data())))
            return;
    }
    payloads.assign(pr.distinctLengths.size(), {});
    for (size_t d = 0; d < pr.distinctLengths.size(); ++d) {
        if (hierarchical)
            referenceBundle(pr.tree, pr.rates.data(), C, pr.distinctLengths[d], payloads[d]);
        else
            denseP(pr.tree, pr.rates, pr.distinctLengths[d], payloads[d]);
    }
}

static void evaluateBeagle(const Options& o, const Problem& pr, BeagleRun& run,
                           const std::vector<std::vector<double>>& payloads, const std::string& scaling,
                           double& setMs, double& upMs, double& scaleMs, double& rootMs, double& lnL) {
    const int N = o.tips;
    const int internal = N - 1;
    const int cumulative = scaling == "none" ? BEAGLE_OP_NONE : internal;
    std::vector<BeagleOperation> ops = pr.ops;
    std::vector<int> writes;
    if (scaling == "manual")
        for (auto& op : ops) {
            op.destinationScaleWrite = op.destinationPartials - N;
            writes.push_back(op.destinationScaleWrite);
        }
    const bool eigen = o.impl == "eigen" || (o.model && o.impl == "hierarchical");
    const bool closedform = o.impl == "closedform";
    const bool rebuild = closedform || (o.rebuild && o.impl == "hierarchical");
    auto t0 = Clock::now();
    if (eigen) {
        std::vector<int> idx(pr.slots);
        std::vector<double> len(pr.slots);
        for (int n = 0; n < pr.slots; ++n) {
            idx[n] = n;
            len[n] = pr.distinctLengths[pr.lengthOf[n]];
        }
        call(run, "beagleUpdateTransitionMatrices",
             beagleUpdateTransitionMatrices(run.instance, 0, idx.data(), nullptr, nullptr, len.data(), pr.slots));
    } else if (rebuild) {
        std::vector<double> fresh;
        for (int n = 0; n < pr.slots; ++n) {
            const double t = pr.distinctLengths[pr.lengthOf[n]];
            if (closedform)
                denseP(pr.tree, pr.rates, t, fresh);
            else
                referenceBundle(pr.tree, pr.rates.data(), (int) pr.rates.size(), t, fresh);
            call(run, "beagleSetTransitionMatrix", beagleSetTransitionMatrix(run.instance, n, fresh.data(), 1.0));
        }
    } else {
        for (int n = 0; n < pr.slots; ++n)
            call(run, "beagleSetTransitionMatrix",
                 beagleSetTransitionMatrix(run.instance, n, payloads[pr.lengthOf[n]].data(), 1.0));
    }
    setMs = since(t0);
    scaleMs = 0.0;
    if (scaling != "none") {
        t0 = Clock::now();
        call(run, "beagleResetScaleFactors", beagleResetScaleFactors(run.instance, cumulative));
        scaleMs += since(t0);
    }
    t0 = Clock::now();
    call(run, "beagleUpdatePartials",
         beagleUpdatePartials(run.instance, ops.data(), (int) ops.size(),
                              scaling == "always" ? cumulative : BEAGLE_OP_NONE));
    upMs = since(t0);
    if (scaling == "manual") {
        t0 = Clock::now();
        call(run, "beagleAccumulateScaleFactors",
             beagleAccumulateScaleFactors(run.instance, writes.data(), (int) writes.size(), cumulative));
        scaleMs += since(t0);
    }
    int rootBuffer = pr.root, weightIndex = 0, freqIndex = 0, cum = cumulative;
    t0 = Clock::now();
    call(run, "beagleCalculateRootLogLikelihoods",
         beagleCalculateRootLogLikelihoods(run.instance, &rootBuffer, &weightIndex, &freqIndex, &cum, 1, &lnL));
    rootMs = since(t0);
}

static std::vector<double> beaglePartials(const Options& o, const Problem& pr, BeagleRun& run) {
    const size_t block = (size_t) o.categories * o.patterns * o.states;
    std::vector<double> all;
    std::vector<double> buf(block);
    for (int m = o.tips; m < pr.nodes; ++m) {
        if (!call(run, "beagleGetPartials", beagleGetPartials(run.instance, m, BEAGLE_OP_NONE, buf.data())))
            return {};
        all.insert(all.end(), buf.begin(), buf.end());
    }
    return all;
}

#ifdef HRATEBENCH_HAVE_BLAS
struct BlasState {
    std::vector<std::vector<double>> partials;
    std::vector<std::vector<double>> payloads;
    std::vector<double> t1, t2;
};

static void setupBlas(const Options& o, const Problem& pr, BlasState& b) {
    const int S = o.states, C = o.categories, K = o.patterns;
    const size_t block = (size_t) C * S * K;
    b.partials.assign(pr.nodes, {});
    for (int t = 0; t < o.tips; ++t) {
        b.partials[t].assign(block, 0.0);
        for (int l = 0; l < C; ++l)
            for (int k = 0; k < K; ++k) {
                int s = pr.tipStates[t][k];
                double* base = b.partials[t].data() + (size_t) l * S * K;
                if (s < S)
                    base[(size_t) s * K + k] = 1.0;
                else
                    for (int x = 0; x < S; ++x)
                        base[(size_t) x * K + k] = 1.0;
            }
    }
    for (int m = o.tips; m < pr.nodes; ++m)
        b.partials[m].assign(block, 0.0);
    b.payloads.assign(pr.distinctLengths.size(), {});
    for (size_t d = 0; d < pr.distinctLengths.size(); ++d)
        denseP(pr.tree, pr.rates, pr.distinctLengths[d], b.payloads[d]);
    b.t1.assign((size_t) S * K, 0.0);
    b.t2.assign((size_t) S * K, 0.0);
}

static void evaluateBlas(const Options& o, const Problem& pr, BlasState& b, double& upMs, double& rootMs,
                         double& lnL) {
    const int S = o.states, C = o.categories, K = o.patterns;
    auto t0 = Clock::now();
    for (const auto& op : pr.ops) {
        for (int l = 0; l < C; ++l) {
            const double* P1 = b.payloads[pr.lengthOf[op.child1TransitionMatrix]].data() + (size_t) l * S * S;
            const double* P2 = b.payloads[pr.lengthOf[op.child2TransitionMatrix]].data() + (size_t) l * S * S;
            const double* L1 = b.partials[op.child1Partials].data() + (size_t) l * S * K;
            const double* L2 = b.partials[op.child2Partials].data() + (size_t) l * S * K;
            double* dest = b.partials[op.destinationPartials].data() + (size_t) l * S * K;
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, S, K, S, 1.0, P1, S, L1, K, 0.0, b.t1.data(), K);
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, S, K, S, 1.0, P2, S, L2, K, 0.0, b.t2.data(), K);
            const double* a = b.t1.data();
            const double* c = b.t2.data();
            const size_t n = (size_t) S * K;
            for (size_t i = 0; i < n; ++i)
                dest[i] = a[i] * c[i];
        }
    }
    upMs = since(t0);
    t0 = Clock::now();
    const double* R = b.partials[pr.root].data();
    std::vector<double> site(K, 0.0);
    for (int l = 0; l < C; ++l)
        for (int x = 0; x < S; ++x) {
            const double w = pr.weights[l] * pr.pi[x];
            const double* row = R + (size_t) l * S * K + (size_t) x * K;
            for (int k = 0; k < K; ++k)
                site[k] += w * row[k];
        }
    lnL = 0.0;
    for (int k = 0; k < K; ++k)
        lnL += std::log(site[k]);
    rootMs = since(t0);
}

static std::vector<double> blasPartials(const Options& o, const Problem& pr, const BlasState& b) {
    const int S = o.states, C = o.categories, K = o.patterns;
    std::vector<double> all;
    all.reserve((size_t) (pr.nodes - o.tips) * C * K * S);
    for (int m = o.tips; m < pr.nodes; ++m)
        for (int l = 0; l < C; ++l)
            for (int k = 0; k < K; ++k)
                for (int x = 0; x < S; ++x)
                    all.push_back(b.partials[m][(size_t) l * S * K + (size_t) x * K + k]);
    return all;
}
#endif

static const char* kHeader =
    "version,implementation,impl,vector,precision,states,patterns,padded,categories,tips,ops,tipdata,phylo,"
    "statetree,seed,threads,threadmode,partitions,scaling,tile,reps,first_ms,set_us_med,up_ms_min,up_ms_med,"
    "up_ms_p25,up_ms_p75,scale_ms_med,root_ms_min,root_ms_med,maxrss_mb,lnL,check_lnL,check_maxdiff,"
    "check_maxrel,partials_sha,pre_ms,grad_ms,kernel";

static void usage() {
    fprintf(stderr,
            "usage: hratebench [--impl hierarchical|standard|eigen|closedform|blas] [--rebuild] [--model] [--vector sse|none] [--precision double|single]\n"
            "                  [--states S] [--patterns K] [--categories C] [--tips N] [--tipdata partials|states]\n"
            "                  [--statetree random|star|balanced|caterpillar] [--kernel tiled|shape|dense|fused] [--group G] [--order category|group] [--tile T] [--lambda m] [--lengths D] [--seed n]\n"
            "                  [--threads T] [--threadmode none|auto] [--scaling none|manual|always]\n"
            "                  [--warmup W] [--minreps R] [--budget s] [--check] [--reps-out FILE] [--load FILE]\n"
            "                  [--header]\n");
}

int main(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](void) -> const char* {
            if (i + 1 >= argc) {
                usage();
                exit(2);
            }
            return argv[++i];
        };
        if (a == "--header") {
            printf("%s\n", kHeader);
            return 0;
        } else if (a == "--impl") o.impl = next();
        else if (a == "--vector") o.vector = next();
        else if (a == "--precision") o.precision = next();
        else if (a == "--states") o.states = atoi(next());
        else if (a == "--patterns") o.patterns = atoi(next());
        else if (a == "--categories") o.categories = atoi(next());
        else if (a == "--tips") o.tips = atoi(next());
        else if (a == "--tipdata") o.tipdata = next();
        else if (a == "--statetree") o.statetree = next();
        else if (a == "--lambda") o.lambda = atof(next());
        else if (a == "--lengths") o.lengths = atoi(next());
        else if (a == "--seed") o.seed = strtoull(next(), nullptr, 10);
        else if (a == "--threads") o.threads = atoi(next());
        else if (a == "--threadmode") o.threadmode = next();
        else if (a == "--scaling") o.scaling = next();
        else if (a == "--warmup") o.warmup = atoi(next());
        else if (a == "--minreps") o.minreps = atoi(next());
        else if (a == "--budget") o.budget = atof(next());
        else if (a == "--check") o.check = true;
        else if (a == "--reps-out") o.repsOut = next();
        else if (a == "--load") o.load = next();
        else if (a == "--rebuild") o.rebuild = true;
        else if (a == "--model") o.model = true;
        else if (a == "--kernel") o.kernel = next();
        else if (a == "--group") o.group = atoi(next());
        else if (a == "--order") o.order = next();
        else if (a == "--tile") o.tile = atoi(next());
        else {
            usage();
            return 2;
        }
    }
    if (o.threadmode.empty())
        o.threadmode = o.threads > 1 ? "auto" : "none";
    const bool blas = o.impl == "blas";
    const bool hierarchical = o.impl == "hierarchical";
    if ((o.impl != "hierarchical" && o.impl != "standard" && o.impl != "eigen" && o.impl != "closedform" && !blas) ||
        (o.vector != "none" && o.vector != "sse") || (o.precision != "double" && o.precision != "single") ||
        (o.tipdata != "partials" && o.tipdata != "states") ||
        (o.statetree != "random" && o.statetree != "star" && o.statetree != "balanced" &&
         o.statetree != "caterpillar" && o.load.empty()) ||
        (!o.kernel.empty() && o.kernel != "tiled" && o.kernel != "shape" && o.kernel != "dense" && o.kernel != "fused") ||
        (!o.order.empty() && o.order != "category" && o.order != "group") ||
        (o.threadmode != "none" && o.threadmode != "auto") ||
        (o.scaling != "none" && o.scaling != "manual" && o.scaling != "always") || o.states < 2 || o.tips < 2 ||
        o.patterns < 1 || o.categories < 1) {
        usage();
        return 2;
    }
    if (!o.kernel.empty())
        setenv("BEAGLE_HRATE_KERNEL", o.kernel.c_str(), 1);
    if (o.group > 0)
        setenv("BEAGLE_HRATE_GROUP", std::to_string(o.group).c_str(), 1);
    if (!o.order.empty())
        setenv("BEAGLE_HRATE_ORDER", o.order.c_str(), 1);
    if (o.tile > 0)
        setenv("BEAGLE_HRATE_TILE", std::to_string(o.tile).c_str(), 1);
    const char* kernelEnv = getenv("BEAGLE_HRATE_KERNEL");
    const std::string kernelName = !hierarchical ? "" : (kernelEnv ? kernelEnv : "default");
    if ((o.impl != "hierarchical" && !blas && o.vector == "sse" && o.precision == "single") ||
        (blas && (o.precision != "double" || o.scaling != "none" || o.threads != 1))) {
        fprintf(stderr, "hratebench: unsupported combination\n");
        return 2;
    }
#ifndef HRATEBENCH_HAVE_BLAS
    if (blas) {
        fprintf(stderr, "hratebench: blas needs Accelerate\n");
        return 2;
    }
#endif
    const char* version = beagleGetVersion();
    if (strncmp(version, "4.3.0", 5) != 0) {
        fprintf(stderr, "hratebench: BEAGLE version %s, expected 4.3.0\n", version);
        return 2;
    }

    Problem pr;
    if (!o.load.empty()) {
        if (!loadProblem(o, pr)) {
            fprintf(stderr, "hratebench: cannot read %s\n", o.load.c_str());
            return 2;
        }
    } else {
        pr = makeProblem(o);
    }
    Timings tm;
    BeagleRun run;
    std::vector<std::vector<double>> payloads;
#ifdef HRATEBENCH_HAVE_BLAS
    BlasState bs;
#endif
    std::string implName;
    if (blas) {
#ifdef HRATEBENCH_HAVE_BLAS
        setupBlas(o, pr, bs);
        implName = "BLAS-Accelerate-Double";
#endif
    } else {
        setupBeagle(o, pr, hierarchical, false, run, payloads);
        if (run.failed) {
            fprintf(stderr, "hratebench: %s failed with %d\n", run.failedCall.c_str(), run.failed);
            return 2;
        }
        implName = run.implName;
        if (hierarchical && implName.rfind("CPU-Hierarchical", 0) != 0) {
            fprintf(stderr, "hratebench: implementation %s is not hierarchical\n", implName.c_str());
            return 2;
        }
    }

    auto evaluate = [&](double& setMs, double& upMs, double& scaleMs, double& rootMs, double& lnL) {
        setMs = scaleMs = 0.0;
        if (blas) {
#ifdef HRATEBENCH_HAVE_BLAS
            evaluateBlas(o, pr, bs, upMs, rootMs, lnL);
#endif
        } else {
            evaluateBeagle(o, pr, run, payloads, o.scaling, setMs, upMs, scaleMs, rootMs, lnL);
        }
    };

    int warmup = std::max(1, o.warmup), minreps = std::max(1, o.minreps);
    double setMs, upMs, scaleMs, rootMs, lnL = 0.0;
    auto firstStart = Clock::now();
    evaluate(setMs, upMs, scaleMs, rootMs, lnL);
    double firstTotal = since(firstStart);
    tm.firstMs = upMs;
    if (firstTotal > 2000.0) {
        warmup = 1;
        minreps = std::min(minreps, 3);
    }
    for (int w = 1; w < warmup; ++w)
        evaluate(setMs, upMs, scaleMs, rootMs, lnL);
    auto budgetStart = Clock::now();
    while ((int) tm.up.size() < minreps || since(budgetStart) < o.budget * 1000.0) {
        evaluate(setMs, upMs, scaleMs, rootMs, lnL);
        tm.set.push_back(setMs * 1000.0 / std::max(1, pr.slots));
        tm.up.push_back(upMs);
        tm.scale.push_back(scaleMs);
        tm.root.push_back(rootMs);
        if (firstTotal > 2000.0 && (int) tm.up.size() >= minreps)
            break;
    }
    tm.lnL = lnL;
    if (run.failed) {
        fprintf(stderr, "hratebench: %s failed with %d\n", run.failedCall.c_str(), run.failed);
        return 2;
    }

    std::vector<double> mine;
    uint64_t sha = 1469598103934665603ULL;
    bool haveSha = false;
    if (o.scaling == "none") {
        haveSha = true;
        for (int m = o.tips; m < pr.nodes; ++m) {
            std::vector<double> one;
#ifdef HRATEBENCH_HAVE_BLAS
            if (blas) {
                Problem single = pr;
                single.nodes = m + 1;
                Options oo = o;
                oo.tips = m;
                one = blasPartials(oo, single, bs);
            } else
#endif
            {
                const size_t block = (size_t) o.categories * o.patterns * o.states;
                one.assign(block, 0.0);
                if (!call(run, "beagleGetPartials", beagleGetPartials(run.instance, m, BEAGLE_OP_NONE, one.data())))
                    one.clear();
            }
            for (double v : one)
                sha = fnv(sha, &v, sizeof(v));
            if (o.check)
                mine.insert(mine.end(), one.begin(), one.end());
        }
    }

    double checkLnL = NAN, maxdiff = NAN, maxrel = NAN;
    bool checkOk = true;
    if (o.check) {
        if (!blas) {
            beagleFinalizeInstance(run.instance);
            run.instance = -1;
#ifdef HRATEBENCH_HAVE_BLAS
        } else {
            bs = BlasState();
#endif
        }
        Options oo = o;
        oo.impl = "standard";
        oo.rebuild = false;
        oo.model = false;
        oo.precision = "double";
        oo.vector = "none";
        oo.threads = 1;
        oo.threadmode = "none";
        oo.scaling = "none";
        BeagleRun oracle;
        std::vector<std::vector<double>> op;
        setupBeagle(oo, pr, false, true, oracle, op);
        double s1, u1, c1, r1;
        if (!oracle.failed)
            evaluateBeagle(oo, pr, oracle, op, "none", s1, u1, c1, r1, checkLnL);
        std::vector<double> ref = oracle.failed ? std::vector<double>() : beaglePartials(oo, pr, oracle);
        if (oracle.failed) {
            fprintf(stderr, "hratebench: oracle %s failed with %d\n", oracle.failedCall.c_str(), oracle.failed);
            checkOk = false;
        } else {
            if (!mine.empty() && mine.size() == ref.size()) {
                maxdiff = 0.0;
                maxrel = 0.0;
                for (size_t i = 0; i < ref.size(); ++i) {
                    double d = std::fabs(mine[i] - ref[i]);
                    if (std::isnan(d))
                        d = INFINITY;
                    maxdiff = std::max(maxdiff, d);
                    maxrel = std::max(maxrel, d / std::max(std::fabs(ref[i]), 1e-300));
                }
            }
            double lnRel = std::fabs(tm.lnL - checkLnL) / std::fabs(checkLnL);
            if (o.precision == "double") {
                checkOk = lnRel <= 1e-10 && (std::isnan(maxdiff) || maxdiff <= 1e-12);
            } else {
                checkOk = lnRel <= 1e-7 && (std::isnan(maxrel) || maxrel <= 1e-6);
            }
            beagleFinalizeInstance(oracle.instance);
        }
    } else if (!blas) {
        beagleFinalizeInstance(run.instance);
    }

    if (!o.repsOut.empty()) {
        FILE* f = fopen(o.repsOut.c_str(), "w");
        if (f) {
            fprintf(f, "rep,set_us,up_ms,scale_ms,root_ms\n");
            for (size_t r = 0; r < tm.up.size(); ++r)
                fprintf(f, "%zu,%.6g,%.6g,%.6g,%.6g\n", r, tm.set[r], tm.up[r], tm.scale[r], tm.root[r]);
            fclose(f);
        }
    }

    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    double maxrssMb = ru.ru_maxrss / 1048576.0;
    int mod = o.precision == "single" ? 16 : 8;
    int padded = hierarchical ? ((o.patterns + mod - 1) / mod) * mod : o.patterns;
    char shaBuf[32];
    snprintf(shaBuf, sizeof(shaBuf), "%016llx", (unsigned long long) sha);

    std::vector<std::string> cols = {
        version, implName, o.impl + (o.rebuild && hierarchical ? "-rebuild" : "") + (o.model && hierarchical ? "-model" : ""), blas ? "none" : (hierarchical ? "none" : o.vector), o.precision,
        std::to_string(o.states), std::to_string(o.patterns), std::to_string(padded),
        std::to_string(o.categories), std::to_string(o.tips), std::to_string(pr.ops.size()), o.tipdata,
        o.load.empty() ? "random" : "loaded",
        o.statetree, std::to_string(o.seed), std::to_string(o.threads), o.threadmode,
        std::to_string(expectedPartitions(o)), o.scaling, "default", std::to_string(tm.up.size()),
        num(tm.firstMs), num(quantile(tm.set, 0.5)), num(minimum(tm.up)), num(quantile(tm.up, 0.5)),
        num(quantile(tm.up, 0.25)), num(quantile(tm.up, 0.75)), num(quantile(tm.scale, 0.5)),
        num(minimum(tm.root)), num(quantile(tm.root, 0.5)), num(maxrssMb, "%.1f"), num(tm.lnL, "%.10f"),
        num(checkLnL, "%.10f"), num(maxdiff, "%.3g"), num(maxrel, "%.3g"), haveSha ? shaBuf : "", "", "", kernelName
    };
    std::string line;
    for (size_t i = 0; i < cols.size(); ++i) {
        std::string c = cols[i];
        if (c.find(',') != std::string::npos)
            c = "\"" + c + "\"";
        line += (i ? "," : "") + c;
    }
    printf("%s\n", line.c_str());
    return checkOk ? 0 : 1;
}
