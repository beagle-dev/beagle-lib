/*
 * Copyright 2026 Phylogenetic Likelihood Working Group
 * This file is part of BEAGLE.
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 *
 * buffergrowthtest: beagleEnsureBufferCounts on the CPU implementations.
 *
 * An instance G is created with only the buffers of a post-order pass and grown in steps between the stages of a
 * computation: post-order and root log likelihood; bottom pre-order; top pre-order, adjoint gradient and edge
 * derivatives (G grows between the two pre-order passes and the adjoint, so stale eigen-information pointers would
 * show); then new operations on new buffers, a degree-2 operation among them, that read old buffers without
 * recomputing them. The growth steps add one buffer of each kind, cross eigen-information chunk boundaries (32 and
 * 64 matrices), ask for smaller counts (no change) and probe with (0, 0, 0). An instance R created with the final
 * counts makes the same calls. After every growth, G's old partials and transition matrices are byte-identical to
 * before, and every result of G equals R's exactly.
 *
 * Cases: CPU, CPU-4State and CPU-Spectral in double and single precision, and CPU-SSE, CPU-4State-SSE,
 * CPU-Spectral-SSE and CPU-4State-Spectral-SSE in double precision; 4, 17 and 20 states; 1 and 64 patterns; 1 and 4
 * rate categories; no scaling, MANUAL, DYNAMIC, ALWAYS and AUTO scaling; and four threads, with 64 patterns (1,536
 * with 4 states, so that the patterns are partitioned between the threads). Each
 * case checks the implementation name, so a fallback to another class fails. DYNAMIC and AUTO are left out on the
 * spectral implementations (their factories do not offer DYNAMIC, and the first AUTO post-order pass exits the
 * process). With threads only no scaling and MANUAL are run: under DYNAMIC, AUTO and ALWAYS the pattern-partition
 * threads all update whole scale buffers, and two identical instances already give different likelihoods without
 * any growth.
 *
 * Contract checks: negative and overflowing counts return BEAGLE_ERROR_OUT_OF_RANGE and leave the instance as it was;
 * invalid instances return BEAGLE_ERROR_UNINITIALIZED_INSTANCE; a GPU instance, standard and spectral, if there is
 * one, returns BEAGLE_ERROR_NO_IMPLEMENTATION and still works. Finally it reports the time of growth calls at 10,000 buffers
 * with 17 states (information only).
 */

#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "libhmsbeagle/beagle.h"

namespace {

int failures = 0;
std::string current; // the case being run

void check(bool ok, const std::string& what) {
    if (!ok) {
        ++failures;
        printf("FAIL %s: %s\n", current.c_str(), what.c_str());
    }
}

bool same(const std::vector<double>& a, const std::vector<double>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), sizeof(double) * a.size()) == 0;
}

// Symmetric eigen decomposition by cyclic Jacobi rotations: a = v diag(d) v^T
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

// Random reversible rate matrix Q = R diag(pi), one substitution per unit time; also returns Q itself
void randomReversible(std::mt19937& rng, int n, std::vector<double>& pi, std::vector<double>& evec,
                      std::vector<double>& ivec, std::vector<double>& eval, std::vector<double>& q) {
    std::uniform_real_distribution<double> u(0.5, 1.5);
    pi.resize(n);
    double sum = 0.0;
    for (auto& p : pi) { p = u(rng); sum += p; }
    for (auto& p : pi) p /= sum;
    std::vector<double> r(n * n, 0.0);
    q.assign(n * n, 0.0);
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

// A random binary tree on tips 0..tips-1 (internal nodes tips..2 tips - 2, the root last) with one degree-2 node,
// 2 tips - 1, inserted on a random branch (as in simdtest)
struct Tree {
    std::vector<int> parent;
    std::vector<std::vector<int>> children;
    std::vector<double> length;
    int root = -1;
};

Tree randomTree(std::mt19937& rng, int tips) {
    Tree t;
    const int binary = 2 * tips - 1, nodes = binary + 1;
    t.parent.assign(nodes, -1);
    t.children.assign(nodes, {});
    std::vector<int> active;
    for (int i = 0; i < tips; ++i) active.push_back(i);
    for (int next = tips; next < binary; ++next) {
        int pick[2];
        for (int& p : pick) {
            const int a = std::uniform_int_distribution<int>(0, (int) active.size() - 1)(rng);
            p = active[a];
            active.erase(active.begin() + a);
        }
        t.children[next] = {pick[0], pick[1]};
        t.parent[pick[0]] = t.parent[pick[1]] = next;
        active.push_back(next);
    }
    t.root = binary - 1;
    const int degree2 = binary;
    const int below = std::uniform_int_distribution<int>(0, binary - 2)(rng); // any node but the root
    const int above = t.parent[below];
    for (int& c : t.children[above]) if (c == below) c = degree2;
    t.parent[degree2] = above;
    t.children[degree2] = {below};
    t.parent[below] = degree2;
    std::uniform_real_distribution<double> u(0.05, 0.3);
    t.length.assign(nodes, 0.0);
    for (int n = 0; n < nodes; ++n) if (n != t.root) t.length[n] = u(rng);
    return t;
}

// Buffer layout. N tips (tips 0..N-2 hold states, tip N-1 partials) and M = 2N nodes including the degree-2 node.
// Partials: post-order n, bottom pre-order M + n, top pre-order 2M + n, then two buffers written only at the end, then
// spare buffers that make the final count exceed the first one (M) times the four threads' pattern partitions.
// Matrices: branch n at n, the infinitesimal matrix at kQ, two matrices written only at the end. Scale buffers
// (MANUAL and DYNAMIC): node n at n - N, cumulative at N, one for the pre-order passes, then three written only at
// the end. MANUAL rescales the bifurcating nodes only. DYNAMIC rescales every operation whose inputs are partials, so
// its degree-2 and pre-order operations need a scale buffer as well (with none, BEAGLE reads scale buffer -1).
const int N = 8, M = 2 * N, kCompact = N - 1;
int bottom(int n) { return M + n; }
int top(int n) { return 2 * M + n; }
const int kNew0 = 3 * M, kNew1 = 3 * M + 1, kPartials = 3 * M + 2 + 4 * M;
const int kQ = 40, kMatrixNew0 = 67, kMatrixNew1 = 68, kMatrices = 70;
const int kCumulative = N, kScalePre = N + 1, kScaleNew0 = N + 2, kScaleNew1 = N + 3, kCumulativeNew = N + 4,
          kScales = N + 5;

enum Scaling { NONE, MANUAL, DYNAMIC, ALWAYS, AUTO };
const char* const scalingName[] = {"none", "MANUAL", "DYNAMIC", "ALWAYS", "AUTO"};

struct Config {
    bool spectral, sse, single;
    int S, P, C;
    Scaling scaling;
    int threads;
};

std::string expectedName(const Config& c) {
    const std::string precision = c.single ? "Single" : "Double";
    if (c.spectral && c.sse) return (c.S == 4) ? "CPU-4State-Spectral-SSE-Double" : "CPU-Spectral-SSE-Double";
    if (c.spectral) return "CPU-Spectral-" + precision;
    if (c.sse) return (c.S == 4) ? "CPU-4State-SSE-Double" : "CPU-SSE-Double";
    // the 4-state factory does not offer DYNAMIC scaling, so the preference selects the general class
    return (c.S == 4 && c.scaling != DYNAMIC) ? "CPU-4State-" + precision : "CPU-" + precision;
}

// One instance of a case and everything it computes
struct Run {
    int instance = -1;
    std::vector<int> codes;            // every return code, in order
    std::vector<double> logL;          // every root log likelihood
    std::vector<double> gradient, derivatives, sums;
    int call(int code) { codes.push_back(code); return code; }
};

class Case {
public:
    explicit Case(const Config& c) : c(c), size(c.S * c.P * c.C), matrixSize(c.S * c.S * c.C) {
        std::mt19937 rng(1000 * c.S + 10 * c.P + c.C);
        tree = randomTree(rng, N);
        randomReversible(rng, c.S, pi, evec, ivec, eval, q);
        rates.resize(c.C);
        weights.resize(c.C);
        for (int k = 0; k < c.C; ++k) {
            rates[k] = (c.C == 1) ? 1.0 : 0.25 + 1.5 * k / (c.C - 1);
            weights[k] = (k + 1.0) / (c.C * (c.C + 1) / 2.0);
        }
        patternWeights.resize(c.P);
        for (auto& w : patternWeights) w = std::uniform_int_distribution<int>(1, 3)(rng);
        for (int tip = 0; tip < kCompact; ++tip) { // states, about one in ten missing
            std::vector<int> s(c.P);
            for (auto& x : s) {
                x = std::uniform_int_distribution<int>(0, c.S - 1)(rng);
                if (std::uniform_int_distribution<int>(0, 9)(rng) == 0) x = c.S;
            }
            states.push_back(s);
        }
        tipPartials.resize(c.S * c.P);
        for (auto& x : tipPartials) x = std::uniform_real_distribution<double>(0.1, 1.0)(rng);
        qk.resize(matrixSize);
        for (int k = 0; k < c.C; ++k)
            for (int i = 0; i < c.S * c.S; ++i) qk[k * c.S * c.S + i] = q[i] * rates[k];
        // the operations, children before parents; a degree-2 node has no second child
        std::function<void(int)> visit = [&](int n) {
            if (tree.children[n].empty()) return;
            preorder.push_back(n);
            for (int child : tree.children[n]) visit(child);
            const int c1 = tree.children[n][0];
            const int c2 = (tree.children[n].size() > 1) ? tree.children[n][1] : BEAGLE_OP_NONE;
            const int w = (c.scaling == DYNAMIC || (scaled() && c2 != BEAGLE_OP_NONE)) ? n - N : BEAGLE_OP_NONE;
            post.push_back({n, w, w, c1, c1, c2, c2});
            if (c2 != BEAGLE_OP_NONE) scaledNodes.push_back(n - N);
        };
        visit(tree.root);
    }

    // false if no implementation matches
    bool run() {
        current = expectedName(c) + " S=" + std::to_string(c.S) + " P=" + std::to_string(c.P) + " C=" +
                  std::to_string(c.C) + " " + scalingName[c.scaling] + " threads=" + std::to_string(c.threads);
        const int failuresBefore = failures;
        Run r, g;
        r.instance = create(kPartials - kCompact, kMatrices, kScales);
        if (r.instance < 0) return false;
        g.instance = create(M - kCompact, M, scaled() ? N + 1 : 1);
        check(g.instance >= 0, "could not create the instance to grow");
        if (g.instance < 0) { beagleFinalizeInstance(r.instance); return true; }

        for (Run* x : {&r, &g}) setUp(*x);
        for (Run* x : {&r, &g}) postOrder(*x);
        snapshot(g);

        grow(g, M + 1, M + 1, (scaled() ? N + 1 : 1) + 1, "by one");
        grow(g, 5, 3, 1, "to smaller counts");
        grow(g, 0, 0, 0, "(0, 0, 0)");
        grow(g, 2 * M, 31, N + 2, "to 31 matrices");
        grow(g, 2 * M, 32, N + 2, "to 32 matrices");
        grow(g, 2 * M, 33, N + 2, "to 33 matrices");
        for (Run* x : {&r, &g}) bottomPreOrder(*x);
        snapshot(g);

        grow(g, 3 * M, 64, N + 2, "to 64 matrices");
        grow(g, 3 * M, 65, N + 2, "to 65 matrices");
        for (Run* x : {&r, &g}) gradients(*x);
        snapshot(g);

        grow(g, kPartials, kMatrices, kScales, "to the final counts");
        for (Run* x : {&r, &g}) newBuffers(*x);
        snapshot(g);

        compare(r, g);
        beagleFinalizeInstance(r.instance);
        beagleFinalizeInstance(g.instance);
        printf("%s %s\n", failures == failuresBefore ? "ok  " : "FAIL", current.c_str());
        return true;
    }

    // the contract checks on one instance
    void contract() {
        current = "contract, " + expectedName(c);
        Run g;
        g.instance = create(M - kCompact, M, N + 1);
        check(g.instance >= 0, "could not create an instance");
        if (g.instance < 0) return;
        setUp(g);
        postOrder(g);
        snapshot(g);
        const double logL = g.logL.back();
        const int out = BEAGLE_ERROR_OUT_OF_RANGE;
        check(beagleEnsureBufferCounts(g.instance, -1, 0, 0) == out, "negative partials count");
        check(beagleEnsureBufferCounts(g.instance, 0, -1, 0) == out, "negative matrix count");
        check(beagleEnsureBufferCounts(g.instance, 0, 0, -1) == out, "negative scale count");
        check(beagleEnsureBufferCounts(g.instance, INT_MAX, 0, 0) == out, "partials + compact count overflows");
        check(beagleEnsureBufferCounts(g.instance, INT_MAX - kCompact, M + 1, N + 2) == out,
              "operation arrays overflow");
        // nothing changed: the new indices are still out of range and the results are the same
        std::vector<double> p(size);
        check(beagleGetPartials(g.instance, M, BEAGLE_OP_NONE, p.data()) == out, "a partials index was added");
        g.codes.clear();
        postOrder(g);
        check(g.logL.back() == logL, "root log likelihood changed");
        unchanged(g, "after the rejected counts");
        check(beagleEnsureBufferCounts(g.instance, M - kCompact + 1, M, N + 1) == BEAGLE_SUCCESS, "growth afterwards");
        beagleFinalizeInstance(g.instance);
        check(beagleEnsureBufferCounts(g.instance, 0, 0, 0) == BEAGLE_ERROR_UNINITIALIZED_INSTANCE,
              "finalized instance");
    }

private:
    const Config c;
    const int size, matrixSize;
    Tree tree;
    std::vector<double> pi, evec, ivec, eval, q, qk, rates, weights, patternWeights, tipPartials;
    std::vector<std::vector<int>> states;
    std::vector<BeagleOperation> post;
    std::vector<int> preorder, scaledNodes;
    std::map<int, std::vector<double>> partials, matrices; // G's snapshot

    bool scaled() const { return c.scaling == MANUAL || c.scaling == DYNAMIC; }
    int preScale() const { return (c.scaling == DYNAMIC) ? kScalePre : BEAGLE_OP_NONE; }
    int preCumulative() const { return (c.scaling == DYNAMIC) ? kCumulative : BEAGLE_OP_NONE; }

    int create(int partialsCount, int matrixCount, int scaleCount) {
        long requirements = BEAGLE_FLAG_PROCESSOR_CPU | BEAGLE_FLAG_PREORDER_TRANSPOSE_AUTO |
                            (c.single ? BEAGLE_FLAG_PRECISION_SINGLE : BEAGLE_FLAG_PRECISION_DOUBLE) |
                            (c.sse ? BEAGLE_FLAG_VECTOR_SSE : BEAGLE_FLAG_VECTOR_NONE) |
                            (c.spectral ? BEAGLE_FLAG_SPECTRAL_REPRESENTATION : 0) |
                            (c.threads > 1 ? BEAGLE_FLAG_THREADING_CPP : 0);
        long preferences = (c.threads > 1) ? 0 : BEAGLE_FLAG_THREADING_NONE;
        switch (c.scaling) {
            case NONE: case MANUAL: preferences |= BEAGLE_FLAG_SCALING_MANUAL; break;
            case DYNAMIC: preferences |= BEAGLE_FLAG_SCALING_DYNAMIC; break; // the spectral factories omit it
            case ALWAYS: requirements |= BEAGLE_FLAG_SCALING_ALWAYS; break;
            case AUTO: requirements |= BEAGLE_FLAG_SCALING_AUTO; break;
        }
        BeagleInstanceDetails details;
        const int instance = beagleCreateInstance(N, partialsCount, kCompact, c.S, c.P, 1, matrixCount, c.C,
                                                  scaleCount, NULL, 0, preferences, requirements, &details);
        if (instance < 0) return instance;
        if (expectedName(c) != details.implName) {
            check(false, std::string("created ") + details.implName);
            beagleFinalizeInstance(instance);
            return -1;
        }
        if (c.threads > 1) check(beagleSetCPUThreadCount(instance, c.threads) == BEAGLE_SUCCESS, "thread count");
        return instance;
    }

    void setUp(Run& x) {
        const int i = x.instance;
        x.call(beagleSetStateFrequencies(i, 0, pi.data()));
        x.call(beagleSetEigenDecomposition(i, 0, evec.data(), ivec.data(), eval.data()));
        x.call(beagleSetCategoryRates(i, rates.data()));
        x.call(beagleSetCategoryWeights(i, 0, weights.data()));
        x.call(beagleSetPatternWeights(i, patternWeights.data()));
        for (int tip = 0; tip < kCompact; ++tip) x.call(beagleSetTipStates(i, tip, states[tip].data()));
        x.call(beagleSetTipPartials(i, N - 1, tipPartials.data()));
    }

    // post-order (none: from the buffers as they are), then the root log likelihood; operations may be repeated (see
    // newBuffers)
    void rootLogLikelihood(Run& x, const std::vector<BeagleOperation>& operations, const std::vector<int>& scales,
                           const std::vector<int>& internal, int root, int cumulative) {
        const int i = x.instance;
        if (c.scaling == MANUAL) x.call(beagleResetScaleFactors(i, cumulative)); // DYNAMIC keeps it up to date
        if (!operations.empty()) {
            if (c.scaling == DYNAMIC) x.call(beagleResetScaleFactors(i, cumulative));
            x.call(beagleUpdatePartials(i, operations.data(), (int) operations.size(),
                                        (c.scaling == DYNAMIC) ? cumulative : BEAGLE_OP_NONE));
        }
        if (c.scaling == MANUAL) {
            x.call(beagleAccumulateScaleFactors(i, scales.data(), (int) scales.size(), cumulative));
        } else if (c.scaling == AUTO) {
            x.call(beagleAccumulateScaleFactors(i, internal.data(), (int) internal.size(), BEAGLE_OP_NONE));
        }
        const int weightsIndex = 0, frequencies = 0, scale = scaled() ? cumulative : BEAGLE_OP_NONE;
        double logL = 0.0;
        x.call(beagleCalculateRootLogLikelihoods(i, &root, &weightsIndex, &frequencies, &scale, 1, &logL));
        x.logL.push_back(logL);
    }

    std::vector<int> internalNodes() const {
        std::vector<int> internal;
        for (const BeagleOperation& op : post) if (op.child2Partials != BEAGLE_OP_NONE) internal.push_back(op.destinationPartials);
        return internal;
    }

    void postOrder(Run& x) {
        std::vector<int> branches;
        std::vector<double> lengths;
        for (int n = 0; n < M; ++n) {
            if (n == tree.root) continue;
            branches.push_back(n);
            lengths.push_back(tree.length[n]);
        }
        x.call(beagleUpdateTransitionMatrices(x.instance, 0, branches.data(), NULL, NULL, lengths.data(),
                                              (int) branches.size()));
        rootLogLikelihood(x, post, scaledNodes, internalNodes(), tree.root, kCumulative);
    }

    void bottomPreOrder(Run& x) {
        const int i = x.instance, root = tree.root, frequencies = 0, rootPre = bottom(root);
        x.call(beagleSetRootPrePartials(i, &rootPre, &frequencies, 1));
        std::vector<BeagleOperation> ops;
        for (int n : preorder) {
            const std::vector<int>& kids = tree.children[n];
            for (size_t k = 0; k < kids.size(); ++k) {
                const int child = kids[k];
                const int sibling = (kids.size() > 1) ? kids[1 - k] : BEAGLE_OP_NONE;
                ops.push_back({bottom(child), preScale(), preScale(), bottom(n), child, sibling, sibling});
            }
        }
        x.call(beagleUpdatePrePartials_v5(i, ops.data(), (int) ops.size(), preCumulative(), BEAGLE_PARTIALS_BOTTOM));
    }

    // the root log likelihood again from the buffers as they are (old scale factors included), then top pre-order,
    // the adjoint gradient and the edge derivatives with the infinitesimal matrix, first written now
    void gradients(Run& x) {
        const int i = x.instance, root = tree.root, frequencies = 0, rootPre = top(root);
        rootLogLikelihood(x, std::vector<BeagleOperation>(), scaledNodes, internalNodes(), root, kCumulative);
        x.call(beagleSetRootPrePartials(i, &rootPre, &frequencies, 1));
        std::vector<BeagleOperation> ops;
        for (int n : preorder) {
            const std::vector<int>& kids = tree.children[n];
            for (size_t k = 0; k < kids.size(); ++k) {
                const int child = kids[k];
                const int sibling = (kids.size() > 1) ? kids[1 - k] : BEAGLE_OP_NONE;
                ops.push_back({top(child), preScale(), preScale(), top(n), (n == root) ? BEAGLE_OP_NONE : n,
                               sibling, sibling});
            }
        }
        x.call(beagleUpdatePrePartials_v5(i, ops.data(), (int) ops.size(), preCumulative(), BEAGLE_PARTIALS_TOP));

        std::vector<BeagleBranchOperation> branches;
        for (int n = 0; n < M; ++n) {
            if (n == root) continue;
            BeagleBranchOperation b;
            b.postOrderPartials = n;
            b.preOrderPartials = top(n);
            b.branchTransitionMatrix = n;
            b.resultSegment = 0;
            branches.push_back(b);
        }
        // S x S, but single precision copies S x S x C values out (BeagleCPUImpl::calcAdjointCrossProducts)
        x.gradient.assign(c.S * c.S * c.C, 0.0);
        x.call(beagleCalculateAdjointDerivative(i, branches.data(), 0, 0, root, 0, (int) branches.size(),
                                                x.gradient.data(), NULL));

        x.call(beagleSetDifferentialMatrix(i, kQ, qk.data()));
        std::vector<int> posts, pres, qs;
        for (int n = N - 1; n < M; ++n) { // tip N - 1 and the internal nodes hold partials
            if (n == root) continue;
            posts.push_back(n);
            pres.push_back(bottom(n));
            qs.push_back(kQ);
        }
        const int weightsIndex = 0;
        x.derivatives.assign(posts.size() * c.P, 0.0);
        x.sums.assign(posts.size(), 0.0);
        x.call(beagleCalculateEdgeDerivatives(i, posts.data(), pres.data(), qs.data(), &weightsIndex,
                                              (int) posts.size(), x.derivatives.data(), x.sums.data(), NULL));
    }

    // two operations on new buffers and matrices that read old post-order buffers without recomputing them: a
    // degree-2 node above the root's first child and a parent of it and the root's second child. They are repeated,
    // so that the list is longer than the first buffer count times four pattern partitions: with auto-partitioning
    // (four threads), operation arrays that did not grow would overflow.
    void newBuffers(Run& x) {
        const int c1 = tree.children[tree.root][0], c2 = tree.children[tree.root][1];
        const int matrices[2] = {kMatrixNew0, kMatrixNew1};
        const double lengths[2] = {0.11, 0.17};
        x.call(beagleUpdateTransitionMatrices(x.instance, 0, matrices, NULL, NULL, lengths, 2));
        const int w0 = (c.scaling == DYNAMIC) ? kScaleNew0 : BEAGLE_OP_NONE, w1 = scaled() ? kScaleNew1 : BEAGLE_OP_NONE;
        std::vector<BeagleOperation> ops;
        while ((int) ops.size() <= 4 * M + 8) {
            ops.push_back({kNew0, w0, w0, c1, kMatrixNew0, BEAGLE_OP_NONE, BEAGLE_OP_NONE});
            ops.push_back({kNew1, w1, w1, kNew0, kMatrixNew1, c2, c2});
        }
        std::vector<int> scales = scaledNodes, internal = internalNodes();
        scales.push_back(kScaleNew1);
        internal.push_back(kNew1);
        rootLogLikelihood(x, ops, scales, internal, kNew1, kCumulativeNew);
    }

    std::vector<double> readPartials(int instance, int index) {
        std::vector<double> p(size);
        check(beagleGetPartials(instance, index, BEAGLE_OP_NONE, p.data()) == BEAGLE_SUCCESS,
              "getPartials " + std::to_string(index));
        return p;
    }

    std::vector<double> readMatrix(int instance, int index) {
        std::vector<double> m(matrixSize);
        check(beagleGetTransitionMatrix(instance, index, m.data()) == BEAGLE_SUCCESS,
              "getTransitionMatrix " + std::to_string(index));
        return m;
    }

    // the buffers each stage writes
    std::vector<int> partialsAfter(int stage) const {
        std::vector<int> indices;
        for (int n = N - 1; n < M; ++n) indices.push_back(n);
        if (stage >= 2) for (int n = 0; n < M; ++n) indices.push_back(bottom(n));
        if (stage >= 3) for (int n = 0; n < M; ++n) indices.push_back(top(n));
        if (stage >= 4) { indices.push_back(kNew0); indices.push_back(kNew1); }
        return indices;
    }

    std::vector<int> matricesAfter(int stage) const {
        std::vector<int> indices;
        for (int n = 0; n < M; ++n) if (n != tree.root) indices.push_back(n);
        if (stage >= 3) indices.push_back(kQ);
        if (stage >= 4) { indices.push_back(kMatrixNew0); indices.push_back(kMatrixNew1); }
        return indices;
    }

    int stage = 0;

    void snapshot(Run& g) {
        ++stage;
        partials.clear();
        matrices.clear();
        for (int index : partialsAfter(stage)) partials[index] = readPartials(g.instance, index);
        for (int index : matricesAfter(stage)) matrices[index] = readMatrix(g.instance, index);
    }

    void unchanged(Run& g, const std::string& when) {
        for (const auto& p : partials)
            check(same(readPartials(g.instance, p.first), p.second),
                  "partials " + std::to_string(p.first) + " changed " + when);
        for (const auto& m : matrices)
            check(same(readMatrix(g.instance, m.first), m.second),
                  "matrix " + std::to_string(m.first) + " changed " + when);
    }

    void grow(Run& g, int partialsSpace, int matrixCount, int scaleCount, const std::string& how) {
        const int partialsCount = (partialsSpace > kCompact) ? partialsSpace - kCompact : 0;
        check(g.call(beagleEnsureBufferCounts(g.instance, partialsCount, matrixCount, scaleCount)) == BEAGLE_SUCCESS,
              "growth " + how);
        unchanged(g, "after growth " + how);
    }

    void compare(Run& r, Run& g) {
        // every call succeeded, R's and G's (which also made the growth calls)
        int errors = 0;
        for (const Run* x : {&r, &g})
            for (int code : x->codes) if (code != BEAGLE_SUCCESS) ++errors;
        check(errors == 0, std::to_string(errors) + " calls failed");
        check(r.logL.size() == g.logL.size() && same(r.logL, g.logL), "root log likelihoods differ");
        for (size_t k = 0; k < r.logL.size() && k < g.logL.size(); ++k)
            if (r.logL[k] != g.logL[k]) printf("     log likelihood %zu: R %.17g, G %.17g\n", k, r.logL[k], g.logL[k]);
        check(same(r.gradient, g.gradient), "adjoint gradients differ");
        check(same(r.derivatives, g.derivatives) && same(r.sums, g.sums), "edge derivatives differ");
        for (int index : partialsAfter(4))
            check(same(readPartials(r.instance, index), readPartials(g.instance, index)),
                  "partials " + std::to_string(index) + " differ");
        for (int index : matricesAfter(4))
            check(same(readMatrix(r.instance, index), readMatrix(g.instance, index)),
                  "matrix " + std::to_string(index) + " differs");
        for (double l : r.logL) check(std::isfinite(l), "root log likelihood is not finite");
        // after growth, the old buffers and scale factors still give the first root log likelihood (DYNAMIC's
        // cumulative scale buffer also takes the pre-order rescaling)
        if (c.scaling != DYNAMIC) check(g.logL.size() > 1 && g.logL[1] == g.logL[0], "old buffers changed by growth");
    }
};

// A 4-taxon Jukes-Cantor log likelihood, without degree-2 nodes (which GPU instances do not support); 0 if a call fails
double gpuLogLikelihood(int instance) {
    bool ok = true;
    const double evec[16] = {1, 1, 1, 1, 1, -1, 1, 1, 1, 0, -2, 1, 1, 0, 0, -3};
    double ivec[16];
    for (int i = 0; i < 4; i++) {
        double norm = 0;
        for (int j = 0; j < 4; j++) norm += evec[j * 4 + i] * evec[j * 4 + i];
        for (int j = 0; j < 4; j++) ivec[i * 4 + j] = evec[j * 4 + i] / norm;
    }
    const double eval[4] = {0, -4.0 / 3, -4.0 / 3, -4.0 / 3}, frequencies[4] = {0.25, 0.25, 0.25, 0.25};
    const double rate = 1.0, weight = 1.0, patternWeights[2] = {1, 2};
    ok &= beagleSetEigenDecomposition(instance, 0, evec, ivec, eval) == BEAGLE_SUCCESS;
    ok &= beagleSetStateFrequencies(instance, 0, frequencies) == BEAGLE_SUCCESS;
    ok &= beagleSetCategoryRates(instance, &rate) == BEAGLE_SUCCESS;
    ok &= beagleSetCategoryWeights(instance, 0, &weight) == BEAGLE_SUCCESS;
    ok &= beagleSetPatternWeights(instance, patternWeights) == BEAGLE_SUCCESS;
    for (int tip = 0; tip < 4; ++tip) {
        const int states[2] = {tip % 4, (tip * 3) % 4};
        ok &= beagleSetTipStates(instance, tip, states) == BEAGLE_SUCCESS;
    }
    const int branches[6] = {0, 1, 2, 3, 4, 5};
    const double lengths[6] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
    ok &= beagleUpdateTransitionMatrices(instance, 0, branches, NULL, NULL, lengths, 6) == BEAGLE_SUCCESS;
    const BeagleOperation ops[3] = {{4, BEAGLE_OP_NONE, BEAGLE_OP_NONE, 0, 0, 1, 1},
                                    {5, BEAGLE_OP_NONE, BEAGLE_OP_NONE, 2, 2, 3, 3},
                                    {6, BEAGLE_OP_NONE, BEAGLE_OP_NONE, 4, 4, 5, 5}};
    ok &= beagleUpdatePartials(instance, ops, 3, BEAGLE_OP_NONE) == BEAGLE_SUCCESS;
    const int root = 6, zero = 0, none = BEAGLE_OP_NONE;
    double logL = 0.0;
    ok &= beagleCalculateRootLogLikelihoods(instance, &root, &zero, &zero, &none, 1, &logL) == BEAGLE_SUCCESS;
    return ok ? logL : 0.0;
}

void gpuCheck(bool spectral) {
    current = spectral ? "GPU spectral" : "GPU";
    BeagleInstanceDetails details;
    const int instance = beagleCreateInstance(4, 3, 4, 4, 2, 1, 6, 1, 0, NULL, 0, 0, BEAGLE_FLAG_PROCESSOR_GPU |
                                              (spectral ? BEAGLE_FLAG_SPECTRAL_REPRESENTATION : 0), &details);
    if (instance < 0) {
        printf("%s: no GPU resource, not checked\n", current.c_str());
        return;
    }
    current = std::string("GPU ") + details.implName + " on " + details.resourceName;
    const double before = gpuLogLikelihood(instance);
    check(beagleEnsureBufferCounts(instance, 0, 0, 0) == BEAGLE_ERROR_NO_IMPLEMENTATION, "probe");
    check(beagleEnsureBufferCounts(instance, 10, 10, 10) == BEAGLE_ERROR_NO_IMPLEMENTATION, "growth");
    const double after = gpuLogLikelihood(instance);
    check(before < 0.0 && after == before, "the instance changed or failed");
    printf("%s %s: NO_IMPLEMENTATION, log likelihood %.10f before and after\n", after == before ? "ok  " : "FAIL",
           current.c_str(), after);
    beagleFinalizeInstance(instance);
}

// time growth at 10,000 buffers, 17 states, 1 pattern, 1 category (information only)
void timing() {
    for (bool spectral : {false, true}) {
        BeagleInstanceDetails details;
        const int count = 10000, compact = 7;
        const int instance = beagleCreateInstance(8, count - compact, compact, 17, 1, 1, count, 1, count, NULL, 0,
                                                  BEAGLE_FLAG_SCALING_MANUAL | BEAGLE_FLAG_THREADING_NONE,
                                                  BEAGLE_FLAG_PROCESSOR_CPU | BEAGLE_FLAG_PRECISION_DOUBLE |
                                                  (spectral ? BEAGLE_FLAG_SPECTRAL_REPRESENTATION : 0), &details);
        if (instance < 0) continue;
        struct Step { const char* what; int partials, matrices, scale; };
        // one buffer of each kind; then one BEAST growth event of the rabies benchmark size (304 augmented nodes
        // with pre-order: 912 partials, 608 matrices, 304 scale buffers); then a factor of 1.5
        const Step steps[3] = {{"+1", count + 1, count + 1, count + 1},
                               {"+912/608/304", count + 1 + 912, count + 1 + 608, count + 1 + 304},
                               {"x1.5", 3 * count / 2 + 1000, 3 * count / 2 + 1000, 3 * count / 2 + 1000}};
        for (const Step& s : steps) {
            const auto start = std::chrono::steady_clock::now();
            const int code = beagleEnsureBufferCounts(instance, s.partials - compact, s.matrices, s.scale);
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            printf("timing %-24s at 10,000 buffers, S=17: grow %-13s %.3f ms%s\n", details.implName, s.what, ms,
                   code == BEAGLE_SUCCESS ? "" : " (failed)");
            current = "timing";
            check(code == BEAGLE_SUCCESS, "timed growth failed");
        }
        beagleFinalizeInstance(instance);
    }
}

} // namespace

int main(int argc, char** argv) {
    const std::string only = (argc > 1) ? argv[1] : ""; // run only the cases whose label contains this
    setvbuf(stdout, NULL, _IONBF, 0); // a crash still shows the cases before it
    current = "invalid instance";
    check(beagleEnsureBufferCounts(0, 0, 0, 0) == BEAGLE_ERROR_UNINITIALIZED_INSTANCE, "before any instance");
    check(beagleEnsureBufferCounts(-1, 0, 0, 0) == BEAGLE_ERROR_UNINITIALIZED_INSTANCE, "negative instance");
    check(beagleEnsureBufferCounts(1000000, 0, 0, 0) == BEAGLE_ERROR_UNINITIALIZED_INSTANCE, "large instance");

    int compared = 0, skipped = 0;
    struct Resource { bool spectral, sse, single; };
    const Resource resources[] = {{false, false, false}, {false, false, true}, {true, false, false},
                                  {true, false, true}, {false, true, false}, {true, true, false}};
    for (const Resource& res : resources)
    for (int S : {4, 17, 20})
    for (int patterns : {1, 64})
    for (int C : {1, 4})
    for (Scaling scaling : {NONE, MANUAL, DYNAMIC, ALWAYS, AUTO})
    for (int threads : {1, 4}) {
        if (threads > 1 && patterns < 64) continue;
        // with 4 states, BEAGLE partitions patterns for threads only from 768 patterns (fewer than 16 cores) or 256
        const int P = (threads > 1 && S == 4) ? 1536 : patterns;
        if ((scaling == DYNAMIC || scaling == AUTO) && res.spectral) continue; // see the header
        if (scaling != NONE && scaling != MANUAL && threads > 1) continue;
        const Config config{res.spectral, res.sse, res.single, S, P, C, scaling, threads};
        if (!only.empty() && (expectedName(config) + " S=" + std::to_string(S) + " P=" + std::to_string(P) + " C=" +
                              std::to_string(C) + " " + scalingName[scaling] + " threads=" +
                              std::to_string(threads)).find(only) == std::string::npos) continue;
        Case test(config);
        if (test.run()) {
            ++compared;
        } else {
            ++skipped;
            printf("skip %s S=%d P=%d C=%d %s threads=%d: no implementation\n", expectedName(config).c_str(), S, P, C,
                   scalingName[scaling], threads);
        }
    }
    for (const Resource& res : resources) {
        const Config config{res.spectral, res.sse, res.single, 17, 1, 1, MANUAL, 1};
        Case(config).contract();
    }
    gpuCheck(false);
    gpuCheck(true);
    timing();
    printf("buffergrowthtest: %d cases, %d skipped, %d failures\n", compared, skipped, failures);
    return (failures == 0 && compared > 0) ? 0 : 1;
}
