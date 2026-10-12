/*
 * Copyright 2026 Phylogenetic Likelihood Working Group
 * This file is part of BEAGLE.
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 *
 * buffergrowthtest: beagleEnsureBufferCounts on the CPU implementations and, with --gpu, on a GPU resource.
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
 * Contract checks: negative and overflowing counts return BEAGLE_ERROR_OUT_OF_RANGE and leave the instance as it
 * was; invalid instances return BEAGLE_ERROR_UNINITIALIZED_INSTANCE. Finally it reports the time of growth calls at
 * 10,000 buffers with 17 states (information only).
 *
 * GPU axis: buffergrowthtest --gpu <resource> [--precision single|double] runs the GPU cases instead, on that resource
 * only, and exits 77 (skipped) if it is not a GPU resource, or has no double precision when only that is asked for.
 * Every instance requires the resource's framework (CUDA, OpenCL or TinyGPU) and checks its implementation name and
 * flags. The flow keeps to what GPU instances implement: bifurcating trees; root pre-order partials through
 * setPartials, so below the partials count; BOTTOM pre-order; getPartials of bound buffers only; edge derivatives and
 * cross products; scale buffers under DYNAMIC on CUDA only. G is created for a first tree, and grown by a probe, by
 * one, to smaller counts, by (0, 0, 0) and by several, then to counts that also hold a second tree, on new buffers,
 * with tips and the first root as leaves. R is created at the final counts. After every growth G's old partials and
 * matrices are byte-identical and the first root log likelihood is unchanged; at the end G's log likelihoods (also by
 * partition), partials, matrices, edge derivatives (more than every buffer count before the final one) and cross
 * products equal R's (memcmp). In the multi-grid cases without scaling, the second tree's post-order is repeated in one
 * call, more operations than G's first internal buffer count. Cases: 4 states (multi-grid) and 17
 * (PREORDER_TRANSPOSE_AUTO, the second pre-order longer than the first, so the transpose scratch grows after a growth)
 * with no scaling, MANUAL, ALWAYS and AUTO; 1,000 patterns, multi-grid, without partitions and with 4 contiguous or
 * reordered ones set before the growth, with root log likelihoods by partition under mixed scaling (the unscaled
 * partitions read the zero scale buffer, which moves); 4 reordered partitions without scaling (a multi-grid post-order
 * with partitions); 1,000 patterns without scaling and 160 post-order operations (a multi-grid operation buffer left at
 * its first size would overflow by far more than its allocation's rounding); no compact tips with 4 reordered
 * partitions, from fewer buffers than 2 x the tips (the last tip's partials then live past the partials count, and the
 * partials pool first moves from there); late tips (16 tips, 2 compact, 4 partials tips set only after the partials
 * pool moved, from counts below the reservation for 2 x the partials tips and grown past it), with and without
 * reordered partitions after them; scale growth from 0 scale buffers; PARALLELOPS_STREAMS; DYNAMIC (off CUDA only
 * multi-grid, without scale buffers, where more of them are BEAGLE_ERROR_NO_IMPLEMENTATION: the per-operation DYNAMIC
 * kernels read scale buffer -1 when there are none, and crash; on CUDA, from 0 scale buffers G grows them before its
 * first post-order). Under AUTO, and without scaling with partitions, only equality is checked: on Apple OpenCL the
 * second tree's root log likelihood is -inf, in a fresh instance too (at 17 states under AUTO; with partitions also
 * before G2b, in the multi-grid post-order). Single and double precision (double is skipped if the resource has none);
 * a creation that fails other than with BEAGLE_ERROR_NO_IMPLEMENTATION is a failure. Spectral instances (if the
 * resource has BEAGLE_FLAG_SPECTRAL_REPRESENTATION, else a skip line per precision), at 4 and 17 states with two eigen decompositions, matrices on both, MANUAL scaling and a TOP
 * pre-order: G grows by a probe and by one before its dense matrix pool exists (a dense read of the new index is
 * BEAGLE_ERROR_OUT_OF_RANGE), then setDifferentialMatrix makes the pool, and G grows to smaller counts, by (0, 0, 0),
 * by several and to the final counts (the second differential matrix written in between; a dense read of a new index,
 * never written, is BEAGLE_ERROR_OUT_OF_RANGE; then a dense write past the dense pool's first size, and a post-order
 * on a new matrix index never updated, whose distance must be 0 as in R). After every growth, tree A's post-order on its matrices as computed
 * before (eigen decomposition 1 included) gives the first root log likelihood exactly, and the old partials and dense
 * matrices are byte-identical. Then tree A's pre-order, edge derivatives on the dense differential matrices, cross
 * products and adjoint gradients on its old matrices, and tree B on new ones. Log likelihoods, partials, dense
 * matrices, edge derivatives and cross products equal R's (memcmp); the adjoint gradients (float atomics) are within 4
 * times the largest difference between R and a second fresh instance F, and at least 1e-6 of the largest gradient.
 * Then the contract checks (at 4 and 17 states, the latter with transpose scratch past the matrix count). If the
 * library prints BEAGLE_DEBUG_MEMORY lines (a
 * -DBEAGLE_DEBUG_MEMORY=ON build): every instance must finalize with nothing left; a spectral growth before the dense
 * pool exists must not make it (the first dense write reports it); 10 growth cycles, after each of
 * which G's buffer, sub-buffer and byte counters equal those of a fresh instance at the same counts that made the same
 * calls; BEAGLE_DEBUG_MEMORY_FAIL, which the test sets itself: 1 (the host-staged fallback, then results equal R's),
 * 1,2 (the fallback fails too) and 4, 5, 6 and 3,4 (a queue, or a later pool and its fallback, after the first pools
 * moved), and 4 from 0 scale buffers (the new scale pool must be freed, also when no growth follows):
 * BEAGLE_ERROR_OUT_OF_MEMORY with the old contents and counts, results equal to a fresh instance at the old counts, and
 * a later growth that succeeds; 1,2,3 (the restore fails too): BEAGLE_ERROR_OUT_OF_MEMORY, the probe too, and a clean
 * finalize; and BEAGLE_DEBUG_MEMORY_SCRIBBLE, which fills the memory growth allocates with garbage, as CUDA and TinyGPU
 * may leave it, for the cases that read the buffer of zeroes and the spectral cases (the new distance pool, also
 * under 4 and 4,5 below). Spectral, with the dense pool made before the growth: 10 growth cycles; and the distance
 * pool's allocation (4: the fallback; 4,5: OUT_OF_MEMORY and the old pool again, uninitialised, which the host mirror
 * fills; 4,5,6: lost), the dense pool's (2; 2,3; 2,3,4) and the derivative queue's (5: after the distance pool moved,
 * whose larger capacity the growth afterwards keeps) forced to fail, the allocation checked by its size, with the
 * outcomes above; tree A's old matrices still give the first root log likelihood exactly.
 */

#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <random>
#include <string>
#include <vector>

#ifdef _WIN32
#include <io.h> // dup, dup2, close
static int setenv(const char* name, const char* value, int) { return _putenv_s(name, value); }
static int unsetenv(const char* name) { return _putenv_s(name, ""); }
#else
#include <unistd.h>
#endif

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

    // the contract checks on one instance; false, as run(), when there is no such implementation here
    bool contract() {
        current = "contract, " + expectedName(c);
        Run g;
        g.instance = create(M - kCompact, M, N + 1);
        if (g.instance < 0) return false;
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
        return true;
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

// The GPU resource of --gpu, which every GPU instance requires, by number and by framework
struct GpuResource {
    int number = -1;
    long framework = 0;  // BEAGLE_FLAG_FRAMEWORK_CUDA, _OPENCL or _TINYGPU
    std::string prefix;  // how its implementation names start: CUDA, OpenCL or TinyGPU
    bool doublePrecision = false;
    unsigned long align = 1; // the offset alignment a debug-memory library reports: BEAGLE_DEBUG_OPENCL_ALIGN's, or
                             // the base alignment of an OpenCL vendor other than Apple and NVIDIA
};
GpuResource gpu;
bool debugMemory = false; // the library prints BEAGLE_DEBUG_MEMORY lines: a -DBEAGLE_DEBUG_MEMORY=ON build

// What the library writes to stderr (its BEAGLE_DEBUG_MEMORY lines) while it exists; finish() passes it on to stderr
class StderrCapture {
public:
    StderrCapture() : file(tmpfile()), saved(-1) {
        fflush(stderr);
        if (file != NULL && (saved = dup(2)) >= 0) dup2(fileno(file), 2);
    }
    ~StderrCapture() { finish(); }
    std::string finish() {
        std::string text;
        if (file == NULL) return text;
        fflush(stderr);
        if (saved >= 0) {
            dup2(saved, 2);
            close(saved);
        }
        rewind(file);
        char buffer[4096];
        for (size_t n; (n = fread(buffer, 1, sizeof(buffer), file)) > 0;) text.append(buffer, n);
        fclose(file);
        file = NULL;
        fputs(text.c_str(), stderr);
        return text;
    }
private:
    FILE* file;
    int saved;
};

// One BEAGLE_DEBUG_MEMORY line: "<event>: <b> buffers, <s> sub-buffers, <n> bytes (peak <p>), <e> errors"
struct MemoryReport {
    std::string event;
    int buffers = -1, subBuffers = -1, errors = -1;
    unsigned long bytes = 0;
    bool sameCounters(const MemoryReport& o) const {
        return buffers == o.buffers && subBuffers == o.subBuffers && bytes == o.bytes && errors == o.errors;
    }
    bool empty() const { return buffers == 0 && subBuffers == 0 && bytes == 0 && errors == 0; }
    std::string str() const {
        return std::to_string(buffers) + " buffers, " + std::to_string(subBuffers) + " sub-buffers, " +
               std::to_string(bytes) + " bytes, " + std::to_string(errors) + " errors";
    }
};

std::vector<MemoryReport> memoryReports(const std::string& text) {
    std::vector<MemoryReport> reports;
    for (size_t at = text.find("BEAGLE_DEBUG_MEMORY "); at != std::string::npos;
         at = text.find("BEAGLE_DEBUG_MEMORY ", at + 1)) {
        char event[64];
        unsigned long peak;
        MemoryReport m;
        if (sscanf(text.c_str() + at,
                   "BEAGLE_DEBUG_MEMORY %63[^:]: %d buffers, %d sub-buffers, %lu bytes (peak %lu), %d errors", event,
                   &m.buffers, &m.subBuffers, &m.bytes, &peak, &m.errors) == 6) {
            m.event = event;
            reports.push_back(m);
        }
    }
    return reports;
}

// the last report of an event, or an empty-named one
MemoryReport lastReport(const std::string& text, const std::string& event) {
    MemoryReport last;
    for (const MemoryReport& m : memoryReports(text)) if (m.event == event) last = m;
    return last;
}

// Finalizes a GPU instance; with the counters, it must leave no buffer, sub-buffer, byte or error
void finalizeGpu(int instance) {
    if (!debugMemory) {
        beagleFinalizeInstance(instance);
        return;
    }
    StderrCapture capture;
    beagleFinalizeInstance(instance);
    const MemoryReport m = lastReport(capture.finish(), "finalize");
    check(m.event == "finalize" && m.empty(), "finalize report: " + (m.event.empty() ? "none" : m.str()));
}

// A random bifurcating tree on leafBuffers. Positions 0..L-1 are the leaves, L..2L-2 the internal nodes (buffers
// internalBase + position - L), children before parents, the root last. Matrices and pre-order buffers are numbered by
// position.
struct GpuTree {
    int leaves = 0;
    std::vector<int> buffer, child; // two children per position, -1 at leaves
    std::vector<double> length;
    int root() const { return 2 * leaves - 2; }
};

GpuTree randomBifurcatingTree(std::mt19937& rng, const std::vector<int>& leafBuffers, int internalBase) {
    GpuTree t;
    t.leaves = (int) leafBuffers.size();
    const int nodes = 2 * t.leaves - 1;
    t.buffer = leafBuffers;
    t.child.assign(2 * nodes, -1);
    std::vector<int> active;
    for (int p = 0; p < t.leaves; ++p) active.push_back(p);
    for (int p = t.leaves; p < nodes; ++p) {
        for (int k = 0; k < 2; ++k) {
            const int a = std::uniform_int_distribution<int>(0, (int) active.size() - 1)(rng);
            t.child[2 * p + k] = active[a];
            active.erase(active.begin() + a);
        }
        active.push_back(p);
        t.buffer.push_back(internalBase + p - t.leaves);
    }
    std::uniform_real_distribution<double> u(0.05, 0.3);
    t.length.assign(nodes, 0.0);
    for (int p = 0; p < t.root(); ++p) t.length[p] = u(rng);
    return t;
}

struct GpuConfig {
    const char* what;     // what the case is for
    bool single;
    int S, P, C;
    int T, K;             // tips, and compact tips (tips 0..K-1 hold states)
    Scaling scaling;
    long preferences;     // PREORDER_TRANSPOSE_AUTO, PARALLELOPS_*: preferences, as TinyGPU's resource lacks the flags
    int late;             // the last partials tips, set only after the partials pool moved
    int partitions;       // pattern partitions (0: none), set before the growth, or after the late tips
    bool reorder;         // partitions that are not contiguous, so that the patterns are reordered
    bool byPartition;     // root log likelihoods by partition, with mixed scaling (MANUAL)
    bool preOrderA;       // pre-order, edge derivatives and cross products on the first tree too
    bool crossProducts;   // which take P x P x 512 values of device scratch
    bool scaleFromZero;   // created without scale buffers
    int repeat = 1;       // tree B's post-order this many times in one call (no scaling), with internal buffers for it
};

struct GpuRun : Run {
    std::vector<double> crossProducts;
};

// Buffer layout. Tree A on the first TA = T - late tips: internal nodes T..T+TA-2, matrices 0..2TA-3, the infinitesimal
// matrix at kQ = 2TA-2, pre-order partials from preA, MANUAL scale buffers 0..TA-2 and the cumulative one TA-1. Tree B
// on every tip and A's root, all on new indices: internal nodes from B0 (the first buffer count), matrices from m0,
// pre-order partials from preB, scale buffers from s0. The first counts hold tree A, the final ones tree B as well.
class GpuCase {
public:
    explicit GpuCase(const GpuConfig& c) : c(c), TA(c.T - c.late), size(c.S * c.P * c.C), matrixSize(c.S * c.S * c.C) {
        std::mt19937 rng(1000 * c.S + 10 * c.P + c.C + 100000 * c.T + 7 * c.late + 3 * c.partitions);
        randomReversible(rng, c.S, pi, evec, ivec, eval, q);
        rates.resize(c.C);
        weights.resize(c.C);
        for (int k = 0; k < c.C; ++k) {
            rates[k] = (c.C == 1) ? 1.0 : 0.25 + 1.5 * k / (c.C - 1);
            weights[k] = (k + 1.0) / (c.C * (c.C + 1) / 2.0);
        }
        patternWeights.resize(c.P);
        for (auto& w : patternWeights) w = std::uniform_int_distribution<int>(1, 3)(rng);
        states.resize(c.T);
        tipPartials.resize(c.T);
        for (int tip = 0; tip < c.T; ++tip) {
            if (tip < c.K) { // states, about one in ten missing
                states[tip].resize(c.P);
                for (auto& x : states[tip]) {
                    x = std::uniform_int_distribution<int>(0, c.S - 1)(rng);
                    if (std::uniform_int_distribution<int>(0, 9)(rng) == 0) x = c.S;
                }
            } else {
                tipPartials[tip].resize(c.S * c.P);
                for (auto& x : tipPartials[tip]) x = std::uniform_real_distribution<double>(0.1, 1.0)(rng);
            }
        }
        qk.resize(matrixSize);
        for (int k = 0; k < c.C; ++k)
            for (int i = 0; i < c.S * c.S; ++i) qk[k * c.S * c.S + i] = q[i] * rates[k];

        kQ = 2 * TA - 2;
        m0 = 2 * TA - 1;
        preA = c.T + TA - 1;
        B0 = c.preOrderA ? preA + 2 * TA - 1 + c.K : preA; // + K: setPartials takes indices below the partials count
        // the reorder swaps the partials tips with internal buffers T..T+T-K-1, whose handles BEAGLE reserves only up to
        // 2(T-K) (with K = 0 they all fit, and the last tip's partials end up past the partials count)
        if (c.reorder && c.late == 0 && c.K > 0) B0 = std::max(B0, 2 * c.T - c.K);
        s0 = (scaleBuffers() && (!c.scaleFromZero || c.scaling == DYNAMIC)) ? TA : 0; // DYNAMIC: see scaleA()
        std::vector<int> leaves;
        for (int tip = 0; tip < TA; ++tip) leaves.push_back(tip);
        treeA = randomBifurcatingTree(rng, leaves, c.T);
        leaves.clear();
        for (int tip = 0; tip < c.T; ++tip) leaves.push_back(tip);
        leaves.push_back(treeA.buffer[treeA.root()]);
        treeB = randomBifurcatingTree(rng, leaves, B0);
        preB = B0 + c.T;
        B1 = std::max(preB + treeB.root() + 1 + c.K + 2, c.T + c.T * c.repeat); // tree B has T post-order operations
        M1 = m0 + treeB.root() + 2;
        S1 = scaleBuffers() ? s0 + c.T + 2 : 0;
        p0 = B0 - c.K;
        p1 = B1 - c.K;
        derivativeCount = (p0 + p1) / 2 + c.K + 1; // more than every buffer count before the final one
    }

    std::string name() const {
        return expectedName() + " S=" + std::to_string(c.S) + " P=" + std::to_string(c.P) + " C=" +
               std::to_string(c.C) + " " + scalingName[c.scaling] + ", " + c.what;
    }

    // false if no implementation matches
    bool run() {
        current = name();
        const int failuresBefore = failures;
        GpuRun r, g;
        r.instance = create(p1, M1, S1);
        if (r.instance < 0) return false;
        g.instance = create(p0, m0, c.scaleFromZero ? 0 : s0);
        check(g.instance >= 0, "could not create the instance to grow");
        if (g.instance < 0) { finalizeGpu(r.instance); return true; }

        for (GpuRun* x : {&r, &g}) begin(*x);
        bool grown = true;
        if (c.scaleFromZero && s0 > 0) grown = grow(g, 0, 0, s0, "from 0 scale buffers"); // DYNAMIC: see scaleA()
        for (GpuRun* x : {&r, &g}) stageA(*x, treeA.root());
        liveA();
        snapshot(g);
        const size_t firstLogL = g.logL.size();

        grown &= grow(g, 0, 0, 0, "probe"); // G uses indices past the old counts only when it has grown
        // without compact tips, the partials pool must first move while the reorder's last tip lies past the partials
        // count, so then the partials grow only by several
        grown &= grow(g, c.K == 0 ? p0 : p0 + 1, m0 + 1, scale(s0 + 1), "by one");
        grown &= grow(g, 5, 3, scale(1), "to smaller counts");
        grown &= grow(g, 0, 0, 0, "(0, 0, 0)");
        if (c.preOrderA) for (GpuRun* x : {&r, &g}) derivatives(*x, treeA, preA, treeA.root(), true);
        grown &= grow(g, (p0 + p1) / 2, (m0 + M1) / 2, scale((s0 + S1) / 2), "by several");
        for (GpuRun* x : {&r, &g}) // from the buffers as they are
            rootLogLikelihood(*x, treeA, std::vector<BeagleOperation>(), scaleA());

        if (c.late > 0 && grown) { // the partials pool has moved
            for (GpuRun* x : {&r, &g}) setTips(*x, TA, c.T);
            for (int tip = std::max(TA, c.K); tip < c.T; ++tip) liveP.push_back(tip);
            if (c.partitions > 0) for (GpuRun* x : {&r, &g}) setPartitions(*x);
            snapshot(g); // the late tips, and every buffer's patterns as reordered
        }
        grown &= grow(g, p1, M1, S1, "to the final counts");
        if (grown) {
            for (GpuRun* x : {&r, &g}) stageB(*x);
            liveB();
            compare(r, g, firstLogL);
        }
        finalizeGpu(r.instance);
        finalizeGpu(g.instance);
        printf("%s %s\n", failures == failuresBefore ? "ok  " : "FAIL", current.c_str());
        return true;
    }

    // Negative and overflowing counts, the instance unchanged, a growth afterwards, a finalized instance; false, as
    // run(), when there is no such implementation here
    bool contract() {
        current = "contract, " + name();
        const int failuresBefore = failures;
        GpuRun g;
        g.instance = create(p0, m0, s0);
        if (g.instance < 0) return false;
        begin(g);
        stageA(g, treeA.root());
        liveA();
        snapshot(g);
        const double logL = g.logL.front();
        const int out = BEAGLE_ERROR_OUT_OF_RANGE;
        check(beagleEnsureBufferCounts(g.instance, -1, 0, 0) == out, "negative partials count");
        check(beagleEnsureBufferCounts(g.instance, 0, -1, 0) == out, "negative matrix count");
        check(beagleEnsureBufferCounts(g.instance, 0, 0, -1) == out, "negative scale count");
        check(beagleEnsureBufferCounts(g.instance, INT_MAX, 0, 0) == out, "partials + compact count overflows");
        check(beagleEnsureBufferCounts(g.instance, INT_MAX - c.K, m0 + 1, s0 + 1) == out, "partials pool overflows");
        check(beagleEnsureBufferCounts(g.instance, p0 + 1, INT_MAX / 8, s0 + 1) == out, "pointer queue overflows");
        check(beagleEnsureBufferCounts(g.instance, p0 + 1, m0 + 1, INT_MAX - 1) == out, "scale pool overflows");
        check(beagleEnsureBufferCounts(g.instance, p0 + 1, m0 + 1, INT_MAX) == out, "scale pool + zeroes overflows");
        check(beagleEnsureBufferCounts(g.instance, 0, INT_MAX, 0) == out, "matrix pool + scratch overflows");
        // nothing changed: the new indices are still out of range and the results are the same
        const std::vector<double> p(size, 0.5);
        std::vector<double> m(matrixSize);
        check(beagleSetPartials(g.instance, p0, p.data()) == out, "a partials index was added");
        check(beagleGetTransitionMatrix(g.instance, m0, m.data()) == out, "a matrix index was added");
        unchanged(g, "after the rejected counts");
        g.codes.clear();
        stageA(g, treeA.root());
        check(g.logL.back() == logL, "root log likelihood changed");
        unchanged(g, "after the rejected counts");
        check(beagleEnsureBufferCounts(g.instance, p0 + 1, m0 + 1, scale(s0 + 1)) == BEAGLE_SUCCESS,
              "growth afterwards");
        check(beagleSetPartials(g.instance, p0, p.data()) == BEAGLE_SUCCESS, "the added partials index");
        check(beagleGetTransitionMatrix(g.instance, m0, m.data()) == BEAGLE_SUCCESS, "the added matrix index");
        for (int code : g.codes) check(code == BEAGLE_SUCCESS, "a call failed");
        finalizeGpu(g.instance);
        check(beagleEnsureBufferCounts(g.instance, 0, 0, 0) == BEAGLE_ERROR_UNINITIALIZED_INSTANCE,
              "finalized instance");
        printf("%s %s\n", failures == failuresBefore ? "ok  " : "FAIL", current.c_str());
        return true;
    }

    // DYNAMIC off CUDA, without scale buffers: more of them are BEAGLE_ERROR_NO_IMPLEMENTATION, instance unchanged
    bool dynamicScaleGrowth() {
        current = "scale growth, " + name();
        const int failuresBefore = failures;
        GpuRun g;
        g.instance = create(p0, m0, 0);
        if (g.instance < 0) return false;
        begin(g);
        stageA(g, treeA.root());
        liveA();
        snapshot(g);
        const double logL = g.logL.front();
        const std::vector<double> p(size, 0.5);
        check(beagleEnsureBufferCounts(g.instance, 0, 0, 1) == BEAGLE_ERROR_NO_IMPLEMENTATION, "scale growth");
        check(beagleEnsureBufferCounts(g.instance, p0 + 1, m0 + 1, 1) == BEAGLE_ERROR_NO_IMPLEMENTATION,
              "growth with a scale buffer");
        check(beagleSetPartials(g.instance, p0, p.data()) == BEAGLE_ERROR_OUT_OF_RANGE, "a partials index was added");
        unchanged(g, "after the refused growth");
        stageA(g, treeA.root());
        check(g.logL.back() == logL, "root log likelihood changed");
        check(beagleEnsureBufferCounts(g.instance, p0 + 1, m0 + 1, 0) == BEAGLE_SUCCESS,
              "growth without scale buffers");
        for (int code : g.codes) check(code == BEAGLE_SUCCESS, "a call failed");
        finalizeGpu(g.instance);
        printf("%s %s: BEAGLE_ERROR_NO_IMPLEMENTATION\n", failures == failuresBefore ? "ok  " : "FAIL",
               current.c_str());
        return true;
    }

    // BEAGLE_DEBUG_MEMORY build: 10 cycles of growth (by one and by several, alternately) and computation, the
    // pre-order one operation longer each time. After each, G and a fresh instance F at G's counts that made the same
    // calls each grow one more scale buffer, so that both report their counters, which must be equal.
    void leakCycles() {
        current = "10 growth cycles, " + name();
        const int failuresBefore = failures;
        GpuRun g;
        g.instance = create(p0, m0, s0);
        check(g.instance >= 0, "could not create the instance to grow");
        if (g.instance < 0) return;
        begin(g);
        cycle(g, 0);
        liveA();
        int partials = p0, matrices = m0, scaleCount = s0;
        MemoryReport last;
        for (int k = 1; k <= 10; ++k) {
            partials += (k % 2) ? 1 : 3;
            matrices += (k % 2) ? 1 : 2;
            scaleCount += 1;
            if (g.call(beagleEnsureBufferCounts(g.instance, partials, matrices, scaleCount)) != BEAGLE_SUCCESS) {
                check(false, "growth in cycle " + std::to_string(k));
                break;
            }
            g.logL.clear();
            g.crossProducts.clear();
            cycle(g, k);
            GpuRun f;
            f.instance = create(partials, matrices, scaleCount);
            check(f.instance >= 0, "could not create the fresh instance");
            if (f.instance < 0) break;
            begin(f);
            cycle(f, k);
            same(f, g, "in cycle " + std::to_string(k));
            ++scaleCount;
            MemoryReport counters[2];
            GpuRun* both[2] = {&g, &f};
            for (int i = 0; i < 2; ++i) {
                StderrCapture capture;
                both[i]->call(beagleEnsureBufferCounts(both[i]->instance, 0, 0, scaleCount));
                counters[i] = lastReport(capture.finish(), "grow");
            }
            check(!counters[0].event.empty() && counters[0].errors == 0 && counters[0].sameCounters(counters[1]),
                  "cycle " + std::to_string(k) + ": " + counters[0].str() + ", but a fresh instance " +
                  counters[1].str());
            last = counters[0];
            for (int code : f.codes) check(code == BEAGLE_SUCCESS, "a call failed on the fresh instance");
            finalizeGpu(f.instance);
        }
        for (int code : g.codes) check(code == BEAGLE_SUCCESS, "a call failed");
        finalizeGpu(g.instance);
        printf("%s %s: %s after the last, as fresh\n", failures == failuresBefore ? "ok  " : "FAIL", current.c_str(),
               last.str().c_str());
    }

    // BEAGLE_DEBUG_MEMORY build: the growth to the final counts with BEAGLE_DEBUG_MEMORY_FAIL=list. A growth that fails
    // must leave G as it was, equal to a fresh instance F at the old counts, and able to grow; unless the pool is lost,
    // when the probe returns BEAGLE_ERROR_OUT_OF_MEMORY too and only the finalize remains. Then G equals R.
    void forcedFailures(const char* list, int expected, bool lost, bool afterwards = true) {
        current = "allocations " + std::string(list) + " forced to fail, " + name() +
                  (afterwards ? "" : ", no growth afterwards");
        const int failuresBefore = failures;
        GpuRun r, g, f;
        r.instance = create(p1, M1, S1);
        g.instance = create(p0, m0, s0);
        f.instance = create(p0, m0, s0);
        check(r.instance >= 0 && g.instance >= 0 && f.instance >= 0, "could not create the instances");
        if (r.instance < 0 || g.instance < 0 || f.instance < 0) return;
        for (GpuRun* x : {&r, &g, &f}) begin(*x);
        for (GpuRun* x : {&r, &g, &f}) stageA(*x, treeA.root());
        liveA();
        snapshot(g);

        StderrCapture capture;
        setenv("BEAGLE_DEBUG_MEMORY_FAIL", list, 1);
        const int code = beagleEnsureBufferCounts(g.instance, p1, M1, S1);
        unsetenv("BEAGLE_DEBUG_MEMORY_FAIL");
        const std::string text = capture.finish();
        for (const char* k = list; *k != '\0';) { // every allocation in the list was made, and failed
            char* end;
            const long n = strtol(k, &end, 10);
            check(text.find("allocation #" + std::to_string(n) + " ") != std::string::npos,
                  "allocation #" + std::to_string(n) + " was not forced to fail");
            k = (*end == ',') ? end + 1 : end;
        }
        check(code == expected, "growth returned " + std::to_string(code));
        const int probe = beagleEnsureBufferCounts(g.instance, 0, 0, 0);
        check(probe == (lost ? BEAGLE_ERROR_OUT_OF_MEMORY : BEAGLE_SUCCESS), "probe returned " + std::to_string(probe));
        if (!lost) {
            unchanged(g, "after the growth");
            if (code != BEAGLE_SUCCESS) { // still at the old counts, as F
                const std::vector<double> p(size, 0.5);
                std::vector<double> m(matrixSize);
                check(beagleSetPartials(g.instance, p0, p.data()) == BEAGLE_ERROR_OUT_OF_RANGE,
                      "a partials index was added");
                check(beagleGetTransitionMatrix(g.instance, m0, m.data()) == BEAGLE_ERROR_OUT_OF_RANGE,
                      "a matrix index was added");
                for (GpuRun* x : {&g, &f}) x->logL.clear();
                for (GpuRun* x : {&r, &g, &f}) stageA(*x, treeA.root());
                same(f, g, "at the old counts");
            }
            if (afterwards && grow(g, p1, M1, S1, "afterwards")) { // a no-op after a growth that succeeded
                for (GpuRun* x : {&r, &g}) x->logL.clear();
                for (GpuRun* x : {&r, &g}) stageB(*x);
                liveB();
                compare(r, g, 0);
            }
        }
        finalizeGpu(r.instance);
        finalizeGpu(g.instance);
        finalizeGpu(f.instance);
        printf("%s %s: growth returned %d, probe %d\n", failures == failuresBefore ? "ok  " : "FAIL", current.c_str(),
               code, probe);
    }

private:
    const GpuConfig c;
    const int TA, size, matrixSize;
    int kQ, m0, preA, B0, s0, preB, B1, M1, S1, p0, p1, derivativeCount;
    GpuTree treeA, treeB;
    std::vector<double> pi, evec, ivec, eval, q, qk, rates, weights, patternWeights;
    std::vector<std::vector<int>> states;
    std::vector<std::vector<double>> tipPartials;
    bool partitioned = false;
    std::vector<int> liveP, liveM; // the partials and matrices written so far
    std::map<int, std::vector<double>> partials, matrices; // G's snapshot

    bool scaleBuffers() const {
        return c.scaling == MANUAL || (c.scaling == DYNAMIC && gpu.framework == BEAGLE_FLAG_FRAMEWORK_CUDA);
    }
    int scale(int count) const { return scaleBuffers() ? count : 0; } // a scale count to ask for
    // DYNAMIC rescales every post-order operation, and without scale buffers reads scale buffer -1: from 0 scale
    // buffers, G grows them before the first post-order
    int scaleA() const { return (scaleBuffers() && (!c.scaleFromZero || c.scaling == DYNAMIC)) ? 0 : BEAGLE_OP_NONE; }

    std::string expectedName() const { return gpu.prefix + (c.single ? "-Single" : "-Double"); }

    int create(int partialsCount, int matrixCount, int scaleCount) {
        const long requirements = BEAGLE_FLAG_PROCESSOR_GPU | gpu.framework |
                                  (c.single ? BEAGLE_FLAG_PRECISION_SINGLE : BEAGLE_FLAG_PRECISION_DOUBLE);
        long preferences = c.preferences;
        switch (c.scaling) {
            case NONE: case MANUAL: preferences |= BEAGLE_FLAG_SCALING_MANUAL; break;
            case DYNAMIC: preferences |= BEAGLE_FLAG_SCALING_DYNAMIC; break;
            case ALWAYS: preferences |= BEAGLE_FLAG_SCALING_ALWAYS; break;
            case AUTO: preferences |= BEAGLE_FLAG_SCALING_AUTO; break;
        }
        int resource = gpu.number;
        BeagleInstanceDetails details;
        const int instance = beagleCreateInstance(c.T, partialsCount, c.K, c.S, c.P, 1, matrixCount, c.C, scaleCount,
                                                  &resource, 1, preferences, requirements, &details);
        if (instance < 0) { // only a missing implementation is a skip
            check(instance == BEAGLE_ERROR_NO_IMPLEMENTATION, "create returned " + std::to_string(instance));
            return instance;
        }
        if (expectedName() != details.implName || (details.flags & preferences) != preferences) {
            check(false, std::string("created ") + details.implName + " with flags " + std::to_string(details.flags));
            finalizeGpu(instance);
            return -1;
        }
        return instance;
    }

    void begin(GpuRun& x) {
        const int i = x.instance;
        x.call(beagleSetStateFrequencies(i, 0, pi.data()));
        x.call(beagleSetEigenDecomposition(i, 0, evec.data(), ivec.data(), eval.data()));
        x.call(beagleSetCategoryRates(i, rates.data()));
        x.call(beagleSetCategoryWeights(i, 0, weights.data()));
        x.call(beagleSetPatternWeights(i, patternWeights.data()));
        setTips(x, 0, TA);
        if (c.partitions > 0 && c.late == 0) setPartitions(x);
    }

    void setTips(GpuRun& x, int from, int to) {
        for (int tip = from; tip < to; ++tip) {
            if (tip < c.K) x.call(beagleSetTipStates(x.instance, tip, states[tip].data()));
            else x.call(beagleSetTipPartials(x.instance, tip, tipPartials[tip].data()));
        }
    }

    void setPartitions(GpuRun& x) {
        std::vector<int> partition(c.P);
        for (int i = 0; i < c.P; ++i)
            partition[i] = c.reorder ? i % c.partitions : (int) ((long) i * c.partitions / c.P);
        x.call(beagleSetPatternPartitions(x.instance, c.partitions, partition.data()));
        partitioned = true;
    }

    // tree A's matrices and post-order, and its pre-order with preCount operations (repeated past the tree's)
    void stageA(GpuRun& x, int preCount) {
        transitionMatrices(x, treeA, 0);
        x.call(beagleSetDifferentialMatrix(x.instance, kQ, qk.data()));
        postOrder(x, treeA, 0, scaleA());
        if (c.preOrderA) preOrder(x, treeA, preA, 0, preCount);
    }

    // tree B, longer than A: its post-order reads A's root, and edge derivatives and cross products exceed every
    // earlier buffer count. Its post-order repeated in one call gives the same partials, with more operations than G's
    // first internal buffer count (multi-grid at 4 states without scaling).
    void stageB(GpuRun& x) {
        transitionMatrices(x, treeB, m0);
        postOrder(x, treeB, m0, scaleBuffers() ? s0 : BEAGLE_OP_NONE, c.repeat);
        preOrder(x, treeB, preB, m0, treeB.root());
        derivatives(x, treeB, preB, derivativeCount, true);
    }

    // a leak cycle's calls: tree A with a pre-order k operations longer, and cross products (edge derivatives allocate
    // their unused dOutFirstDeriv at the buffer count of their first call, so a fresh instance would differ)
    void cycle(GpuRun& x, int k) {
        stageA(x, treeA.root() + k);
        derivatives(x, treeA, preA, treeA.root(), false);
    }

    void transitionMatrices(GpuRun& x, const GpuTree& t, int matrixBase) {
        std::vector<int> indices;
        for (int p = 0; p < t.root(); ++p) indices.push_back(matrixBase + p);
        x.call(beagleUpdateTransitionMatrices(x.instance, 0, indices.data(), NULL, NULL, t.length.data(),
                                              (int) indices.size()));
    }

    void postOrder(GpuRun& x, const GpuTree& t, int matrixBase, int scaleBase, int repeat = 1) {
        std::vector<BeagleOperation> ops;
        for (int k = 0; k < repeat; ++k) {
            for (int p = t.leaves; p <= t.root(); ++p) {
                const int a = t.child[2 * p], b = t.child[2 * p + 1];
                const int w = (scaleBase != BEAGLE_OP_NONE) ? scaleBase + p - t.leaves : BEAGLE_OP_NONE;
                ops.push_back({t.buffer[p], w, w, t.buffer[a], matrixBase + a, t.buffer[b], matrixBase + b});
            }
        }
        rootLogLikelihood(x, t, ops, scaleBase);
    }

    // post-order (none: from the buffers as they are), then the root log likelihood, and by partition with every
    // other partition unscaled, which reads the buffer of zeroes
    void rootLogLikelihood(GpuRun& x, const GpuTree& t, const std::vector<BeagleOperation>& ops, int scaleBase) {
        const int i = x.instance;
        const int cumulative = (scaleBase != BEAGLE_OP_NONE) ? scaleBase + t.leaves - 1 : BEAGLE_OP_NONE;
        const bool manual = c.scaling == MANUAL && cumulative != BEAGLE_OP_NONE;
        const bool dynamic = c.scaling == DYNAMIC && cumulative != BEAGLE_OP_NONE;
        if (manual) x.call(beagleResetScaleFactors(i, cumulative)); // DYNAMIC keeps it up to date
        if (!ops.empty()) {
            if (dynamic) x.call(beagleResetScaleFactors(i, cumulative));
            x.call(beagleUpdatePartials(i, ops.data(), (int) ops.size(), dynamic ? cumulative : BEAGLE_OP_NONE));
        }
        if (manual) {
            std::vector<int> scales;
            for (int p = t.leaves; p <= t.root(); ++p) scales.push_back(scaleBase + p - t.leaves);
            x.call(beagleAccumulateScaleFactors(i, scales.data(), (int) scales.size(), cumulative));
        } else if (c.scaling == AUTO) {
            const std::vector<int> internal(t.buffer.begin() + t.leaves, t.buffer.end());
            x.call(beagleAccumulateScaleFactors(i, internal.data(), (int) internal.size(), BEAGLE_OP_NONE));
        }
        const int root = t.buffer[t.root()], zero = 0, scaling = (manual || dynamic) ? cumulative : BEAGLE_OP_NONE;
        double logL = 0.0;
        x.call(beagleCalculateRootLogLikelihoods(i, &root, &zero, &zero, &scaling, 1, &logL));
        x.logL.push_back(logL);
        if (c.byPartition && partitioned && manual) {
            const int n = c.partitions;
            std::vector<int> roots(n, root), zeros(n, 0), partitionIndices(n), cumulatives(n);
            for (int k = 0; k < n; ++k) {
                partitionIndices[k] = k;
                cumulatives[k] = (k % 2) ? cumulative : BEAGLE_OP_NONE;
            }
            std::vector<double> byPartition(n);
            double sum = 0.0;
            x.call(beagleCalculateRootLogLikelihoodsByPartition(i, roots.data(), zeros.data(), zeros.data(),
                                                                cumulatives.data(), partitionIndices.data(), n, 1,
                                                                byPartition.data(), &sum));
            x.logL.insert(x.logL.end(), byPartition.begin(), byPartition.end());
            x.logL.push_back(sum);
        }
    }

    // bottom pre-order from the root's pre-order partials (the frequencies, set with setPartials), with count
    // operations: the tree's, repeated
    void preOrder(GpuRun& x, const GpuTree& t, int preBase, int matrixBase, int count) {
        std::vector<double> rootPre(size);
        for (int k = 0; k < c.C * c.P; ++k) for (int s = 0; s < c.S; ++s) rootPre[k * c.S + s] = pi[s];
        x.call(beagleSetPartials(x.instance, preBase + t.root(), rootPre.data()));
        std::vector<BeagleOperation> ops;
        for (int p = t.root(); p >= t.leaves; --p) {
            for (int k = 0; k < 2; ++k) {
                const int child = t.child[2 * p + k], sibling = t.child[2 * p + 1 - k];
                ops.push_back({preBase + child, BEAGLE_OP_NONE, BEAGLE_OP_NONE, preBase + p, matrixBase + child,
                               t.buffer[sibling], matrixBase + sibling});
            }
        }
        for (size_t k = 0; (int) ops.size() < count; ++k) {
            const BeagleOperation op = ops[k];
            ops.push_back(op);
        }
        x.call(beagleUpdatePrePartials(x.instance, ops.data(), (int) ops.size(), BEAGLE_OP_NONE));
    }

    // edge derivatives with the infinitesimal matrix (without squared sums, which overrun dMultipleDerivativeSum)
    // and cross products, for count branches: the tree's, repeated
    void derivatives(GpuRun& x, const GpuTree& t, int preBase, int count, bool edge) {
        std::vector<int> posts, pres, qs;
        std::vector<double> lengths;
        for (int p = 0; (int) posts.size() < count; p = (p + 1) % t.root()) {
            posts.push_back(t.buffer[p]);
            pres.push_back(preBase + p);
            qs.push_back(kQ);
            lengths.push_back(t.length[p]);
        }
        const int zero = 0;
        if (edge) {
            std::vector<double> d(count * c.P), sums(count);
            x.call(beagleCalculateEdgeDerivatives(x.instance, posts.data(), pres.data(), qs.data(), &zero, count,
                                                  d.data(), sums.data(), NULL));
            x.derivatives.insert(x.derivatives.end(), d.begin(), d.end());
            x.sums.insert(x.sums.end(), sums.begin(), sums.end());
        }
        if (c.crossProducts) {
            std::vector<double> cross(c.S * c.S);
            x.call(beagleCalculateCrossProductDerivative(x.instance, posts.data(), pres.data(), &zero, &zero,
                                                         lengths.data(), count, cross.data(), NULL));
            x.crossProducts.insert(x.crossProducts.end(), cross.begin(), cross.end());
        }
    }

    void liveA() {
        liveP.clear();
        liveM.clear();
        for (int tip = c.K; tip < TA; ++tip) liveP.push_back(tip);
        for (int p = treeA.leaves; p <= treeA.root(); ++p) liveP.push_back(treeA.buffer[p]);
        if (c.preOrderA) for (int p = 0; p <= treeA.root(); ++p) liveP.push_back(preA + p);
        for (int p = 0; p < treeA.root(); ++p) liveM.push_back(p);
        liveM.push_back(kQ);
    }

    void liveB() {
        for (int p = treeB.leaves; p <= treeB.root(); ++p) liveP.push_back(treeB.buffer[p]);
        for (int p = 0; p <= treeB.root(); ++p) liveP.push_back(preB + p);
        for (int p = 0; p < treeB.root(); ++p) liveM.push_back(m0 + p);
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

    void snapshot(GpuRun& g) {
        partials.clear();
        matrices.clear();
        for (int index : liveP) partials[index] = readPartials(g.instance, index);
        for (int index : liveM) matrices[index] = readMatrix(g.instance, index);
    }

    void unchanged(GpuRun& g, const std::string& when) {
        for (const auto& p : partials)
            check(::same(readPartials(g.instance, p.first), p.second),
                  "partials " + std::to_string(p.first) + " changed " + when);
        for (const auto& m : matrices)
            check(::same(readMatrix(g.instance, m.first), m.second),
                  "matrix " + std::to_string(m.first) + " changed " + when);
    }

    bool grow(GpuRun& g, int partialsCount, int matrixCount, int scaleCount, const std::string& how) {
        const int code = g.call(beagleEnsureBufferCounts(g.instance, partialsCount, matrixCount, scaleCount));
        check(code == BEAGLE_SUCCESS, "growth " + how + " returned " + std::to_string(code));
        unchanged(g, "after growth " + how);
        return code == BEAGLE_SUCCESS;
    }

    // a's results and live buffers equal b's
    void same(GpuRun& a, GpuRun& b, const std::string& when) {
        check(::same(a.logL, b.logL), "root log likelihoods differ " + when);
        for (size_t k = 0; k < a.logL.size() && k < b.logL.size(); ++k)
            if (a.logL[k] != b.logL[k]) printf("     log likelihood %zu: %.17g and %.17g\n", k, a.logL[k], b.logL[k]);
        check(::same(a.derivatives, b.derivatives) && ::same(a.sums, b.sums), "edge derivatives differ " + when);
        check(::same(a.crossProducts, b.crossProducts), "cross products differ " + when);
        for (int index : liveP)
            check(::same(readPartials(a.instance, index), readPartials(b.instance, index)),
                  "partials " + std::to_string(index) + " differ " + when);
        for (int index : liveM)
            check(::same(readMatrix(a.instance, index), readMatrix(b.instance, index)),
                  "matrix " + std::to_string(index) + " differs " + when);
    }

    void compare(GpuRun& r, GpuRun& g, size_t firstLogL) {
        // every call succeeded, R's and G's (which also made the growth calls)
        int errors = 0;
        for (const GpuRun* x : {&r, &g})
            for (int code : x->codes) if (code != BEAGLE_SUCCESS) ++errors;
        check(errors == 0, std::to_string(errors) + " calls failed");
        same(r, g, "from R's");
        if (c.scaling != AUTO && !(c.scaling == NONE && c.partitions > 0)) // see the header
            for (double l : r.logL) check(std::isfinite(l), "root log likelihood is not finite");
        // after growth, the old buffers and scale factors still give the first root log likelihoods
        if (firstLogL > 0)
            check(g.logL.size() >= 2 * firstLogL &&
                  std::memcmp(g.logL.data(), g.logL.data() + firstLogL, sizeof(double) * firstLogL) == 0,
                  "old buffers changed by growth");
    }
};

struct SpectralRun : GpuRun {
    std::vector<double> adjoint; // adjoint gradients: float atomics, so compared with a tolerance
};

// Spectral instances (BEAGLE_FLAG_SPECTRAL_REPRESENTATION): a matrix index holds branch lengths (a distance pool with
// a host mirror) on an eigen decomposition; dense matrices live in a pool made at the first dense write. T = 8 tips,
// K = 2 compact, 2 categories, MANUAL scaling, two eigen decompositions: the matrix of tree position p is on eigen
// decomposition p % 2. Tree A on every tip: internal buffers T..2T-2, matrices 0..2T-3, TOP pre-order partials from
// preA, scale buffers 0..T-1. The differential matrices of the two decompositions at kQ0 = 2T-2 and kQ1 = m0 (dense).
// Tree B on every tip and A's root, on new buffers, matrices from m0 + 1 and scale buffers from s0. After the final
// growth, M1 - 2 is written densely only (past the dense pool's first size), and M1 - 1 is never written: a
// post-order on it reads the zero tail of the distances (the identity).
class SpectralCase {
public:
    SpectralCase(bool single, int S, int P) : single(single), S(S), P(P), size(S * P * C), matrixSize(S * S * C) {
        std::mt19937 rng(70000 + 1000 * S + P);
        for (int e = 0; e < 2; ++e) {
            std::vector<double> q;
            randomReversible(rng, S, pi[e], evec[e], ivec[e], eval[e], q);
            qk[e].resize(matrixSize);
            for (int k = 0; k < C; ++k)
                for (int i = 0; i < S * S; ++i) qk[e][k * S * S + i] = q[i] * rates[k];
        }
        patternWeights.resize(P);
        for (auto& w : patternWeights) w = std::uniform_int_distribution<int>(1, 3)(rng);
        states.resize(T);
        tipPartials.resize(T);
        for (int tip = 0; tip < T; ++tip) {
            if (tip < K) { // states, about one in ten missing
                states[tip].resize(P);
                for (auto& x : states[tip]) {
                    x = std::uniform_int_distribution<int>(0, S - 1)(rng);
                    if (std::uniform_int_distribution<int>(0, 9)(rng) == 0) x = S;
                }
            } else {
                tipPartials[tip].resize(S * P);
                for (auto& x : tipPartials[tip]) x = std::uniform_real_distribution<double>(0.1, 1.0)(rng);
            }
        }
        kQ0 = 2 * T - 2;
        m0 = 2 * T - 1;
        kQ1 = m0;
        preA = 2 * T - 1;
        B0 = preA + 2 * T - 1 + K; // + K: setPartials takes indices below the partials count
        s0 = T;
        std::vector<int> leaves;
        for (int tip = 0; tip < T; ++tip) leaves.push_back(tip);
        treeA = randomBifurcatingTree(rng, leaves, T);
        leaves.push_back(treeA.buffer[treeA.root()]);
        treeB = randomBifurcatingTree(rng, leaves, B0);
        preB = B0 + T;
        B1 = preB + treeB.root() + 1 + K + 2;
        M1 = m0 + 1 + treeB.root() + 2; // the last two: see above
        S1 = s0 + T + 2;
        p0 = B0 - K;
        p1 = B1 - K;
    }

    std::string name() const {
        return expectedName() + " S=" + std::to_string(S) + " P=" + std::to_string(P) + " C=" + std::to_string(C) +
               " MANUAL, 2 eigen decompositions";
    }

    // G grows; R and F are fresh at the final counts. False if there is no spectral implementation here.
    bool run(const std::string& what = "") {
        current = name() + what;
        const int failuresBefore = failures;
        SpectralRun r, f, g;
        r.instance = create(p1, M1, S1);
        if (r.instance < 0) return false;
        f.instance = create(p1, M1, S1);
        g.instance = create(p0, m0, s0);
        check(f.instance >= 0 && g.instance >= 0, "could not create the instances");
        if (f.instance < 0 || g.instance < 0) {
            for (int i : {r.instance, f.instance, g.instance}) if (i >= 0) finalizeGpu(i);
            return true;
        }
        std::vector<double> m(matrixSize);
        for (SpectralRun* x : {&r, &f, &g}) begin(*x);
        for (SpectralRun* x : {&r, &f, &g}) stageA(*x);
        firstLogL = g.logL.front();
        check(beagleGetTransitionMatrix(g.instance, 0, m.data()) == BEAGLE_ERROR_OUT_OF_RANGE,
              "a dense read of a matrix with eigen-system contents only");
        liveA();
        snapshot(g);

        bool grown = grow(g, 0, 0, 0, "probe");
        grown &= grow(g, p0 + 1, m0 + 1, s0 + 1, "by one, without a dense pool");
        check(beagleGetTransitionMatrix(g.instance, m0, m.data()) == BEAGLE_ERROR_OUT_OF_RANGE,
              "a dense read of a new matrix index, never written");
        for (SpectralRun* x : {&r, &f}) x->call(beagleSetDifferentialMatrix(x->instance, kQ0, qk[0].data()));
        { // G's dense pool, made by this write: the growths without one must not have made it
            StderrCapture capture;
            g.call(beagleSetDifferentialMatrix(g.instance, kQ0, qk[0].data()));
            const std::string text = capture.finish();
            if (debugMemory)
                check(!lastReport(text, "dense matrices").event.empty(), "a growth without a dense pool made one");
        }
        liveM.push_back(kQ0);
        snapshot(g);
        grown &= grow(g, 5, 3, 1, "to smaller counts");
        grown &= grow(g, 0, 0, 0, "(0, 0, 0)");
        grown &= grow(g, (p0 + p1) / 2, (m0 + M1) / 2, (s0 + S1) / 2, "by several");
        for (SpectralRun* x : {&r, &f, &g}) x->call(beagleSetDifferentialMatrix(x->instance, kQ1, qk[1].data()));
        liveM.push_back(kQ1);
        snapshot(g);
        grown &= grow(g, p1, M1, S1, "to the final counts");
        if (grown) {
            check(beagleGetTransitionMatrix(g.instance, M1 - 1, m.data()) == BEAGLE_ERROR_OUT_OF_RANGE,
                  "a dense read of a new matrix index, never written");
            for (SpectralRun* x : {&r, &f, &g}) x->call(beagleSetDifferentialMatrix(x->instance, M1 - 2, qk[0].data()));
            liveM.push_back(M1 - 2);
            for (SpectralRun* x : {&r, &f, &g}) afterA(*x, true);
            for (SpectralRun* x : {&r, &f, &g}) stageB(*x);
            liveB();
            for (SpectralRun* x : {&r, &f, &g}) unwritten(*x);
            liveP.push_back(p1 - 1);
            compare(r, g);
            adjointWithin(r, f, g);
        }
        for (int i : {r.instance, f.instance, g.instance}) finalizeGpu(i);
        printf("%s %s%s\n", failures == failuresBefore ? "ok  " : "FAIL", current.c_str(), adjointNote.c_str());
        return true;
    }

    // BEAGLE_DEBUG_MEMORY build: 10 cycles of growth and computation, the dense pool made in the first; after each,
    // G and a fresh instance F at G's counts that made the same calls grow one more scale buffer, and the counters
    // they report must be equal
    void leakCycles() {
        current = "10 growth cycles, " + name();
        const int failuresBefore = failures;
        SpectralRun g;
        g.instance = create(p0, m0, s0);
        check(g.instance >= 0, "could not create the instance to grow");
        if (g.instance < 0) return;
        begin(g);
        cycle(g);
        firstLogL = g.logL.front();
        liveA();
        liveM.push_back(kQ0);
        int partials = p0, matrices = m0, scaleCount = s0;
        MemoryReport last;
        for (int k = 1; k <= 10; ++k) {
            partials += (k % 2) ? 1 : 3;
            matrices += (k % 2) ? 1 : 2;
            scaleCount += 1;
            if (g.call(beagleEnsureBufferCounts(g.instance, partials, matrices, scaleCount)) != BEAGLE_SUCCESS) {
                check(false, "growth in cycle " + std::to_string(k));
                break;
            }
            reuseA(g, "in cycle " + std::to_string(k));
            g.logL.clear();
            g.crossProducts.clear();
            cycle(g);
            SpectralRun f;
            f.instance = create(partials, matrices, scaleCount);
            check(f.instance >= 0, "could not create the fresh instance");
            if (f.instance < 0) break;
            begin(f);
            cycle(f);
            same(f, g, "in cycle " + std::to_string(k));
            ++scaleCount;
            MemoryReport counters[2];
            SpectralRun* both[2] = {&g, &f};
            for (int i = 0; i < 2; ++i) {
                StderrCapture capture;
                both[i]->call(beagleEnsureBufferCounts(both[i]->instance, 0, 0, scaleCount));
                counters[i] = lastReport(capture.finish(), "grow");
            }
            check(!counters[0].event.empty() && counters[0].errors == 0 && counters[0].sameCounters(counters[1]),
                  "cycle " + std::to_string(k) + ": " + counters[0].str() + ", but a fresh instance " +
                  counters[1].str());
            last = counters[0];
            for (int code : f.codes) check(code == BEAGLE_SUCCESS, "a call failed on the fresh instance");
            finalizeGpu(f.instance);
        }
        for (int code : g.codes) check(code == BEAGLE_SUCCESS, "a call failed");
        finalizeGpu(g.instance);
        printf("%s %s: %s after the last, as fresh\n", failures == failuresBefore ? "ok  " : "FAIL", current.c_str(),
               last.str().c_str());
    }

    // BEAGLE_DEBUG_MEMORY build: the growth to the final counts, with the dense pool made, under
    // BEAGLE_DEBUG_MEMORY_FAIL=list, whose first allocation is the pool (or queue) named (its size is checked). As
    // GpuCase's, plus: G's old matrices, whose distances the growth may have left in new memory, still give the first
    // root log likelihood exactly.
    void forcedFailures(const char* list, const char* pool, int expected, bool lost, const std::string& what = "") {
        const std::string named(pool);
        current = "allocations " + std::string(list) + " forced to fail (the " + named +
                  (named == "derivative queue" ? "), " : " pool), ") + name() + what;
        const int failuresBefore = failures;
        SpectralRun r, g, f;
        r.instance = create(p1, M1, S1);
        g.instance = create(p0, m0, s0);
        f.instance = create(p0, m0, s0);
        check(r.instance >= 0 && g.instance >= 0 && f.instance >= 0, "could not create the instances");
        if (r.instance < 0 || g.instance < 0 || f.instance < 0) return;
        for (SpectralRun* x : {&r, &g, &f}) begin(*x);
        for (SpectralRun* x : {&r, &g, &f}) stageA(*x);
        for (SpectralRun* x : {&r, &g, &f}) x->call(beagleSetDifferentialMatrix(x->instance, kQ0, qk[0].data()));
        firstLogL = g.logL.front();
        liveA();
        liveM.push_back(kQ0);
        snapshot(g);

        StderrCapture capture;
        setenv("BEAGLE_DEBUG_MEMORY_FAIL", list, 1);
        const int code = beagleEnsureBufferCounts(g.instance, p1, M1, S1);
        unsetenv("BEAGLE_DEBUG_MEMORY_FAIL");
        const std::string text = capture.finish();
        const unsigned long poolBytes = named == "derivative queue" ? (unsigned long) sizeof(unsigned int) * B1 * 3 :
            (unsigned long) M1 * aligned((named == "distance" ? 1 : padded() * padded()) * C * realSize());
        for (const char* k = list; *k != '\0';) { // every allocation in the list was made, and failed
            char* end;
            const long n = strtol(k, &end, 10);
            const size_t at = text.find("allocation #" + std::to_string(n) + " (");
            check(at != std::string::npos, "allocation #" + std::to_string(n) + " was not forced to fail");
            if (at != std::string::npos && k == list) {
                const unsigned long bytes = strtoul(text.c_str() + text.find('(', at) + 1, NULL, 10);
                check(bytes == poolBytes, "allocation #" + std::to_string(n) + " has " + std::to_string(bytes) +
                      " bytes, not the " + named + (named == "derivative queue" ? "'s " : " pool's ") +
                      std::to_string(poolBytes));
            }
            k = (*end == ',') ? end + 1 : end;
        }
        check(code == expected, "growth returned " + std::to_string(code));
        const int probe = beagleEnsureBufferCounts(g.instance, 0, 0, 0);
        check(probe == (lost ? BEAGLE_ERROR_OUT_OF_MEMORY : BEAGLE_SUCCESS), "probe returned " + std::to_string(probe));
        if (!lost) {
            unchanged(g, "after the growth");
            reuseA(g, "after the growth");
            if (code != BEAGLE_SUCCESS) { // still at the old counts, as F
                const std::vector<double> p(size, 0.5);
                std::vector<double> m(matrixSize);
                check(beagleSetPartials(g.instance, p0, p.data()) == BEAGLE_ERROR_OUT_OF_RANGE,
                      "a partials index was added");
                check(beagleSetTransitionMatrix(g.instance, m0, m.data(), 0.0) == BEAGLE_ERROR_OUT_OF_RANGE,
                      "a matrix index was added"); // (a dense read is OUT_OF_RANGE at any index never written)
                for (SpectralRun* x : {&r, &g, &f}) afterA(*x, false);
                same(f, g, "at the old counts");
                check(grow(g, p1, M1, S1, "afterwards"), "no growth afterwards");
            }
            for (SpectralRun* x : {&r, &g}) x->logL.clear();
            for (SpectralRun* x : {&r, &g}) x->call(beagleSetDifferentialMatrix(x->instance, kQ1, qk[1].data()));
            liveM.push_back(kQ1);
            for (SpectralRun* x : {&r, &g}) afterA(*x, true);
            for (SpectralRun* x : {&r, &g}) stageB(*x);
            liveB();
            compare(r, g);
        }
        for (int i : {r.instance, g.instance, f.instance}) finalizeGpu(i);
        printf("%s %s: growth returned %d, probe %d\n", failures == failuresBefore ? "ok  " : "FAIL", current.c_str(),
               code, probe);
    }

private:
    static const int T = 8, K = 2, C = 2;
    const bool single;
    const int S, P, size, matrixSize;
    const double rates[C] = {0.5, 1.5}, weights[C] = {0.4, 0.6};
    int kQ0, kQ1, m0, preA, B0, s0, preB, B1, M1, S1, p0, p1;
    GpuTree treeA, treeB;
    std::vector<double> pi[2], evec[2], ivec[2], eval[2], qk[2], patternWeights;
    std::vector<std::vector<int>> states;
    std::vector<std::vector<double>> tipPartials;
    double firstLogL = 0.0;
    std::string adjointNote;
    std::vector<int> liveP, liveM; // the partials and dense matrices written so far
    std::map<int, std::vector<double>> partials, matrices; // G's snapshot

    std::string expectedName() const { return gpu.prefix + (single ? "-Single" : "-Double") + "-Spectral"; }
    size_t realSize() const { return single ? sizeof(float) : sizeof(double); }
    int padded() const { return S <= 4 ? 4 : S <= 16 ? 16 : 32; } // the GPU's padded state count, up to 32 states
    unsigned long aligned(unsigned long bytes) const { return (bytes + gpu.align - 1) / gpu.align * gpu.align; }

    int create(int partialsCount, int matrixCount, int scaleCount) {
        const long requirements = BEAGLE_FLAG_PROCESSOR_GPU | gpu.framework | BEAGLE_FLAG_SPECTRAL_REPRESENTATION |
                                  (single ? BEAGLE_FLAG_PRECISION_SINGLE : BEAGLE_FLAG_PRECISION_DOUBLE);
        int resource = gpu.number;
        BeagleInstanceDetails details;
        const int instance = beagleCreateInstance(T, partialsCount, K, S, P, 2, matrixCount, C, scaleCount,
                                                  &resource, 1, BEAGLE_FLAG_SCALING_MANUAL, requirements, &details);
        if (instance < 0) { // only a missing implementation is a skip
            check(instance == BEAGLE_ERROR_NO_IMPLEMENTATION, "create returned " + std::to_string(instance));
            return instance;
        }
        if (expectedName() != details.implName || !(details.flags & BEAGLE_FLAG_SPECTRAL_REPRESENTATION) ||
            !(details.flags & BEAGLE_FLAG_SCALING_MANUAL)) {
            check(false, std::string("created ") + details.implName + " with flags " + std::to_string(details.flags));
            finalizeGpu(instance);
            return -1;
        }
        return instance;
    }

    void begin(SpectralRun& x) {
        const int i = x.instance;
        x.call(beagleSetStateFrequencies(i, 0, pi[0].data()));
        for (int e = 0; e < 2; ++e)
            x.call(beagleSetEigenDecomposition(i, e, evec[e].data(), ivec[e].data(), eval[e].data()));
        x.call(beagleSetCategoryRates(i, rates));
        x.call(beagleSetCategoryWeights(i, 0, weights));
        x.call(beagleSetPatternWeights(i, patternWeights.data()));
        for (int tip = 0; tip < T; ++tip) {
            if (tip < K) x.call(beagleSetTipStates(i, tip, states[tip].data()));
            else x.call(beagleSetTipPartials(i, tip, tipPartials[tip].data()));
        }
    }

    // tree A's matrices on both eigen decompositions, post-order, TOP pre-order and adjoint gradients
    void stageA(SpectralRun& x) {
        transitionMatrices(x, treeA, 0);
        x.logL.push_back(postOrder(x, treeA, 0, 0));
        preOrder(x, treeA, preA, 0);
        adjoint(x, treeA, preA, 0);
    }

    // after growth, on tree A's matrices as they are: its pre-order, edge derivatives on the dense differential
    // matrices (kQ1 too when it is written), cross products and adjoint gradients
    void afterA(SpectralRun& x, bool bothQ) {
        preOrder(x, treeA, preA, 0);
        derivatives(x, treeA, preA, 0, bothQ);
        adjoint(x, treeA, preA, 0);
    }

    // tree B, on matrices computed after the growth
    void stageB(SpectralRun& x) {
        transitionMatrices(x, treeB, m0 + 1);
        x.logL.push_back(postOrder(x, treeB, m0 + 1, s0));
        preOrder(x, treeB, preB, m0 + 1);
        derivatives(x, treeB, preB, m0 + 1, true);
        adjoint(x, treeB, preB, m0 + 1);
    }

    // a leak cycle's calls (edge derivatives allocate their unused dOutFirstDeriv at the buffer count of their first
    // call, so a fresh instance would differ)
    void cycle(SpectralRun& x) {
        stageA(x);
        x.call(beagleSetDifferentialMatrix(x.instance, kQ0, qk[0].data()));
        std::vector<int> posts, pres;
        std::vector<double> lengths;
        for (int p = 0; p < treeA.root(); ++p) {
            posts.push_back(treeA.buffer[p]);
            pres.push_back(preA + p);
            lengths.push_back(treeA.length[p]);
        }
        const int zero = 0;
        std::vector<double> cross(S * S);
        x.call(beagleCalculateCrossProductDerivative(x.instance, posts.data(), pres.data(), &zero, &zero,
                                                     lengths.data(), (int) posts.size(), cross.data(), NULL));
        x.crossProducts.insert(x.crossProducts.end(), cross.begin(), cross.end());
    }

    void transitionMatrices(SpectralRun& x, const GpuTree& t, int matrixBase) {
        for (int e = 0; e < 2; ++e) {
            std::vector<int> indices;
            std::vector<double> lengths;
            for (int p = e; p < t.root(); p += 2) {
                indices.push_back(matrixBase + p);
                lengths.push_back(t.length[p]);
            }
            x.call(beagleUpdateTransitionMatrices(x.instance, e, indices.data(), NULL, NULL, lengths.data(),
                                                  (int) indices.size()));
        }
    }

    // MANUAL post-order and the root log likelihood
    double postOrder(SpectralRun& x, const GpuTree& t, int matrixBase, int scaleBase) {
        const int i = x.instance;
        std::vector<BeagleOperation> ops;
        std::vector<int> scales;
        for (int p = t.leaves; p <= t.root(); ++p) {
            const int a = t.child[2 * p], b = t.child[2 * p + 1], w = scaleBase + p - t.leaves;
            ops.push_back({t.buffer[p], w, w, t.buffer[a], matrixBase + a, t.buffer[b], matrixBase + b});
            scales.push_back(w);
        }
        const int cumulative = scaleBase + t.leaves - 1, root = t.buffer[t.root()], zero = 0;
        x.call(beagleUpdatePartials(i, ops.data(), (int) ops.size(), BEAGLE_OP_NONE));
        x.call(beagleResetScaleFactors(i, cumulative));
        x.call(beagleAccumulateScaleFactors(i, scales.data(), (int) scales.size(), cumulative));
        double logL = 0.0;
        x.call(beagleCalculateRootLogLikelihoods(i, &root, &zero, &zero, &cumulative, 1, &logL));
        return logL;
    }

    // a post-order on matrix M1 - 1, never updated or written (distance 0, the identity), into buffer p1 - 1, and its
    // root log likelihood: the new tail of the distances must be zero, as in a fresh instance
    void unwritten(SpectralRun& x) {
        const int i = x.instance, buffer = p1 - 1, none = BEAGLE_OP_NONE, zero = 0;
        const BeagleOperation op = {buffer, BEAGLE_OP_NONE, BEAGLE_OP_NONE, K, M1 - 1, K + 1, M1 - 1};
        x.call(beagleUpdatePartials(i, &op, 1, BEAGLE_OP_NONE));
        double logL = 0.0;
        x.call(beagleCalculateRootLogLikelihoods(i, &buffer, &zero, &zero, &none, 1, &logL));
        x.logL.push_back(logL);
    }

    // TOP pre-order from the root's pre-order partials (the frequencies, set with setPartials)
    void preOrder(SpectralRun& x, const GpuTree& t, int preBase, int matrixBase) {
        std::vector<double> rootPre(size);
        for (int k = 0; k < C * P; ++k) for (int s = 0; s < S; ++s) rootPre[k * S + s] = pi[0][s];
        x.call(beagleSetPartials(x.instance, preBase + t.root(), rootPre.data()));
        std::vector<BeagleOperation> ops;
        for (int p = t.root(); p >= t.leaves; --p) {
            for (int k = 0; k < 2; ++k) {
                const int child = t.child[2 * p + k], sibling = t.child[2 * p + 1 - k];
                ops.push_back({preBase + child, BEAGLE_OP_NONE, BEAGLE_OP_NONE, preBase + p,
                               p == t.root() ? BEAGLE_OP_NONE : matrixBase + p, t.buffer[sibling],
                               matrixBase + sibling});
            }
        }
        x.call(beagleUpdatePrePartials_v5(x.instance, ops.data(), (int) ops.size(), BEAGLE_OP_NONE,
                                          BEAGLE_PARTIALS_TOP));
    }

    // edge derivatives on the differential matrix of each branch's eigen decomposition (kQ0 only unless bothQ), and
    // cross products
    void derivatives(SpectralRun& x, const GpuTree& t, int preBase, int matrixBase, bool bothQ) {
        std::vector<int> posts, pres, qs;
        std::vector<double> lengths;
        for (int p = 0; p < t.root(); ++p) {
            posts.push_back(t.buffer[p]);
            pres.push_back(preBase + p);
            qs.push_back(bothQ && p % 2 ? kQ1 : kQ0);
            lengths.push_back(t.length[p]);
        }
        const int count = (int) posts.size(), zero = 0;
        std::vector<double> d(count * P), sums(count), cross(S * S);
        x.call(beagleCalculateEdgeDerivatives(x.instance, posts.data(), pres.data(), qs.data(), &zero, count,
                                              d.data(), sums.data(), NULL));
        x.derivatives.insert(x.derivatives.end(), d.begin(), d.end());
        x.sums.insert(x.sums.end(), sums.begin(), sums.end());
        x.call(beagleCalculateCrossProductDerivative(x.instance, posts.data(), pres.data(), &zero, &zero,
                                                     lengths.data(), count, cross.data(), NULL));
        x.crossProducts.insert(x.crossProducts.end(), cross.begin(), cross.end());
    }

    // the adjoint gradient of every branch, twice: it returns the gradient of the first branch's eigen
    // decomposition, so first a branch on decomposition 0 leads, then one on 1
    void adjoint(SpectralRun& x, const GpuTree& t, int preBase, int matrixBase) {
        const int count = t.root(), root = t.buffer[t.root()];
        for (int first = 0; first < 2; ++first) {
            std::vector<BeagleBranchOperation> ops;
            for (int k = 0; k < count; ++k) {
                const int p = (first + k) % count;
                ops.push_back({t.buffer[p], preBase + p, matrixBase + p, 0});
            }
            std::vector<double> gradient(S * S);
            x.call(beagleCalculateAdjointDerivative(x.instance, ops.data(), 0, 0, root, 0, count, gradient.data(),
                                                    NULL));
            x.adjoint.insert(x.adjoint.end(), gradient.begin(), gradient.end());
        }
    }

    void liveA() {
        liveP.clear();
        liveM.clear();
        for (int tip = K; tip < T; ++tip) liveP.push_back(tip);
        for (int p = treeA.leaves; p <= treeA.root(); ++p) liveP.push_back(treeA.buffer[p]);
        for (int p = 0; p <= treeA.root(); ++p) liveP.push_back(preA + p);
    }

    void liveB() {
        for (int p = treeB.leaves; p <= treeB.root(); ++p) liveP.push_back(treeB.buffer[p]);
        for (int p = 0; p <= treeB.root(); ++p) liveP.push_back(preB + p);
    }

    std::vector<double> readPartials(int instance, int index) {
        std::vector<double> p(size);
        check(beagleGetPartials(instance, index, BEAGLE_OP_NONE, p.data()) == BEAGLE_SUCCESS,
              "getPartials " + std::to_string(index));
        return p;
    }

    std::vector<double> readMatrix(int instance, int index) { // dense contents only
        std::vector<double> m(matrixSize);
        check(beagleGetTransitionMatrix(instance, index, m.data()) == BEAGLE_SUCCESS,
              "getTransitionMatrix " + std::to_string(index));
        return m;
    }

    void snapshot(SpectralRun& g) {
        partials.clear();
        matrices.clear();
        for (int index : liveP) partials[index] = readPartials(g.instance, index);
        for (int index : liveM) matrices[index] = readMatrix(g.instance, index);
    }

    void unchanged(SpectralRun& g, const std::string& when) {
        for (const auto& p : partials)
            check(::same(readPartials(g.instance, p.first), p.second),
                  "partials " + std::to_string(p.first) + " changed " + when);
        for (const auto& m : matrices)
            check(::same(readMatrix(g.instance, m.first), m.second),
                  "matrix " + std::to_string(m.first) + " changed " + when);
    }

    // tree A's post-order again on its matrices as computed before the growth (distances and eigen decompositions):
    // the first root log likelihood exactly
    void reuseA(SpectralRun& g, const std::string& when) {
        const double logL = postOrder(g, treeA, 0, 0);
        check(std::memcmp(&logL, &firstLogL, sizeof(double)) == 0,
              "tree A's root log likelihood " + std::to_string(logL) + " is not " + std::to_string(firstLogL) + " " +
              when);
    }

    bool grow(SpectralRun& g, int partialsCount, int matrixCount, int scaleCount, const std::string& how) {
        const int code = g.call(beagleEnsureBufferCounts(g.instance, partialsCount, matrixCount, scaleCount));
        check(code == BEAGLE_SUCCESS, "growth " + how + " returned " + std::to_string(code));
        unchanged(g, "after growth " + how);
        reuseA(g, "after growth " + how);
        unchanged(g, "after growth " + how + " and tree A's post-order");
        return code == BEAGLE_SUCCESS;
    }

    // a's results and live buffers equal b's, apart from the adjoint gradients
    void same(SpectralRun& a, SpectralRun& b, const std::string& when) {
        check(::same(a.logL, b.logL), "root log likelihoods differ " + when);
        for (size_t k = 0; k < a.logL.size() && k < b.logL.size(); ++k)
            if (a.logL[k] != b.logL[k]) printf("     log likelihood %zu: %.17g and %.17g\n", k, a.logL[k], b.logL[k]);
        check(::same(a.derivatives, b.derivatives) && ::same(a.sums, b.sums), "edge derivatives differ " + when);
        check(::same(a.crossProducts, b.crossProducts), "cross products differ " + when);
        for (int index : liveP)
            check(::same(readPartials(a.instance, index), readPartials(b.instance, index)),
                  "partials " + std::to_string(index) + " differ " + when);
        for (int index : liveM)
            check(::same(readMatrix(a.instance, index), readMatrix(b.instance, index)),
                  "matrix " + std::to_string(index) + " differs " + when);
    }

    void compare(SpectralRun& r, SpectralRun& g) {
        int errors = 0;
        for (const SpectralRun* x : {&r, &g})
            for (int code : x->codes) if (code != BEAGLE_SUCCESS) ++errors;
        check(errors == 0, std::to_string(errors) + " calls failed");
        same(r, g, "from R's");
        for (double l : r.logL) check(std::isfinite(l), "root log likelihood is not finite");
        for (double v : g.adjoint) check(std::isfinite(v), "adjoint gradient is not finite");
    }

    // The adjoint gradients (float atomics) of G against R's, within a tolerance from the two fresh instances R and F:
    // 4 times their largest difference, and at least 1e-6 of the largest |gradient|
    void adjointWithin(SpectralRun& r, SpectralRun& f, SpectralRun& g) {
        double fresh = 0.0, grown = 0.0, largest = 0.0;
        check(r.adjoint.size() == f.adjoint.size() && r.adjoint.size() == g.adjoint.size() && !r.adjoint.empty(),
              "adjoint gradients missing");
        for (size_t k = 0; k < r.adjoint.size() && k < f.adjoint.size() && k < g.adjoint.size(); ++k) {
            fresh = std::max(fresh, std::fabs(r.adjoint[k] - f.adjoint[k]));
            grown = std::max(grown, std::fabs(r.adjoint[k] - g.adjoint[k]));
            largest = std::max(largest, std::fabs(r.adjoint[k]));
        }
        const double tolerance = std::max(4.0 * fresh, 1e-6 * largest);
        check(grown <= tolerance, "adjoint gradients differ by " + std::to_string(grown) + " > " +
              std::to_string(tolerance));
        char note[160];
        snprintf(note, sizeof(note), ": adjoint |G - R| %.3g, fresh |F - R| %.3g, tolerance %.3g (largest %.3g)",
                 grown, fresh, tolerance, largest);
        adjointNote = note;
    }
};

// The GPU axis on resource number; 77 if it is not a GPU resource
int gpuAxis(int number, const std::string& precision, const std::string& only) {
    BeagleResourceList* list = beagleGetResourceList();
    if (number < 0 || number >= list->length || !(list->list[number].supportFlags & BEAGLE_FLAG_PROCESSOR_GPU)) {
        printf("SKIP: no GPU resource %d\n", number);
        return 77;
    }
    const BeagleResource& resource = list->list[number];
    gpu.number = number;
    if (resource.supportFlags & BEAGLE_FLAG_FRAMEWORK_CUDA) {
        gpu.framework = BEAGLE_FLAG_FRAMEWORK_CUDA;
        gpu.prefix = "CUDA";
    } else if (resource.supportFlags & BEAGLE_FLAG_FRAMEWORK_OPENCL) {
        gpu.framework = BEAGLE_FLAG_FRAMEWORK_OPENCL;
        gpu.prefix = "OpenCL";
    } else if (resource.supportFlags & BEAGLE_FLAG_FRAMEWORK_TINYGPU) {
        gpu.framework = BEAGLE_FLAG_FRAMEWORK_TINYGPU;
        gpu.prefix = "TinyGPU";
    } else {
        printf("SKIP: resource %d (%s) has no CUDA, OpenCL or TinyGPU framework\n", number, resource.name);
        return 77;
    }
    gpu.doublePrecision = (resource.supportFlags & BEAGLE_FLAG_PRECISION_DOUBLE) != 0;
    if (precision == "double" && !gpu.doublePrecision) {
        printf("SKIP: resource %d (%s) has no double precision\n", number, resource.name);
        return 77;
    }
    { // does the library count buffers? An instance created and finalized says so on stderr
        StderrCapture capture;
        BeagleInstanceDetails details;
        const int instance = beagleCreateInstance(4, 3, 4, 4, 2, 1, 6, 1, 0, &number, 1, 0,
                                                  BEAGLE_FLAG_PROCESSOR_GPU | gpu.framework |
                                                  BEAGLE_FLAG_PRECISION_SINGLE, &details);
        if (instance >= 0) beagleFinalizeInstance(instance);
        const std::string text = capture.finish();
        debugMemory = !lastReport(text, "create").event.empty();
        const size_t at = text.find("offsets aligned to ");
        if (at != std::string::npos) gpu.align = strtoul(text.c_str() + at + strlen("offsets aligned to "), NULL, 10);
    }
    printf("GPU resource %d: %s, %s%s; memory counters %s\n", number, resource.name, gpu.prefix.c_str(),
           gpu.doublePrecision ? ", double precision" : "", debugMemory ? "on" : "off");

    const long transpose = BEAGLE_FLAG_PREORDER_TRANSPOSE_AUTO;
    int cases = 0, checks = 0, skipped = 0;
    for (bool single : {true, false}) {
        if (!precision.empty() && precision != (single ? "single" : "double")) continue;
        if (!single && !gpu.doublePrecision) {
            printf("skip %s-Double: the resource has no double precision\n", gpu.prefix.c_str());
            ++skipped;
            continue;
        }
        const bool s = single;
        // what, single, S, P, C, T, K, scaling, preferences, late, partitions, reorder, byPartition, preOrderA,
        // crossProducts, scaleFromZero
        const GpuConfig configs[] = {
            // first: a multi-grid operation buffer left at its first size overruns by far, which this case then always
            // shows; after other cases the overrun can land in memory that hides it
            {"multi-grid, 160 post-order operations", s, 4, 1000, 2, 8, 2, NONE, transpose, 0, 0, false, false, true,
             false, false, 20},
            {"multi-grid", s, 4, 100, 2, 8, 2, NONE, transpose, 0, 0, false, false, true, true, false, 4},
            {"multi-grid", s, 4, 100, 2, 8, 2, MANUAL, transpose, 0, 0, false, false, true, true, false},
            {"multi-grid", s, 4, 100, 2, 8, 2, ALWAYS, transpose, 0, 0, false, false, true, true, false},
            {"multi-grid", s, 4, 100, 2, 8, 2, AUTO, transpose, 0, 0, false, false, true, true, false},
            {"transposed pre-order", s, 17, 33, 2, 8, 2, NONE, transpose, 0, 0, false, false, true, true, false},
            {"transposed pre-order", s, 17, 33, 2, 8, 2, MANUAL, transpose, 0, 0, false, false, true, true, false},
            {"transposed pre-order", s, 17, 33, 1, 8, 2, MANUAL, transpose, 0, 0, false, false, true, true, false},
            {"transposed pre-order", s, 17, 33, 2, 8, 2, ALWAYS, transpose, 0, 0, false, false, true, true, false},
            {"transposed pre-order", s, 17, 33, 2, 8, 2, AUTO, transpose, 0, 0, false, false, true, true, false},
            {"multi-grid", s, 4, 1000, 2, 8, 2, MANUAL, transpose, 0, 0, false, false, true, false, false},
            {"4 partitions, by partition", s, 4, 1000, 2, 8, 2, MANUAL, transpose, 0, 4, false, true, true, false,
             false},
            {"4 partitions reordered, by partition", s, 4, 1000, 2, 8, 2, MANUAL, transpose, 0, 4, true, true, true,
             false, false},
            {"4 late tips", s, 4, 100, 2, 16, 2, MANUAL, transpose, 4, 0, false, false, false, true, false},
            {"4 late tips, then 3 partitions reordered, by partition", s, 4, 100, 2, 16, 2, MANUAL, transpose, 4, 3,
             true, true, false, true, false},
            {"from 0 scale buffers, 2 partitions, by partition", s, 4, 100, 2, 8, 2, MANUAL, transpose, 0, 2, false,
             true, true, true, true},
            {"from 0 scale buffers", s, 17, 33, 2, 8, 2, MANUAL, transpose, 0, 0, false, false, true, true, true},
            {"streams", s, 4, 100, 2, 8, 2, MANUAL, transpose | BEAGLE_FLAG_PARALLELOPS_STREAMS, 0, 0, false, false,
             true, true, false},
            {"multi-grid", s, 4, 100, 2, 8, 2, DYNAMIC, transpose, 0, 0, false, false, true, true, false},
            {"from 0 scale buffers", s, 17, 33, 2, 8, 2, DYNAMIC, transpose, 0, 0, false, false, true, true, true},
            {"4 partitions reordered, multi-grid", s, 4, 100, 2, 8, 2, NONE, transpose, 0, 4, true, false, true, true,
             false, 4},
            {"no compact tips, 4 partitions reordered, by partition", s, 4, 100, 2, 8, 0, MANUAL, transpose, 0, 4,
             true, true, false, true, false},
        };
        for (const GpuConfig& config : configs) {
            if (config.scaling == DYNAMIC && gpu.framework != BEAGLE_FLAG_FRAMEWORK_CUDA && config.S != 4)
                continue; // see the header
            GpuCase test(config);
            if (!only.empty() && test.name().find(only) == std::string::npos) continue;
            if (test.run()) {
                ++cases;
            } else {
                ++skipped;
                printf("skip %s: no implementation\n", test.name().c_str());
            }
        }
        // a spectral implementation here; without the flag (TinyGPU, or no spectral plugin) creation would fail with
        // BEAGLE_ERROR_NO_RESOURCE, so G2b's skip is printed after the contract checks
        bool spectral = false;
        const bool spectralResource = (resource.supportFlags & BEAGLE_FLAG_SPECTRAL_REPRESENTATION) != 0;
        for (int S : {4, 17}) {
            if (!spectralResource) break;
            SpectralCase test(single, S, S == 4 ? 100 : 33);
            if (!only.empty() && test.name().find(only) == std::string::npos) continue;
            if (test.run()) {
                ++cases;
                spectral = true;
            } else {
                ++skipped;
                printf("skip %s: no implementation\n", test.name().c_str());
            }
        }
        if (!only.empty()) continue;
        const GpuConfig manual = configs[2], transposed = configs[6], dynamic = configs[18], fromZero = configs[16];
        for (const GpuConfig& config : {manual, transposed}) { // transposed: scratch matrices past the matrix count
            if (GpuCase(config).contract()) {
                ++checks;
            } else {
                ++skipped;
                printf("skip contract, %s: no implementation\n", GpuCase(config).name().c_str());
            }
        }
        if (gpu.framework != BEAGLE_FLAG_FRAMEWORK_CUDA) {
            if (GpuCase(dynamic).dynamicScaleGrowth()) {
                ++checks;
            } else {
                ++skipped;
                printf("skip scale growth, %s: no implementation\n", GpuCase(dynamic).name().c_str());
            }
        }
        if (!spectralResource) {
            ++skipped;
            printf("skip %s-%s-Spectral: no implementation\n", gpu.prefix.c_str(), single ? "Single" : "Double");
        }
        if (debugMemory) {
            for (const GpuConfig& config : {manual, transposed}) {
                GpuCase(config).leakCycles();
                GpuCase(config).forcedFailures("1", BEAGLE_SUCCESS, false);       // the host-staged fallback
                GpuCase(config).forcedFailures("1,2", BEAGLE_ERROR_OUT_OF_MEMORY, false); // and its allocation
                GpuCase(config).forcedFailures("4", BEAGLE_ERROR_OUT_OF_MEMORY, false);   // the derivative queue
                GpuCase(config).forcedFailures("5", BEAGLE_ERROR_OUT_OF_MEMORY, false);   // the pointer queue
                GpuCase(config).forcedFailures("6", BEAGLE_ERROR_OUT_OF_MEMORY, false);   // the distance queue
                GpuCase(config).forcedFailures("3,4", BEAGLE_ERROR_OUT_OF_MEMORY, false); // the scale pool's fallback
                GpuCase(config).forcedFailures("1,2,3", BEAGLE_ERROR_OUT_OF_MEMORY, true); // and the restore
                checks += 8;
            }
            // a scale pool made from 0 scale buffers by a growth that fails later: the finalize must free it, also while
            // the scale count is still 0
            GpuCase(fromZero).forcedFailures("4", BEAGLE_ERROR_OUT_OF_MEMORY, false);
            GpuCase(fromZero).forcedFailures("4", BEAGLE_ERROR_OUT_OF_MEMORY, false, false);
            checks += 2;
            // new device memory full of garbage, which CUDA and TinyGPU may give: G's by-partition log likelihoods read
            // the buffer of zeroes, which must be written
            setenv("BEAGLE_DEBUG_MEMORY_SCRIBBLE", "1", 1);
            for (int k : {11, 12, 14, 15, 21}) { // the cases by partition
                GpuConfig config = configs[k];
                const std::string what = std::string(config.what) + ", new memory scribbled";
                config.what = what.c_str();
                if (GpuCase(config).run()) ++checks;
            }
            for (int S : {4, 17}) { // the new distance pool, which the host mirror must fill again
                if (!spectral) break;
                const int P = (S == 4) ? 100 : 33;
                if (SpectralCase(single, S, P).run(", new memory scribbled")) ++checks;
                // the fallback's block and the restored one, which may start at the old address with the old contents
                SpectralCase(single, S, P).forcedFailures("4", "distance", BEAGLE_SUCCESS, false,
                                                          ", new memory scribbled");
                SpectralCase(single, S, P).forcedFailures("4,5", "distance", BEAGLE_ERROR_OUT_OF_MEMORY, false,
                                                          ", new memory scribbled");
                checks += 2;
            }
            unsetenv("BEAGLE_DEBUG_MEMORY_SCRIBBLE");
            for (int S : {4, 17}) {
                if (!spectral) break;
                const int P = (S == 4) ? 100 : 33;
                SpectralCase(single, S, P).leakCycles();
                // the dense pool is made before the growth: partials (1), dense (2), scale (3) and distance pool (4)
                SpectralCase(single, S, P).forcedFailures("4", "distance", BEAGLE_SUCCESS, false); // the fallback
                SpectralCase(single, S, P).forcedFailures("4,5", "distance", BEAGLE_ERROR_OUT_OF_MEMORY, false);
                SpectralCase(single, S, P).forcedFailures("4,5,6", "distance", BEAGLE_ERROR_OUT_OF_MEMORY, true);
                SpectralCase(single, S, P).forcedFailures("2", "dense matrix", BEAGLE_SUCCESS, false);
                SpectralCase(single, S, P).forcedFailures("2,3", "dense matrix", BEAGLE_ERROR_OUT_OF_MEMORY, false);
                SpectralCase(single, S, P).forcedFailures("2,3,4", "dense matrix", BEAGLE_ERROR_OUT_OF_MEMORY, true);
                // a queue after the distance pool moved: its capacity stays above the matrix count, and the growth
                // afterwards keeps its handles
                SpectralCase(single, S, P).forcedFailures("5", "derivative queue", BEAGLE_ERROR_OUT_OF_MEMORY, false);
                checks += 8;
            }
        }
    }
    printf("buffergrowthtest --gpu %d: %d cases, %d other checks, %d skipped, %d failures\n", number, cases, checks,
           skipped, failures);
    const bool pass = failures == 0 && cases > 0;
    printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
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
    std::string only, precision; // run only the cases whose label contains only; GPU cases of one precision
    int gpuResource = -1;
    bool gpuCases = false;
    for (int a = 1; a < argc; ++a) {
        const std::string arg = argv[a];
        if (arg == "--gpu" && a + 1 < argc) {
            gpuCases = true;
            gpuResource = atoi(argv[++a]);
        } else if (arg == "--precision" && a + 1 < argc && (std::string(argv[a + 1]) == "single" ||
                                                            std::string(argv[a + 1]) == "double")) {
            precision = argv[++a];
        } else if (arg.compare(0, 2, "--") == 0) {
            printf("usage: buffergrowthtest [--gpu <resource> [--precision single|double]] [label]\n");
            return 1;
        } else {
            only = arg;
        }
    }
    setvbuf(stdout, NULL, _IONBF, 0); // a crash still shows the cases before it
    current = "invalid instance";
    check(beagleEnsureBufferCounts(0, 0, 0, 0) == BEAGLE_ERROR_UNINITIALIZED_INSTANCE, "before any instance");
    check(beagleEnsureBufferCounts(-1, 0, 0, 0) == BEAGLE_ERROR_UNINITIALIZED_INSTANCE, "negative instance");
    check(beagleEnsureBufferCounts(1000000, 0, 0, 0) == BEAGLE_ERROR_UNINITIALIZED_INSTANCE, "large instance");
    if (gpuCases)
        return gpuAxis(gpuResource, precision, only);

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
    int contracts = 0;
    for (const Resource& res : resources) {
        const Config config{res.spectral, res.sse, res.single, 17, 1, 1, MANUAL, 1};
        if (Case(config).contract()) {
            ++contracts;
        } else {
            printf("skip contract, %s: no implementation\n", expectedName(config).c_str());
        }
    }
    printf("GPU: not run (buffergrowthtest --gpu <resource>)\n");
    timing();
    printf("buffergrowthtest: %d cases, %d skipped, %d failures\n", compared, skipped, failures);
    const bool pass = failures == 0 && compared > 0 && contracts > 0;
    printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
