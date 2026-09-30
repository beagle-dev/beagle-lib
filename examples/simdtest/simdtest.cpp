/*
 * Copyright 2026 Phylogenetic Likelihood Working Group
 * This file is part of BEAGLE.
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 *
 * simdtest: compares each SIMD CPU implementation with its scalar counterpart,
 *   standard: CPU-SSE-Double and CPU-4State-SSE-Double with CPU-Double and CPU-4State-Double;
 *   spectral: CPU-Spectral-SSE-Double and CPU-4State-Spectral-SSE-Double with CPU-Spectral-Double.
 * For state counts 2 to 64, 1 to 64 patterns, 1 and 4 rate categories, and real (reversible) and complex
 * (asymmetric circulant) eigenvalues, it compares the root log likelihood, the post-order partials, the bottom and
 * top pre-order partials of every node and the adjoint gradient. The tree has tips as states, some of them
 * missing, one tip as partials and a degree-2 node; with 64 patterns the SIMD implementation also runs with
 * three threads. Exits non-zero if a relative difference exceeds 1e-10.
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "libhmsbeagle/beagle.h"

namespace {

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

// Random reversible rate matrix Q = R diag(pi), one substitution per unit time
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

// Random asymmetric circulant rate matrix: complex conjugate eigenvalue pairs a_m +/- i b_m in the real Fourier basis
void randomCirculant(std::mt19937& rng, int n, std::vector<double>& pi, std::vector<double>& evec,
                     std::vector<double>& ivec, std::vector<double>& eval) {
    std::uniform_real_distribution<double> u(0.5, 1.5);
    std::vector<double> r(n, 0.0);
    double sum = 0.0;
    for (int k = 1; k < n; ++k) { r[k] = u(rng); sum += r[k]; }
    for (auto& x : r) x /= sum;
    const double w = 2.0 * M_PI / n;
    const bool even = (n % 2 == 0);
    const int pairs = even ? n / 2 - 1 : (n - 1) / 2;
    pi.assign(n, 1.0 / n);
    evec.assign(n * n, 0.0); ivec.assign(n * n, 0.0); eval.assign(2 * n, 0.0);
    for (int j = 0; j < n; ++j) { evec[j * n] = 1.0 / n; ivec[j] = 1.0; }
    for (int m = 1; m <= pairs; ++m) {
        const int re = 2 * m - 1, im = 2 * m;
        const double theta = w * m;
        double a = 0.0, b = 0.0;
        for (int k = 1; k < n; ++k) { a += r[k] * (std::cos(theta * k) - 1.0); b += r[k] * std::sin(theta * k); }
        eval[re] = a; eval[n + re] = b; eval[im] = a; eval[n + im] = -b;
        for (int j = 0; j < n; ++j) {
            evec[j * n + re] = std::cos(theta * j) / n; evec[j * n + im] = std::sin(theta * j) / n;
            ivec[re * n + j] = 2.0 * std::cos(theta * j); ivec[im * n + j] = 2.0 * std::sin(theta * j);
        }
    }
    if (even) {
        const int last = n - 1;
        double a = 0.0;
        for (int k = 1; k < n; ++k) a += r[k] * (((k % 2 == 0) ? 1.0 : -1.0) - 1.0);
        eval[last] = a;
        for (int j = 0; j < n; ++j) {
            const double s = (j % 2 == 0) ? 1.0 : -1.0;
            evec[j * n + last] = s / n; ivec[last * n + j] = s;
        }
    }
}

// A random binary tree on tips 0..tips-1 (internal nodes tips..2 tips - 2, the root last) with one degree-2 node,
// 2 tips - 1, inserted on a random branch
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
    std::replace(t.children[above].begin(), t.children[above].end(), below, degree2);
    t.parent[degree2] = above;
    t.children[degree2] = {below};
    t.parent[below] = degree2;
    std::uniform_real_distribution<double> u(0.05, 0.3);
    t.length.assign(nodes, 0.0);
    for (int n = 0; n < nodes; ++n) if (n != t.root) t.length[n] = u(rng);
    return t;
}

struct Result {
    std::string name;
    double lnL = 0.0;
    std::vector<std::vector<double>> post, bottom, top; // per node
    std::vector<double> gradient;
};

struct Config {
    bool spectral, complexModel;
    int states, patterns, categories, threads;
};

// One evaluation of everything compared; false if no implementation matches the flags
bool evaluate(const Config& c, bool simd, Result& out) {
    const int S = c.states, C = c.patterns, K = c.categories, N = 8;
    std::mt19937 rng(1000 * S + 10 * C + K + (c.complexModel ? 7 : 0));
    const Tree tree = randomTree(rng, N);
    const int M = (int) tree.parent.size();
    auto bottom = [M](int n) { return M + n; };
    auto top = [M](int n) { return 2 * M + n; };

    long requirements = BEAGLE_FLAG_PROCESSOR_CPU | BEAGLE_FLAG_PRECISION_DOUBLE |
                        (simd ? BEAGLE_FLAG_VECTOR_SSE : BEAGLE_FLAG_VECTOR_NONE) |
                        BEAGLE_FLAG_PREORDER_TRANSPOSE_AUTO |
                        (c.spectral ? BEAGLE_FLAG_SPECTRAL_REPRESENTATION : 0) |
                        (c.complexModel ? BEAGLE_FLAG_EIGEN_COMPLEX : 0);
    const bool threaded = simd && c.threads > 1;
    if (threaded) requirements |= BEAGLE_FLAG_THREADING_CPP;
    const long preferences = BEAGLE_FLAG_SCALING_MANUAL | (threaded ? 0 : BEAGLE_FLAG_THREADING_NONE);

    BeagleInstanceDetails details;
    const int instance = beagleCreateInstance(N, 3 * M, N - 1, S, C, 1, M, K, 0, NULL, 0, preferences,
                                              requirements, &details);
    if (instance < 0) return false;
    out.name = details.implName;
    if (threaded) beagleSetCPUThreadCount(instance, c.threads);

    std::vector<double> pi, evec, ivec, eval;
    if (c.complexModel) randomCirculant(rng, S, pi, evec, ivec, eval);
    else randomReversible(rng, S, pi, evec, ivec, eval);
    beagleSetStateFrequencies(instance, 0, pi.data());
    beagleSetEigenDecomposition(instance, 0, evec.data(), ivec.data(), eval.data());
    std::vector<double> rates(K), weights(K);
    for (int k = 0; k < K; ++k) {
        rates[k] = (K == 1) ? 1.0 : 0.25 + 1.5 * k / (K - 1);
        weights[k] = (k + 1.0) / (K * (K + 1) / 2.0);
    }
    beagleSetCategoryRates(instance, rates.data());
    beagleSetCategoryWeights(instance, 0, weights.data());
    std::vector<double> patternWeights(C);
    for (auto& w : patternWeights) w = std::uniform_int_distribution<int>(1, 3)(rng);
    beagleSetPatternWeights(instance, patternWeights.data());

    for (int tip = 0; tip < N - 1; ++tip) { // states, about one in ten missing
        std::vector<int> states(C);
        for (auto& s : states) {
            s = std::uniform_int_distribution<int>(0, S - 1)(rng);
            if (std::uniform_int_distribution<int>(0, 9)(rng) == 0) s = S;
        }
        beagleSetTipStates(instance, tip, states.data());
    }
    std::vector<double> tipPartials(S * C);
    for (auto& x : tipPartials) x = std::uniform_real_distribution<double>(0.1, 1.0)(rng);
    beagleSetTipPartials(instance, N - 1, tipPartials.data());

    std::vector<int> matrixIndices;
    std::vector<double> lengths;
    for (int n = 0; n < M; ++n) {
        if (n == tree.root) continue;
        matrixIndices.push_back(n);
        lengths.push_back(tree.length[n]);
    }
    beagleUpdateTransitionMatrices(instance, 0, matrixIndices.data(), NULL, NULL, lengths.data(),
                                   (int) matrixIndices.size());

    // post-order: children before parents; a degree-2 node has no second child
    std::vector<BeagleOperation> post;
    std::vector<int> preorder; // internal nodes, parents first
    std::function<void(int)> visit = [&](int n) {
        if (tree.children[n].empty()) return;
        preorder.push_back(n);
        for (int child : tree.children[n]) visit(child);
        const int c1 = tree.children[n][0];
        const int c2 = (tree.children[n].size() > 1) ? tree.children[n][1] : BEAGLE_OP_NONE;
        post.push_back({n, BEAGLE_OP_NONE, BEAGLE_OP_NONE, c1, c1, c2, c2});
    };
    visit(tree.root);
    beagleUpdatePartials(instance, post.data(), (int) post.size(), BEAGLE_OP_NONE);
    const int root = tree.root, weightsIndex = 0, frequencies = 0, noScale = BEAGLE_OP_NONE;
    beagleCalculateRootLogLikelihoods(instance, &root, &weightsIndex, &frequencies, &noScale, 1, &out.lnL);

    // pre-order: bottom partials include the branch above the node, top partials do not
    for (const int rootPre : {bottom(root), top(root)}) { // one buffer per call
        beagleSetRootPrePartials(instance, &rootPre, &frequencies, 1);
    }
    std::vector<BeagleOperation> bottomOps, topOps;
    for (int n : preorder) {
        const std::vector<int>& kids = tree.children[n];
        for (size_t k = 0; k < kids.size(); ++k) {
            const int child = kids[k];
            const int sibling = (kids.size() > 1) ? kids[1 - k] : BEAGLE_OP_NONE;
            bottomOps.push_back({bottom(child), BEAGLE_OP_NONE, BEAGLE_OP_NONE, bottom(n), child, sibling, sibling});
            topOps.push_back({top(child), BEAGLE_OP_NONE, BEAGLE_OP_NONE, top(n),
                              (n == root) ? BEAGLE_OP_NONE : n, sibling, sibling});
        }
    }
    beagleUpdatePrePartials_v5(instance, bottomOps.data(), (int) bottomOps.size(), BEAGLE_OP_NONE,
                               BEAGLE_PARTIALS_BOTTOM);
    beagleUpdatePrePartials_v5(instance, topOps.data(), (int) topOps.size(), BEAGLE_OP_NONE, BEAGLE_PARTIALS_TOP);

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
    out.gradient.assign(S * S, 0.0);
    beagleCalculateAdjointDerivative(instance, branches.data(), 0, 0, root, 0, (int) branches.size(),
                                     out.gradient.data(), NULL);

    auto partials = [&](int buffer) {
        std::vector<double> p(S * C * K);
        beagleGetPartials(instance, buffer, BEAGLE_OP_NONE, p.data());
        return p;
    };
    out.post.clear(); out.bottom.clear(); out.top.clear();
    for (int n = 0; n < M; ++n) {
        if (!tree.children[n].empty()) out.post.push_back(partials(n));
        if (n != root) {
            out.bottom.push_back(partials(bottom(n)));
            out.top.push_back(partials(top(n)));
        }
    }
    beagleFinalizeInstance(instance);
    return true;
}

// max |a - b| / max |b|
double difference(const std::vector<double>& a, const std::vector<double>& b) {
    double error = 0.0, scale = 0.0;
    for (size_t i = 0; i < b.size(); ++i) {
        error = std::max(error, std::fabs(a[i] - b[i]));
        scale = std::max(scale, std::fabs(b[i]));
    }
    return (scale > 0.0) ? error / scale : error;
}

double difference(const std::vector<std::vector<double>>& a, const std::vector<std::vector<double>>& b) {
    double error = 0.0;
    for (size_t i = 0; i < b.size(); ++i) error = std::max(error, difference(a[i], b[i]));
    return error;
}

} // namespace

int main() {
    const double tolerance = 1e-10;
    int compared = 0, failed = 0;
    for (bool spectral : {false, true})
    for (bool complexModel : {false, true})
    for (int S : {2, 3, 4, 5, 6, 7, 8, 9, 16, 17, 20, 31, 32, 61, 64})
    for (int C : {1, 5, 64})
    for (int K : {1, 4})
    for (int threads : {1, 3}) {
        if (threads > 1 && C < 64) continue;
        const Config config{spectral, complexModel, S, C, K, threads};
        Result simd, scalar;
        if (!evaluate(config, true, simd) || !evaluate(config, false, scalar)) {
            printf("skipped %s S=%d C=%d K=%d %s: no implementation\n", spectral ? "spectral" : "standard", S, C, K,
                   complexModel ? "complex" : "real");
            continue;
        }
        const double errors[5] = {std::fabs(simd.lnL - scalar.lnL) / std::fabs(scalar.lnL),
                                  difference(simd.post, scalar.post), difference(simd.bottom, scalar.bottom),
                                  difference(simd.top, scalar.top), difference(simd.gradient, scalar.gradient)};
        const bool ok = std::all_of(errors, errors + 5, [tolerance](double e) { return e <= tolerance; });
        ++compared;
        if (!ok) ++failed;
        printf("%s %-30s vs %-20s S=%2d C=%2d K=%d %-7s threads=%d: lnL %.1e post %.1e bottom %.1e top %.1e "
               "gradient %.1e\n", ok ? "ok  " : "FAIL", simd.name.c_str(), scalar.name.c_str(), S, C, K,
               complexModel ? "complex" : "real", threads, errors[0], errors[1], errors[2], errors[3], errors[4]);
    }
    printf("%d comparisons, %d failed\n", compared, failed);
    return (failed == 0 && compared > 0) ? 0 : 1;
}
