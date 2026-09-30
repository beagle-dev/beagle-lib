/*
 * Copyright 2026 Phylogenetic Likelihood Working Group
 * This file is part of BEAGLE.
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 *
 * spectralbench: times a log likelihood and its adjoint gradient with one BEAGLE CPU implementation, the way an
 * MCMC step after a change of the substitution model computes them, and prints one CSV line.
 *
 * Phases of one evaluation:
 *   eigen     beagleSetEigenDecomposition
 *   matrices  beagleUpdateTransitionMatrices for every branch
 *   post      beagleUpdatePartials (post-order traversal)
 *   root      beagleCalculateRootLogLikelihoods
 *   pre       beagleUpdatePrePartials_v5 with TOP partials (pre-order traversal)
 *   adjoint   beagleCalculateAdjointDerivative over every branch
 * likelihood = eigen + matrices + post + root; gradient = likelihood + pre + adjoint.
 *
 * Usage:
 *   spectralbench --impl standard|spectral --states S --patterns C --categories K
 *                 [--vector sse|none] [--tips N] [--budget seconds] [--minreps n] [--seed n]
 *                 [--complex] [--tippartials] [--nogradient] [--header]
 *
 * The model is a random reversible (GTR-like) rate matrix, or with --complex a random asymmetric circulant rate
 * matrix, whose eigenvalues are complex conjugate pairs. Tips are random states (compact buffers), or random partials with --tippartials. The tree
 * is a random binary tree with branch lengths uniform on (0.05, 0.3). Evaluations repeat until --budget seconds
 * have passed and at least --minreps evaluations were timed, after one untimed evaluation. Each reported time is
 * the median over the timed evaluations (likelihood and gradient are medians of the per-evaluation sums).
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "libhmsbeagle/beagle.h"

namespace {

struct Options {
    std::string impl;              // standard | spectral
    std::string vector = "sse";    // sse | none
    int states = 0, patterns = 0, categories = 1, tips = 64, minReps = 3, seed = 1;
    double budget = 0.5;
    bool complexModel = false, tipPartials = false, gradient = true, header = false;
};

void usage() {
    fprintf(stderr, "usage: spectralbench --impl standard|spectral --states S --patterns C --categories K\n"
                    "                     [--vector sse|none] [--tips N] [--budget seconds] [--minreps n]\n"
                    "                     [--seed n] [--complex] [--tippartials] [--nogradient] [--header]\n");
    exit(2);
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

// Random reversible rate matrix Q = R diag(pi), one substitution per unit time; V = diag(pi)^-1/2 U, V^-1 = U^T diag(pi)^1/2
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

// Random asymmetric circulant rate matrix: rate r_k from state i to state i + k (mod n), sum r_k = 1. The real
// Fourier basis diagonalizes it into complex conjugate eigenvalue pairs a_m +/- i b_m (and real eigenvalues for
// m = 0 and, for even n, m = n / 2), with a_m = sum_k r_k (cos(w m k) - 1) and b_m = sum_k r_k sin(w m k).
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

struct Tree {
    std::vector<int> parent;
    std::vector<std::vector<int>> children;
    std::vector<double> length;
    int root = -1;
};

// Random binary tree: tips 0..tips-1, internal nodes numbered in the order they are joined, the root last
Tree randomTree(std::mt19937& rng, int tips) {
    Tree t;
    const int nodes = 2 * tips - 1;
    t.parent.assign(nodes, -1);
    t.children.assign(nodes, {});
    t.length.assign(nodes, 0.0);
    std::uniform_real_distribution<double> u(0.05, 0.3);
    std::vector<int> active;
    for (int i = 0; i < tips; ++i) active.push_back(i);
    for (int next = tips; next < nodes; ++next) {
        const int a = std::uniform_int_distribution<int>(0, (int) active.size() - 1)(rng);
        const int x = active[a];
        active.erase(active.begin() + a);
        const int b = std::uniform_int_distribution<int>(0, (int) active.size() - 1)(rng);
        const int y = active[b];
        active.erase(active.begin() + b);
        t.children[next] = {x, y};
        t.parent[x] = t.parent[y] = next;
        active.push_back(next);
    }
    t.root = nodes - 1;
    for (int n = 0; n < nodes; ++n) if (n != t.root) t.length[n] = u(rng);
    return t;
}

double milliseconds(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

double median(std::vector<double> x) {
    std::sort(x.begin(), x.end());
    const size_t n = x.size();
    return (n % 2 == 1) ? x[n / 2] : 0.5 * (x[n / 2 - 1] + x[n / 2]);
}

} // namespace

int main(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* { if (i + 1 >= argc) usage(); return argv[++i]; };
        if (a == "--impl") o.impl = next();
        else if (a == "--vector") o.vector = next();
        else if (a == "--states") o.states = atoi(next());
        else if (a == "--patterns") o.patterns = atoi(next());
        else if (a == "--categories") o.categories = atoi(next());
        else if (a == "--tips") o.tips = atoi(next());
        else if (a == "--budget") o.budget = atof(next());
        else if (a == "--minreps") o.minReps = atoi(next());
        else if (a == "--seed") o.seed = atoi(next());
        else if (a == "--complex") o.complexModel = true;
        else if (a == "--tippartials") o.tipPartials = true;
        else if (a == "--nogradient") o.gradient = false;
        else if (a == "--header") o.header = true;
        else usage();
    }
    if (o.header) {
        printf("implementation,impl,vector,states,patterns,categories,tips,model,tipdata,reps,"
               "eigen_ms,matrices_ms,post_ms,root_ms,pre_ms,adjoint_ms,likelihood_ms,gradient_ms,lnL\n");
        return 0;
    }
    if ((o.impl != "standard" && o.impl != "spectral") || (o.vector != "sse" && o.vector != "none") ||
        o.states < 2 || o.patterns < 1 || o.categories < 1 || o.tips < 3) {
        usage();
    }

    const int S = o.states, C = o.patterns, K = o.categories, N = o.tips;
    const int nodes = 2 * N - 1;
    auto pre = [N](int node) { return 2 * N - 1 + node; }; // pre-order buffers follow the post-order ones

    long requirements = BEAGLE_FLAG_PROCESSOR_CPU | BEAGLE_FLAG_PRECISION_DOUBLE |
                        (o.vector == "sse" ? BEAGLE_FLAG_VECTOR_SSE : BEAGLE_FLAG_VECTOR_NONE);
    if (o.gradient) requirements |= BEAGLE_FLAG_PREORDER_TRANSPOSE_AUTO;
    if (o.impl == "spectral") requirements |= BEAGLE_FLAG_SPECTRAL_REPRESENTATION;
    if (o.complexModel) requirements |= BEAGLE_FLAG_EIGEN_COMPLEX;
    const long preferences = BEAGLE_FLAG_THREADING_NONE | BEAGLE_FLAG_SCALING_MANUAL;

    BeagleInstanceDetails details;
    const int instance = beagleCreateInstance(N, 4 * N - 2, o.tipPartials ? 0 : N, S, C, 1, nodes, K, 0, NULL, 0,
                                              preferences, requirements, &details);
    if (instance < 0) {
        fprintf(stderr, "spectralbench: no implementation for these flags (error %d)\n", instance);
        return 1;
    }

    std::mt19937 rng(o.seed);
    const Tree tree = randomTree(rng, N);

    std::vector<double> pi, evec, ivec, eval;
    if (o.complexModel) randomCirculant(rng, S, pi, evec, ivec, eval);
    else randomReversible(rng, S, pi, evec, ivec, eval);

    beagleSetStateFrequencies(instance, 0, pi.data());
    std::vector<double> rates(K), weights(K, 1.0 / K);
    for (int k = 0; k < K; ++k) rates[k] = (K == 1) ? 1.0 : 0.25 + 1.5 * k / (K - 1); // mean 1
    beagleSetCategoryRates(instance, rates.data());
    beagleSetCategoryWeights(instance, 0, weights.data());
    std::vector<double> patternWeights(C, 1.0);
    beagleSetPatternWeights(instance, patternWeights.data());

    for (int tip = 0; tip < N; ++tip) {
        if (o.tipPartials) {
            std::uniform_real_distribution<double> u(0.0, 1.0);
            std::vector<double> partials(S * C);
            for (auto& x : partials) x = u(rng);
            beagleSetTipPartials(instance, tip, partials.data());
        } else {
            std::uniform_int_distribution<int> state(0, S - 1);
            std::vector<int> states(C);
            for (auto& s : states) s = state(rng);
            beagleSetTipStates(instance, tip, states.data());
        }
    }

    // post-order: internal nodes in the order they were joined (children first)
    std::vector<BeagleOperation> post;
    for (int n = N; n < nodes; ++n) {
        const int c1 = tree.children[n][0], c2 = tree.children[n][1];
        post.push_back({n, BEAGLE_OP_NONE, BEAGLE_OP_NONE, c1, c1, c2, c2});
    }
    // TOP pre-order: parents first; at the root the parent partials are the root prior
    std::vector<BeagleOperation> top;
    for (int n = nodes - 1; n >= N; --n) {
        for (int k = 0; k < 2; ++k) {
            const int c = tree.children[n][k], s = tree.children[n][1 - k];
            top.push_back({pre(c), BEAGLE_OP_NONE, BEAGLE_OP_NONE, pre(n), (n == tree.root) ? BEAGLE_OP_NONE : n, s, s});
        }
    }
    std::vector<int> matrixIndices;
    std::vector<double> lengths;
    std::vector<BeagleBranchOperation> branches;
    for (int n = 0; n < nodes; ++n) {
        if (n == tree.root) continue;
        matrixIndices.push_back(n);
        lengths.push_back(tree.length[n]);
        BeagleBranchOperation b;
        b.postOrderPartials = n;
        b.preOrderPartials = pre(n);
        b.branchTransitionMatrix = n;
        b.resultSegment = 0;
        branches.push_back(b);
    }
    const int rootPre = pre(tree.root), frequencies = 0;
    beagleSetRootPrePartials(instance, &rootPre, &frequencies, 1);

    const int root = tree.root, weightsIndex = 0, scaleIndex = BEAGLE_OP_NONE;
    std::vector<double> gradient(S * S);
    std::vector<double> t[8]; // eigen, matrices, post, root, pre, adjoint, likelihood, gradient
    double lnL = 0.0;
    int reps = 0;
    const auto start = std::chrono::steady_clock::now();
    for (int r = -1; ; ++r) { // r = -1 is an untimed first evaluation
        const auto a = std::chrono::steady_clock::now();
        beagleSetEigenDecomposition(instance, 0, evec.data(), ivec.data(), eval.data());
        const auto b = std::chrono::steady_clock::now();
        beagleUpdateTransitionMatrices(instance, 0, matrixIndices.data(), NULL, NULL, lengths.data(),
                                       (int) matrixIndices.size());
        const auto c = std::chrono::steady_clock::now();
        beagleUpdatePartials(instance, post.data(), (int) post.size(), BEAGLE_OP_NONE);
        const auto d = std::chrono::steady_clock::now();
        beagleCalculateRootLogLikelihoods(instance, &root, &weightsIndex, &frequencies, &scaleIndex, 1, &lnL);
        const auto e = std::chrono::steady_clock::now();
        auto f = e, g = e;
        if (o.gradient) {
            beagleUpdatePrePartials_v5(instance, top.data(), (int) top.size(), BEAGLE_OP_NONE, BEAGLE_PARTIALS_TOP);
            f = std::chrono::steady_clock::now();
            beagleCalculateAdjointDerivative(instance, branches.data(), 0, 0, root, 0, (int) branches.size(),
                                             gradient.data(), NULL);
            g = std::chrono::steady_clock::now();
        }
        if (r < 0) continue;
        const double phase[6] = {milliseconds(a, b), milliseconds(b, c), milliseconds(c, d), milliseconds(d, e),
                                 milliseconds(e, f), milliseconds(f, g)};
        for (int p = 0; p < 6; ++p) t[p].push_back(phase[p]);
        t[6].push_back(milliseconds(a, e));
        t[7].push_back(milliseconds(a, g));
        ++reps;
        if (reps >= o.minReps && std::chrono::duration<double>(g - start).count() >= o.budget) break;
    }
    double m[8];
    for (int p = 0; p < 8; ++p) m[p] = median(t[p]);
    printf("%s,%s,%s,%d,%d,%d,%d,%s,%s,%d,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.10g\n",
           details.implName, o.impl.c_str(), o.vector.c_str(), S, C, K, N, o.complexModel ? "complex" : "reversible",
           o.tipPartials ? "partials" : "states", reps, m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], lnL);
    beagleFinalizeInstance(instance);
    return 0;
}
