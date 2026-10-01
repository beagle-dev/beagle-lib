/*
 * gpuspectraltest: every GPU spectral kernel against CPU spectral (double), in isolation. Each post-order and
 * pre-order (BOTTOM and TOP) operation runs on its own from the CPU's inputs, and the adjoint gradient from the
 * CPU's partials, so a wrong kernel is reported as such rather than through the errors it propagates. Each
 * post-order operation also runs with fixed scaling, reading the scale factors it wrote.
 *
 * The models are a random reversible one (real eigenvalues) and a random asymmetric circulant one (complex
 * conjugate pairs; the first eigenvalue of a pair is a + bi or a - bi), the latter also with the sign of every
 * pair's first imaginary part reversed (with the matching eigenvectors) and under a diagonal similarity, so that
 * P^T x differs from P x.
 *
 * usage: gpuspectraltest [--states S ...] [--scale x] [--resource r] [--tolerance t]
 * Prints SKIP and succeeds when no GPU resource offers the spectral representation.
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
                const double x = a[k * n + p], y = a[k * n + q];
                a[k * n + p] = c * x - s * y; a[k * n + q] = s * x + c * y;
            }
            for (int k = 0; k < n; ++k) {
                const double x = a[p * n + k], y = a[q * n + k];
                a[p * n + k] = c * x - s * y; a[q * n + k] = s * x + c * y;
            }
            for (int k = 0; k < n; ++k) {
                const double x = v[k * n + p], y = v[k * n + q];
                v[k * n + p] = c * x - s * y; v[k * n + q] = s * x + c * y;
            }
        }
    }
    d.resize(n);
    for (int i = 0; i < n; ++i) d[i] = a[i * n + i];
}

struct Model {
    std::string name;
    bool complex = false;
    std::vector<double> pi, evec, ivec, eval; // eval: n real parts, then n imaginary parts
};

// Random reversible rate matrix Q = R diag(pi); V = diag(pi)^-1/2 U, V^-1 = U^T diag(pi)^1/2
Model reversible(std::mt19937& rng, int n) {
    Model m;
    m.name = "reversible";
    std::uniform_real_distribution<double> u(0.5, 1.5);
    m.pi.resize(n);
    double sum = 0.0;
    for (auto& p : m.pi) { p = u(rng); sum += p; }
    for (auto& p : m.pi) p /= sum;
    std::vector<double> r(n * n, 0.0), q(n * n, 0.0);
    for (int i = 0; i < n; ++i) for (int j = i + 1; j < n; ++j) r[i * n + j] = r[j * n + i] = u(rng);
    double rate = 0.0;
    for (int i = 0; i < n; ++i) {
        double row = 0.0;
        for (int j = 0; j < n; ++j) if (i != j) { q[i * n + j] = r[i * n + j] * m.pi[j]; row += q[i * n + j]; }
        q[i * n + i] = -row;
        rate += m.pi[i] * row;
    }
    std::vector<double> b(n * n), uvec, d;
    for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j)
        b[i * n + j] = std::sqrt(m.pi[i]) * q[i * n + j] / rate / std::sqrt(m.pi[j]);
    jacobi(b, n, uvec, d);
    m.evec.resize(n * n); m.ivec.resize(n * n);
    for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) {
        m.evec[i * n + j] = uvec[i * n + j] / std::sqrt(m.pi[i]);
        m.ivec[i * n + j] = uvec[j * n + i] * std::sqrt(m.pi[j]);
    }
    m.eval = d;
    m.eval.resize(2 * n, 0.0);
    return m;
}

// Random asymmetric circulant rate matrix, rate r_k from state i to i + k (mod n). The real Fourier basis
// diagonalizes it into conjugate pairs a_m +/- i b_m, with b_m of either sign.
Model circulant(std::mt19937& rng, int n) {
    Model m;
    m.name = "circulant";
    m.complex = true;
    std::uniform_real_distribution<double> u(0.5, 1.5);
    std::vector<double> r(n, 0.0);
    double sum = 0.0;
    for (int k = 1; k < n; ++k) { r[k] = u(rng); sum += r[k]; }
    for (auto& x : r) x /= sum;
    const double w = 2.0 * M_PI / n;
    const bool even = (n % 2 == 0);
    const int pairs = even ? n / 2 - 1 : (n - 1) / 2;
    m.pi.assign(n, 1.0 / n);
    m.evec.assign(n * n, 0.0); m.ivec.assign(n * n, 0.0); m.eval.assign(2 * n, 0.0);
    for (int j = 0; j < n; ++j) { m.evec[j * n] = 1.0 / n; m.ivec[j] = 1.0; }
    for (int p = 1; p <= pairs; ++p) {
        const int re = 2 * p - 1, im = 2 * p;
        const double theta = w * p;
        double a = 0.0, b = 0.0;
        for (int k = 1; k < n; ++k) { a += r[k] * (std::cos(theta * k) - 1.0); b += r[k] * std::sin(theta * k); }
        m.eval[re] = a; m.eval[n + re] = b; m.eval[im] = a; m.eval[n + im] = -b;
        for (int j = 0; j < n; ++j) {
            m.evec[j * n + re] = std::cos(theta * j) / n; m.evec[j * n + im] = std::sin(theta * j) / n;
            m.ivec[re * n + j] = 2.0 * std::cos(theta * j); m.ivec[im * n + j] = 2.0 * std::sin(theta * j);
        }
    }
    if (even) {
        const int last = n - 1;
        double a = 0.0;
        for (int k = 1; k < n; ++k) a += r[k] * (((k % 2 == 0) ? 1.0 : -1.0) - 1.0);
        m.eval[last] = a;
        for (int j = 0; j < n; ++j) {
            const double s = (j % 2 == 0) ? 1.0 : -1.0;
            m.evec[j * n + last] = s / n; m.ivec[last * n + j] = s;
        }
    }
    return m;
}

// The same transition matrices with every pair's first eigenvalue a - bi: negate both imaginary parts, the pair's
// second column of V and second row of V^-1
Model flipped(Model m, int n) {
    m.name += ", pairs a - bi first";
    for (int i = 0; i < n; ) {
        if (m.eval[n + i] == 0.0) { ++i; continue; }
        m.eval[n + i] = -m.eval[n + i];
        m.eval[n + i + 1] = -m.eval[n + i + 1];
        for (int j = 0; j < n; ++j) {
            m.evec[j * n + i + 1] = -m.evec[j * n + i + 1];
            m.ivec[(i + 1) * n + j] = -m.ivec[(i + 1) * n + j];
        }
        i += 2;
    }
    return m;
}

// D Q D^-1 for a random positive diagonal D: V -> D V, V^-1 -> V^-1 D^-1. P is then not doubly stochastic, so
// P^T x differs from P x for every x (the kernels do not need a rate matrix).
Model skewed(Model m, std::mt19937& rng, int n) {
    m.name += ", skewed";
    std::uniform_real_distribution<double> u(0.5, 2.0);
    std::vector<double> d(n);
    for (auto& x : d) x = u(rng);
    for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) { m.evec[i * n + j] *= d[i]; m.ivec[i * n + j] /= d[j]; }
    return m;
}

// The largest error of a (category, pattern) relative to that pattern's largest partial; stride 0: one block.
// Infinite if x has a NaN or an infinity.
double relativeError(const std::vector<double>& x, const std::vector<double>& ref, size_t stride) {
    if (stride == 0) stride = ref.size();
    double worst = 0.0;
    for (size_t p = 0; p + stride <= ref.size(); p += stride) {
        double d = 0.0, m = 0.0;
        for (size_t i = p; i < p + stride; ++i) {
            if (!std::isfinite(x[i])) return INFINITY;
            d = std::max(d, std::fabs(x[i] - ref[i]));
            m = std::max(m, std::fabs(ref[i]));
        }
        if (m > 0.0) worst = std::max(worst, d / m);
    }
    return worst;
}

// (((0,1),(2,3)),(4,(5,6))),7) with random branch lengths: every kind of operation occurs, each child type
// in the post-order and each sibling type in the pre-order, at the root and below it. Tips 0-7, root 14.
struct Tree {
    int tips = 8, nodes = 15, root = 14;
    std::vector<std::vector<int>> children = {{}, {}, {}, {}, {}, {}, {}, {},
                                              {0, 1}, {2, 3}, {8, 9}, {5, 6}, {4, 11}, {10, 12}, {13, 7}};
    std::vector<double> lengths;
};

Tree randomLengths(std::mt19937& rng, double scale) {
    Tree t;
    for (int n = 0; n < t.nodes; n++) t.lengths.push_back(scale * std::uniform_real_distribution<double>(0.1, 0.5)(rng));
    return t;
}

} // namespace

int main(int argc, char** argv) {
    std::vector<int> stateCounts;
    double scale = 10.0, tolerance = -1.0;
    int gpuResource = -1;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--states")) { while (i + 1 < argc && argv[i + 1][0] != '-') stateCounts.push_back(atoi(argv[++i])); }
        else if (!strcmp(argv[i], "--scale") && i + 1 < argc) scale = atof(argv[++i]);
        else if (!strcmp(argv[i], "--resource") && i + 1 < argc) gpuResource = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tolerance") && i + 1 < argc) tolerance = atof(argv[++i]);
        else { fprintf(stderr, "usage: gpuspectraltest [--states S ...] [--scale x] [--resource r] [--tolerance t]\n"); return 2; }
    }
    if (stateCounts.empty()) stateCounts = {4, 17, 61};

    BeagleResourceList* resources = beagleGetResourceList();
    if (gpuResource < 0) {
        for (int r = 0; r < resources->length && gpuResource < 0; r++)
            if (resources->list[r].supportFlags & BEAGLE_FLAG_PROCESSOR_GPU) gpuResource = r;
    }
    if (gpuResource < 0 || gpuResource >= resources->length) {
        printf("SKIP: no GPU resource\n");
        return 0;
    }
    const bool gpuDouble = (resources->list[gpuResource].supportFlags & BEAGLE_FLAG_PRECISION_DOUBLE) != 0;
    if (tolerance < 0.0) tolerance = gpuDouble ? 1e-9 : 1e-3;

    const int N = 8, C = 37, K = 2;
    const double rates[K] = {0.5, 1.5}, weights[K] = {0.5, 0.5};
    bool ok = true;

    for (const int S : stateCounts) {
        std::mt19937 rng(S);
        const Tree tree = randomLengths(rng, scale);
        const int M = tree.nodes, root = tree.root;
        std::vector<std::vector<int>> states(N, std::vector<int>(C));
        for (auto& tip : states) for (int k = 0; k < C; k++) // every 11th pattern missing
            tip[k] = (k % 11 == 5) ? S : std::uniform_int_distribution<int>(0, S - 1)(rng);
        std::vector<double> prior((size_t) K * C * S); // an arbitrary root prior, so no pre-order partials are uniform
        for (auto& x : prior) x = std::uniform_real_distribution<double>(0.5, 1.5)(rng);

        const Model circ = circulant(rng, S);
        const std::vector<Model> models = {reversible(rng, S), circ, skewed(flipped(circ, S), rng, S)};

        // buffers: post-order 0..M-1, BOTTOM pre-order M..2M-1, TOP pre-order 2M..3M-1
        std::vector<BeagleOperation> post, bottom, top;
        for (int n = N; n < M; n++) {
            const int c1 = tree.children[n][0], c2 = tree.children[n][1];
            post.push_back({n, BEAGLE_OP_NONE, BEAGLE_OP_NONE, c1, c1, c2, c2});
        }
        for (int n = M - 1; n >= N; n--) for (int k = 0; k < 2; k++) {
            const int c = tree.children[n][k], s = tree.children[n][1 - k];
            bottom.push_back({M + c, BEAGLE_OP_NONE, BEAGLE_OP_NONE, M + n, c, s, s});
            top.push_back({2 * M + c, BEAGLE_OP_NONE, BEAGLE_OP_NONE, 2 * M + n, n == root ? BEAGLE_OP_NONE : n, s, s});
        }
        std::vector<BeagleBranchOperation> branches;
        for (int n = 0; n < M; n++) if (n != root) branches.push_back({n, 2 * M + n, n, 0});
        std::vector<int> matrices;
        std::vector<double> lengths;
        for (int n = 0; n < M; n++) if (n != root) { matrices.push_back(n); lengths.push_back(tree.lengths[n]); }

        for (const Model& model : models) {
            auto create = [&](bool gpu, std::string& name) {
                const long flags = (model.complex ? BEAGLE_FLAG_EIGEN_COMPLEX : BEAGLE_FLAG_EIGEN_REAL) |
                                   BEAGLE_FLAG_SPECTRAL_REPRESENTATION | BEAGLE_FLAG_PREORDER_TRANSPOSE_AUTO |
                                   (gpu ? BEAGLE_FLAG_PROCESSOR_GPU : BEAGLE_FLAG_PROCESSOR_CPU) |
                                   (gpu && !gpuDouble ? BEAGLE_FLAG_PRECISION_SINGLE : BEAGLE_FLAG_PRECISION_DOUBLE);
                int resource = gpu ? gpuResource : 0;
                BeagleInstanceDetails details;
                const int instance = beagleCreateInstance(N, 3 * M, N, S, C, 1, M, K, (int) post.size(), &resource, 1,
                                                          BEAGLE_FLAG_SCALERS_RAW | BEAGLE_FLAG_SCALING_MANUAL, flags,
                                                          &details);
                if (instance < 0) return instance;
                name = details.implName;
                for (int t = 0; t < N; t++) beagleSetTipStates(instance, t, states[t].data());
                std::vector<double> patternWeights(C, 1.0);
                beagleSetCategoryRates(instance, rates);
                beagleSetPatternWeights(instance, patternWeights.data());
                beagleSetStateFrequencies(instance, 0, model.pi.data());
                beagleSetCategoryWeights(instance, 0, weights);
                beagleSetEigenDecomposition(instance, 0, model.evec.data(), model.ivec.data(), model.eval.data());
                beagleUpdateTransitionMatrices(instance, 0, matrices.data(), NULL, NULL, lengths.data(),
                                               (int) matrices.size());
                return instance;
            };
            auto partials = [&](int instance, int buffer) {
                std::vector<double> out((size_t) K * C * S);
                beagleGetPartials(instance, buffer, BEAGLE_OP_NONE, out.data());
                return out;
            };

            std::string cpuName, gpuName;
            const int cpu = create(false, cpuName);
            if (cpu < 0) { printf("FAIL: no CPU spectral instance (%d)\n", cpu); return 1; }
            beagleUpdatePartials(cpu, post.data(), (int) post.size(), BEAGLE_OP_NONE);
            beagleSetPartials(cpu, M + root, prior.data());
            beagleSetPartials(cpu, 2 * M + root, prior.data());
            beagleUpdatePrePartials_v5(cpu, bottom.data(), (int) bottom.size(), BEAGLE_OP_NONE, BEAGLE_PARTIALS_BOTTOM);
            beagleUpdatePrePartials_v5(cpu, top.data(), (int) top.size(), BEAGLE_OP_NONE, BEAGLE_PARTIALS_TOP);
            std::vector<std::vector<double>> ref(3 * M);
            for (int b = N; b < 3 * M; b++) ref[b] = partials(cpu, b);
            std::vector<double> cpuGradient(S * S), gpuGradient(S * S);
            beagleCalculateAdjointDerivative(cpu, branches.data(), 0, 0, root, 0, (int) branches.size(),
                                             cpuGradient.data(), NULL);
            beagleFinalizeInstance(cpu);

            const int gpu = create(true, gpuName);
            if (gpu < 0) {
                printf("SKIP: no GPU spectral instance on resource %d (%d)\n", gpuResource, gpu);
                return 0;
            }
            // the worst error of each kind of operation
            std::vector<std::pair<std::string, double>> worst;
            auto record = [&](const std::string& kind, double error) {
                for (auto& w : worst) if (w.first == kind) { w.second = std::max(w.second, error); return; }
                worst.push_back({kind, error});
            };
            for (auto& op : post) {
                for (int c : {op.child1Partials, op.child2Partials}) if (c >= N) beagleSetPartials(gpu, c, ref[c].data());
                beagleUpdatePartials(gpu, &op, 1, BEAGLE_OP_NONE);
                const int tipChildren = (op.child1Partials < N) + (op.child2Partials < N);
                record(tipChildren == 2 ? "post-order, states x states" :
                       tipChildren == 1 ? "post-order, states x partials" : "post-order, partials x partials",
                       relativeError(partials(gpu, op.destinationPartials), ref[op.destinationPartials], S));
            }
            for (int n = N; n < M; n++) beagleSetPartials(gpu, n, ref[n].data());
            for (const bool isTop : {false, true}) {
                for (auto& op : isTop ? top : bottom) {
                    const int parent = op.child1Partials;
                    const bool atRoot = (parent == M + root || parent == 2 * M + root);
                    beagleSetPartials(gpu, parent, atRoot ? prior.data() : ref[parent].data());
                    beagleUpdatePrePartials_v5(gpu, &op, 1, BEAGLE_OP_NONE,
                                               isTop ? BEAGLE_PARTIALS_TOP : BEAGLE_PARTIALS_BOTTOM);
                    const std::string sibling = op.child2Partials < N ? "states sibling" : "partials sibling";
                    record(isTop ? (atRoot ? "TOP at the root, " : "TOP, ") + sibling : "BOTTOM, " + sibling,
                           relativeError(partials(gpu, op.destinationPartials), ref[op.destinationPartials], S));
                }
            }
            for (int b = 2 * M; b < 3 * M; b++) beagleSetPartials(gpu, b, b == 2 * M + root ? prior.data() : ref[b].data());
            beagleCalculateAdjointDerivative(gpu, branches.data(), 0, 0, root, 0, (int) branches.size(),
                                             gpuGradient.data(), NULL);
            record("adjoint gradient", relativeError(gpuGradient, cpuGradient, 0));
            // fixed scaling: each post-order operation writing its scale factors (into buffer M + n), then reading
            // them back (the fixed-scale kernels, into buffer 2M + n); the children are still the CPU's
            for (size_t i = 0; i < post.size(); i++) {
                const int n = post[i].destinationPartials;
                BeagleOperation writing = post[i], reading = post[i];
                writing.destinationPartials = M + n;
                writing.destinationScaleWrite = (int) i;
                reading.destinationPartials = 2 * M + n;
                reading.destinationScaleRead = (int) i;
                beagleUpdatePartials(gpu, &writing, 1, BEAGLE_OP_NONE);
                beagleUpdatePartials(gpu, &reading, 1, BEAGLE_OP_NONE);
                record("post-order, fixed scaling", relativeError(partials(gpu, 2 * M + n), partials(gpu, M + n), S));
            }
            beagleFinalizeInstance(gpu);

            printf("%d states, %s, branch scale %g: %s against %s\n", S, model.name.c_str(), scale,
                   gpuName.c_str(), cpuName.c_str());
            for (auto& w : worst) {
                const bool pass = w.second < tolerance;
                ok = ok && pass;
                printf("  %-34s %.1e%s\n", w.first.c_str(), w.second, pass ? "" : "  FAIL");
            }
        }
    }
    printf("%s (tolerance %.0e, relative to each pattern's largest partial)\n", ok ? "PASS" : "FAIL", tolerance);
    return ok ? 0 : 1;
}
