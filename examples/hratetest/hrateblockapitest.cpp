#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>
#include "libhmsbeagle/beagle.h"
#include "hratereference.h"

static int checks = 0;
static int failures = 0;

static void report(const std::string& name, bool pass, const std::string& detail) {
    ++checks;
    if (!pass)
        ++failures;
    if (!pass || getenv("HRATE_VERBOSE"))
        printf("%-78s %s  %s\n", name.c_str(), pass ? "PASS" : "FAIL", detail.c_str());
}

static const int N = 16;
static const int S = 4;
static const int K = 2400;
static const int C = 2;
static const int INTERNAL = N - 1;
static const int NODES = 2 * N - 1;
static const int ROOT = NODES - 1;

struct Phylogeny {
    std::vector<int> left, right, parent;
    std::vector<double> length;
    std::vector<int> order;
};

static Phylogeny makePhylogeny(std::mt19937_64& rng) {
    Phylogeny p;
    p.left.assign(NODES, -1);
    p.right.assign(NODES, -1);
    p.parent.assign(NODES, -1);
    p.length.assign(NODES, 0.0);
    std::vector<int> pool;
    for (int t = 0; t < N; t++)
        pool.push_back(t);
    std::uniform_real_distribution<double> len(0.02, 0.25);
    for (int m = N; m < NODES; m++) {
        int i = (int) (rng() % pool.size());
        const int a = pool[i];
        pool.erase(pool.begin() + i);
        i = (int) (rng() % pool.size());
        const int b = pool[i];
        pool.erase(pool.begin() + i);
        p.left[m] = a;
        p.right[m] = b;
        p.parent[a] = m;
        p.parent[b] = m;
        p.length[a] = len(rng);
        p.length[b] = len(rng);
        p.order.push_back(m);
        pool.push_back(m);
    }
    return p;
}

static std::vector<std::vector<int>> makeTips(std::mt19937_64& rng) {
    std::vector<std::vector<int>> tips(N, std::vector<int>(K));
    const int pools[3][4] = { { 0, 2, 0, 2 }, { 0, 1, 2, 3 }, { 1, 3, 1, 3 } };
    for (int k = 0; k < K; k++) {
        const int group = k < K / 3 ? 0 : k < 2 * K / 3 ? 1 : 2;
        for (int t = 0; t < N; t++)
            tips[t][k] = (rng() % 20 == 0) ? S : pools[group][rng() % 4];
    }
    return tips;
}

struct Run {
    int instance = -1;
    std::string label;
};

static Run create(bool aggregate, bool single, int threads) {
    Run r;
    long req = BEAGLE_FLAG_PROCESSOR_CPU | BEAGLE_FLAG_HIERARCHICAL_REPRESENTATION |
               (single ? BEAGLE_FLAG_PRECISION_SINGLE : BEAGLE_FLAG_PRECISION_DOUBLE) |
               (aggregate ? BEAGLE_FLAG_HIERARCHICAL_AGGREGATE_ON : BEAGLE_FLAG_HIERARCHICAL_AGGREGATE_OFF);
    long pref = (single ? BEAGLE_FLAG_PRECISION_SINGLE : BEAGLE_FLAG_PRECISION_DOUBLE) |
                (threads > 1 ? BEAGLE_FLAG_THREADING_CPP : BEAGLE_FLAG_THREADING_NONE);
    BeagleInstanceDetails det;
    r.instance = beagleCreateInstance(N, INTERNAL, N, S, K, 2, NODES - 1, C, 0, NULL, 0, pref, req, &det);
    if (r.instance >= 0 && threads > 1)
        beagleSetCPUThreadCount(r.instance, threads);
    return r;
}

static std::vector<BeagleOperation> fullOps(const Phylogeny& p) {
    std::vector<BeagleOperation> ops;
    for (int m : p.order)
        ops.push_back({ m, BEAGLE_OP_NONE, BEAGLE_OP_NONE, p.left[m], p.left[m], p.right[m], p.right[m] });
    return ops;
}

static std::vector<BeagleOperation> pathOps(const Phylogeny& p, int from) {
    std::vector<BeagleOperation> ops;
    for (int m = from < N ? p.parent[from] : from; m >= 0; m = p.parent[m])
        ops.push_back({ m, BEAGLE_OP_NONE, BEAGLE_OP_NONE, p.left[m], p.left[m], p.right[m], p.right[m] });
    return ops;
}

struct Pair {
    Run on, off;
    bool single;
    std::string tag;
    template <typename F>
    void both(F f) {
        f(on.instance);
        f(off.instance);
    }
    double lnL(int instance) {
        int root = ROOT, zero = 0, none = BEAGLE_OP_NONE;
        double v = 0.0;
        beagleCalculateRootLogLikelihoods(instance, &root, &zero, &zero, &none, 1, &v);
        return v;
    }
    void compare(const std::string& step, const std::vector<int>& buffers, bool blocksExpected = false) {
        const double tol = single ? 2e-5 : 1e-12;
        const double a = lnL(on.instance), b = lnL(off.instance);
        double worst = 0.0;
        std::vector<double> pa((size_t) K * S * C), pb((size_t) K * S * C);
        for (int buffer : buffers) {
            beagleGetPartials(on.instance, buffer, BEAGLE_OP_NONE, pa.data());
            beagleGetPartials(off.instance, buffer, BEAGLE_OP_NONE, pb.data());
            for (size_t i = 0; i < pa.size(); i++)
                worst = std::max(worst, std::fabs(pa[i] - pb[i]) / std::max(std::fabs(pb[i]), 1e-290));
        }
        const double rel = std::fabs(a - b) / std::fabs(b);
        char detail[160];
        snprintf(detail, sizeof(detail), "lnL %.10f vs %.10f (rel %.1e), partials maxrel %.1e", a, b, rel, worst);
        report(tag + " " + step, std::isfinite(a) && rel <= tol && worst <= tol, detail);
        if (blocksExpected)
            report(tag + " " + step + ": blocks ran", worst > 0.0, "partials differ from aggregation off in the last bits");
    }
};

static void setModel(Pair& pr, const StateTree& tree, const Phylogeny& p, bool frequencies) {
    ModelArrays m = modelArrays(tree);
    std::vector<double> pi(tree.pi.begin(), tree.pi.begin() + S);
    std::vector<int> idx;
    std::vector<double> len;
    for (int n = 0; n < NODES - 1; n++) {
        idx.push_back(n);
        len.push_back(p.length[n]);
    }
    pr.both([&](int i) {
        if (frequencies)
            beagleSetStateFrequencies(i, 0, pi.data());
        beagleSetEigenDecomposition(i, 0, m.vectors.data(), m.inverse.data(), m.values.data());
        beagleUpdateTransitionMatrices(i, 0, idx.data(), NULL, NULL, len.data(), (int) idx.size());
    });
}

static void scenario(bool single, int threads, const char* store) {
    setenv("BEAGLE_HRATE_BLOCK_STORE", store, 1);
    std::mt19937_64 rng(20261119);
    Pair pr;
    pr.single = single;
    char tag[96];
    snprintf(tag, sizeof(tag), "%s %d thread%s %s:", single ? "single" : "double", threads, threads > 1 ? "s" : "",
             store);
    pr.tag = tag;
    pr.on = create(true, single, threads);
    pr.off = create(false, single, threads);
    unsetenv("BEAGLE_HRATE_BLOCK_STORE");
    if (pr.on.instance < 0 || pr.off.instance < 0) {
        report(pr.tag + " create", false, "no hierarchical CPU instance");
        return;
    }
    Phylogeny p = makePhylogeny(rng);
    std::vector<std::vector<int>> tips = makeTips(rng);
    const std::vector<double> rates = { 0.4, 1.6 }, weights = { 0.5, 0.5 };
    std::vector<double> patternWeights(K, 1.0);
    for (int k = 0; k < K; k++)
        patternWeights[k] = 1.0 + (double) (k % 3);
    std::vector<double> pi0 = { 0.3, 0.2, 0.25, 0.25 }, pi1 = { 0.22, 0.31, 0.19, 0.28 };
    StateTree tn93 = makeAGCTTree(1.1, 2.3, 0.7, pi0);
    pr.both([&](int i) {
        for (int t = 0; t < N; t++)
            beagleSetTipStates(i, t, tips[t].data());
        beagleSetCategoryRates(i, rates.data());
        beagleSetCategoryWeights(i, 0, weights.data());
        beagleSetPatternWeights(i, patternWeights.data());
    });
    std::vector<int> internal;
    for (int m = N; m < NODES; m++)
        internal.push_back(m);
    std::vector<BeagleOperation> full = fullOps(p);
    auto update = [&](const std::vector<BeagleOperation>& ops) {
        pr.both([&](int i) { beagleUpdatePartials(i, ops.data(), (int) ops.size(), BEAGLE_OP_NONE); });
    };

    setModel(pr, tn93, p, true);
    update(full);
    pr.compare("1 full evaluation", { ROOT, p.order[0], p.order[INTERNAL / 2] }, true);

    pr.both([&](int i) { beagleSetStateFrequencies(i, 0, pi1.data()); });
    update(full);
    pr.compare("2 frequencies changed, model not re-uploaded", { ROOT });

    StateTree tn93b = makeAGCTTree(1.1, 2.3, 0.7, pi1);
    setModel(pr, tn93b, p, false);
    update(full);
    pr.compare("3 model re-uploaded", { ROOT }, true);

    const int branch = p.order[2];
    p.length[branch] *= 1.7;
    {
        ModelArrays m = modelArrays(tn93b);
        int index = branch;
        double length = p.length[branch];
        pr.both([&](int i) { beagleUpdateTransitionMatrices(i, 0, &index, NULL, NULL, &length, 1); });
    }
    update(pathOps(p, branch));
    pr.compare("4 partial traversal after a branch change", { ROOT }, true);

    for (int k = 0; k < K; k += 7)
        tips[0][k] = tips[0][k] == S ? S : tips[0][k] ^ 2;
    pr.both([&](int i) { beagleSetTipStates(i, 0, tips[0].data()); });
    update(pathOps(p, 0));
    pr.compare("5 tip changed, partial traversal from it", { ROOT });
    update(full);
    pr.compare("5b then a full traversal", { ROOT }, true);

    const int written = p.order[INTERNAL / 2];
    {
        std::vector<double> values((size_t) K * S * C);
        beagleGetPartials(pr.off.instance, written, BEAGLE_OP_NONE, values.data());
        for (size_t i = 0; i < values.size(); i++)
            values[i] *= 1.0 + 1e-2 * (double) (i % S);
        pr.both([&](int i) { beagleSetPartials(i, written, values.data()); });
    }
    update(pathOps(p, p.parent[written]));
    pr.compare("6 internal buffer written by the client, then read", { ROOT, written });
    update(full);

    StateTree caterpillar;
    caterpillar.S = S;
    caterpillar.parent.assign(2 * S - 1, -1);
    caterpillar.left.assign(2 * S - 1, -1);
    caterpillar.right.assign(2 * S - 1, -1);
    caterpillar.lambda.assign(2 * S - 1, 0.0);
    join(caterpillar, 4, 0, 2);
    join(caterpillar, 5, 4, 1);
    join(caterpillar, 6, 5, 3);
    caterpillar.lambda[4] = 1.9;
    caterpillar.lambda[5] = 0.8;
    caterpillar.lambda[6] = 1.3;
    finishTree(caterpillar, pi1);
    if (threads == 1) {
        std::vector<int> partition(K);
        for (int k = 0; k < K; k++)
            partition[k] = k < K / 2 ? 0 : 1;
        int rc = 0;
        pr.both([&](int i) { rc |= beagleSetPatternPartitions(i, 2, partition.data()); });
        if (rc == 0) {
            std::vector<BeagleOperationByPartition> ops;
            for (int part = 0; part < 2; part++)
                for (const BeagleOperation& o : full)
                    ops.push_back({ o.destinationPartials, BEAGLE_OP_NONE, BEAGLE_OP_NONE, o.child1Partials,
                                    o.child1TransitionMatrix, o.child2Partials, o.child2TransitionMatrix, part,
                                    BEAGLE_OP_NONE });
            setModel(pr, caterpillar, p, false);
            pr.both([&](int i) { beagleUpdatePartialsByPartition(i, ops.data(), (int) ops.size()); });
            pr.compare("7 client partitions under another character tree", { ROOT });
            setModel(pr, tn93b, p, false);
            update(pathOps(p, p.order[1]));
            pr.compare("7b then a partial traversal through updatePartials under the first tree", { ROOT });
            update(full);
            pr.compare("7c then a full traversal", { ROOT }, true);
        } else {
            report(pr.tag + " 7 client partitions", false, "setPatternPartitions failed");
        }
    }

    setModel(pr, caterpillar, p, true);
    update(full);
    pr.compare("8 character tree changed, full traversal", { ROOT }, true);
    p.length[p.order[3]] *= 0.6;
    {
        int index = p.order[3];
        double length = p.length[index];
        pr.both([&](int i) { beagleUpdateTransitionMatrices(i, 0, &index, NULL, NULL, &length, 1); });
    }
    update(pathOps(p, p.order[3]));
    pr.compare("8b partial traversal under the new tree", { ROOT }, true);

    std::vector<BeagleOperation> twice = full;
    twice.push_back(full.back());
    update(twice);
    pr.compare("9 one destination written twice in a call", { ROOT });

    pr.compare("10 every internal buffer", internal);

    pr.both([&](int i) { beagleSetCPUThreadCount(i, threads == 1 ? 4 : 1); });
    update(pathOps(p, p.order[4]));
    pr.compare("11 thread count changed, partial traversal", { ROOT });
    update(full);
    pr.compare("11b then a full traversal", internal, true);
    beagleFinalizeInstance(pr.on.instance);
    beagleFinalizeInstance(pr.off.instance);
}

int main() {
    for (bool single : { false, true })
        for (int threads : { 1, 4 })
            for (const char* store : { "compact", "broadcast" })
                scenario(single, threads, store);
    printf("%d of %d checks passed\n", checks - failures, checks);
    return failures == 0 ? 0 : 1;
}
