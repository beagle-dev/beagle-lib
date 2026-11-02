#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "libhmsbeagle/beagle.h"
#include "libhmsbeagle/CPU/BeagleCPUHierarchicalImpl.h"
#include "hratereference.h"

using beagle::cpu::BeagleCPUHierarchicalImpl;
using beagle::cpu::HierarchicalOps;
using beagle::cpu::HierarchicalSlot;
using beagle::cpu::HierarchicalClass;
using beagle::cpu::HierarchicalReduced;

static int checks = 0;
static int failures = 0;

static void report(const std::string& name, bool pass, const std::string& detail) {
    ++checks;
    if (!pass)
        ++failures;
    if (!pass || getenv("HRATE_VERBOSE"))
        printf("%-70s %s  %s\n", name.c_str(), pass ? "PASS" : "FAIL", detail.c_str());
}

static std::string relDetail(double rel) {
    char buf[64];
    snprintf(buf, sizeof(buf), "max rel %.3g", rel);
    return buf;
}

template <typename Hier, typename Real>
struct Probe : public Hier {
    const HierarchicalSlot<Real>& slot(int i) const { return this->gSlots[i]; }
    const HierarchicalOps& modelOps(int i) const { return this->gModels[i].ops; }
    bool mayAggregate() const { return this->kMayAggregate; }
    unsigned long frequencyStamp() const { return this->gFrequencyStamp; }
    bool ready(int i) { return this->prZ0Ready(this->gSlots[i]); }
    void prepare(const int* operations, int count) { this->preparePrZ0(operations, count, BEAGLE_OP_COUNT); }
    const double* piNode(int f, const HierarchicalOps& ops) { return this->nodeFrequencies(f, ops); }
    void classes(const int* operations, int count) { this->prepareClasses(operations, count, BEAGLE_OP_COUNT); }
    bool aggregated() const { return this->gAggregated; }
    bool classesValid() const { return this->gClassesValid; }
    const std::vector<HierarchicalClass<Real>>& classList() const { return this->gClasses; }
    const std::vector<int>& classOf() const { return this->gClassOfPattern; }
    const std::vector<int>& capacity() const { return this->gPatternCapacity; }
    const std::vector<size_t>& offset() const { return this->gPatternOffset; }
    size_t raggedSize() const { return this->gRaggedSize; }
    int tipState(int t, int k) const { return this->gTipStates[t][k]; }
    Real tipValue(int t, int l, int k, int i) const { return this->gPartials[t][this->pIdx(l, k, i)]; }
    double weight(int k) const { return this->gPatternWeights[k]; }
    int newOrder(int k) const { return this->gPatternsNewOrder[k]; }
    int categories() const { return this->kCategoryCount; }
    int accumulate(const int* scales, int count, int cumulative) {
        return this->accumulateScaleFactors(scales, count, cumulative);
    }
    bool raggedLive(int b) const { return this->raggedReady(b); }
    const Real* ragged(int b) const { return this->gRagged[b]; }
    unsigned long raggedGeneration(int b) const { return this->gRaggedGeneration[b]; }
    unsigned long classGeneration() const { return this->gClassGeneration; }
    const HierarchicalOps& classOps() const { return this->gClassOps; }
    Real denseValue(int b, int l, int k, int i) const { return this->gPartials[b][this->pIdx(l, k, i)]; }
};

static std::vector<double> tipFrequencies(const StateTree& tree) {
    return std::vector<double>(tree.pi.begin(), tree.pi.begin() + tree.S);
}

static StateTree withFrequencies(const StateTree& tree, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> weight(0.2, 1.0);
    std::vector<double> pi(tree.S);
    double total = 0.0;
    for (int x = 0; x < tree.S; ++x)
        total += (pi[x] = weight(rng));
    for (int x = 0; x < tree.S; ++x)
        pi[x] /= total;
    StateTree out = tree;
    finishTree(out, pi);
    return out;
}

template <typename Real>
static double internalError(const HierarchicalSlot<Real>& slot, const StateTree& tree,
                            const std::vector<double>& rates, double t) {
    const int S = tree.S, n = S - 1, C = (int) rates.size();
    if (slot.prZ0Internal.size() != (size_t) C * n)
        return INFINITY;
    double worst = 0.0;
    std::vector<double> prZ0, H;
    for (int l = 0; l < C; ++l) {
        nodeScalars(tree, t * rates[l], prZ0, H);
        for (int m = 0; m < n; ++m) {
            const double ref = prZ0[S + m];
            const double got = (double) slot.prZ0Internal[(size_t) l * n + m];
            worst = std::max(worst, std::fabs(got - ref) / ref);
        }
    }
    return worst;
}

template <typename Real>
static size_t recurrenceMismatches(const HierarchicalSlot<Real>& slot, const StateTree& tree, int C) {
    const int S = tree.S, n = S - 1, block = 2 * S - 1;
    std::vector<int> post;
    postOrder(tree, tree.root(), post);
    size_t bad = 0;
    std::vector<double> z(2 * S - 1);
    for (int l = 0; l < C; ++l) {
        const Real* H = slot.values.data() + (size_t) l * block;
        for (int x = 0; x < S; ++x)
            z[x] = (double) H[n + x];
        for (int p : post) {
            z[p] = tree.pi[p] * (double) H[p - S] + z[tree.left[p]];
            if (slot.prZ0Internal[(size_t) l * n + p - S] != (Real) z[p])
                ++bad;
        }
    }
    return bad;
}

template <typename Real, typename Hier>
static void instanceCase(const char* label, int S, const char* env, std::mt19937_64& rng) {
    char tag[128];
    snprintf(tag, sizeof(tag), "%s S=%d AGGREGATE=%s", label, S, env ? env : "unset");
    const std::string name(tag);
    if (env)
        setenv("BEAGLE_HRATE_AGGREGATE", env, 1);
    else
        unsetenv("BEAGLE_HRATE_AGGREGATE");

    const int C = 3;
    const std::vector<double> rates = { 0.3, 1.0, 2.7 };
    const double t = 0.41;
    const double tolerance = sizeof(Real) == 8 ? 1e-12 : 1e-6;

    Probe<Hier, Real> h;
    int rc = h.createInstance(3, 5, 0, S, 8, 2, 6, C, 0, 0, 0, 0, 0);
    unsetenv("BEAGLE_HRATE_AGGREGATE");
    if (rc != 0) {
        report(name + " create", false, "code " + std::to_string(rc));
        return;
    }
    const bool may = (env && std::string(env) == "1") || (!env && S >= 9);
    report(name + " may aggregate", h.mayAggregate() == may, may ? "expected true" : "expected false");

    StateTree tree = makeRandomTree(S, rng, 0.7);
    ModelArrays model = modelArrays(tree);
    std::vector<double> pi = tipFrequencies(tree);
    rc = h.setStateFrequencies(0, pi.data()) | h.setCategoryRates(rates.data()) |
         h.setEigenDecomposition(0, model.vectors.data(), model.inverse.data(), model.values.data());
    const int matrix0 = 0;
    rc |= h.updateTransitionMatrices(0, &matrix0, NULL, NULL, &t, 1);
    std::vector<double> bundle;
    referenceBundle(tree, rates.data(), C, t, bundle);
    rc |= h.setTransitionMatrix(1, bundle.data(), 0.0);
    if (rc != 0) {
        report(name + " setup", false, "code " + std::to_string(rc));
        return;
    }

    const int ops[7] = { 3, BEAGLE_OP_NONE, BEAGLE_OP_NONE, 0, 0, 1, 1 };
    if (!may) {
        h.prepare(ops, 1);
        const bool untouched = h.slot(0).prZ0Internal.empty() && h.slot(1).prZ0Internal.empty() &&
                               h.modelOps(0).topologyId == -1 && h.slot(0).topologyId == -1 &&
                               h.slot(1).topologyId == -1;
        report(name + " nothing written", untouched, "prZ0Internal empty and topology ids -1");
        return;
    }

    report(name + " model path internal Pr(Z=0)", internalError(h.slot(0), tree, rates, t) <= tolerance,
           relDetail(internalError(h.slot(0), tree, rates, t)));
    report(name + " model path ready without prepare", h.ready(0), "");
    report(name + " bundle path stale before prepare", !h.ready(1), "");
    h.prepare(ops, 1);
    report(name + " bundle path recovered", h.ready(1) && internalError(h.slot(1), tree, rates, t) <= tolerance,
           relDetail(internalError(h.slot(1), tree, rates, t)));

    const std::vector<Real> recovered = h.slot(1).prZ0Internal;
    h.prepare(ops, 1);
    report(name + " prepare is idempotent", h.slot(1).prZ0Internal == recovered, "");

    StateTree other = withFrequencies(tree, rng);
    std::vector<double> otherPi = tipFrequencies(other);
    const unsigned long before = h.frequencyStamp();
    h.setStateFrequencies(0, otherPi.data());
    report(name + " frequency change marks bundle slot stale",
           h.frequencyStamp() != before && !h.ready(1) && h.ready(0), "");
    h.prepare(ops, 1);
    report(name + " recovery follows the new frequencies",
           h.ready(1) && recurrenceMismatches(h.slot(1), other, C) == 0,
           std::to_string(recurrenceMismatches(h.slot(1), other, C)) + " values differ");
    const double* piNode = h.piNode(0, h.modelOps(0));
    size_t piBad = 0;
    for (int m = 0; m < 2 * S - 1; ++m)
        if (piNode[m] != other.pi[m])
            ++piBad;
    report(name + " piNode follows frequency buffer 0", piBad == 0, std::to_string(piBad) + " nodes differ");
    h.setStateFrequencies(1, pi.data());
    const double* piNode1 = h.piNode(1, h.modelOps(0));
    piBad = 0;
    for (int m = 0; m < 2 * S - 1; ++m)
        if (piNode1[m] != tree.pi[m])
            ++piBad;
    report(name + " piNode per frequency buffer", piBad == 0 && h.piNode(0, h.modelOps(0))[2 * S - 2] == other.pi[2 * S - 2],
           std::to_string(piBad) + " nodes differ");
    h.setStateFrequencies(0, pi.data());

    const int id = h.modelOps(0).topologyId;
    bool idsOk = id >= 0 && h.slot(0).topologyId == id && h.slot(1).topologyId == id;
    std::vector<int> order = randomPostOrder(tree, rng);
    referenceBundle(tree, rates.data(), C, t, bundle, &order, true);
    h.setTransitionMatrix(2, bundle.data(), 0.0);
    idsOk = idsOk && h.slot(2).topologyId == id;
    StateTree second = makeRandomTree(S, rng, 0.7);
    while (S > 3 && second.parent == tree.parent)
        second = makeRandomTree(S, rng, 0.7);
    ModelArrays secondModel = modelArrays(second);
    h.setEigenDecomposition(1, secondModel.vectors.data(), secondModel.inverse.data(), secondModel.values.data());
    const int secondId = h.modelOps(1).topologyId;
    idsOk = idsOk && secondId >= 0 && (secondId != id || second.parent == tree.parent);
    referenceBundle(second, rates.data(), C, t, bundle);
    h.setTransitionMatrix(3, bundle.data(), 0.0);
    idsOk = idsOk && h.slot(3).topologyId == secondId;
    h.setEigenDecomposition(1, model.vectors.data(), model.inverse.data(), model.values.data());
    idsOk = idsOk && h.modelOps(1).topologyId == id;
    report(name + " topology ids", idsOk,
           "first " + std::to_string(id) + ", second " + std::to_string(secondId));

    const int ops2[7] = { 4, BEAGLE_OP_NONE, BEAGLE_OP_NONE, 0, 2, 1, 1 };
    h.prepare(ops2, 1);
    report(name + " recovery under a reordered bundle", internalError(h.slot(2), tree, rates, t) <= tolerance,
           relDetail(internalError(h.slot(2), tree, rates, t)));
}

struct SparseData {
    int S, K, stateTips, partialTips, C;
    std::vector<std::vector<int>> states;
    std::vector<std::vector<double>> partials;
    std::vector<std::vector<double>> planes;
};

static SparseData makeSparse(int S, int K, int C, std::mt19937_64& rng) {
    SparseData d{ S, K, 4, 3, C, {}, {}, {} };
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    d.states.assign(d.stateTips, std::vector<int>(K));
    d.partials.assign(d.partialTips, std::vector<double>((size_t) K * S, 0.0));
    for (int k = 0; k < K; ++k) {
        std::vector<int> pool;
        const int n = 1 + (int) (rng() % 3);
        for (int j = 0; j < n; ++j)
            pool.push_back((int) (rng() % S));
        const bool allGap = k == 3;
        for (int t = 0; t < d.stateTips; ++t)
            d.states[t][k] = (allGap || unit(rng) < 0.1) ? S + (int) (rng() % 3) : pool[rng() % pool.size()];
        for (int t = 0; t < d.partialTips; ++t) {
            double* col = d.partials[t].data() + (size_t) k * S;
            const int kind = allGap ? 3 : (int) (rng() % 6);
            if (kind == 0) {
                col[pool[rng() % pool.size()]] = 1.0;
            } else if (kind == 1) {
                for (int x : pool)
                    col[x] = 1.0;
                col[(int) (rng() % S)] = 1.0;
            } else if (kind == 2) {
                for (int i = 0; i < S; ++i)
                    col[i] = 0.5;
            } else if (kind == 3) {
                for (int i = 0; i < S; ++i)
                    col[i] = 1.0;
            } else if (kind == 5 && k % 5 == 0) {
                for (int i = 0; i < S; ++i)
                    col[i] = (rng() % 2) ? 0.1 + 0.01 * i : 0.0;
            } else if (kind == 4 && k % 7 == 0) {
                for (int i = 0; i < S; ++i)
                    col[i] = 0.1 + unit(rng);
            } else {
                col[pool[0]] = 0.3;
                col[pool.back()] = 0.9;
            }
        }
    }
    d.planes.assign(C, std::vector<double>((size_t) K * S, 0.0));
    for (int l = 0; l < C; ++l)
        for (int k = 0; k < K; ++k)
            for (int i = 0; i < S; ++i)
                d.planes[l][(size_t) k * S + i] = (k % 4 != 0 || i == (k + l) % S) ? 1.0 : 0.0;
    return d;
}

template <typename Hier, typename Real>
static size_t roundTripMismatches(Probe<Hier, Real>& h, const HierarchicalOps& ops, int S, int K, int tips) {
    size_t bad = 0;
    std::vector<int> rowOf;
    std::vector<Real> col(S), red, back(S);
    for (int k = 0; k < K; ++k) {
        const HierarchicalReduced<Real>& r = h.classList()[h.classOf()[k]].reduced[0];
        beagle::cpu::hierarchicalRowOfState(ops, S, r, rowOf);
        for (int t = 0; t < tips; ++t) {
            const int planes = t < 4 ? 1 : h.categories();
            for (int l = 0; l < planes; ++l) {
                for (int i = 0; i < S; ++i)
                    col[i] = t < 4 ? ((h.tipState(t, k) >= S || h.tipState(t, k) == i) ? Real(1) : Real(0))
                                   : h.tipValue(t, l, k, i);
                red.resize(r.stateCount);
                for (int x = 0; x < r.stateCount; ++x)
                    red[x] = col[r.repOfRow[x]];
                for (int i = 0; i < S; ++i)
                    back[i] = red[rowOf[i]];
                if (back != col)
                    ++bad;
            }
        }
    }
    return bad;
}

template <typename Real, typename Hier>
static void classCase(const char* label, int S, std::mt19937_64& rng) {
    char tag[128];
    snprintf(tag, sizeof(tag), "%s classes S=%d", label, S);
    const std::string name(tag);
    const int K = 60, C = 2, tips = 8;
    SparseData d = makeSparse(S, K, C, rng);
    setenv("BEAGLE_HRATE_AGGREGATE", "1", 1);
    Probe<Hier, Real> h;
    int rc = h.createInstance(tips, 2 * tips, 4, S, K, 2, 6, C, 0, 0, 0, 0, 0);
    unsetenv("BEAGLE_HRATE_AGGREGATE");
    StateTree tree = makeRandomTree(S, rng, 0.7);
    ModelArrays model = modelArrays(tree);
    std::vector<double> pi = tipFrequencies(tree), rates = { 0.5, 1.5 }, weights(K);
    for (int k = 0; k < K; ++k)
        weights[k] = 1.0 + k;
    rc |= h.setStateFrequencies(0, pi.data()) | h.setCategoryRates(rates.data()) | h.setPatternWeights(weights.data()) |
          h.setEigenDecomposition(0, model.vectors.data(), model.inverse.data(), model.values.data());
    for (int t = 0; t < 4; ++t)
        rc |= h.setTipStates(t, d.states[t].data());
    for (int t = 0; t < 3; ++t)
        rc |= h.setTipPartials(4 + t, d.partials[t].data());
    std::vector<double> full((size_t) C * K * S);
    for (int l = 0; l < C; ++l)
        std::copy(d.planes[l].begin(), d.planes[l].end(), full.begin() + (size_t) l * K * S);
    rc |= h.setPartials(7, full.data());
    const int matrices[2] = { 0, 1 };
    const double lengths[2] = { 0.2, 0.3 };
    rc |= h.updateTransitionMatrices(0, matrices, NULL, NULL, lengths, 2);
    if (rc != 0) {
        report(name + " setup", false, "code " + std::to_string(rc));
        return;
    }
    const int ops[7] = { 9, BEAGLE_OP_NONE, BEAGLE_OP_NONE, 0, 0, 1, 1 };
    h.classes(ops, 1);
    const HierarchicalOps& full0 = h.modelOps(0);
    report(name + " built and forced on", h.classesValid() && h.aggregated() && (int) h.classOf().size() == K, "");
    report(name + " round trip of every tip column", roundTripMismatches(h, full0, S, K, tips) == 0,
           std::to_string(roundTripMismatches(h, full0, S, K, tips)) + " columns differ");
    const HierarchicalReduced<Real>& gap = h.classList()[h.classOf()[3]].reduced[0];
    report(name + " all-gap pattern has S_c = 1", gap.stateCount == 1 && h.classList()[h.classOf()[3]].observed.empty(),
           "S_c " + std::to_string(gap.stateCount));
    bool fullClass = false;
    for (int k = 0; k < K; k += 7)
        fullClass = fullClass || h.classList()[h.classOf()[k]].reduced[0].stateCount == S;
    report(name + " a non-constant column without zeros gives the full class", fullClass, "");

    bool offsetsOk = h.offset().size() == (size_t) K + 1 && h.offset()[K] == h.raggedSize();
    for (int k = 0; k < K && offsetsOk; ++k)
        offsetsOk = h.offset()[k] % 8 == 0 && h.offset()[k + 1] >= h.offset()[k] + (size_t) C * h.capacity()[k] &&
                    h.capacity()[k] == h.classList()[h.classOf()[k]].reduced[0].stateCount;
    report(name + " ragged offsets", offsetsOk, "size " + std::to_string(h.raggedSize()));

    const auto* before = h.classList().data();
    h.classes(ops, 1);
    report(name + " no rebuild without a change", h.classList().data() == before, "");

    StateTree other = withFrequencies(tree, rng);
    std::vector<double> otherPi = tipFrequencies(other);
    h.setStateFrequencies(0, otherPi.data());
    h.classes(ops, 1);
    size_t piBad = 0;
    for (const auto& c : h.classList())
        for (int x = 0; x < c.reduced[0].stateCount; ++x)
            if (c.reduced[0].piRed[x] != (Real) other.pi[c.reduced[0].localToFull[x]])
                ++piBad;
    report(name + " piRed follows frequency buffer 0", piBad == 0, std::to_string(piBad) + " rows differ");

    std::vector<int> changed = d.states[0];
    for (int k = 0; k < K; ++k)
        changed[k] = (int) ((k * 5 + 1) % S);
    h.setTipStates(0, changed.data());
    h.classes(ops, 1);
    report(name + " tip change rebuilds", roundTripMismatches(h, full0, S, K, tips) == 0 &&
           h.classList()[h.classOf()[3]].observed.size() == 1, "");

    std::vector<int> partitions(K);
    for (int k = 0; k < K; ++k)
        partitions[k] = (k * 7 + k / 5) % 3;
    h.setPatternPartitions(3, partitions.data());
    report(name + " reorder invalidates", !h.classesValid() && !h.aggregated(), "");
    h.setTipStates(1, d.states[1].data());
    h.setPatternWeights(weights.data());
    bool permuted = true;
    for (int k = 0; k < K; ++k) {
        const int s = d.states[1][k];
        permuted = permuted && h.tipState(1, h.newOrder(k)) == (s < S ? s : S) && h.weight(h.newOrder(k)) == weights[k];
    }
    report(name + " setTipStates and setPatternWeights after a reorder", permuted, "");
    h.classes(ops, 1);
    report(name + " round trip after a reorder", h.aggregated() && roundTripMismatches(h, full0, S, K, tips) == 0, "");

    StateTree second = makeRandomTree(S, rng, 0.7);
    ModelArrays secondModel = modelArrays(second);
    h.setEigenDecomposition(1, secondModel.vectors.data(), secondModel.inverse.data(), secondModel.values.data());
    const int matrix2 = 2;
    h.updateTransitionMatrices(1, &matrix2, NULL, NULL, lengths, 1);
    const int mixedOps[7] = { 9, BEAGLE_OP_NONE, BEAGLE_OP_NONE, 0, 0, 1, 2 };
    h.classes(mixedOps, 1);
    report(name + " two topologies in one call turn it off", !h.aggregated() && !h.classesValid(), "");
    const int secondOps[7] = { 9, BEAGLE_OP_NONE, BEAGLE_OP_NONE, 0, 2, 1, 2 };
    h.classes(secondOps, 1);
    report(name + " a single new topology rebuilds", h.aggregated() &&
           roundTripMismatches(h, h.modelOps(1), S, K, tips) == 0, "");
}

template <typename Real, typename Hier>
static void ruleCase(const char* label, int S, int K, const char* env, const char* kernel, bool expectMay,
                     std::mt19937_64& rng) {
    char tag[160];
    snprintf(tag, sizeof(tag), "%s rule S=%d K=%d AGGREGATE=%s KERNEL=%s", label, S, K, env ? env : "unset",
             kernel ? kernel : "default");
    const std::string name(tag);
    if (env)
        setenv("BEAGLE_HRATE_AGGREGATE", env, 1);
    if (kernel)
        setenv("BEAGLE_HRATE_KERNEL", kernel, 1);
    Probe<Hier, Real> h;
    int rc = h.createInstance(4, 8, 4, S, K, 1, 4, 1, 0, 0, 0, 0, 0);
    unsetenv("BEAGLE_HRATE_AGGREGATE");
    unsetenv("BEAGLE_HRATE_KERNEL");
    report(name, rc == 0 && h.mayAggregate() == expectMay, expectMay ? "expected may" : "expected off");
    if (rc != 0 || !expectMay)
        return;
    StateTree tree = makeShapedTree(S, false, rng, 0.7);
    ModelArrays model = modelArrays(tree);
    std::vector<double> pi = tipFrequencies(tree), rates = { 1.0 };
    h.setStateFrequencies(0, pi.data());
    h.setCategoryRates(rates.data());
    h.setEigenDecomposition(0, model.vectors.data(), model.inverse.data(), model.values.data());
    const int matrices[2] = { 0, 1 };
    const double lengths[2] = { 0.2, 0.3 };
    h.updateTransitionMatrices(0, matrices, NULL, NULL, lengths, 2);
    std::vector<int> sparse(K, 0), dense(K);
    for (int t = 0; t < 4; ++t) {
        for (int k = 0; k < K; ++k)
            dense[k] = (int) (rng() % S);
        h.setTipStates(t, dense.data());
    }
    const int ops[7] = { 5, BEAGLE_OP_NONE, BEAGLE_OP_NONE, 0, 0, 1, 1 };
    h.classes(ops, 1);
    const bool forced = env && std::string(env) == "1";
    const bool denseDecision = h.aggregated();
    for (int t = 0; t < 4; ++t)
        h.setTipStates(t, sparse.data());
    h.classes(ops, 1);
    const bool sparseDecision = h.aggregated();
    if (forced)
        report(name + " forced on for dense and sparse data", denseDecision && sparseDecision, "");
    else if (S >= 256)
        report(name + " decision follows r", !denseDecision && sparseDecision, "");
    else
        report(name + " r above the threshold at C = 1 keeps it off", !denseDecision && !sparseDecision, "");
}

static double maxRel(const std::vector<double>& a, const std::vector<double>& b) {
    double worst = a.size() == b.size() ? 0.0 : 1.0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
        const double scale = std::max(std::fabs(b[i]), 1e-300);
        if (a[i] != b[i])
            worst = std::max(worst, std::fabs(a[i] - b[i]) / scale);
    }
    return worst;
}

template <typename Hier, typename Real>
struct EntryRun {
    Probe<Hier, Real> h;
    int rc = 0;
    double logL = 0.0;
};

template <typename Hier, typename Real>
static void entrySetup(EntryRun<Hier, Real>& run, const char* env, int S, int K, int C, const SparseData& d,
                       const StateTree& tree, const StateTree& other, bool partitioned) {
    const int tips = 8;
    setenv("BEAGLE_HRATE_AGGREGATE", env, 1);
    Probe<Hier, Real>& h = run.h;
    run.rc = h.createInstance(tips, 12, 4, S, K, 2, 14, C, 8, 0, 0,
                              BEAGLE_FLAG_SCALING_MANUAL | BEAGLE_FLAG_SCALERS_LOG, 0);
    unsetenv("BEAGLE_HRATE_AGGREGATE");
    ModelArrays model = modelArrays(tree);
    std::vector<double> pi = tipFrequencies(tree), otherPi = tipFrequencies(other), weights(K);
    std::vector<double> rates(C), catWeights(C, 1.0 / C);
    for (int l = 0; l < C; ++l)
        rates[l] = 0.4 + 1.2 * l;
    for (int k = 0; k < K; ++k)
        weights[k] = 1.0 + k % 5;
    run.rc |= h.setStateFrequencies(0, pi.data()) | h.setStateFrequencies(1, otherPi.data()) |
              h.setCategoryRates(rates.data()) | h.setCategoryWeights(0, catWeights.data()) |
              h.setPatternWeights(weights.data()) |
              h.setEigenDecomposition(0, model.vectors.data(), model.inverse.data(), model.values.data());
    if (partitioned) {
        std::vector<int> partitions(K);
        for (int k = 0; k < K; ++k)
            partitions[k] = (k * 7 + k / 5) % 3;
        run.rc |= h.setPatternPartitions(3, partitions.data());
        run.rc |= h.setPatternWeights(weights.data());
    }
    for (int t = 0; t < 4; ++t)
        run.rc |= h.setTipStates(t, d.states[t].data());
    for (int t = 0; t < 3; ++t)
        run.rc |= h.setTipPartials(4 + t, d.partials[t].data());
    std::vector<double> full((size_t) C * K * S);
    for (int l = 0; l < C; ++l)
        std::copy(d.planes[l].begin(), d.planes[l].end(), full.begin() + (size_t) l * K * S);
    run.rc |= h.setPartials(7, full.data());
    int matrices[14];
    double lengths[14];
    for (int m = 0; m < 14; ++m) {
        matrices[m] = m;
        lengths[m] = 0.05 + 0.03 * m;
    }
    run.rc |= h.updateTransitionMatrices(0, matrices, NULL, NULL, lengths, 14);
}

static const int kEntryChildren[7][2] = { { 0, 1 }, { 2, 3 }, { 4, 5 }, { 6, 7 }, { 8, 9 }, { 10, 11 }, { 12, 13 } };

template <typename Hier, typename Real>
static double entryEvaluate(EntryRun<Hier, Real>& run, bool partitioned, int frequency, double* byPartition) {
    Probe<Hier, Real>& h = run.h;
    const int cumulative = 7;
    if (!partitioned) {
        int ops[7 * 7];
        for (int n = 0; n < 7; ++n) {
            const int o[7] = { 8 + n, n, BEAGLE_OP_NONE, kEntryChildren[n][0], kEntryChildren[n][0],
                               kEntryChildren[n][1], kEntryChildren[n][1] };
            std::copy(o, o + 7, ops + 7 * n);
        }
        run.rc |= h.updatePartials(ops, 7, BEAGLE_OP_NONE, BEAGLE_PARTIALS_BOTTOM);
        const int scales[7] = { 0, 1, 2, 3, 4, 5, 6 };
        run.rc |= h.resetScaleFactors(cumulative);
        run.rc |= h.accumulate(scales, 7, cumulative);
        const int root = 14, weightsIndex = 0;
        double logL = 0.0;
        run.rc |= h.calculateRootLogLikelihoods(&root, &weightsIndex, &frequency, &cumulative, 1, &logL);
        return logL;
    }
    int ops[3 * 7 * 9];
    int count = 0;
    for (int p = 0; p < 3; ++p)
        for (int n = 0; n < 7; ++n) {
            const int o[9] = { 8 + n, n, BEAGLE_OP_NONE, kEntryChildren[n][0], kEntryChildren[n][0],
                               kEntryChildren[n][1], kEntryChildren[n][1], p, BEAGLE_OP_NONE };
            std::copy(o, o + 9, ops + 9 * count++);
        }
    run.rc |= h.updatePartialsByPartition(ops, count);
    const int scales[7] = { 0, 1, 2, 3, 4, 5, 6 };
    for (int p = 0; p < 3; ++p) {
        run.rc |= h.resetScaleFactorsByPartition(cumulative, p);
        run.rc |= h.accumulateScaleFactorsByPartition(scales, 7, cumulative, p);
    }
    const int roots[3] = { 14, 14, 14 }, weightsIndex[3] = { 0, 0, 0 }, freqs[3] = { frequency, frequency, frequency },
              cums[3] = { cumulative, cumulative, cumulative }, parts[3] = { 0, 1, 2 };
    double logL = 0.0;
    run.rc |= h.calculateRootLogLikelihoodsByPartition(roots, weightsIndex, freqs, cums, parts, 3, 1, byPartition,
                                                       &logL);
    return logL;
}

template <typename Hier, typename Real>
static std::vector<double> entryPartials(EntryRun<Hier, Real>& run, int buffer, int scale, int S, int K, int C) {
    std::vector<double> out((size_t) C * K * S);
    run.rc |= run.h.getPartials(buffer, scale, out.data());
    return out;
}

template <typename Real, typename Hier>
static void entryCase(const char* label, int S, bool partitioned, std::mt19937_64& rng) {
    char tag[128];
    snprintf(tag, sizeof(tag), "%s entry points S=%d%s", label, S, partitioned ? " partitioned" : "");
    const std::string name(tag);
    const int K = 60, C = 2;
    const bool dp = sizeof(Real) == 8;
    const double tolL = dp ? 1e-13 : 1e-6, tolP = dp ? 1e-12 : 2e-5;
    SparseData d = makeSparse(S, K, C, rng);
    StateTree tree = makeRandomTree(S, rng, 0.7);
    StateTree other = withFrequencies(tree, rng);
    EntryRun<Hier, Real> a, b;
    entrySetup(a, "0", S, K, C, d, tree, other, partitioned);
    entrySetup(b, "1", S, K, C, d, tree, other, partitioned);
    double partsA[3] = { 0, 0, 0 }, partsB[3] = { 0, 0, 0 };
    const double logLA = entryEvaluate(a, partitioned, 0, partsA);
    const double logLB = entryEvaluate(b, partitioned, 0, partsB);
    if (a.rc != 0 || b.rc != 0) {
        report(name + " setup", false, "codes " + std::to_string(a.rc) + " " + std::to_string(b.rc));
        return;
    }
    report(name + " ragged root used only when forced", b.h.raggedLive(14) && !a.h.raggedLive(14) && b.h.aggregated(),
           "");
    double dl = std::fabs(logLB - logLA) / std::fabs(logLA);
    for (int p = 0; partitioned && p < 3; ++p)
        dl = std::max(dl, std::fabs(partsB[p] - partsA[p]) / std::fabs(partsA[p]));
    report(name + " root likelihood", dl <= tolL, relDetail(dl));

    double dp1 = 0.0, dp2 = 0.0;
    for (int n = 8; n <= 14; ++n) {
        dp1 = std::max(dp1, maxRel(entryPartials(b, n, BEAGLE_OP_NONE, S, K, C), entryPartials(a, n, BEAGLE_OP_NONE, S, K, C)));
        dp2 = std::max(dp2, maxRel(entryPartials(b, n, 7, S, K, C), entryPartials(a, n, 7, S, K, C)));
    }
    report(name + " getPartials broadcast", dp1 <= tolP, relDetail(dp1));
    report(name + " getPartials with the cumulative scale", dp2 <= tolP, relDetail(dp2));

    size_t tipBad = 0;
    for (int t = 4; t < 8; ++t) {
        if (b.h.raggedGeneration(t) != b.h.classGeneration())
            ++tipBad;
        for (int k = 0; k < K; ++k) {
            const HierarchicalReduced<Real>& r = b.h.classList()[b.h.classOf()[k]].reduced[0];
            for (int x = 0; x < r.stateCount; ++x)
                for (int l = 0; l < C; ++l)
                    if (b.h.ragged(t)[b.h.offset()[k] + x * C + l] != b.h.denseValue(t, l, k, r.repOfRow[x]))
                        ++tipBad;
        }
    }
    report(name + " ragged tip copies", tipBad == 0, std::to_string(tipBad) + " values differ");

    double logL1A = 0.0, logL1B = 0.0;
    const int root = 14, weightsIndex = 0, frequency = 1, cumulative = 7;
    a.rc |= a.h.calculateRootLogLikelihoods(&root, &weightsIndex, &frequency, &cumulative, 1, &logL1A);
    b.rc |= b.h.calculateRootLogLikelihoods(&root, &weightsIndex, &frequency, &cumulative, 1, &logL1B);
    const double dlf = std::fabs(logL1B - logL1A) / std::fabs(logL1A);
    report(name + " root likelihood with frequency buffer 1", dlf <= tolL && logL1A != logLA, relDetail(dlf));

    const int spare = 15;
    const std::vector<double> rootA = entryPartials(a, 14, BEAGLE_OP_NONE, S, K, C);
    b.rc |= b.h.setPartials(spare, rootA.data());
    double logLS = 0.0;
    const int zero = 0;
    b.rc |= b.h.calculateRootLogLikelihoods(&spare, &weightsIndex, &zero, &cumulative, 1, &logLS);
    const double dls = std::fabs(logLS - logLA) / std::fabs(logLA);
    report(name + " setPartials compresses an internal buffer", b.h.raggedLive(spare) && dls <= tolL, relDetail(dls));
    const double dlsP = maxRel(entryPartials(b, spare, BEAGLE_OP_NONE, S, K, C), rootA);
    report(name + " setPartials then getPartials", dlsP <= tolP, relDetail(dlsP));

    b.rc |= b.h.setRootPrePartials(&spare, &frequency, 1);
    std::vector<double> pre = entryPartials(b, spare, BEAGLE_OP_NONE, S, K, C);
    const std::vector<double> otherPi = tipFrequencies(other);
    size_t preBad = b.h.raggedLive(spare) ? 1 : 0;
    for (int l = 0; l < C; ++l)
        for (int k = 0; k < K; ++k)
            for (int i = 0; i < S; ++i)
                if (pre[((size_t) l * K + k) * S + i] != (double) (Real) otherPi[i])
                    ++preBad;
    const double* piNode = b.h.piNode(1, b.h.classOps());
    for (int k = 0; k < K; ++k) {
        const HierarchicalReduced<Real>& r = b.h.classList()[b.h.classOf()[k]].reduced[0];
        for (int x = 0; x < r.stateCount; ++x)
            for (int l = 0; l < C; ++l)
                if (b.h.ragged(spare)[b.h.offset()[k] + x * C + l] != (Real) piNode[r.localToFull[x]])
                    ++preBad;
    }
    report(name + " setRootPrePartials writes piNode per row, getPartials stays dense", preBad == 0,
           std::to_string(preBad) + " values differ");

    std::vector<int> changed = d.states[0];
    for (int k = 0; k < K; ++k)
        changed[k] = (int) ((k * 5 + 1) % S);
    a.rc |= a.h.setTipStates(0, changed.data());
    b.rc |= b.h.setTipStates(0, changed.data());
    const unsigned long generation = b.h.classGeneration();
    const double logL2A = entryEvaluate(a, partitioned, 0, partsA);
    const double logL2B = entryEvaluate(b, partitioned, 0, partsB);
    const double dl2 = std::fabs(logL2B - logL2A) / std::fabs(logL2A);
    report(name + " a tip change rebuilds and recompresses", b.h.classGeneration() == generation + 1 &&
           b.h.raggedLive(14) && !b.h.raggedLive(spare) && dl2 <= tolL && logL2A != logLA, relDetail(dl2));
}

template <typename Real, typename Hier>
static void autoEntryCase(const char* label, std::mt19937_64& rng) {
    const std::string name = std::string(label) + " entry points S=256 automatic rule";
    const int S = 256, K = 60, C = 1;
    SparseData d = makeSparse(S, K, C, rng);
    for (int t = 0; t < 3; ++t)
        for (int k = 0; k < K; ++k)
            for (int i = 0; i < S; ++i)
                d.partials[t][(size_t) k * S + i] = (d.states[t][k] >= S || d.states[t][k] == i) ? 1.0 : 0.0;
    std::fill(d.planes[0].begin(), d.planes[0].end(), 1.0);
    StateTree tree = makeRandomTree(S, rng, 0.7);
    StateTree other = withFrequencies(tree, rng);
    EntryRun<Hier, Real> a, b;
    entrySetup(a, "0", S, K, C, d, tree, other, false);
    entrySetup(b, "auto", S, K, C, d, tree, other, false);
    const double logLA = entryEvaluate(a, false, 0, NULL);
    const double logLB = entryEvaluate(b, false, 0, NULL);
    const bool samePartials = entryPartials(a, 14, 7, S, K, C) == entryPartials(b, 14, 7, S, K, C);
    report(name + " decides on but keeps the dense path bitwise", a.rc == 0 && b.rc == 0 && b.h.aggregated() &&
           !b.h.raggedLive(14) && logLA == logLB && samePartials, "");
}

int main() {
    std::mt19937_64 rng(20261101);
    const int stateCounts[5] = { 4, 8, 9, 17, 61 };
    const char* envs[3] = { NULL, "0", "1" };
    for (int S : stateCounts)
        for (const char* env : envs) {
            instanceCase<double, BeagleCPUHierarchicalImpl<double, 1, 0>>("double", S, env, rng);
            instanceCase<float, BeagleCPUHierarchicalImpl<float, 1, 0>>("single", S, env, rng);
        }
    for (int S : { 9, 17, 61 }) {
        classCase<double, BeagleCPUHierarchicalImpl<double, 1, 0>>("double", S, rng);
        classCase<float, BeagleCPUHierarchicalImpl<float, 1, 0>>("single", S, rng);
    }
    for (int S : { 17, 61 })
        for (bool partitioned : { false, true }) {
            entryCase<double, BeagleCPUHierarchicalImpl<double, 1, 0>>("double", S, partitioned, rng);
            entryCase<float, BeagleCPUHierarchicalImpl<float, 1, 0>>("single", S, partitioned, rng);
        }
    autoEntryCase<double, BeagleCPUHierarchicalImpl<double, 1, 0>>("double", rng);
    ruleCase<double, BeagleCPUHierarchicalImpl<double, 1, 0>>("double", 8, 200, NULL, NULL, false, rng);
    ruleCase<double, BeagleCPUHierarchicalImpl<double, 1, 0>>("double", 8, 200, "1", NULL, true, rng);
    ruleCase<double, BeagleCPUHierarchicalImpl<double, 1, 0>>("double", 64, 200, NULL, NULL, true, rng);
    ruleCase<double, BeagleCPUHierarchicalImpl<double, 1, 0>>("double", 64, 200, "1", NULL, true, rng);
    ruleCase<double, BeagleCPUHierarchicalImpl<double, 1, 0>>("double", 64, 200, "0", NULL, false, rng);
    ruleCase<double, BeagleCPUHierarchicalImpl<double, 1, 0>>("double", 64, 1200, NULL, NULL, false, rng);
    ruleCase<double, BeagleCPUHierarchicalImpl<double, 1, 0>>("double", 64, 1200, NULL, "tiled", true, rng);
    ruleCase<double, BeagleCPUHierarchicalImpl<double, 1, 0>>("double", 64, 200, NULL, "fused", false, rng);
    ruleCase<double, BeagleCPUHierarchicalImpl<double, 1, 0>>("double", 256, 1200, NULL, NULL, true, rng);
    ruleCase<float, BeagleCPUHierarchicalImpl<float, 1, 0>>("single", 256, 200, NULL, NULL, true, rng);
    printf("%d of %d checks passed\n", checks - failures, checks);
    return failures == 0 ? 0 : 1;
}
