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

int main() {
    std::mt19937_64 rng(20261101);
    const int stateCounts[5] = { 4, 8, 9, 17, 61 };
    const char* envs[3] = { NULL, "0", "1" };
    for (int S : stateCounts)
        for (const char* env : envs) {
            instanceCase<double, BeagleCPUHierarchicalImpl<double, 1, 0>>("double", S, env, rng);
            instanceCase<float, BeagleCPUHierarchicalImpl<float, 1, 0>>("single", S, env, rng);
        }
    printf("%d of %d checks passed\n", checks - failures, checks);
    return failures == 0 ? 0 : 1;
}
