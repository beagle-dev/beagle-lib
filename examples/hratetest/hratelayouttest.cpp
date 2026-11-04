#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "libhmsbeagle/beagle.h"
#include "libhmsbeagle/CPU/BeagleCPUHierarchicalImpl.h"

using beagle::cpu::BeagleCPUImpl;
using beagle::cpu::BeagleCPUHierarchicalImpl;

static int checks = 0;
static int failures = 0;

static void report(const std::string& name, bool pass, const std::string& detail) {
    ++checks;
    if (!pass)
        ++failures;
    if (!pass || getenv("HRATE_VERBOSE"))
        printf("%-70s %s  %s\n", name.c_str(), pass ? "PASS" : "FAIL", detail.c_str());
}

template <typename T>
static size_t countMismatches(const std::vector<T>& a, const std::vector<T>& b) {
    size_t bad = 0;
    for (size_t i = 0; i < a.size(); ++i)
        if (!(a[i] == b[i]) && !(std::isnan((double) a[i]) && std::isnan((double) b[i])))
            ++bad;
    return bad + (a.size() > b.size() ? a.size() - b.size() : b.size() - a.size());
}

static std::string mismatchDetail(size_t bad, size_t total) {
    return std::to_string(bad) + " of " + std::to_string(total) + " values differ";
}

static std::vector<double> randomPartials(std::mt19937_64& rng, size_t n, bool wide) {
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::uniform_real_distribution<double> exponent(-280.0, 0.0);
    std::vector<double> v(n);
    for (size_t i = 0; i < n; ++i)
        v[i] = wide ? unit(rng) * std::pow(10.0, exponent(rng)) : unit(rng);
    return v;
}

static std::vector<double> roundTo(const std::vector<double>& v, bool doublePrecision) {
    if (doublePrecision)
        return v;
    std::vector<double> r(v.size());
    for (size_t i = 0; i < v.size(); ++i)
        r[i] = (double) (float) v[i];
    return r;
}

static int create(long requirements, int S, int P, int C, BeagleInstanceDetails* det) {
    return beagleCreateInstance(3, 5, 0, S, P, 1, 4, C, 0, nullptr, 0, 0, requirements, det);
}

static void apiCase(bool dp, int S, int C, int P, std::mt19937_64& rng) {
    long prec = dp ? BEAGLE_FLAG_PRECISION_DOUBLE : BEAGLE_FLAG_PRECISION_SINGLE;
    char tag[96];
    snprintf(tag, sizeof(tag), "%s S=%d C=%d P=%d", dp ? "double" : "single", S, C, P);
    std::string name(tag);

    BeagleInstanceDetails hd, pd;
    int h = create(BEAGLE_FLAG_HIERARCHICAL_REPRESENTATION | BEAGLE_FLAG_PROCESSOR_CPU | prec, S, P, C, &hd);
    int p = create(BEAGLE_FLAG_VECTOR_NONE | BEAGLE_FLAG_PROCESSOR_CPU | prec, S, P, C, &pd);
    if (h < 0 || p < 0 || std::string(hd.implName).find("Hierarchical") == std::string::npos) {
        report(name + " create", false, "hierarchical " + std::to_string(h) + ", plain " + std::to_string(p));
        return;
    }

    std::vector<double> tip = randomPartials(rng, (size_t) P * S, false);
    std::vector<double> full = randomPartials(rng, (size_t) C * P * S, true);
    std::vector<double> freqs = randomPartials(rng, S, false);
    std::vector<double> tipAll(tip.size() * C);
    for (int l = 0; l < C; ++l)
        std::copy(tip.begin(), tip.end(), tipAll.begin() + (size_t) l * P * S);

    int rc = 0;
    for (int inst : { h, p }) {
        rc |= beagleSetTipPartials(inst, 0, tip.data());
        rc |= beagleSetTipPartials(inst, 1, tip.data());
        rc |= beagleSetTipPartials(inst, 2, tip.data());
        rc |= beagleSetPartials(inst, 3, full.data());
        rc |= beagleSetStateFrequencies(inst, 0, freqs.data());
    }
    int root = 4, freqIndex = 0;
    int rootRc = beagleSetRootPrePartials(h, &root, &freqIndex, 1) | beagleSetRootPrePartials(p, &root, &freqIndex, 1);
    report(name + " set calls", rc == 0 && rootRc == 0, "codes " + std::to_string(rc) + "/" + std::to_string(rootRc));

    size_t n = (size_t) C * P * S;
    std::vector<double> got(n), want(n);

    beagleGetPartials(h, 0, BEAGLE_OP_NONE, got.data());
    std::vector<double> expect = roundTo(tipAll, dp);
    report(name + " tip round trip", countMismatches(got, expect) == 0, mismatchDetail(countMismatches(got, expect), n));

    beagleGetPartials(h, 3, BEAGLE_OP_NONE, got.data());
    expect = roundTo(full, dp);
    report(name + " buffer round trip", countMismatches(got, expect) == 0, mismatchDetail(countMismatches(got, expect), n));

    beagleGetPartials(h, 4, BEAGLE_OP_NONE, got.data());
    beagleGetPartials(p, 4, BEAGLE_OP_NONE, want.data());
    report(name + " root pre-partials vs CPU", countMismatches(got, want) == 0, mismatchDetail(countMismatches(got, want), n));

    int buffers[3] = { 0, 3, 4 };
    int scales[3] = { BEAGLE_OP_NONE, BEAGLE_OP_NONE, BEAGLE_OP_NONE };
    std::vector<double> batchH(3 * n), batchP(3 * n);
    int brc = beagleGetPartialsBatch(h, buffers, scales, 3, batchH.data()) |
              beagleGetPartialsBatch(p, buffers, scales, 3, batchP.data());
    report(name + " batch vs CPU", brc == 0 && countMismatches(batchH, batchP) == 0,
           mismatchDetail(countMismatches(batchH, batchP), 3 * n));

    std::vector<int> partitions(P);
    for (int k = 0; k < P; ++k)
        partitions[k] = (k * 7 + 3) % 3;
    int prc = beagleSetPatternPartitions(h, 3, partitions.data()) | beagleSetPatternPartitions(p, 3, partitions.data());
    beagleGetPartials(h, 0, BEAGLE_OP_NONE, got.data());
    beagleGetPartials(p, 0, BEAGLE_OP_NONE, want.data());
    report(name + " reordered tip vs CPU", prc == 0 && countMismatches(got, want) == 0,
           "code " + std::to_string(prc) + ", " + mismatchDetail(countMismatches(got, want), n));

    beagleFinalizeInstance(h);
    beagleFinalizeInstance(p);
}

template <typename Impl, typename Real>
struct Probe : public Impl {
    Real* scale(int s) { return this->gScaleBuffers[s]; }
    void rescale(int b, int s, int cumulative) {
        this->rescalePartials(this->gPartials[b], this->gScaleBuffers[s],
                              cumulative >= 0 ? this->gScaleBuffers[cumulative] : nullptr, 0);
    }
    void rescaleByPartition(int b, int s, int cumulative, int partition) {
        this->rescalePartialsByPartition(this->gPartials[b], this->gScaleBuffers[s],
                                         cumulative >= 0 ? this->gScaleBuffers[cumulative] : nullptr, 0, partition);
    }
    void autoRescale(int b, std::vector<signed short>& out) {
        this->autoRescalePartials(this->gPartials[b], out.data());
    }
    int patterns() const { return this->kPatternCount; }
    int paddedPatterns() const { return this->kPaddedPatternCount; }
};

template <typename Real>
static std::vector<Real> copyScale(Real* s, int n) {
    return std::vector<Real>(s, s + n);
}

template <typename Real, typename Hier, typename Base>
static void scalingCase(const char* label, int S, int C, int P, bool logScalers, std::mt19937_64& rng) {
    char tag[128];
    snprintf(tag, sizeof(tag), "%s S=%d C=%d P=%d %s", label, S, C, P, logScalers ? "log" : "raw");
    std::string name(tag);
    long prefs = logScalers ? BEAGLE_FLAG_SCALERS_LOG : BEAGLE_FLAG_SCALERS_RAW;

    Probe<Hier, Real> hier;
    Probe<Base, Real> base;
    int rc = hier.createInstance(3, 5, 0, S, P, 1, 4, C, 4, 0, 0, prefs, 0) |
             base.createInstance(3, 5, 0, S, P, 1, 4, C, 4, 0, 0, prefs, 0);
    if (rc != 0) {
        report(name + " create", false, "code " + std::to_string(rc));
        return;
    }

    std::vector<double> data = randomPartials(rng, (size_t) C * P * S, true);
    for (int i = 0; i < S; ++i)
        for (int l = 0; l < C; ++l)
            data[((size_t) l * P + (P / 2)) * S + i] = 0.0;
    std::vector<double> tip = randomPartials(rng, (size_t) P * S, false);

    hier.setPartials(3, data.data());
    base.setPartials(3, data.data());
    hier.setPartials(4, data.data());
    base.setPartials(4, data.data());
    for (int t = 0; t < 3; ++t) {
        hier.setTipPartials(t, tip.data());
        base.setTipPartials(t, tip.data());
    }
    hier.resetScaleFactors(2);
    base.resetScaleFactors(2);

    size_t n = (size_t) C * P * S;
    std::vector<double> got(n), want(n);

    hier.rescale(3, 0, 2);
    base.rescale(3, 0, 2);
    hier.getPartials(3, BEAGLE_OP_NONE, got.data());
    base.getPartials(3, BEAGLE_OP_NONE, want.data());
    report(name + " rescalePartials partials", countMismatches(got, want) == 0, mismatchDetail(countMismatches(got, want), n));
    std::vector<Real> hs = copyScale(hier.scale(0), P), bs = copyScale(base.scale(0), P);
    std::vector<Real> hc = copyScale(hier.scale(2), P), bc = copyScale(base.scale(2), P);
    report(name + " rescalePartials factors", countMismatches(hs, bs) == 0 && countMismatches(hc, bc) == 0,
           mismatchDetail(countMismatches(hs, bs) + countMismatches(hc, bc), 2 * (size_t) P));

    hier.getPartials(3, 2, got.data());
    base.getPartials(3, 2, want.data());
    report(name + " getPartials with cumulative scale", countMismatches(got, want) == 0,
           mismatchDetail(countMismatches(got, want), n));

    std::vector<signed short> ha(hier.paddedPatterns(), 0), ba(base.paddedPatterns(), 0);
    hier.autoRescale(4, ha);
    base.autoRescale(4, ba);
    ha.resize(P);
    ba.resize(P);
    hier.getPartials(4, BEAGLE_OP_NONE, got.data());
    base.getPartials(4, BEAGLE_OP_NONE, want.data());
    report(name + " autoRescalePartials", countMismatches(got, want) == 0 && countMismatches(ha, ba) == 0,
           mismatchDetail(countMismatches(got, want) + countMismatches(ha, ba), n + P));

    std::vector<int> partitions(P);
    for (int k = 0; k < P; ++k)
        partitions[k] = (k % 5 == 1) ? 1 : 0;
    rc = hier.setPatternPartitions(2, partitions.data()) | base.setPatternPartitions(2, partitions.data());
    hier.getPartials(0, BEAGLE_OP_NONE, got.data());
    base.getPartials(0, BEAGLE_OP_NONE, want.data());
    report(name + " reordered tip", rc == 0 && countMismatches(got, want) == 0, mismatchDetail(countMismatches(got, want), n));

    hier.setPartials(3, data.data());
    base.setPartials(3, data.data());
    hier.resetScaleFactors(2);
    base.resetScaleFactors(2);
    // Scale buffer 1 is written only for partition 1's patterns; clear it in both
    // instances so the untouched entries compare equal instead of uninitialized memory.
    hier.resetScaleFactors(1);
    base.resetScaleFactors(1);
    hier.rescaleByPartition(3, 1, 2, 1);
    base.rescaleByPartition(3, 1, 2, 1);
    hier.getPartials(3, BEAGLE_OP_NONE, got.data());
    base.getPartials(3, BEAGLE_OP_NONE, want.data());
    hs = copyScale(hier.scale(1), P);
    bs = copyScale(base.scale(1), P);
    hc = copyScale(hier.scale(2), P);
    bc = copyScale(base.scale(2), P);
    size_t partitionBad = countMismatches(got, want) + countMismatches(hs, bs) + countMismatches(hc, bc);
    report(name + " rescalePartialsByPartition", partitionBad == 0, mismatchDetail(partitionBad, n + 2 * (size_t) P));
}

int main() {
    std::mt19937_64 rng(20261013);

    const int stateCounts[3] = { 4, 17, 61 };
    const int categoryCounts[2] = { 1, 4 };
    const int patternCounts[2] = { 48, 45 };

    for (bool dp : { true, false })
        for (int S : stateCounts)
            for (int C : categoryCounts)
                for (int P : patternCounts)
                    apiCase(dp, S, C, P, rng);

    for (int S : stateCounts)
        for (int C : categoryCounts)
            for (int P : patternCounts)
                for (bool logScalers : { true, false }) {
                    scalingCase<double, BeagleCPUHierarchicalImpl<double, 1, 0>, BeagleCPUImpl<double, 1, 0>>(
                            "double", S, C, P, logScalers, rng);
                    scalingCase<float, BeagleCPUHierarchicalImpl<float, 1, 0>, BeagleCPUImpl<float, 1, 0>>(
                            "single", S, C, P, logScalers, rng);
                    scalingCase<double, BeagleCPUHierarchicalImpl<double, 1, 1>, BeagleCPUImpl<double, 1, 0>>(
                            "double P_PAD=1", S, C, P, logScalers, rng);
                }

    printf("%d of %d checks passed\n", checks - failures, checks);
    return failures == 0 ? 0 : 1;
}
