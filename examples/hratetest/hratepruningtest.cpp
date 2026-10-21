#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
        printf("%-72s %s  %s\n", name.c_str(), pass ? "PASS" : "FAIL", detail.c_str());
}

static double maxRelDiff(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size())
        return INFINITY;
    double d = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] == b[i])
            continue;
        double denom = std::max(std::fabs(b[i]), 1e-300);
        d = std::max(d, std::fabs(a[i] - b[i]) / denom);
    }
    return d;
}

enum class Scaling { None, Manual, ManualRaw, Always, Dynamic };

struct Variant {
    std::string name;
    bool states = false;
    int threads = 1;
    int partitions = 0;
    Scaling scaling = Scaling::None;
    bool reencode = false;
    bool reread = false;
};

struct Problem {
    int S, C, P;
    bool dp;
    StateTree stateTree;
    std::vector<StateTree> branchTrees;
    std::string label;
    std::vector<double> rates, weights;
    std::vector<std::vector<int>> tipStates;
    std::vector<int> nodeParent;
    std::vector<double> edge;
    std::vector<BeagleOperation> ops;
    std::vector<int> partitionOf;
    const StateTree& treeOf(int n) const {
        return branchTrees.empty() ? stateTree : branchTrees[n % branchTrees.size()];
    }
};

static const int kTips = 6;
static const int kNodes = 2 * kTips;
static const int kStem = kNodes - 1;
static const int kRoot = kNodes - 2;

static Problem makeProblem(bool dp, int S, int C, int P, std::mt19937_64& rng) {
    Problem pr;
    pr.S = S;
    pr.C = C;
    pr.P = P;
    pr.dp = dp;
    pr.stateTree = makeRandomTree(S, rng, 2.0);
    std::gamma_distribution<double> gamma(0.5, 2.0);
    double mean = 0.0;
    for (int l = 0; l < C; ++l) {
        pr.rates.push_back(C == 1 ? 1.0 : gamma(rng) + 0.01);
        mean += pr.rates.back() / C;
    }
    for (double& r : pr.rates)
        r /= mean;
    std::uniform_real_distribution<double> weight(0.2, 1.0);
    double totalWeight = 0.0;
    for (int l = 0; l < C; ++l)
        totalWeight += pr.weights.emplace_back(weight(rng));
    for (double& w : pr.weights)
        w /= totalWeight;

    std::uniform_int_distribution<int> state(0, S - 1);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    pr.tipStates.assign(kTips, std::vector<int>(P));
    std::vector<int> ancestor(P);
    for (int k = 0; k < P; ++k)
        ancestor[k] = state(rng);
    for (int t = 0; t < kTips; ++t)
        for (int k = 0; k < P; ++k) {
            double u = unit(rng);
            pr.tipStates[t][k] = u < 0.08 ? S : (u < 0.6 ? ancestor[k] : state(rng));
        }

    pr.nodeParent.assign(kNodes, -1);
    std::vector<int> pool;
    for (int t = 0; t < kTips; ++t)
        pool.push_back(t);
    for (int m = kTips; m <= kRoot; ++m) {
        int i = (int) (rng() % pool.size());
        int a = pool[i];
        pool.erase(pool.begin() + i);
        int j = (int) (rng() % pool.size());
        int b = pool[j];
        pool.erase(pool.begin() + j);
        pr.nodeParent[a] = m;
        pr.nodeParent[b] = m;
        pr.ops.push_back({ m, BEAGLE_OP_NONE, BEAGLE_OP_NONE, a, a, b, b });
        pool.push_back(m);
    }
    pr.nodeParent[kRoot] = kStem;
    pr.ops.push_back({ kStem, BEAGLE_OP_NONE, BEAGLE_OP_NONE, kRoot, kRoot, BEAGLE_OP_NONE, BEAGLE_OP_NONE });

    std::uniform_real_distribution<double> length(0.02, 0.6);
    pr.edge.assign(kNodes - 1, 0.0);
    for (int n = 0; n < kNodes - 1; ++n)
        pr.edge[n] = length(rng);

    pr.partitionOf.resize(P);
    for (int k = 0; k < P; ++k)
        pr.partitionOf[k] = (k * 7 + k / 5) % 3;
    return pr;
}

struct Result {
    int code = BEAGLE_SUCCESS;
    std::string call;
    std::string impl;
    std::vector<double> partials;
    double logL = 0.0;
    std::vector<double> logLByPartition;
};

static bool ok(Result& r, const char* call, int code) {
    if (code < 0 && r.code == BEAGLE_SUCCESS) {
        r.code = code;
        r.call = call;
    }
    return code >= 0;
}

static long scalingFlags(Scaling s) {
    switch (s) {
        case Scaling::Manual: return BEAGLE_FLAG_SCALING_MANUAL | BEAGLE_FLAG_SCALERS_LOG;
        case Scaling::ManualRaw: return BEAGLE_FLAG_SCALING_MANUAL | BEAGLE_FLAG_SCALERS_RAW;
        case Scaling::Always: return BEAGLE_FLAG_SCALING_ALWAYS;
        case Scaling::Dynamic: return BEAGLE_FLAG_SCALING_DYNAMIC;
        default:              return 0;
    }
}

static Result run(const Problem& pr, const Variant& v, bool hierarchical, std::mt19937_64& rng,
                  bool model = false, const std::vector<int>* fixedOrder = nullptr) {
    Result res;
    const int S = pr.S, C = pr.C, P = pr.P;
    long prec = pr.dp ? BEAGLE_FLAG_PRECISION_DOUBLE : BEAGLE_FLAG_PRECISION_SINGLE;
    long requirements = BEAGLE_FLAG_PROCESSOR_CPU | prec |
                        (hierarchical ? BEAGLE_FLAG_HIERARCHICAL_REPRESENTATION : BEAGLE_FLAG_VECTOR_NONE);
    long prefs = prec | scalingFlags(v.scaling) | (v.threads > 1 ? BEAGLE_FLAG_THREADING_CPP : 0);
    bool states = v.states && hierarchical;
    BeagleInstanceDetails det;
    int inst = beagleCreateInstance(kTips, states ? kNodes - kTips : kNodes, states ? kTips : 0, S, P, 1,
                                    kNodes - 1, C, kNodes, nullptr, 0, prefs, requirements, &det);
    if (inst < 0) {
        ok(res, "beagleCreateInstance", inst);
        return res;
    }
    res.impl = det.implName;

    for (int t = 0; t < kTips; ++t) {
        if (states) {
            if (!ok(res, "beagleSetTipStates", beagleSetTipStates(inst, t, pr.tipStates[t].data())))
                return res;
        } else {
            std::vector<double> p = partialsFromStates(pr.tipStates[t], S);
            if (!ok(res, "beagleSetTipPartials", beagleSetTipPartials(inst, t, p.data())))
                return res;
        }
    }
    std::vector<double> pi(pr.stateTree.pi.begin(), pr.stateTree.pi.begin() + S);
    std::vector<double> patternWeights(P, 1.0);
    ok(res, "beagleSetStateFrequencies", beagleSetStateFrequencies(inst, 0, pi.data()));
    ok(res, "beagleSetCategoryRates", beagleSetCategoryRates(inst, pr.rates.data()));
    ok(res, "beagleSetCategoryWeights", beagleSetCategoryWeights(inst, 0, pr.weights.data()));
    ok(res, "beagleSetPatternWeights", beagleSetPatternWeights(inst, patternWeights.data()));
    if (v.threads > 1)
        ok(res, "beagleSetCPUThreadCount", beagleSetCPUThreadCount(inst, v.threads));
    if (v.partitions > 0)
        ok(res, "beagleSetPatternPartitions", beagleSetPatternPartitions(inst, v.partitions, pr.partitionOf.data()));

    std::vector<double> payload;
    if (hierarchical && model) {
        ModelArrays m = modelArrays(pr.stateTree, fixedOrder, fixedOrder != nullptr);
        if (!ok(res, "beagleSetEigenDecomposition",
                beagleSetEigenDecomposition(inst, 0, m.vectors.data(), m.inverse.data(), m.values.data())))
            return res;
        std::vector<int> indices(kNodes - 1);
        for (int n = 0; n < kNodes - 1; ++n)
            indices[n] = n;
        if (!ok(res, "beagleUpdateTransitionMatrices",
                beagleUpdateTransitionMatrices(inst, 0, indices.data(), nullptr, nullptr, pr.edge.data(), kNodes - 1)))
            return res;
    }
    for (int n = 0; n < kNodes - 1 && !(hierarchical && model); ++n) {
        if (hierarchical) {
            if (fixedOrder) {
                referenceBundle(pr.treeOf(n), pr.rates.data(), C, pr.edge[n], payload, fixedOrder, true);
            } else if (v.reencode) {
                std::vector<int> order = randomPostOrder(pr.treeOf(n), rng);
                referenceBundle(pr.treeOf(n), pr.rates.data(), C, pr.edge[n], payload, &order, true);
            } else {
                referenceBundle(pr.treeOf(n), pr.rates.data(), C, pr.edge[n], payload);
            }
        } else {
            referenceP(pr.treeOf(n), pr.rates.data(), C, pr.edge[n], payload);
        }
        if (!ok(res, "beagleSetTransitionMatrix", beagleSetTransitionMatrix(inst, n, payload.data(), 1.0)))
            return res;
    }
    if (res.code != BEAGLE_SUCCESS)
        return res;

    std::vector<BeagleOperation> ops = pr.ops;
    if (v.scaling == Scaling::Manual || v.scaling == Scaling::ManualRaw || v.scaling == Scaling::Dynamic)
        for (auto& op : ops)
            op.destinationScaleWrite = op.destinationPartials - kTips;
    if (v.scaling == Scaling::Dynamic)
        for (auto& op : ops)
            op.destinationScaleRead = op.destinationScaleWrite;
    const int cumulative = (v.scaling == Scaling::Dynamic) ? kNodes - kTips : BEAGLE_OP_NONE;
    if (cumulative != BEAGLE_OP_NONE)
        ok(res, "beagleResetScaleFactors", beagleResetScaleFactors(inst, cumulative));

    if (v.partitions > 0) {
        std::vector<BeagleOperationByPartition> pops;
        for (int part = 0; part < v.partitions; ++part)
            for (auto& op : ops)
                pops.push_back({ op.destinationPartials, op.destinationScaleWrite, op.destinationScaleRead,
                                 op.child1Partials, op.child1TransitionMatrix, op.child2Partials,
                                 op.child2TransitionMatrix, part, BEAGLE_OP_NONE });
        if (!ok(res, "beagleUpdatePartialsByPartition",
                beagleUpdatePartialsByPartition(inst, pops.data(), (int) pops.size())))
            return res;
    } else {
        const int passes = (v.scaling == Scaling::Dynamic) ? 2 : 1;
        for (int pass = 0; pass < passes; ++pass)
            if (!ok(res, "beagleUpdatePartials", beagleUpdatePartials(inst, ops.data(), (int) ops.size(), cumulative)))
                return res;
        if (v.reread) {
            for (auto& op : ops) {
                op.destinationScaleRead = op.destinationScaleWrite;
                op.destinationScaleWrite = BEAGLE_OP_NONE;
            }
            if (!ok(res, "beagleUpdatePartials", beagleUpdatePartials(inst, ops.data(), (int) ops.size(),
                                                                     BEAGLE_OP_NONE)))
                return res;
        }
    }

    const int internal = kNodes - kTips;
    std::vector<int> buffers, scales(internal, BEAGLE_OP_NONE);
    for (int n = kTips; n < kNodes; ++n)
        buffers.push_back(n);
    res.partials.assign((size_t) internal * C * P * S, 0.0);
    ok(res, "beagleGetPartialsBatch",
       beagleGetPartialsBatch(inst, buffers.data(), scales.data(), internal, res.partials.data()));
    int weightIndex = 0, freqIndex = 0, stem = kStem;
    if (v.partitions > 0) {
        std::vector<int> stems(v.partitions, kStem), zeros(v.partitions, 0), none(v.partitions, BEAGLE_OP_NONE),
                         parts(v.partitions);
        for (int part = 0; part < v.partitions; ++part)
            parts[part] = part;
        res.logLByPartition.assign(v.partitions, 0.0);
        ok(res, "beagleCalculateRootLogLikelihoodsByPartition",
           beagleCalculateRootLogLikelihoodsByPartition(inst, stems.data(), zeros.data(), zeros.data(), none.data(),
                                                        parts.data(), v.partitions, 1,
                                                        res.logLByPartition.data(), &res.logL));
    } else {
        int rootScale = BEAGLE_OP_NONE;
        if (v.scaling == Scaling::Manual || v.scaling == Scaling::ManualRaw) {
            std::vector<int> written;
            for (auto& op : ops)
                written.push_back(op.destinationPartials - kTips);
            rootScale = kNodes - kTips;
            ok(res, "beagleResetScaleFactors", beagleResetScaleFactors(inst, rootScale));
            ok(res, "beagleAccumulateScaleFactors",
               beagleAccumulateScaleFactors(inst, written.data(), (int) written.size(), rootScale));
        } else if (v.scaling == Scaling::Dynamic) {
            rootScale = cumulative;
        } else if (v.scaling == Scaling::Always) {
            rootScale = kStem - kTips;
        }
        ok(res, "beagleCalculateRootLogLikelihoods",
           beagleCalculateRootLogLikelihoods(inst, &stem, &weightIndex, &freqIndex, &rootScale, 1, &res.logL));
    }

    if (cumulative != BEAGLE_OP_NONE) {
        std::vector<double> stem((size_t) C * P * S);
        ok(res, "beagleGetPartials", beagleGetPartials(inst, kStem, cumulative, stem.data()));
        res.partials.insert(res.partials.end(), stem.begin(), stem.end());
    }

    beagleFinalizeInstance(inst);
    return res;
}

static double siteMaxDeviation(const Problem& pr, const std::vector<double>& partials) {
    const int S = pr.S, C = pr.C, P = pr.P;
    double worst = 0.0;
    for (int n = 0; n < kNodes - kTips; ++n) {
        const double* buffer = partials.data() + (size_t) n * C * P * S;
        for (int k = 0; k < P; ++k) {
            double max = 0.0;
            for (int l = 0; l < C; ++l)
                for (int i = 0; i < S; ++i)
                    max = std::max(max, buffer[((size_t) l * P + k) * S + i]);
            if (max > 0.0)
                worst = std::max(worst, std::fabs(max - 1.0));
        }
    }
    return worst;
}

static void compare(const Problem& pr, const Variant& v, std::mt19937_64& rng) {
    char tag[160];
    snprintf(tag, sizeof(tag), "%s S=%d C=%d P=%d%s %s", pr.dp ? "double" : "single", pr.S, pr.C, pr.P,
             pr.label.c_str(), v.name.c_str());
    std::string name(tag);
    Variant oracleVariant = v;
    oracleVariant.states = false;
    oracleVariant.reencode = false;
    Result h = run(pr, v, true, rng);
    Result o = run(pr, oracleVariant, false, rng);
    if (h.code != BEAGLE_SUCCESS || o.code != BEAGLE_SUCCESS || h.impl.find("Hierarchical") == std::string::npos) {
        report(name, false, "hierarchical " + h.call + " " + std::to_string(h.code) + " (" + h.impl + "), oracle " +
                            o.call + " " + std::to_string(o.code));
        return;
    }
    const double tol = pr.dp ? 1e-12 : 2e-5;
    double dp = maxRelDiff(h.partials, o.partials);
    std::string extra;
    bool extraPass = true;
    if (v.scaling != Scaling::None) {
        double worst = std::max(siteMaxDeviation(pr, h.partials), siteMaxDeviation(pr, o.partials));
        const double ulps = v.reread ? 8.0 : 2.0;
        extraPass = worst <= ulps * (pr.dp ? 2.22e-16 : 1.19e-7);
        char buf[80];
        snprintf(buf, sizeof(buf), ", rescaled site maxima deviate from 1 by %.1e", worst);
        extra = buf;
    }
    double dl = std::fabs(h.logL - o.logL) / std::fabs(o.logL);
    for (size_t p = 0; p < h.logLByPartition.size() && p < o.logLByPartition.size(); ++p)
        dl = std::max(dl, std::fabs(h.logLByPartition[p] - o.logLByPartition[p]) / std::fabs(o.logLByPartition[p]));
    const double tolL = pr.dp ? 1e-13 : 1e-6;
    char detail[240];
    snprintf(detail, sizeof(detail), "partials %.2e, tolerance %.0e; logL %.6f, relative %.2e, tolerance %.0e (oracle %s)",
             dp, tol, h.logL, dl, tolL, o.impl.c_str());
    report(name, dp < tol && dl < tolL && std::isfinite(h.logL) && extraPass, detail + extra);
}

static void modelMatchesBundles(const Problem& pr, const Variant& v, std::mt19937_64& rng) {
    char tag[160];
    snprintf(tag, sizeof(tag), "%s S=%d C=%d P=%d%s %s: model path == bundle path", pr.dp ? "double" : "single",
             pr.S, pr.C, pr.P, pr.label.c_str(), v.name.c_str());
    std::vector<int> order;
    if (v.reencode)
        order = randomPostOrder(pr.stateTree, rng);
    const std::vector<int>* fixedOrder = v.reencode ? &order : nullptr;
    Result a = run(pr, v, true, rng, false, fixedOrder);
    Result b = run(pr, v, true, rng, true, fixedOrder);
    bool same = a.code == 0 && b.code == 0 && a.partials.size() == b.partials.size() &&
                std::memcmp(a.partials.data(), b.partials.data(), a.partials.size() * sizeof(double)) == 0 &&
                std::memcmp(&a.logL, &b.logL, sizeof(double)) == 0 &&
                a.logLByPartition.size() == b.logLByPartition.size() &&
                std::memcmp(a.logLByPartition.data(), b.logLByPartition.data(),
                            a.logLByPartition.size() * sizeof(double)) == 0;
    report(tag, same, "codes " + std::to_string(a.code) + " " + b.call + " " + std::to_string(b.code) +
                      ", max relative difference " + std::to_string(maxRelDiff(b.partials, a.partials)));
}

static void rescaleMatchesLegacy(const Problem& pr, const Variant& v, std::mt19937_64& rng) {
    char tag[160];
    snprintf(tag, sizeof(tag), "%s S=%d C=%d P=%d%s %s: rescale == legacy rescale", pr.dp ? "double" : "single",
             pr.S, pr.C, pr.P, pr.label.c_str(), v.name.c_str());
    std::mt19937_64 rngA = rng, rngB = rng;
    setenv("BEAGLE_HRATE_RESCALE", "legacy", 1);
    Result a = run(pr, v, true, rngA);
    unsetenv("BEAGLE_HRATE_RESCALE");
    Result b = run(pr, v, true, rngB);
    rng = rngB;
    bool same = a.code == 0 && b.code == 0 && a.partials.size() == b.partials.size() &&
                std::memcmp(a.partials.data(), b.partials.data(), a.partials.size() * sizeof(double)) == 0 &&
                std::memcmp(&a.logL, &b.logL, sizeof(double)) == 0;
    report(tag, same, "codes " + std::to_string(a.code) + " " + std::to_string(b.code) +
                      ", max relative difference " + std::to_string(maxRelDiff(b.partials, a.partials)));
}

static void fusedMatchesTiled(const Problem& pr, const Variant& v, std::mt19937_64& rng) {
    char tag[160];
    snprintf(tag, sizeof(tag), "%s S=%d C=%d P=%d%s %s: fused == tiled", pr.dp ? "double" : "single", pr.S, pr.C,
             pr.P, pr.label.c_str(), v.name.c_str());
    std::mt19937_64 rngA = rng, rngB = rng;
    const char* saved = getenv("BEAGLE_HRATE_KERNEL");
    std::string restore = saved ? saved : "";
    setenv("BEAGLE_HRATE_KERNEL", "tiled", 1);
    Result a = run(pr, v, true, rngA);
    setenv("BEAGLE_HRATE_KERNEL", "fused", 1);
    setenv("BEAGLE_HRATE_GATHER", "off", 1);
    Result b = run(pr, v, true, rngB);
    unsetenv("BEAGLE_HRATE_GATHER");
    if (saved)
        setenv("BEAGLE_HRATE_KERNEL", restore.c_str(), 1);
    else
        unsetenv("BEAGLE_HRATE_KERNEL");
    rng = rngB;
    bool same = a.code == 0 && b.code == 0 && a.partials.size() == b.partials.size() &&
                std::memcmp(a.partials.data(), b.partials.data(), a.partials.size() * sizeof(double)) == 0 &&
                std::memcmp(&a.logL, &b.logL, sizeof(double)) == 0 &&
                a.logLByPartition.size() == b.logLByPartition.size() &&
                std::memcmp(a.logLByPartition.data(), b.logLByPartition.data(),
                            a.logLByPartition.size() * sizeof(double)) == 0;
    report(tag, same, "codes " + std::to_string(a.code) + " " + std::to_string(b.code) +
                      ", max relative difference " + std::to_string(maxRelDiff(b.partials, a.partials)));
}

static void statesMatchPartials(const Problem& pr, std::mt19937_64& rng) {
    char tag[160];
    snprintf(tag, sizeof(tag), "%s S=%d C=%d P=%d%s states == one-hot partials", pr.dp ? "double" : "single",
             pr.S, pr.C, pr.P, pr.label.c_str());
    Variant partials, states;
    states.states = true;
    Result a = run(pr, partials, true, rng);
    Result b = run(pr, states, true, rng);
    double d = maxRelDiff(a.partials, b.partials);
    const double tol = pr.dp ? 1e-12 : 2e-5;
    char detail[96];
    snprintf(detail, sizeof(detail), "max relative difference %.2e, tolerance %.0e", d, tol);
    report(tag, a.code == 0 && b.code == 0 && d <= tol, detail);
}

static void failsLoudly(std::mt19937_64& rng) {
    Problem pr = makeProblem(true, 4, 2, 40, rng);
    BeagleInstanceDetails det;
    int inst = beagleCreateInstance(kTips, kNodes, 0, 4, 40, 1, kNodes - 1, 2, 0, nullptr, 0,
                                    BEAGLE_FLAG_THREADING_CPP,
                                    BEAGLE_FLAG_HIERARCHICAL_REPRESENTATION | BEAGLE_FLAG_PRECISION_DOUBLE, &det);
    for (int t = 0; t < kTips; ++t) {
        std::vector<double> p = partialsFromStates(pr.tipStates[t], 4);
        beagleSetTipPartials(inst, t, p.data());
    }
    std::vector<double> payload;
    for (int n = 0; n < kNodes - 2; ++n) {
        referenceBundle(pr.stateTree, pr.rates.data(), 2, pr.edge[n], payload);
        beagleSetTransitionMatrix(inst, n, payload.data(), 1.0);
    }
    int noFrequencies = beagleUpdatePartials(inst, pr.ops.data(), (int) pr.ops.size(), BEAGLE_OP_NONE);
    std::vector<double> pi(pr.stateTree.pi.begin(), pr.stateTree.pi.begin() + 4);
    beagleSetStateFrequencies(inst, 0, pi.data());
    int missingSlot = beagleUpdatePartials(inst, pr.ops.data(), (int) pr.ops.size(), BEAGLE_OP_NONE);
    beagleSetCPUThreadCount(inst, 8);
    int missingSlotThreaded = beagleUpdatePartials(inst, pr.ops.data(), (int) pr.ops.size(), BEAGLE_OP_NONE);
    int noPreOrder = beagleUpdatePrePartials(inst, pr.ops.data(), (int) pr.ops.size(), BEAGLE_OP_NONE);
    BeagleInstanceDetails autoDet;
    int autoRequired = beagleCreateInstance(kTips, kNodes, 0, 4, 40, 1, kNodes - 1, 2, 0, nullptr, 0, 0,
                                            BEAGLE_FLAG_HIERARCHICAL_REPRESENTATION | BEAGLE_FLAG_SCALING_AUTO, &autoDet);
    int autoPreferred = beagleCreateInstance(kTips, kNodes, 0, 4, 40, 1, kNodes - 1, 2, 0, nullptr, 0,
                                             BEAGLE_FLAG_SCALING_AUTO, BEAGLE_FLAG_HIERARCHICAL_REPRESENTATION, &autoDet);
    report("SCALING_AUTO required or preferred: no hierarchical instance", autoRequired < 0 && autoPreferred < 0,
           "codes " + std::to_string(autoRequired) + " " + std::to_string(autoPreferred));
    report("updatePartials without frequencies, with an unloaded slot (1 and 8 threads), pre-order",
           noFrequencies == BEAGLE_ERROR_OUT_OF_RANGE && missingSlot == BEAGLE_ERROR_OUT_OF_RANGE &&
           missingSlotThreaded == BEAGLE_ERROR_OUT_OF_RANGE && noPreOrder == BEAGLE_ERROR_NO_IMPLEMENTATION,
           "codes " + std::to_string(noFrequencies) + " " + std::to_string(missingSlot) + " " +
           std::to_string(missingSlotThreaded) + " " + std::to_string(noPreOrder));
    beagleFinalizeInstance(inst);
}

int main() {
    std::mt19937_64 rng(20261013);
    std::vector<Variant> variants(10);
    variants[0].name = "default";
    variants[1].name = "tip states";
    variants[1].states = true;
    variants[2].name = "8 threads";
    variants[2].threads = 8;
    variants[3].name = "3 partitions";
    variants[3].partitions = 3;
    variants[4].name = "3 partitions, 8 threads";
    variants[4].partitions = 3;
    variants[4].threads = 8;
    variants[5].name = "scaling manual";
    variants[5].scaling = Scaling::Manual;
    variants[6].name = "scaling always";
    variants[6].scaling = Scaling::Always;
    variants[7].name = "re-encoded bundles";
    variants[7].reencode = true;
    variants[8].name = "scaling manual raw, factors re-read";
    variants[8].scaling = Scaling::ManualRaw;
    variants[8].reread = true;
    variants[9].name = "scaling dynamic, two passes";
    variants[9].scaling = Scaling::Dynamic;

    for (bool dp : { true, false }) {
        for (int S : { 2, 3, 4, 5, 6, 7, 8, 17, 61 })
            for (int C : { 1, 4 }) {
                Problem pr = makeProblem(dp, S, C, 1003, rng);
                for (const Variant& v : variants) {
                    compare(pr, v, rng);
                    modelMatchesBundles(pr, v, rng);
                    if (v.scaling != Scaling::None)
                        rescaleMatchesLegacy(pr, v, rng);
                    fusedMatchesTiled(pr, v, rng);
                }
                statesMatchPartials(pr, rng);
            }
        for (int kind = 0; kind < 3; ++kind) {
            Problem pr = makeProblem(dp, 4, 4, 1003, rng);
            if (kind == 0) {
                pr.stateTree = makeShapedTree(4, false, rng, 2.0);
                pr.label = " balanced";
            } else if (kind == 1) {
                pr.stateTree = makeShapedTree(4, true, rng, 2.0);
                pr.label = " caterpillar";
            } else {
                std::vector<double> tipPi(pr.stateTree.pi.begin(), pr.stateTree.pi.begin() + 4);
                pr.branchTrees = { makeShapedTree(4, false, rng, 2.0, &tipPi), makeShapedTree(4, true, rng, 2.0, &tipPi),
                                   makeShapedTree(4, true, rng, 2.0, &tipPi) };
                std::swap(pr.branchTrees[2].lambda[4], pr.branchTrees[2].lambda[5]);
                pr.stateTree = pr.branchTrees[0];
                pr.label = " two state trees";
            }
            for (const Variant& v : variants) {
                compare(pr, v, rng);
                if (pr.branchTrees.empty())
                    modelMatchesBundles(pr, v, rng);
                if (v.scaling != Scaling::None)
                    rescaleMatchesLegacy(pr, v, rng);
                fusedMatchesTiled(pr, v, rng);
            }
            statesMatchPartials(pr, rng);
        }
        for (int S : { 4, 8, 17, 20, 61 })
            for (int P : { 5, 32, 33, 64, 95 }) {
                Problem pr = makeProblem(dp, S, 4, P, rng);
                for (int i : { 0, 1, 3, 4, 6, 9 })
                    fusedMatchesTiled(pr, variants[i], rng);
            }
    }
    failsLoudly(rng);

    printf("%d of %d checks passed\n", checks - failures, checks);
    return failures == 0 ? 0 : 1;
}
