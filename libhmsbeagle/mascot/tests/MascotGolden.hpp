/*
 * MascotGolden.hpp -- reader for the golden cases written by BEAST's MascotGoldenCorpus ("mascot-golden 1", "2" or "3";
 * version 2 adds the step rule and the per-segment step counts, version 3 the step budget). Every double is the 16 hex
 * digits of its raw bits.
 */
#ifndef MASCOT_GOLDEN_HPP
#define MASCOT_GOLDEN_HPP

#include "../MascotProblem.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mascot {

struct GoldenCase {
    std::string name;
    int K = 0, lineageCount = 0, intervalCount = 0, initialSampleNode = 0, epochCount = 0;
    double maxStep = 0.0, initialTime = 0.0;
    int stepRuleKind = STEPS_FIXED, maxStepsPerSegment = 0, stepBudget = 0;   /* stepBudget: version 3 only */
    double tau = 0.0, minStep = 0.0;
    std::vector<double> boundaries, migration, inversePopulation, theta, branchRates;
    std::vector<double> intervalLengths;
    std::vector<int> eventTypes, sampleNodes, child1, child2, parent;
    std::vector<double> tipPartials;
    std::string status;                 /* "ok", "numerical" (a NumericalException in Java), "stepBudget" (its
                                           StepBudgetException, version 3), "invalidRates"                       */
    double logL = 0.0;
    std::vector<double> thetaGradient, clockGradient, ancestralScores;
    std::vector<int> stepCounts;          /* version 2 only                                                   */

    Problem problem(bool checkProbabilities = false) const {
        Problem p;
        p.K = K;
        p.lineageCount = lineageCount;
        p.intervalCount = intervalCount;
        p.initialSampleNode = initialSampleNode;
        p.epochCount = epochCount;
        p.stepRuleKind = stepRuleKind;
        p.maxStep = maxStep;
        p.tau = tau;
        p.minStep = minStep;
        p.maxStepsPerSegment = maxStepsPerSegment;
        p.stepBudget = stepBudget;
        p.initialTime = initialTime;
        p.boundaries = boundaries.data();
        p.migration = migration.data();
        p.inversePopulation = inversePopulation.data();
        p.branchRates = branchRates.empty() ? nullptr : branchRates.data();
        p.intervalLengths = intervalLengths.data();
        p.eventTypes = eventTypes.data();
        p.sampleNodes = sampleNodes.data();
        p.child1 = child1.data();
        p.child2 = child2.data();
        p.parent = parent.data();
        p.tipPartials = tipPartials.data();
        p.checkProbabilities = checkProbabilities;
        p.checkpointReadSet = -1;
        p.checkpointWriteSet = -1;
        return p;
    }
};

/*
 * The strict kernels match Java bit for bit except where logL adds log(lambda) at each coalescence: the platform's
 * log may differ from Java's by an ulp of log(lambda), which is many ulp of logL when terms cancel. Allowed: 2 ulp of
 * logL plus one ulp of a log(lambda) up to 16 in magnitude (3.6e-15) per coalescence.
 */
inline bool logLWithinTolerance(double actual, double expected, const std::vector<int>& eventTypes) {
    if (actual == expected) return true;
    if (!std::isfinite(actual) || !std::isfinite(expected)) return false;
    int coalescences = 0;
    for (int type : eventTypes) coalescences += type == COALESCENT;
    double ulp = std::nextafter(std::fabs(expected), INFINITY) - std::fabs(expected);
    return std::fabs(actual - expected) <= 2.0 * ulp + coalescences * 3.6e-15;
}

/* Entries of actual that differ from expected: +0 and -0 compare equal, and NaN must meet NaN. */
inline size_t bitwiseMismatches(const std::vector<double>& expected, const double* actual) {
    size_t mismatches = 0;
    for (size_t i = 0; i < expected.size(); i++) {
        double e = expected[i], a = actual[i];
        if (std::isnan(e) || std::isnan(a)) mismatches += std::isnan(e) != std::isnan(a);
        else mismatches += !(a == e);
    }
    return mismatches;
}

/*
 * Worst |actual - expected| / max(|expected|, 1e-3 * max |expected|) over the finite entries, so near-zero entries are
 * judged on the scale of their array; infinity when the NaN patterns differ. Used for FMA-against-strict tolerances.
 */
inline double relativeDifference(const std::vector<double>& expected, const double* actual) {
    double scale = 0.0;
    for (double e : expected) if (std::isfinite(e)) scale = std::max(scale, std::fabs(e));
    double worst = 0.0;
    for (size_t i = 0; i < expected.size(); i++) {
        double e = expected[i], a = actual[i];
        if (std::isnan(e) || std::isnan(a)) {
            if (std::isnan(e) != std::isnan(a)) return INFINITY;
            continue;
        }
        double r = std::fabs(a - e) / std::max(std::fabs(e), 1.0e-3 * scale);
        if (!(r <= worst)) worst = std::isnan(r) ? INFINITY : r;
    }
    return worst;
}

inline double goldenDouble(const std::string& hex) {
    if (hex.size() != 16) throw std::runtime_error("bad hex double: " + hex);
    uint64_t bits = std::stoull(hex, nullptr, 16);
    double x;
    std::memcpy(&x, &bits, sizeof x);
    return x;
}

inline std::vector<double> goldenDoubles(std::istringstream& line) {
    std::vector<double> v;
    std::string t;
    while (line >> t) v.push_back(goldenDouble(t));
    return v;
}

inline GoldenCase readGolden(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open " + path);
    GoldenCase g;
    std::string line;
    std::getline(in, line);
    if (line != "mascot-golden 1" && line != "mascot-golden 2" && line != "mascot-golden 3") {
        throw std::runtime_error(path + ": not a mascot-golden 1, 2 or 3 file");
    }
    const int version = line[line.size() - 1] - '0';
    auto expect = [&](const std::string& key) -> std::istringstream {
        if (!std::getline(in, line)) throw std::runtime_error(path + ": missing " + key);
        std::istringstream s(line);
        std::string k;
        s >> k;
        if (k != key) throw std::runtime_error(path + ": expected " + key + ", found " + k);
        return s;
    };
    { auto s = expect("name"); s >> g.name; }
    { auto s = expect("dims"); s >> g.K >> g.lineageCount >> g.intervalCount >> g.initialSampleNode >> g.epochCount; }
    { auto s = expect("maxStep"); std::string t; s >> t; g.maxStep = goldenDouble(t); }
    if (version >= 2) {
        auto s = expect("stepRule");
        std::string kind, tau, minStep;
        s >> kind >> tau >> minStep >> g.maxStepsPerSegment;
        if (version >= 3) s >> g.stepBudget;
        if (kind != "fixed" && kind != "adaptive") throw std::runtime_error(path + ": unknown step rule " + kind);
        g.stepRuleKind = kind == "adaptive" ? STEPS_ADAPTIVE : STEPS_FIXED;
        g.tau = goldenDouble(tau);
        g.minStep = goldenDouble(minStep);
    }
    { auto s = expect("initialTime"); std::string t; s >> t; g.initialTime = goldenDouble(t); }
    { auto s = expect("boundaries"); g.boundaries = goldenDoubles(s); }
    { auto s = expect("migration"); g.migration = goldenDoubles(s); }
    { auto s = expect("inversePopulation"); g.inversePopulation = goldenDoubles(s); }
    { auto s = expect("theta"); g.theta = goldenDoubles(s); }
    {
        auto s = expect("branchRates");
        std::string rest;
        std::getline(s >> std::ws, rest);
        if (rest != "none") { std::istringstream r(rest); g.branchRates = goldenDoubles(r); }
    }
    expect("intervals");
    for (int i = 0; i < g.intervalCount; i++) {
        std::getline(in, line);
        std::istringstream s(line);
        std::string len;
        int type, sample, c1, c2, par;
        s >> len >> type >> sample >> c1 >> c2 >> par;
        g.intervalLengths.push_back(goldenDouble(len));
        g.eventTypes.push_back(type);
        g.sampleNodes.push_back(sample);
        g.child1.push_back(c1);
        g.child2.push_back(c2);
        g.parent.push_back(par);
    }
    expect("tips");
    g.tipPartials.assign((size_t) g.lineageCount * g.K, 0.0);
    for (int n = 0; n < g.lineageCount; n++) {
        std::getline(in, line);
        std::istringstream s(line);
        int node, has;
        s >> node >> has;
        if (has) {
            std::vector<double> p = goldenDoubles(s);
            for (int k = 0; k < g.K; k++) g.tipPartials[(size_t) node * g.K + k] = p[k];
        }
    }
    { auto s = expect("status"); s >> g.status; }
    if (g.status == "ok") {
        { auto s = expect("logL"); std::string t; s >> t; g.logL = goldenDouble(t); }
        { auto s = expect("thetaGradient"); g.thetaGradient = goldenDoubles(s); }
        {
            auto s = expect("clockGradient");
            std::string rest;
            std::getline(s >> std::ws, rest);
            if (rest != "none") { std::istringstream r(rest); g.clockGradient = goldenDoubles(r); }
        }
        { auto s = expect("ancestralScores"); g.ancestralScores = goldenDoubles(s); }
        if (version >= 2) {
            auto s = expect("stepCounts");
            int n;
            while (s >> n) g.stepCounts.push_back(n);
        }
    }
    expect("end");
    return g;
}

} // namespace mascot

#endif
