/*
 * mascot_api_test.cpp -- the MASCOT C API (mascot.h) on golden cases from BEAST's Java reference, with the default
 * kernel choice (argument checks, dispatch and concurrency are in mascot_selftest).
 *
 * Usage: mascot_api_test [--fma] [--rtol R] case.golden...
 * Comparison rules are those of mascot_golden_test (strict: bitwise outputs, logL within logLWithinTolerance).
 */
#include "../mascot.h"
#include "MascotGolden.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace mascot;

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::printf("FAIL %s\n", what.c_str());
        failures++;
    }
}

struct Result {
    int statusForward, status;
    double logLForward, logL;
    std::vector<double> theta, clock, ancestral;
    std::vector<int> stepCounts;
};

int create(const GoldenCase& g, long flags) {
    int instance = mascotCreateInstance(g.K, g.lineageCount, g.epochCount, flags);
    if (instance >= 0) mascotSetTipPartials(instance, g.tipPartials.data());
    return instance;
}

Result evaluate(int instance, const GoldenCase& g) {
    Result r;
    MascotStepRule rule = {g.stepRuleKind, g.maxStep, g.tau, g.minStep, g.maxStepsPerSegment};
    const double* br = g.branchRates.empty() ? nullptr : g.branchRates.data();
    r.theta.assign((size_t) g.epochCount * (g.K * (g.K - 1) + g.K), NAN);
    r.clock.assign(g.lineageCount, NAN);
    r.ancestral.assign((size_t) g.lineageCount * g.K, 0.0);
    r.statusForward = mascotEvaluate(instance, g.intervalCount, g.initialTime, g.initialSampleNode,
            g.intervalLengths.data(), g.eventTypes.data(), g.sampleNodes.data(), g.child1.data(), g.child2.data(),
            g.parent.data(), g.epochCount, g.boundaries.data(), g.migration.data(), g.inversePopulation.data(), br,
            &rule, 0, -1, -1, MASCOT_OUT_LOGL, &r.logLForward, nullptr, nullptr, nullptr);
    int outputs = MASCOT_OUT_LOGL | MASCOT_OUT_THETA_GRADIENT | MASCOT_OUT_ANCESTRAL_SCORES |
                  (br ? MASCOT_OUT_CLOCK_GRADIENT : 0);
    r.status = mascotEvaluate(instance, g.intervalCount, g.initialTime, g.initialSampleNode,
            g.intervalLengths.data(), g.eventTypes.data(), g.sampleNodes.data(), g.child1.data(), g.child2.data(),
            g.parent.data(), g.epochCount, g.boundaries.data(), g.migration.data(), g.inversePopulation.data(), br,
            &rule, 0, -1, -1, outputs, &r.logL, r.theta.data(), br ? r.clock.data() : nullptr, r.ancestral.data());
    if (r.status == MASCOT_OK) {
        std::vector<int> steps(g.intervalCount + g.epochCount + 1);
        int n = 0;
        if (mascotGetLastStepCounts(instance, steps.data(), (int) steps.size(), &n) == MASCOT_OK) {
            r.stepCounts.assign(steps.begin(), steps.begin() + n);
        }
    }
    return r;
}

void compareGolden(const GoldenCase& g, const Result& r, bool strict, double rtol) {
    const std::string n = g.name + ": ";
    if (g.status == "numerical") {
        check(r.statusForward == MASCOT_ERROR_COALESCENT_RATE || r.statusForward == MASCOT_ERROR_PROBABILITIES,
              n + "forward status " + std::to_string(r.statusForward));
        check(r.status == MASCOT_ERROR_COALESCENT_RATE || r.status == MASCOT_ERROR_PROBABILITIES,
              n + "gradient status " + std::to_string(r.status));
        return;
    }
    check(r.statusForward == MASCOT_OK && r.status == MASCOT_OK,
          n + "status " + std::to_string(r.statusForward) + "/" + std::to_string(r.status));
    if (r.status != MASCOT_OK) return;
    if (!g.stepCounts.empty()) check(r.stepCounts == g.stepCounts, n + "step counts");
    if (strict) {
        check(logLWithinTolerance(r.logLForward, g.logL, g.eventTypes) &&
              logLWithinTolerance(r.logL, g.logL, g.eventTypes), n + "logL");
        check(bitwiseMismatches(g.thetaGradient, r.theta.data()) == 0, n + "theta gradient");
        if (!g.branchRates.empty()) check(bitwiseMismatches(g.clockGradient, r.clock.data()) == 0, n + "clock gradient");
        check(bitwiseMismatches(g.ancestralScores, r.ancestral.data()) == 0, n + "ancestral scores");
    } else {
        double worst = std::max(std::fabs(r.logL - g.logL), std::fabs(r.logLForward - g.logL)) / std::fabs(g.logL);
        worst = std::max(worst, relativeDifference(g.thetaGradient, r.theta.data()));
        if (!g.branchRates.empty()) worst = std::max(worst, relativeDifference(g.clockGradient, r.clock.data()));
        worst = std::max(worst, relativeDifference(g.ancestralScores, r.ancestral.data()));
        check(worst <= rtol, n + "relative difference " + std::to_string(worst));
    }
}

} // namespace

int main(int argc, char** argv) {
    bool fma = false;
    double rtol = 1.0e-9;
    std::vector<std::string> files;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--fma") fma = true;
        else if (a == "--rtol" && i + 1 < argc) rtol = std::atof(argv[++i]);
        else files.push_back(a);
    }
    if (files.empty()) {
        std::fprintf(stderr, "usage: mascot_api_test [--fma] [--rtol R] case.golden...\n");
        return 2;
    }
    int major, minor;
    check(mascotGetApiVersion(&major, &minor) == MASCOT_OK && major == 1 && minor == 2, "API version 1.2");

    const long flags = fma ? MASCOT_FP_FMA : MASCOT_FP_STRICT;
    std::vector<GoldenCase> cases;
    for (const std::string& f : files) cases.push_back(readGolden(f));
    for (const GoldenCase& g : cases) {
        if (g.status == "invalidRates") continue;
        int instance = create(g, flags);
        check(instance >= 0, g.name + ": create");
        compareGolden(g, evaluate(instance, g), !fma, rtol);
        mascotFinalizeInstance(instance);
    }
    std::printf("mascot_api_test %s: %zu cases, %d failures\n", fma ? "fma" : "strict", cases.size(), failures);
    return failures == 0 ? 0 : 1;
}
