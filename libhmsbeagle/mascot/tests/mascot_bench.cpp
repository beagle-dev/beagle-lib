/*
 * mascot_bench.cpp -- times the MASCOT C API on golden cases (not a test).
 *
 * Usage: mascot_bench [--reps N] case.golden...
 * For each case, kernels A and B, strict and FMA, are interleaved in a shuffled order every rep after a warm-up;
 * the medians of the forward and the forward + reverse times are printed in ms.
 */
#include "../mascot.h"
#include "MascotGolden.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

using namespace mascot;

namespace {

struct Variant {
    const char* name;
    long flags;
    int instance = -1;
    std::vector<double> forward, full;
};

double millis(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

} // namespace

int main(int argc, char** argv) {
    int reps = 31;
    std::vector<std::string> files;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--reps" && i + 1 < argc) reps = std::atoi(argv[++i]);
        else files.push_back(a);
    }
    std::mt19937 rng(12345);
    std::printf("%-20s %-8s %10s %14s\n", "case", "variant", "fwd ms", "fwd+rev ms");
    for (const std::string& file : files) {
        GoldenCase g = readGolden(file);
        if (g.status != "ok") continue;
        std::vector<Variant> variants = {
            {"A", MASCOT_FP_STRICT | MASCOT_FORCE_KERNEL_A},
            {"B", MASCOT_FP_STRICT | MASCOT_FORCE_KERNEL_B},
            {"A-fma", MASCOT_FP_FMA | MASCOT_FORCE_KERNEL_A},
            {"B-fma", MASCOT_FP_FMA | MASCOT_FORCE_KERNEL_B},
        };
        for (Variant& v : variants) {
            v.instance = mascotCreateInstance(g.K, g.lineageCount, g.epochCount, v.flags);
            if (v.instance >= 0) mascotSetTipPartials(v.instance, g.tipPartials.data());
        }
        // kernel B is not built everywhere (MSVC)
        variants.erase(std::remove_if(variants.begin(), variants.end(),
                                      [](const Variant& v) { return v.instance < 0; }), variants.end());
        MascotStepRule rule = {g.stepRuleKind, g.maxStep, g.tau, g.minStep, g.maxStepsPerSegment, g.stepBudget};
        const double* br = g.branchRates.empty() ? nullptr : g.branchRates.data();
        std::vector<double> theta(g.thetaGradient.size()), clock(g.lineageCount);
        double logL;
        int outputs = MASCOT_OUT_LOGL | MASCOT_OUT_THETA_GRADIENT | (br ? MASCOT_OUT_CLOCK_GRADIENT : 0);
        auto run = [&](const Variant& v, bool gradient) {
            return mascotEvaluate(v.instance, g.intervalCount, g.initialTime, g.initialSampleNode,
                    g.intervalLengths.data(), g.eventTypes.data(), g.sampleNodes.data(), g.child1.data(),
                    g.child2.data(), g.parent.data(), g.epochCount, g.boundaries.data(), g.migration.data(),
                    g.inversePopulation.data(), br, &rule, 0, -1, -1, gradient ? outputs : MASCOT_OUT_LOGL, &logL,
                    gradient ? theta.data() : nullptr, gradient && br ? clock.data() : nullptr, nullptr);
        };
        std::vector<int> order(variants.size());
        for (size_t i = 0; i < order.size(); i++) order[i] = (int) i;
        for (int rep = -3; rep < reps; rep++) {
            std::shuffle(order.begin(), order.end(), rng);
            for (int i : order) {
                Variant& v = variants[i];
                auto t0 = std::chrono::steady_clock::now();
                run(v, false);
                auto t1 = std::chrono::steady_clock::now();
                run(v, true);
                auto t2 = std::chrono::steady_clock::now();
                if (rep >= 0) {
                    v.forward.push_back(millis(t0, t1));
                    v.full.push_back(millis(t1, t2));
                }
            }
        }
        for (Variant& v : variants) {
            std::printf("%-20s %-8s %10.3f %14.3f\n", g.name.c_str(), v.name, median(v.forward), median(v.full));
            mascotFinalizeInstance(v.instance);
        }
    }
    return 0;
}
