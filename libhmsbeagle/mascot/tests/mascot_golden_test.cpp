/*
 * mascot_golden_test.cpp -- runs MASCOT kernels on golden cases from BEAST's Java reference and compares.
 *
 * Usage: mascot_golden_test --kernel A|B [--fma] [--rtol R] case.golden...
 *
 * Strict builds (default): the theta gradient, clock gradient and ancestral scores must equal Java's bit for bit
 * (+0 and -0 compare equal; NaN must be NaN), and logL must be within logLWithinTolerance (MascotGolden.hpp):
 * the kernel's log() is the platform's, which may differ from Java's by an ulp of log(lambda).
 * FMA builds (--fma): every output must be within the relative tolerance R (default 1e-9) of Java's, as measured
 * by relativeDifference (MascotGolden.hpp).
 * Golden cases with status "numerical" must return ERROR_COALESCENT_RATE or ERROR_PROBABILITIES.
 */
#include "MascotGolden.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace mascot;

namespace {

int64_t ulpDistance(double a, double b) {
    if (a == b) return 0;
    if (std::isnan(a) || std::isnan(b) || std::signbit(a) != std::signbit(b)) return INT64_MAX;
    int64_t x, y;
    std::memcpy(&x, &a, sizeof x);
    std::memcpy(&y, &b, sizeof y);
    return x > y ? x - y : y - x;
}

struct Comparison {
    size_t mismatches = 0;
    double maxRelative = 0.0;
};

Comparison compare(const std::vector<double>& expected, const double* actual, bool strict) {
    Comparison c;
    if (strict) c.mismatches = bitwiseMismatches(expected, actual);
    else c.maxRelative = relativeDifference(expected, actual);
    return c;
}

/* MASCOT_TEST_ONLY_A / MASCOT_TEST_ONLY_B link a single kernel (for building one kernel in isolation). */
std::unique_ptr<Kernel> make(char kernel, bool fma) {
#ifndef MASCOT_TEST_ONLY_B
    if (kernel == 'A') return fma ? makeKernelAFma() : makeKernelAStrict();
#endif
#ifndef MASCOT_TEST_ONLY_A
    if (kernel == 'B') return fma ? makeKernelBFma() : makeKernelBStrict();
#endif
    return nullptr;
}

} // namespace

int main(int argc, char** argv) {
    char kernelName = 0;
    bool fma = false;
    double rtol = 1.0e-9;
    std::vector<std::string> files;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--kernel" && i + 1 < argc) kernelName = argv[++i][0];
        else if (a == "--fma") fma = true;
        else if (a == "--rtol" && i + 1 < argc) rtol = std::atof(argv[++i]);
        else files.push_back(a);
    }
    if ((kernelName != 'A' && kernelName != 'B') || files.empty()) {
        std::fprintf(stderr, "usage: mascot_golden_test --kernel A|B [--fma] [--rtol R] case.golden...\n");
        return 2;
    }
    const bool strict = !fma;
    std::unique_ptr<Kernel> kernel = make(kernelName, fma);
    if (!kernel) {
        std::fprintf(stderr, "kernel %c is not linked into this build\n", kernelName);
        return 2;
    }
    int failures = 0;
    int64_t worstUlp = 0;
    double worstRelative = 0.0;

    for (const std::string& file : files) {
        GoldenCase g = readGolden(file);
        if (g.status == "invalidRates") continue;
        Problem p = g.problem();
        size_t thetaSize = (size_t) g.epochCount * (g.K * (g.K - 1) + g.K);
        std::vector<double> theta(thetaSize, NAN), clock(g.lineageCount, NAN),
                ancestral((size_t) g.lineageCount * g.K, 0.0);
        double logLForward = NAN, logL = NAN;
        int statusForward = kernel->forward(p, &logLForward);
        int status = kernel->forwardAndReverse(p, &logL, theta.data(), p.branchRates ? clock.data() : nullptr,
                                               ancestral.data());

        std::string problems;
        if (g.status == "numerical") {
            bool ok1 = statusForward == ERROR_COALESCENT_RATE || statusForward == ERROR_PROBABILITIES;
            bool ok2 = status == ERROR_COALESCENT_RATE || status == ERROR_PROBABILITIES;
            if (!ok1 || !ok2) problems += " expected an error status, got " + std::to_string(statusForward) + "/" +
                                          std::to_string(status);
            std::printf("%-28s %s%s\n", g.name.c_str(), problems.empty() ? "ok (error status)" : "FAIL",
                        problems.c_str());
            if (!problems.empty()) failures++;
            continue;
        }
        if (statusForward != OK || status != OK) {
            std::printf("%-28s FAIL status %d/%d\n", g.name.c_str(), statusForward, status);
            failures++;
            continue;
        }

        int64_t ulpF = ulpDistance(logLForward, g.logL), ulpR = ulpDistance(logL, g.logL);
        double relLogL = std::max(std::fabs(logLForward - g.logL), std::fabs(logL - g.logL)) / std::fabs(g.logL);
        Comparison ct = compare(g.thetaGradient, theta.data(), strict);
        Comparison cc = p.branchRates ? compare(g.clockGradient, clock.data(), strict) : Comparison();
        Comparison ca = compare(g.ancestralScores, ancestral.data(), strict);

        if (strict) {
            if (!logLWithinTolerance(logLForward, g.logL, g.eventTypes) ||
                !logLWithinTolerance(logL, g.logL, g.eventTypes)) {
                problems += " logL ulp " + std::to_string(std::max(ulpF, ulpR));
            }
            if (ct.mismatches) problems += " theta " + std::to_string(ct.mismatches) + "/" + std::to_string(thetaSize);
            if (cc.mismatches) problems += " clock " + std::to_string(cc.mismatches);
            if (ca.mismatches) problems += " ancestral " + std::to_string(ca.mismatches);
            worstUlp = std::max(worstUlp, std::max(ulpF, ulpR));
            std::printf("%-28s %s logL ulp %lld%s\n", g.name.c_str(), problems.empty() ? "ok  " : "FAIL",
                        (long long) std::max(ulpF, ulpR), problems.c_str());
        } else {
            double worst = std::max({relLogL, ct.maxRelative, cc.maxRelative, ca.maxRelative});
            if (!(worst <= rtol)) problems += " exceeds rtol";
            worstRelative = std::max(worstRelative, worst);
            std::printf("%-28s %s rel logL %.2e theta %.2e clock %.2e ancestral %.2e%s\n", g.name.c_str(),
                        problems.empty() ? "ok  " : "FAIL", relLogL, ct.maxRelative, cc.maxRelative, ca.maxRelative,
                        problems.c_str());
        }
        if (!problems.empty()) failures++;
    }
    if (strict) std::printf("kernel %c strict: %d failures; worst logL ulp %lld\n", kernelName, failures,
                            (long long) worstUlp);
    else std::printf("kernel %c fma: %d failures; worst relative difference %.2e (rtol %.1e)\n", kernelName,
                     failures, worstRelative, rtol);
    return failures == 0 ? 0 : 1;
}
