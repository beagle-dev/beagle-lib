/*
 * mascot_selftest.cpp -- self-contained MASCOT tests (no data files), for CI.
 *
 * On generated problems (MascotSynthetic.hpp):
 *  - kernels A and B agree bit for bit in the strict build (each is bitwise equal to BEAST's Java reference, which
 *    the opt-in golden tests check), and each FMA build is within --rtol of its strict build;
 *  - forward() and forwardAndReverse() give the same logL, and every build takes the same step counts (the step rule,
 *    fixed or adaptive, is evaluated in strict arithmetic);
 *  - theta and clock gradients match central finite differences;
 *  - lambda = 0 gives ERROR_COALESCENT_RATE, and checkProbabilities is silent on valid problems and fires on invalid tips;
 *  - the C API: version, the K -> kernel table, FP modes, argument and unsupported statuses, instance reuse, partial
 *    outputs, and concurrent instances reproducing their serial results.
 *
 * Usage: mascot_selftest [--rtol R]
 */
#include "../mascot.h"
#include "MascotSynthetic.hpp"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace mascot;

namespace {

int failures = 0;
int checks = 0;

void check(bool ok, const std::string& what) {
    checks++;
    if (!ok) {
        std::printf("FAIL %s\n", what.c_str());
        failures++;
    }
}

struct Run {
    int statusForward = 0, status = 0;
    double logLForward = NAN, logL = NAN;
    std::vector<double> theta, clock, ancestral;
    std::vector<int> steps;
};

std::vector<int> stepCounts(const Kernel& kernel, const GoldenCase& g) {
    std::vector<int> steps(g.intervalCount + g.epochCount + 1);
    int n = 0;
    if (kernel.lastStepCounts(steps.data(), (int) steps.size(), &n) != OK) return {};
    steps.resize(n);
    return steps;
}

Run run(Kernel& kernel, const GoldenCase& g, bool checkProbabilities = false) {
    Run r;
    Problem p = g.problem(checkProbabilities);
    r.theta.assign((size_t) g.epochCount * (g.K * (g.K - 1) + g.K), NAN);
    r.clock.assign(g.lineageCount, NAN);
    r.ancestral.assign((size_t) g.lineageCount * g.K, 0.0);
    r.statusForward = kernel.forward(p, &r.logLForward);
    r.status = kernel.forwardAndReverse(p, &r.logL, r.theta.data(), p.branchRates ? r.clock.data() : nullptr,
                                        r.ancestral.data());
    r.steps = stepCounts(kernel, g);
    return r;
}

double logLAt(Kernel& kernel, const GoldenCase& g) {
    double logL = NAN;
    Problem p = g.problem();
    return kernel.forward(p, &logL) == OK ? logL : NAN;
}

/* Strict A and strict B must agree bit for bit; each FMA build must be within rtol of its strict build. */
void testKernels(const GoldenCase& g, double rtol) {
    std::unique_ptr<Kernel> as = makeKernelAStrict(), bs = makeKernelBStrict(), af = makeKernelAFma(),
                            bf = makeKernelBFma();
    Run ra = run(*as, g), rb = run(*bs, g), rfa = run(*af, g), rfb = run(*bf, g);
    const std::string n = g.name + ": ";
    for (const Run* r : {&ra, &rb, &rfa, &rfb}) {
        check(r->statusForward == OK && r->status == OK, n + "status");
        check(r->logLForward == r->logL, n + "forward and forwardAndReverse logL differ");
    }
    if (ra.status != OK || rb.status != OK) return;
    check(!ra.steps.empty() && ra.steps == rb.steps && ra.steps == rfa.steps && ra.steps == rfb.steps,
          n + "step counts differ between kernels or FP modes");
    check(ra.logL == rb.logL, n + "A and B strict logL differ");
    check(bitwiseMismatches(ra.theta, rb.theta.data()) == 0, n + "A and B strict theta gradients differ");
    if (!g.branchRates.empty()) {
        check(bitwiseMismatches(ra.clock, rb.clock.data()) == 0, n + "A and B strict clock gradients differ");
    }
    check(bitwiseMismatches(ra.ancestral, rb.ancestral.data()) == 0, n + "A and B strict ancestral scores differ");
    for (const auto& pair : {std::make_pair(&ra, &rfa), std::make_pair(&rb, &rfb)}) {
        const Run& s = *pair.first;
        const Run& f = *pair.second;
        double worst = std::fabs(f.logL - s.logL) / std::fabs(s.logL);
        worst = std::max(worst, relativeDifference(s.theta, f.theta.data()));
        if (!g.branchRates.empty()) worst = std::max(worst, relativeDifference(s.clock, f.clock.data()));
        worst = std::max(worst, relativeDifference(s.ancestral, f.ancestral.data()));
        check(worst <= rtol, n + "FMA differs from strict by " + std::to_string(worst));
    }
}

/* Central differences of logL (strict kernel A) against the analytic theta and clock gradients. */
void testFiniteDifferences(const GoldenCase& g0) {
    std::unique_ptr<Kernel> kernel = makeKernelAStrict();
    Run r = run(*kernel, g0);
    if (r.status != OK) {
        check(false, g0.name + ": finite differences need a valid problem");
        return;
    }
    double scale = 0.0;
    for (double x : r.theta) scale = std::max(scale, std::fabs(x));
    const int K = g0.K, per = K * (K - 1) + K;
    std::vector<int> entries;
    for (int e = 0; e < g0.epochCount; e++) {
        entries.push_back(e * per);                           // a migration rate
        entries.push_back(e * per + K * (K - 1) - 1);         // the last migration rate
        entries.push_back(e * per + K * (K - 1));             // a log Ne
        entries.push_back(e * per + per - 1);                 // the last log Ne
    }
    const double relativeStep = 1.0e-5;
    for (int i : entries) {
        double x = g0.theta[i];
        if (x == 0.0 && i % per < K * (K - 1)) continue;      // an inactive rate cannot be perturbed downwards
        double h = relativeStep * std::max(std::fabs(x), 1.0);
        GoldenCase up = g0, down = g0;
        up.theta[i] = x + h;
        down.theta[i] = x - h;
        ratesFromTheta(up);
        ratesFromTheta(down);
        double logLUp = logLAt(*kernel, up);
        bool sameSteps = stepCounts(*kernel, up) == r.steps;
        double logLDown = logLAt(*kernel, down);
        if (!sameSteps || stepCounts(*kernel, down) != r.steps) continue;   // logL jumps where a count changes
        double numeric = (logLUp - logLDown) / (2.0 * h);
        double error = std::fabs(numeric - r.theta[i]) / std::max(std::fabs(r.theta[i]), 1.0e-2 * scale + 1.0e-8);
        check(error < 1.0e-5, g0.name + ": theta[" + std::to_string(i) + "] analytic " +
                              std::to_string(r.theta[i]) + " numeric " + std::to_string(numeric));
    }
    if (g0.branchRates.empty()) return;
    double clockScale = 0.0;
    for (double x : r.clock) clockScale = std::max(clockScale, std::fabs(x));
    for (int lineage : {0, g0.lineageCount / 2, g0.lineageCount - 2}) {
        double x = g0.branchRates[lineage];
        double h = relativeStep * x;
        GoldenCase up = g0, down = g0;
        up.branchRates[lineage] = x + h;
        down.branchRates[lineage] = x - h;
        double logLUp = logLAt(*kernel, up);
        bool sameSteps = stepCounts(*kernel, up) == r.steps;
        double logLDown = logLAt(*kernel, down);
        if (!sameSteps || stepCounts(*kernel, down) != r.steps) continue;
        double numeric = (logLUp - logLDown) / (2.0 * h);
        double error = std::fabs(numeric - r.clock[lineage]) /
                       std::max(std::fabs(r.clock[lineage]), 1.0e-2 * clockScale + 1.0e-8);
        check(error < 1.0e-5, g0.name + ": clock[" + std::to_string(lineage) + "] analytic " +
                              std::to_string(r.clock[lineage]) + " numeric " + std::to_string(numeric));
    }
}

void testStatuses(const GoldenCase& valid) {
    GoldenCase bad = coalescentRateErrorCase();
    for (auto make : {makeKernelAStrict, makeKernelBStrict, makeKernelAFma, makeKernelBFma}) {
        std::unique_ptr<Kernel> kernel = make();
        const std::string n = std::string("kernel ") + kernel->name() + (kernel->fusedMultiplyAdd() ? " fma" : "");
        Run r = run(*kernel, bad);
        check(r.statusForward == ERROR_COALESCENT_RATE && r.status == ERROR_COALESCENT_RATE,
              n + ": lambda = 0 gives " + std::to_string(r.statusForward) + "/" + std::to_string(r.status));

        Run unchecked = run(*kernel, valid), checked = run(*kernel, valid, true);
        check(checked.status == OK && checked.logL == unchecked.logL &&
              bitwiseMismatches(unchecked.theta, checked.theta.data()) == 0,
              n + ": checkProbabilities changes a valid problem");

        GoldenCase corrupt = valid;
        int tip = -1;
        for (int i = 0; i < corrupt.intervalCount && tip < 0; i++)
            if (corrupt.eventTypes[i] == SAMPLE) tip = corrupt.sampleNodes[i];
        double* p = corrupt.tipPartials.data() + (size_t) tip * corrupt.K;
        p[0] = -0.5;
        p[1] = 1.5;
        Run c = run(*kernel, corrupt, true);
        check(c.statusForward == ERROR_PROBABILITIES && c.status == ERROR_PROBABILITIES,
              n + ": checkProbabilities on a negative tip gives " + std::to_string(c.status));
    }
}

int create(const GoldenCase& g, long flags) {
    int instance = mascotCreateInstance(g.K, g.lineageCount, g.epochCount, flags);
    if (instance >= 0) mascotSetTipPartials(instance, g.tipPartials.data());
    return instance;
}

int evaluate(int instance, const GoldenCase& g, const MascotStepRule* rule, int outputs, double* logL,
             double* theta, double* clock, double* ancestral, int epochCount = -1, int read = -1, int write = -1) {
    return mascotEvaluate(instance, g.intervalCount, g.initialTime, g.initialSampleNode, g.intervalLengths.data(),
            g.eventTypes.data(), g.sampleNodes.data(), g.child1.data(), g.child2.data(), g.parent.data(),
            epochCount < 0 ? g.epochCount : epochCount, g.boundaries.data(), g.migration.data(),
            g.inversePopulation.data(), g.branchRates.empty() ? nullptr : g.branchRates.data(), rule, 0, read, write,
            outputs, logL, theta, clock, ancestral);
}

void testApi(const std::vector<GoldenCase>& byK) {
    int major = 0, minor = 0;
    check(mascotGetApiVersion(&major, &minor) == MASCOT_OK && major == 1 && minor == 2, "API version 1.2");

    for (const GoldenCase& g : byK) {
        const char expected = (g.K <= 7 && g.K != 6) ? 'A' : 'B';
        for (long fp : {(long) MASCOT_FP_STRICT, 0L}) {
            int instance = create(g, fp);
            MascotInstanceDetails d;
            check(instance >= 0 && mascotGetInstanceDetails(instance, &d) == MASCOT_OK, g.name + ": details");
            check(d.kernel == expected, g.name + ": kernel " + d.kernel);
            if (fp == MASCOT_FP_STRICT) check(d.fpMode == MASCOT_FP_STRICT, g.name + ": strict instance not strict");
            else check(d.fpMode == MASCOT_FP_FMA || d.fpMode == MASCOT_FP_STRICT, g.name + ": default FP mode");
            mascotFinalizeInstance(instance);
        }
    }

    const GoldenCase& g = byK.front();
    int instance = create(g, MASCOT_FP_STRICT);
    double logL;
    MascotStepRule fixed = {MASCOT_STEPS_FIXED, g.maxStep, 0.0, 0.0, 0};
    MascotStepRule adaptive = {MASCOT_STEPS_ADAPTIVE, g.maxStep, 0.25, 0.0, 0};
    MascotStepRule badTau = {MASCOT_STEPS_ADAPTIVE, g.maxStep, 0.0, 0.0, 0};
    MascotStepRule zeroStep = {MASCOT_STEPS_FIXED, 0.0, 0.0, 0.0, 0};
    check(evaluate(instance, g, &adaptive, MASCOT_OUT_LOGL, &logL, nullptr, nullptr, nullptr) == MASCOT_OK,
          "ADAPTIVE evaluates (API 1.1)");
    check(evaluate(instance, g, &badTau, MASCOT_OUT_LOGL, &logL, nullptr, nullptr, nullptr) == MASCOT_ERROR_ARGUMENT,
          "ADAPTIVE with tau 0 rejected");
    check(evaluate(instance, g, &fixed, MASCOT_OUT_LOGL, &logL, nullptr, nullptr, nullptr, -1, 0, 1) == MASCOT_OK,
          "checkpoint sets accepted (API 1.2)");
    check(evaluate(instance, g, &fixed, MASCOT_OUT_LOGL, &logL, nullptr, nullptr, nullptr, -1, 2, 1) ==
          MASCOT_ERROR_ARGUMENT, "checkpoint set 2 rejected");
    int restart = -2;
    check(evaluate(instance, g, &fixed, MASCOT_OUT_LOGL, &logL, nullptr, nullptr, nullptr, -1, 1, 0) == MASCOT_OK &&
          mascotGetLastRestartInterval(instance, &restart) == MASCOT_OK && restart == g.intervalCount,
          "unchanged inputs resume after the last interval (" + std::to_string(restart) + ")");
    check(evaluate(instance, g, &zeroStep, MASCOT_OUT_LOGL, &logL, nullptr, nullptr, nullptr) ==
          MASCOT_ERROR_ARGUMENT, "maxStep 0 rejected");
    check(evaluate(instance, g, nullptr, MASCOT_OUT_LOGL, &logL, nullptr, nullptr, nullptr) ==
          MASCOT_ERROR_ARGUMENT, "null step rule rejected");
    check(evaluate(instance, g, &fixed, MASCOT_OUT_LOGL | MASCOT_OUT_THETA_GRADIENT, &logL, nullptr, nullptr,
                   nullptr) == MASCOT_ERROR_ARGUMENT, "gradient requested without a buffer rejected");
    check(evaluate(instance, g, &fixed, MASCOT_OUT_LOGL, &logL, nullptr, nullptr, nullptr, g.epochCount + 1) ==
          MASCOT_ERROR_ARGUMENT, "more epochs than the instance holds rejected");

    // ancestral scores alone equal those of a full request
    std::vector<double> theta(g.theta.size()), clock(g.lineageCount), a1((size_t) g.lineageCount * g.K),
            a2((size_t) g.lineageCount * g.K);
    int all = MASCOT_OUT_LOGL | MASCOT_OUT_THETA_GRADIENT | MASCOT_OUT_ANCESTRAL_SCORES |
              (g.branchRates.empty() ? 0 : MASCOT_OUT_CLOCK_GRADIENT);
    check(evaluate(instance, g, &fixed, all, &logL, theta.data(), clock.data(), a1.data()) == MASCOT_OK, "all outputs");
    check(evaluate(instance, g, &fixed, MASCOT_OUT_LOGL | MASCOT_OUT_ANCESTRAL_SCORES, &logL, nullptr, nullptr,
                   a2.data()) == MASCOT_OK && bitwiseMismatches(a1, a2.data()) == 0,
          "ancestral scores alone equal those of a full request");

    check(mascotFinalizeInstance(instance) == MASCOT_OK, "finalize");
    check(mascotFinalizeInstance(instance) == MASCOT_ERROR_ARGUMENT, "double finalize rejected");
    check(mascotCreateInstance(1, 3, 1, 0) == MASCOT_ERROR_ARGUMENT, "K = 1 rejected");
    check(mascotCreateInstance(3, 3, 1, MASCOT_FP_STRICT | MASCOT_FP_FMA) == MASCOT_ERROR_ARGUMENT,
          "both FP modes rejected");
    check(mascotCreateInstance(3, 3, 1, MASCOT_FORCE_KERNEL_A | MASCOT_FORCE_KERNEL_B) == MASCOT_ERROR_ARGUMENT,
          "both forced kernels rejected");
    int reused = mascotCreateInstance(3, 3, 1, 0);
    check(reused == instance, "finalized slot reused");
    mascotFinalizeInstance(reused);
}

/* Two instances evaluated concurrently, many times, must reproduce their serial results bit for bit. */

/* The first schedule interval at which two problems differ, or intervalCount if none; -1 for any global difference. */
int firstDifference(const GoldenCase& a, const GoldenCase& b) {
    if (a.K != b.K || a.intervalCount != b.intervalCount || a.initialTime != b.initialTime ||
        a.initialSampleNode != b.initialSampleNode || a.boundaries != b.boundaries || a.migration != b.migration ||
        a.inversePopulation != b.inversePopulation || a.branchRates != b.branchRates ||
        a.tipPartials != b.tipPartials || a.stepRuleKind != b.stepRuleKind || a.maxStep != b.maxStep ||
        a.tau != b.tau || a.minStep != b.minStep || a.maxStepsPerSegment != b.maxStepsPerSegment) {
        return 0;
    }
    for (int i = 0; i < a.intervalCount; i++) {
        if (a.intervalLengths[i] != b.intervalLengths[i] || a.eventTypes[i] != b.eventTypes[i] ||
            a.sampleNodes[i] != b.sampleNodes[i] || a.child1[i] != b.child1[i] || a.child2[i] != b.child2[i] ||
            a.parent[i] != b.parent[i]) {
            return i;
        }
    }
    return a.intervalCount;
}

double fullLogL(Kernel& fresh, const GoldenCase& g) {
    double logL = NAN;
    Problem p = g.problem();
    return fresh.forward(p, &logL) == OK ? logL : NAN;
}

/*
 * Incremental restart (R4): random "tree moves" (interval lengths changed from a random interval on) accepted or
 * rejected, with checkpoint sets used as Java uses them: read the current state's set, write the set the stored state
 * does not hold. Every pass must resume at the first changed interval and give the bits of a full pass. Then global
 * changes (rates, tips, step rule) and in-place updates.
 */
void testIncremental(const GoldenCase& g0, std::unique_ptr<Kernel> (*make)()) {
    std::unique_ptr<Kernel> kernel = make(), fresh = make();
    const std::string n = g0.name + " (" + kernel->name() + (kernel->fusedMultiplyAdd() ? " fma" : "") + "): ";
    std::mt19937_64 rng(77);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    GoldenCase built[2] = {g0, g0};       // the inputs each set was last written from
    bool written[2] = {false, false};
    int stored = 0, current = 0;
    GoldenCase accepted = g0;

    auto evaluate = [&](const GoldenCase& g, int read, int write, int expectedRestart, const std::string& what) {
        Problem p = g.problem();
        p.checkpointReadSet = read;
        p.checkpointWriteSet = write;
        double logL = NAN;
        int status = kernel->forward(p, &logL);
        double expected = fullLogL(*fresh, g);
        check(status == OK && logL == expected, n + what + ": logL " + std::to_string(logL) + " vs full " +
                                               std::to_string(expected));
        check(kernel->lastRestartInterval() == expectedRestart, n + what + ": restart " +
              std::to_string(kernel->lastRestartInterval()) + ", expected " + std::to_string(expectedRestart));
        if (write >= 0) {
            built[write] = g;
            written[write] = true;
        }
    };

    evaluate(g0, -1, 0, -1, "first pass");
    stored = current = 0;
    for (int move = 0; move < 12; move++) {
        GoldenCase proposed = accepted;
        int j = (int) (u(rng) * proposed.intervalCount);
        for (int i = j; i < proposed.intervalCount; i++) {
            if (proposed.intervalLengths[i] > 0.0 && (i == j || u(rng) < 0.3)) proposed.intervalLengths[i] *= 0.9 + 0.2 * u(rng);
        }
        int write = 1 - stored;
        int expected = written[current] ? firstDifference(built[current], proposed) : 0;
        evaluate(proposed, current, write, expected, "move " + std::to_string(move));
        current = write;
        if (u(rng) < 0.5) {                 // accept: the proposal is the new stored state
            accepted = proposed;
            stored = current;
        } else {                            // reject: back to the stored state's set
            current = stored;
        }
    }

    // global changes start from the beginning
    GoldenCase rates = accepted;
    rates.inversePopulation[0] *= 1.01;
    evaluate(rates, current, 1 - stored, 0, "changed inverse population");
    current = stored;
    GoldenCase tips = accepted;                 // the first sample becomes one-hot in the state after its likeliest
    double* first = tips.tipPartials.data() + (size_t) tips.initialSampleNode * tips.K;
    int likeliest = (int) (std::max_element(first, first + tips.K) - first);
    std::fill(first, first + tips.K, 0.0);
    first[(likeliest + 1) % tips.K] = 1.0;
    evaluate(tips, current, 1 - stored, 0, "changed tips");
    current = stored;
    GoldenCase rule = accepted;
    rule.stepRuleKind = rule.stepRuleKind == STEPS_FIXED ? STEPS_ADAPTIVE : STEPS_FIXED;
    if (rule.stepRuleKind == STEPS_ADAPTIVE) rule.tau = 0.25;
    if (rule.stepRuleKind == STEPS_FIXED && !(rule.maxStep > 0.0)) rule.maxStep = 0.05;
    evaluate(rule, current, 1 - stored, 0, "changed step rule");
    current = stored;

    // in place: read and write the same set
    GoldenCase moved = accepted;
    int j = moved.intervalCount / 2;
    moved.intervalLengths[j] *= 1.05;
    evaluate(moved, stored, stored, firstDifference(built[stored], moved), "in place");
    evaluate(moved, stored, stored, moved.intervalCount, "in place, unchanged");
}

void testConcurrentInstances(const GoldenCase& a, const GoldenCase& b) {
    MascotStepRule fixed = {MASCOT_STEPS_FIXED, a.maxStep, 0.0, 0.0, 0};
    auto full = [&](int instance, const GoldenCase& g, std::vector<double>& theta) {
        double logL = NAN;
        theta.assign(g.theta.size(), NAN);
        evaluate(instance, g, &fixed, MASCOT_OUT_LOGL | MASCOT_OUT_THETA_GRADIENT, &logL, theta.data(), nullptr,
                 nullptr);
        return logL;
    };
    int ia = create(a, 0), ib = create(b, 0);
    std::vector<double> ta, tb;
    double la = full(ia, a, ta), lb = full(ib, b, tb);
    std::atomic<bool> same(true);
    auto loop = [&](int instance, const GoldenCase& g, double logL, const std::vector<double>& theta) {
        std::vector<double> t;
        for (int i = 0; i < 20; i++) {
            if (full(instance, g, t) != logL || bitwiseMismatches(theta, t.data()) != 0) same = false;
        }
    };
    std::thread first(loop, ia, std::cref(a), la, std::cref(ta)), second(loop, ib, std::cref(b), lb, std::cref(tb));
    first.join();
    second.join();
    check(same, "concurrent instances reproduce their serial results");
    mascotFinalizeInstance(ia);
    mascotFinalizeInstance(ib);
}

} // namespace

int main(int argc, char** argv) {
    double rtol = 1.0e-11;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--rtol" && i + 1 < argc) rtol = std::atof(argv[++i]);
    }

    std::vector<GoldenCase> cases;
    const double steps[] = {0.02, 0.005, 0.1};
    int variant = 0;
    for (int tips : {2, 3, 5, 10, 20, 40}) {
        for (int K : {2, 3, 4, 5, 6, 7, 8, 9, 12, 17}) {
            SyntheticOptions o;
            o.tipCount = tips;
            o.K = K;
            o.epochCount = 1 + variant % 3;
            o.maxStep = steps[variant % 3];
            o.branchRates = variant % 2 == 0;
            o.simultaneousSamples = variant % 4 == 1;
            o.zeroRates = variant % 5 == 2;
            o.ambiguousTips = variant % 3 == 2;
            o.logNeShift = variant % 6 == 3 ? -3.0 : 0.0;
            cases.push_back(syntheticCase("n" + std::to_string(tips) + "K" + std::to_string(K) + "v" +
                                          std::to_string(variant), 9000 + variant, o));
            variant++;
        }
    }
    for (int K : {5, 16, 33, 64}) {
        SyntheticOptions o;
        o.tipCount = 150;
        o.K = K;
        cases.push_back(syntheticCase("n150K" + std::to_string(K), 9500 + K, o));
    }

    // every problem again under the adaptive rule: uncapped, capped at 0.1, and with each guard
    const size_t fixedCount = cases.size();
    for (size_t i = 0; i < fixedCount; i++) {
        GoldenCase a = cases[i];
        a.name = "adaptive-" + a.name;
        a.stepRuleKind = STEPS_ADAPTIVE;
        a.tau = 0.25;
        a.maxStep = i % 4 == 1 ? 0.1 : 0.0;
        a.minStep = i % 4 == 2 ? 0.05 : 0.0;
        // a guard that binds on the stiffest problems (they want 62-474 steps) but keeps RK4 well inside its
        // stability limit (about a tenth of the uncapped count here)
        a.maxStepsPerSegment = i % 4 == 3 ? 64 : 0;
        cases.push_back(a);
    }

    for (const GoldenCase& g : cases) testKernels(g, rtol);
    for (const GoldenCase& g : cases) {
        bool small = g.lineageCount <= 39 && g.K <= 9;
        if (small || g.name == "n150K5" || g.name == "adaptive-n150K5") testFiniteDifferences(g);
    }
    testStatuses(cases[fixedCount - 4]);   // n150K5, fixed rule

    std::vector<GoldenCase> byK;
    for (int K : {2, 3, 4, 5, 6, 7, 8, 9, 12}) {
        SyntheticOptions o;
        o.tipCount = 8;
        o.K = K;
        o.epochCount = 2;
        byK.push_back(syntheticCase("api" + std::to_string(K), 9700 + K, o));
    }
    testApi(byK);
    testConcurrentInstances(byK[3], byK[6]);

    for (const GoldenCase& g : cases) {
        bool chosen = g.name == "n40K3v51" || g.name == "n20K8v46" || g.name == "n150K5" || g.name == "n150K16" ||
                      g.name == "adaptive-n40K3v51" || g.name == "adaptive-n150K5" || g.name == "n10K5v33" || g.name == "n20K9v47";
        if (!chosen) continue;
        for (auto make : {makeKernelAStrict, makeKernelBStrict, makeKernelAFma, makeKernelBFma}) testIncremental(g, make);
    }

    std::printf("mascot_selftest: %zu problems, %d checks, %d failures\n", cases.size(), checks, failures);
    return failures == 0 ? 0 : 1;
}
