/*
 * MascotSynthetic.hpp -- random MASCOT problems for the self-contained tests, in the GoldenCase layout (without
 * reference outputs). Mirrors BEAST's MascotGoldenCorpus generator: a heterochronous tree simulated as a schedule,
 * one-hot or ambiguous tips, 1-3 epochs with one boundary on a coalescence and one between events, optional zero
 * migration rates, branch rates and a log Ne shift (stiff cases).
 */
#ifndef MASCOT_SYNTHETIC_HPP
#define MASCOT_SYNTHETIC_HPP

#include "MascotGolden.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <vector>

namespace mascot {

struct SyntheticOptions {
    int tipCount = 10;
    int K = 3;
    int epochCount = 1;
    double maxStep = 0.02;
    bool branchRates = true;
    bool simultaneousSamples = false;
    bool zeroRates = false;
    bool ambiguousTips = false;
    double logNeShift = 0.0;
    int stepRuleKind = STEPS_FIXED;   /* ADAPTIVE uses maxStep as its cap (<= 0: none)                          */
    double tau = 0.25;
    double minStep = 0.0;
    int maxStepsPerSegment = 0;
};

/* Fills migration (diagonal = -row sum, summed in sink order) and inversePopulation = exp(-log Ne) from theta. */
inline void ratesFromTheta(GoldenCase& g) {
    const int K = g.K, per = K * (K - 1) + K;
    g.migration.assign((size_t) g.epochCount * K * K, 0.0);
    g.inversePopulation.assign((size_t) g.epochCount * K, 0.0);
    for (int e = 0; e < g.epochCount; e++) {
        const double* th = g.theta.data() + (size_t) e * per;
        double* M = g.migration.data() + (size_t) e * K * K;
        int index = 0;
        for (int source = 0; source < K; source++) {
            double rowSum = 0.0;
            for (int sink = 0; sink < K; sink++) {
                if (sink == source) continue;
                M[source * K + sink] = th[index];
                rowSum += th[index];
                index++;
            }
            M[source * K + source] = -rowSum;
        }
        for (int s = 0; s < K; s++) g.inversePopulation[(size_t) e * K + s] = std::exp(-th[K * (K - 1) + s]);
    }
}

inline GoldenCase syntheticCase(const std::string& name, uint64_t seed, const SyntheticOptions& o) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    GoldenCase g;
    g.name = name;
    g.status = "ok";
    g.K = o.K;
    g.epochCount = o.epochCount;
    g.maxStep = o.maxStep;
    g.stepRuleKind = o.stepRuleKind;
    if (o.stepRuleKind == STEPS_ADAPTIVE) {
        g.tau = o.tau;
        g.minStep = o.minStep;
        g.maxStepsPerSegment = o.maxStepsPerSegment;
    }
    const int n = o.tipCount, nodes = 2 * n - 1;
    g.lineageCount = nodes;
    g.intervalCount = nodes - 1;

    std::vector<double> sampleTimes(n, 0.0);
    for (int i = 1; i < n; i++) {
        double t = 4.0 * u(rng);
        sampleTimes[i] = o.simultaneousSamples ? std::floor(t) : t;
    }
    std::sort(sampleTimes.begin(), sampleTimes.end());
    std::vector<int> tipOrder(n);
    for (int i = 0; i < n; i++) tipOrder[i] = i;
    std::shuffle(tipOrder.begin(), tipOrder.end(), rng);

    std::vector<int> active = {tipOrder[0]};
    std::vector<double> eventTimes;
    double time = sampleTimes[0];
    g.initialTime = time;
    g.initialSampleNode = tipOrder[0];
    int nextSample = 1, nextParent = n;
    for (int i = 0; i < g.intervalCount; i++) {
        int a = (int) active.size();
        double rate = a >= 2 ? 0.25 * a * (a - 1) : 0.0;
        double wait = rate > 0.0 ? -std::log(1.0 - u(rng)) / rate : INFINITY;
        int sample = -1, c1 = -1, c2 = -1, parent = -1, type;
        double length;
        if (nextSample < n && time + wait >= sampleTimes[nextSample]) {
            length = sampleTimes[nextSample] - time;
            type = SAMPLE;
            sample = tipOrder[nextSample];
            active.push_back(sample);
            time = sampleTimes[nextSample];
            nextSample++;
        } else {
            length = wait;
            type = COALESCENT;
            size_t i1 = (size_t) (u(rng) * active.size()) % active.size();
            c1 = active[i1];
            active.erase(active.begin() + i1);
            size_t i2 = (size_t) (u(rng) * active.size()) % active.size();
            c2 = active[i2];
            active.erase(active.begin() + i2);
            parent = nextParent++;
            active.push_back(parent);
            time += wait;
        }
        g.intervalLengths.push_back(length);
        g.eventTypes.push_back(type);
        g.sampleNodes.push_back(sample);
        g.child1.push_back(c1);
        g.child2.push_back(c2);
        g.parent.push_back(parent);
        eventTimes.push_back(time);
    }

    g.tipPartials.assign((size_t) nodes * o.K, 0.0);
    for (int tip = 0; tip < n; tip++) {
        double* p = g.tipPartials.data() + (size_t) tip * o.K;
        p[(size_t) (u(rng) * o.K) % o.K] = 1.0;
        if (o.ambiguousTips && u(rng) < 0.2) p[(size_t) (u(rng) * o.K) % o.K] = 1.0;
        double sum = 0.0;
        for (int s = 0; s < o.K; s++) sum += p[s];
        for (int s = 0; s < o.K; s++) p[s] /= sum;
    }

    g.boundaries.assign(o.epochCount + 1, 0.0);
    g.boundaries[o.epochCount] = INFINITY;
    if (o.epochCount >= 2) {
        std::vector<double> coalescentTimes;
        for (int i = 0; i < g.intervalCount; i++)
            if (g.eventTypes[i] == COALESCENT && eventTimes[i] > 0.0) coalescentTimes.push_back(eventTimes[i]);
        double onEvent = coalescentTimes[coalescentTimes.size() / 2];
        g.boundaries[1] = onEvent;
        if (o.epochCount == 3) {
            double span = std::max(eventTimes.back() - onEvent, 1.0);
            g.boundaries[2] = onEvent + span * (0.25 + 0.5 * u(rng));
        }
    }

    const int K = o.K, mig = K * (K - 1), per = mig + K;
    g.theta.assign((size_t) o.epochCount * per, 0.0);
    for (int e = 0; e < o.epochCount; e++) {
        for (int i = 0; i < mig; i++) {
            bool off = o.zeroRates && u(rng) < 0.3;
            g.theta[(size_t) e * per + i] = off ? 0.0 : (0.5 + u(rng)) * 4.0 / (K - 1);
        }
        for (int s = 0; s < K; s++) g.theta[(size_t) e * per + mig + s] = 0.672 + (u(rng) - 0.5) + o.logNeShift;
    }
    ratesFromTheta(g);

    if (o.branchRates) {
        g.branchRates.resize(nodes);
        for (double& r : g.branchRates) r = 0.5 + u(rng);
    }
    return g;
}

/* Two lineages sampled in different demes with no migration: lambda = 0 at their coalescence. */
inline GoldenCase coalescentRateErrorCase() {
    GoldenCase g;
    g.name = "errorCoalescentRate";
    g.status = "numerical";
    g.K = 2;
    g.lineageCount = 3;
    g.intervalCount = 2;
    g.initialSampleNode = 0;
    g.epochCount = 1;
    g.maxStep = 0.02;
    g.initialTime = 0.0;
    g.boundaries = {0.0, INFINITY};
    g.theta = {0.0, 0.0, 0.0, 0.0};
    ratesFromTheta(g);
    g.intervalLengths = {0.5, 1.0};
    g.eventTypes = {SAMPLE, COALESCENT};
    g.sampleNodes = {1, -1};
    g.child1 = {-1, 0};
    g.child2 = {-1, 1};
    g.parent = {-1, 2};
    g.tipPartials = {1.0, 0.0, 0.0, 1.0, 0.0, 0.0};
    return g;
}

} // namespace mascot

#endif
