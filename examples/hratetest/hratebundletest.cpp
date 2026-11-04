#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
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
        printf("%-64s %s  %s\n", name.c_str(), pass ? "PASS" : "FAIL", detail.c_str());
}

static std::string fmt(const char* f, double a, double b = 0.0) {
    char buf[128];
    snprintf(buf, sizeof(buf), f, a, b);
    return buf;
}

static void reportDelta(const std::string& name, double delta, double tol) {
    report(name, delta < tol, fmt("max |delta| %.2e, tolerance %.0e", delta, tol));
}

static double maxAbsDiff(const std::vector<double>& a, const std::vector<double>& b) {
    double d = 0.0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i)
        d = std::max(d, std::fabs(a[i] - b[i]));
    return a.size() == b.size() ? d : INFINITY;
}

static int createInstance(int S, int C, bool dp, int matrices) {
    long prec = dp ? BEAGLE_FLAG_PRECISION_DOUBLE : BEAGLE_FLAG_PRECISION_SINGLE;
    BeagleInstanceDetails det;
    int instance = beagleCreateInstance(3, 5, 0, S, 10, 1, matrices, C, 0, nullptr, 0, 0,
                                        BEAGLE_FLAG_HIERARCHICAL_REPRESENTATION | BEAGLE_FLAG_PROCESSOR_CPU | prec,
                                        &det);
    if (instance >= 0 && std::string(det.implName).find("Hierarchical") == std::string::npos) {
        beagleFinalizeInstance(instance);
        return -1;
    }
    return instance;
}

static int setFrequencies(int instance, const StateTree& tree) {
    std::vector<double> pi(tree.pi.begin(), tree.pi.begin() + tree.S);
    return beagleSetStateFrequencies(instance, 0, pi.data());
}

static std::vector<double> getP(int instance, int slot, int S, int C, int* code = nullptr) {
    std::vector<double> P((size_t) C * S * S, NAN);
    int rc = beagleGetTransitionMatrix(instance, slot, P.data());
    if (code)
        *code = rc;
    return P;
}

static void workedExample(bool dp) {
    std::string tag = dp ? "double " : "single ";
    const int S = 4, C = 1;
    const double rate[1] = { 1.0 };
    std::vector<double> uniform(S, 0.25);
    StateTree tree = makeAGCTTree(4.0, 2.0, 3.0, uniform);

    std::vector<double> prZ0, H;
    nodeScalars(tree, 0.1, prZ0, H);
    std::vector<double> gotScalars = { prZ0[6], prZ0[4], prZ0[5], prZ0[0], prZ0[1], H[6], H[4], H[5] };
    std::vector<double> wantScalars = { 1.0, 0.670320, 0.670320, 0.548812, 0.496585, 0.329680, 0.243017, 0.347469 };
    reportDelta(tag + "worked example: Pr(Z=0) and H", maxAbsDiff(gotScalars, wantScalars), 1e-6);

    int instance = createInstance(S, C, dp, 4);
    if (instance < 0) {
        report(tag + "worked example: create", false, "no hierarchical instance");
        return;
    }
    std::vector<double> bundle;
    referenceBundle(tree, rate, C, 0.1, bundle);
    int rc = beagleSetTransitionMatrix(instance, 0, bundle.data(), 0.0);
    report(tag + "worked example: upload", rc == BEAGLE_SUCCESS, "code " + std::to_string(rc));
    int noFrequencies;
    getP(instance, 0, S, C, &noFrequencies);
    report(tag + "getTransitionMatrix before frequencies are set", noFrequencies == BEAGLE_ERROR_OUT_OF_RANGE,
           "code " + std::to_string(noFrequencies));
    setFrequencies(instance, tree);

    std::vector<double> P = getP(instance, 0, S, C, &rc);
    const double d = 0.691986, dCT = 0.665873, ag = 0.143174, ct = 0.169287, tv = 0.082420;
    std::vector<double> want = { d,   tv,  ag,  tv,
                                 tv,  dCT, tv,  ct,
                                 ag,  tv,  d,   tv,
                                 tv,  ct,  tv,  dCT };
    report(tag + "worked example: getTransitionMatrix code", rc == BEAGLE_SUCCESS, "code " + std::to_string(rc));
    reportDelta(tag + "worked example: P vs summary numbers", maxAbsDiff(P, want), 1e-6);

    std::vector<double> ref;
    referenceP(tree, rate, C, 0.1, ref);
    reportDelta(tag + "worked example: P vs referenceP", maxAbsDiff(P, ref), dp ? 1e-14 : 1e-6);

    const int n = S - 1;
    const int pre = 1 + 3 * n;
    const int blocks = 1 + 6 * n;
    std::vector<std::pair<std::string, std::function<void(std::vector<double>&)>>> bad = {
        { "post-order parent before child", [&](std::vector<double>& b) {
              std::rotate(b.begin() + 1, b.begin() + 1 + 3 * (n - 1), b.begin() + 1 + 3 * n); } },
        { "pre-order child before parent", [&](std::vector<double>& b) {
              std::rotate(b.begin() + pre, b.begin() + pre + 3, b.begin() + pre + 3 * n); } },
        { "lists not the same set of triples", [&](std::vector<double>& b) {
              std::swap(b[pre + 4], b[pre + 5]); } },
        { "unknown version", [&](std::vector<double>& b) { b[0] = 2 * 65536 + n; } },
        { "internal count != S - 1", [&](std::vector<double>& b) { b[0] = 65536 + n + 1; } },
        { "non-integer header", [&](std::vector<double>& b) { b[0] += 0.5; } },
        { "node out of range", [&](std::vector<double>& b) { b[2] = 2 * S - 1; } },
        { "parent that is a tip", [&](std::vector<double>& b) { b[1] = 0; } },
        { "non-integer node", [&](std::vector<double>& b) { b[2] += 0.25; } },
        { "node twice as a child", [&](std::vector<double>& b) { b[5] = 0; b[pre + 7] = 0; } },
        { "negative H", [&](std::vector<double>& b) { b[blocks + 0] = -1e-3; } },
        { "Pr(Z=0) above one", [&](std::vector<double>& b) { b[blocks + n + 1] = 1.5; } },
        { "NaN H", [&](std::vector<double>& b) { b[blocks + 1] = NAN; } },
    };
    for (auto& c : bad) {
        std::vector<double> b = bundle;
        c.second(b);
        int code = beagleSetTransitionMatrix(instance, 0, b.data(), 0.0);
        std::vector<double> after = getP(instance, 0, S, C);
        report(tag + "rejects " + c.first, code == BEAGLE_ERROR_OUT_OF_RANGE && maxAbsDiff(after, P) == 0.0,
               "code " + std::to_string(code) + (maxAbsDiff(after, P) == 0.0 ? ", slot unchanged" : ", slot changed"));
    }

    int unsetCode, rangeCode, setRangeCode;
    getP(instance, 1, S, C, &unsetCode);
    getP(instance, 4, S, C, &rangeCode);
    setRangeCode = beagleSetTransitionMatrix(instance, 4, bundle.data(), 0.0);
    report(tag + "unset slot and out-of-range index",
           unsetCode == BEAGLE_ERROR_OUT_OF_RANGE && rangeCode == BEAGLE_ERROR_OUT_OF_RANGE &&
           setRangeCode == BEAGLE_ERROR_OUT_OF_RANGE,
           "codes " + std::to_string(unsetCode) + " " + std::to_string(rangeCode) + " " + std::to_string(setRangeCode));

    std::vector<double> b1, b2, both;
    referenceBundle(tree, rate, C, 0.3, b1);
    referenceBundle(tree, rate, C, 0.7, b2);
    both = b1;
    both.insert(both.end(), b2.begin(), b2.end());
    int indices[2] = { 2, 3 };
    double padded[2] = { 1.0, 1.0 };
    rc = beagleSetTransitionMatrices(instance, indices, both.data(), padded, 2);
    std::vector<double> r1, r2;
    referenceP(tree, rate, C, 0.3, r1);
    referenceP(tree, rate, C, 0.7, r2);
    double batchDelta = std::max(maxAbsDiff(getP(instance, 2, S, C), r1), maxAbsDiff(getP(instance, 3, S, C), r2));
    report(tag + "setTransitionMatrices, two bundles back to back",
           rc == BEAGLE_SUCCESS && batchDelta < (dp ? 1e-14 : 1e-6), fmt("code %.0f, max |delta| %.2e", rc, batchDelta));

    beagleFinalizeInstance(instance);
}

static void properties(bool dp, int S, std::mt19937_64& rng) {
    std::string tag = std::string(dp ? "double " : "single ") + "S=" + std::to_string(S) + " ";
    const int C = 4;
    const double rates[C] = { 0.03338775, 0.25191592, 0.82026848, 2.89442785 };
    const double tolP = dp ? 1e-12 : 1e-5;
    StateTree tree = makeRandomTree(S, rng, 2.0);

    int instance = createInstance(S, C, dp, 8);
    if (instance < 0) {
        report(tag + "create", false, "no hierarchical instance");
        return;
    }
    const double t = 0.37, s = 0.81;
    const double times[4] = { t, s, t + s, 0.0 };
    std::vector<double> bundle;
    int rc = setFrequencies(instance, tree);
    for (int k = 0; k < 4; ++k) {
        referenceBundle(tree, rates, C, times[k], bundle);
        rc |= beagleSetTransitionMatrix(instance, k, bundle.data(), 0.0);
    }
    std::vector<int> order = randomPostOrder(tree, rng);
    referenceBundle(tree, rates, C, t, bundle, &order, true);
    rc |= beagleSetTransitionMatrix(instance, 4, bundle.data(), 0.0);
    report(tag + "uploads", rc == 0, "code " + std::to_string(rc));

    std::vector<double> Pt = getP(instance, 0, S, C), Ps = getP(instance, 1, S, C), Pts = getP(instance, 2, S, C),
                        P0 = getP(instance, 3, S, C), Pt2 = getP(instance, 4, S, C);

    std::vector<double> ref;
    referenceP(tree, rates, C, t, ref);
    reportDelta(tag + "P vs referenceP", maxAbsDiff(Pt, ref), tolP);
    reportDelta(tag + "P independent of encoding (order, left/right)", maxAbsDiff(Pt, Pt2), dp ? 1e-15 : 1e-7);

    double rowSum = 0.0, balance = 0.0, ck = 0.0, identity = 0.0, negative = 0.0;
    for (int l = 0; l < C; ++l) {
        const double* A = Pt.data() + l * S * S;
        const double* B = Ps.data() + l * S * S;
        const double* AB = Pts.data() + l * S * S;
        const double* I = P0.data() + l * S * S;
        for (int i = 0; i < S; ++i) {
            double sum = 0.0;
            for (int j = 0; j < S; ++j) {
                sum += A[i * S + j];
                negative = std::max(negative, -A[i * S + j]);
                balance = std::max(balance, std::fabs(tree.pi[i] * A[i * S + j] - tree.pi[j] * A[j * S + i]));
                double prod = 0.0;
                for (int k = 0; k < S; ++k)
                    prod += A[i * S + k] * B[k * S + j];
                ck = std::max(ck, std::fabs(prod - AB[i * S + j]));
                identity = std::max(identity, std::fabs(I[i * S + j] - (i == j ? 1.0 : 0.0)));
            }
            rowSum = std::max(rowSum, std::fabs(sum - 1.0));
        }
    }
    reportDelta(tag + "rows sum to 1", rowSum, dp ? 1e-12 : 1e-5);
    reportDelta(tag + "detailed balance", balance, tolP);
    reportDelta(tag + "P(t) P(s) = P(t+s)", ck, dp ? 1e-12 : 1e-5);
    reportDelta(tag + "P(0) = I", identity, dp ? 1e-15 : 1e-7);
    reportDelta(tag + "no negative entries", negative, 1e-300);

    beagleFinalizeInstance(instance);
}

static void tn93(bool dp, std::mt19937_64& rng) {
    std::string tag = dp ? "double " : "single ";
    const int S = 4, C = 4;
    const double rates[C] = { 0.03338775, 0.25191592, 0.82026848, 2.89442785 };
    std::uniform_real_distribution<double> u(0.1, 1.0);
    std::vector<double> pi = { u(rng), u(rng), u(rng), u(rng) };
    double total = pi[0] + pi[1] + pi[2] + pi[3];
    for (double& p : pi)
        p /= total;
    double lambdaRoot = 3 * u(rng), lambdaAG = 3 * u(rng), lambdaCT = 3 * u(rng);
    StateTree tree = makeAGCTTree(lambdaRoot, lambdaAG, lambdaCT, pi);

    const double edge = 0.43;
    std::vector<double> closed;
    tn93P(pi, lambdaRoot, lambdaAG, lambdaCT, rates, C, edge, closed);

    int instance = createInstance(S, C, dp, 4);
    if (instance < 0) {
        report(tag + "TN93 create", false, "no hierarchical instance");
        return;
    }
    std::vector<double> bundle;
    referenceBundle(tree, rates, C, edge, bundle);
    setFrequencies(instance, tree);
    int rc = beagleSetTransitionMatrix(instance, 0, bundle.data(), 0.0);
    std::vector<double> P = getP(instance, 0, S, C);
    report(tag + "TN93 upload", rc == BEAGLE_SUCCESS, "code " + std::to_string(rc));
    reportDelta(tag + "TN93 closed form", maxAbsDiff(P, closed), dp ? 1e-14 : 1e-6);
    beagleFinalizeInstance(instance);
}

int main() {
    std::mt19937_64 rng(20261013);
    for (bool dp : { true, false }) {
        workedExample(dp);
        for (int S : { 4, 17, 61 })
            properties(dp, S, rng);
        tn93(dp, rng);
    }
    printf("%d of %d checks passed\n", checks - failures, checks);
    return failures == 0 ? 0 : 1;
}
