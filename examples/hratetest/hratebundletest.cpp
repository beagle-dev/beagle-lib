#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
    if (a.size() != b.size())
        return INFINITY;
    double d = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double e = std::fabs(a[i] - b[i]);
        if (!(e <= d))
            d = std::isnan(e) ? INFINITY : e;
    }
    return d;
}

static bool sameBytes(const std::vector<double>& a, const std::vector<double>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0;
}

static int createInstance(int S, int C, bool dp, int matrices, int models = 1) {
    long prec = dp ? BEAGLE_FLAG_PRECISION_DOUBLE : BEAGLE_FLAG_PRECISION_SINGLE;
    BeagleInstanceDetails det;
    int instance = beagleCreateInstance(1, 2, 0, S, S, models, matrices, C, 0, nullptr, 0, 0,
                                        BEAGLE_FLAG_HIERARCHICAL_REPRESENTATION | BEAGLE_FLAG_PROCESSOR_CPU | prec,
                                        &det);
    if (instance >= 0 && std::string(det.implName).find("Hierarchical") == std::string::npos) {
        beagleFinalizeInstance(instance);
        return -1;
    }
    if (instance >= 0) {
        std::vector<double> unit((size_t) S * S, 0.0);
        for (int k = 0; k < S; ++k)
            unit[(size_t) k * S + k] = 1.0;
        beagleSetTipPartials(instance, 0, unit.data());
    }
    return instance;
}

static int setFrequencies(int instance, const StateTree& tree) {
    std::vector<double> pi(tree.pi.begin(), tree.pi.begin() + tree.S);
    return beagleSetStateFrequencies(instance, 0, pi.data());
}

static std::vector<double> getP(int instance, int slot, int S, int C, int* code = nullptr) {
    std::vector<double> P((size_t) C * S * S, NAN);
    BeagleOperation op = { 1, BEAGLE_OP_NONE, BEAGLE_OP_NONE, 0, slot, BEAGLE_OP_NONE, BEAGLE_OP_NONE };
    int rc = beagleUpdatePartials(instance, &op, 1, BEAGLE_OP_NONE);
    if (code)
        *code = rc;
    if (rc < 0)
        return P;
    std::vector<double> parent((size_t) C * S * S);
    rc = beagleGetPartials(instance, 1, BEAGLE_OP_NONE, parent.data());
    if (rc < 0) {
        if (code)
            *code = rc;
        return P;
    }
    for (int l = 0; l < C; ++l)
        for (int x = 0; x < S; ++x)
            for (int y = 0; y < S; ++y)
                P[((size_t) l * S + x) * S + y] = parent[((size_t) l * S + y) * S + x];
    return P;
}

static std::vector<double> getDense(int instance, int slot, int S, int C, int* code = nullptr) {
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
    report(tag + "updatePartials before frequencies are set", noFrequencies == BEAGLE_ERROR_OUT_OF_RANGE,
           "code " + std::to_string(noFrequencies));
    int denseNoFrequencies;
    getDense(instance, 0, S, C, &denseNoFrequencies);
    report(tag + "getTransitionMatrix before frequencies are set", denseNoFrequencies == BEAGLE_ERROR_OUT_OF_RANGE,
           "code " + std::to_string(denseNoFrequencies));
    setFrequencies(instance, tree);

    std::vector<double> P = getP(instance, 0, S, C, &rc);
    const double d = 0.691986, dCT = 0.665873, ag = 0.143174, ct = 0.169287, tv = 0.082420;
    std::vector<double> want = { d,   tv,  ag,  tv,
                                 tv,  dCT, tv,  ct,
                                 ag,  tv,  d,   tv,
                                 tv,  ct,  tv,  dCT };
    report(tag + "worked example: P through a single-child update", rc == BEAGLE_SUCCESS, "code " + std::to_string(rc));
    reportDelta(tag + "worked example: P vs summary numbers", maxAbsDiff(P, want), 1e-6);

    std::vector<double> ref;
    referenceP(tree, rate, C, 0.1, ref);
    reportDelta(tag + "worked example: P vs referenceP", maxAbsDiff(P, ref), dp ? 1e-14 : 1e-6);
    int denseCode;
    std::vector<double> dense = getDense(instance, 0, S, C, &denseCode);
    reportDelta(tag + "worked example: getTransitionMatrix vs referenceP",
                denseCode == BEAGLE_SUCCESS ? maxAbsDiff(dense, ref) : INFINITY, dp ? 1e-14 : 1e-6);

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
    reportDelta(tag + "getTransitionMatrix vs referenceP", maxAbsDiff(getDense(instance, 0, S, C), ref), tolP);
    reportDelta(tag + "P independent of encoding (order, left/right)", maxAbsDiff(Pt, Pt2), dp ? 1e-14 : 1e-6);

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

static int setModel(int instance, int index, const ModelArrays& m) {
    return beagleSetEigenDecomposition(instance, index, m.vectors.data(), m.inverse.data(), m.values.data());
}

static void modelIdentity(bool dp, int S, int C, int variant, std::mt19937_64& rng) {
    const char* variants[3] = { "DFS", "DFS swapped", "random swapped" };
    std::string tag = std::string(dp ? "double " : "single ") + "model path S=" + std::to_string(S) +
                      " C=" + std::to_string(C) + " " + variants[variant] + " ";
    const double rates4[4] = { 0.03338775, 0.25191592, 0.82026848, 2.89442785 };
    const double rates1[1] = { 0.7 };
    const double* rates = C == 4 ? rates4 : rates1;
    const double lengths[6] = { 0.0, 1e-9, 0.02, 0.37, 1.3, 12.0 };
    const int L = 6;
    StateTree tree = makeRandomTree(S, rng, 2.0);
    std::vector<int> order;
    if (variant == 2)
        order = randomPostOrder(tree, rng);
    const std::vector<int>* orderPtr = variant == 2 ? &order : nullptr;
    const bool swap = variant != 0;

    int instance = createInstance(S, C, dp, 2 * L);
    if (instance < 0) {
        report(tag + "create", false, "no hierarchical instance");
        return;
    }
    int rc = setFrequencies(instance, tree);
    rc |= beagleSetCategoryRates(instance, rates);
    rc |= setModel(instance, 0, modelArrays(tree, orderPtr, swap));
    int indices[L];
    for (int k = 0; k < L; ++k)
        indices[k] = k;
    rc |= beagleUpdateTransitionMatrices(instance, 0, indices, nullptr, nullptr, lengths, L);
    std::vector<double> bundle;
    for (int k = 0; k < L; ++k) {
        referenceBundle(tree, rates, C, lengths[k], bundle, orderPtr, swap);
        rc |= beagleSetTransitionMatrix(instance, L + k, bundle.data(), 0.0);
    }
    bool same = rc == BEAGLE_SUCCESS;
    for (int k = 0; k < L && same; ++k) {
        int c1, c2, c3, c4;
        same = sameBytes(getP(instance, k, S, C, &c1), getP(instance, L + k, S, C, &c2)) &&
               sameBytes(getDense(instance, k, S, C, &c3), getDense(instance, L + k, S, C, &c4)) &&
               c1 == 0 && c2 == 0 && c3 == 0 && c4 == 0;
    }
    report(tag + "byte-identical to the bundle path", same, "code " + std::to_string(rc));
    beagleFinalizeInstance(instance);
}

static void modelErrors(bool dp, std::mt19937_64& rng) {
    std::string tag = std::string(dp ? "double " : "single ") + "model path ";
    const int S = 4, C = 4, n = S - 1;
    const double rates[C] = { 0.03338775, 0.25191592, 0.82026848, 2.89442785 };
    const double otherRates[C] = { 0.1, 0.5, 1.0, 2.4 };
    StateTree tree = makeRandomTree(S, rng, 2.0);
    StateTree other = tree;
    for (int m = S; m < tree.nodeCount(); ++m)
        other.lambda[m] = 1.5 * tree.lambda[m] + 0.1;
    const ModelArrays good = modelArrays(tree);
    const int one[1] = { 0 };
    const double t[1] = { 0.37 };

    int instance = createInstance(S, C, dp, 8, 2);
    if (instance < 0) {
        report(tag + "create", false, "no hierarchical instance");
        return;
    }
    int rc = setModel(instance, 0, good);
    report(tag + "setEigenDecomposition accepts the encoding", rc == BEAGLE_SUCCESS, "code " + std::to_string(rc));
    beagleSetCategoryRates(instance, rates);
    rc = beagleUpdateTransitionMatrices(instance, 0, one, nullptr, nullptr, t, 1);
    report(tag + "updateTransitionMatrices before setStateFrequencies", rc == BEAGLE_ERROR_OUT_OF_RANGE,
           "code " + std::to_string(rc));
    beagleFinalizeInstance(instance);

    instance = createInstance(S, C, dp, 8, 2);
    setModel(instance, 0, good);
    setFrequencies(instance, tree);
    rc = beagleUpdateTransitionMatrices(instance, 0, one, nullptr, nullptr, t, 1);
    report(tag + "updateTransitionMatrices before setCategoryRates", rc == BEAGLE_ERROR_OUT_OF_RANGE,
           "code " + std::to_string(rc));
    beagleSetCategoryRates(instance, rates);
    int unsetModel = beagleUpdateTransitionMatrices(instance, 1, one, nullptr, nullptr, t, 1);
    int modelRange = beagleUpdateTransitionMatrices(instance, 2, one, nullptr, nullptr, t, 1);
    int setRange = setModel(instance, 2, good);
    int setNegative = setModel(instance, -1, good);
    report(tag + "unset model and out-of-range model index",
           unsetModel == BEAGLE_ERROR_OUT_OF_RANGE && modelRange == BEAGLE_ERROR_OUT_OF_RANGE &&
           setRange == BEAGLE_ERROR_OUT_OF_RANGE && setNegative == BEAGLE_ERROR_OUT_OF_RANGE,
           "codes " + std::to_string(unsetModel) + " " + std::to_string(modelRange) + " " +
           std::to_string(setRange) + " " + std::to_string(setNegative));

    rc = beagleUpdateTransitionMatrices(instance, 0, one, nullptr, nullptr, t, 1);
    const std::vector<double> P = getDense(instance, 0, S, C);
    std::vector<double> ref;
    referenceP(tree, rates, C, t[0], ref);
    reportDelta(tag + "P vs referenceP", rc == BEAGLE_SUCCESS ? maxAbsDiff(P, ref) : INFINITY, dp ? 1e-14 : 1e-6);

    auto slotUnchanged = [&](int slot, const std::vector<double>& want) {
        return sameBytes(getDense(instance, slot, S, C), want);
    };
    auto rebuildSame = [&]() {
        int code = beagleUpdateTransitionMatrices(instance, 0, one, nullptr, nullptr, t, 1);
        return code == BEAGLE_SUCCESS && slotUnchanged(0, P);
    };

    std::vector<std::pair<std::string, std::function<void(ModelArrays&)>>> bad = {
        { "unknown version", [&](ModelArrays& m) { m.values[0] = 2 * 65536 + n; } },
        { "internal count != S - 1", [&](ModelArrays& m) { m.values[0] = 65536 + n + 1; } },
        { "genuine eigenvalues", [&](ModelArrays& m) { m.values[0] = 0.0; m.values[1] = -1.3; } },
        { "node out of range", [&](ModelArrays& m) { m.vectors[1] = 2 * S - 1; } },
        { "node twice as a child", [&](ModelArrays& m) { m.vectors[4] = m.vectors[1]; } },
        { "parent that is a tip", [&](ModelArrays& m) { m.vectors[0] = 0; } },
        { "non-integer node", [&](ModelArrays& m) { m.vectors[2] += 0.25; } },
        { "post-order parent before child", [&](ModelArrays& m) {
              std::rotate(m.vectors.begin(), m.vectors.begin() + 3 * (n - 1), m.vectors.begin() + 3 * n); } },
        { "negative lambda", [&](ModelArrays& m) { m.values[1] = -1e-3; } },
        { "NaN lambda", [&](ModelArrays& m) { m.values[2] = NAN; } },
        { "infinite lambda", [&](ModelArrays& m) { m.values[3] = INFINITY; } },
    };
    for (auto& c : bad) {
        ModelArrays m = good;
        c.second(m);
        int code = setModel(instance, 0, m);
        bool kept = rebuildSame();
        report(tag + "rejects " + c.first, code == BEAGLE_ERROR_OUT_OF_RANGE && kept,
               "code " + std::to_string(code) + (kept ? ", model unchanged" : ", model changed"));
    }

    const int derivative[1] = { 1 };
    const double other37[1] = { 0.9 };
    int first = beagleUpdateTransitionMatrices(instance, 0, one, derivative, nullptr, other37, 1);
    int second = beagleUpdateTransitionMatrices(instance, 0, one, nullptr, derivative, other37, 1);
    report(tag + "rejects derivative indices",
           first == BEAGLE_ERROR_NO_IMPLEMENTATION && second == BEAGLE_ERROR_NO_IMPLEMENTATION && slotUnchanged(0, P),
           "codes " + std::to_string(first) + " " + std::to_string(second));

    for (double badLength : { -0.1, (double) NAN, (double) INFINITY }) {
        const double length[1] = { badLength };
        int code = beagleUpdateTransitionMatrices(instance, 0, one, nullptr, nullptr, length, 1);
        report(tag + "rejects edge length " + fmt("%g", badLength),
               code == BEAGLE_ERROR_OUT_OF_RANGE && slotUnchanged(0, P), "code " + std::to_string(code));
    }

    int matrixRange[1] = { 8 };
    int code = beagleUpdateTransitionMatrices(instance, 0, matrixRange, nullptr, nullptr, t, 1);
    report(tag + "rejects matrix index out of range", code == BEAGLE_ERROR_OUT_OF_RANGE, "code " + std::to_string(code));

    rc = beagleUpdateTransitionMatrices(instance, 0, one, nullptr, nullptr, t, 1);
    const int batch[3] = { 1, 2, 3 };
    const double batchLengths[3] = { 0.2, 0.6, -1.0 };
    const double priorLength[1] = { 0.05 };
    const int three[1] = { 3 };
    rc |= beagleUpdateTransitionMatrices(instance, 0, three, nullptr, nullptr, priorLength, 1);
    const std::vector<double> before3 = getDense(instance, 3, S, C);
    code = beagleUpdateTransitionMatrices(instance, 0, batch, nullptr, nullptr, batchLengths, 3);
    std::vector<double> r1, r2;
    referenceP(tree, rates, C, 0.2, r1);
    referenceP(tree, rates, C, 0.6, r2);
    double batchDelta = std::max(maxAbsDiff(getDense(instance, 1, S, C), r1), maxAbsDiff(getDense(instance, 2, S, C), r2));
    report(tag + "batch failing at its third index",
           rc == BEAGLE_SUCCESS && code == BEAGLE_ERROR_OUT_OF_RANGE && batchDelta < (dp ? 1e-14 : 1e-6) &&
           slotUnchanged(3, before3),
           fmt("code %.0f, first two max |delta| %.2e", code, batchDelta));

    rc = setModel(instance, 1, modelArrays(other));
    rc |= beagleSetCategoryRatesWithIndex(instance, 1, otherRates);
    const int models[2] = { 0, 1 };
    const int rateIndices[2] = { 1, 0 };
    const int slots[2] = { 4, 5 };
    const double multiLengths[2] = { 0.3, 0.8 };
    rc |= beagleUpdateTransitionMatricesWithMultipleModels(instance, models, rateIndices, slots, nullptr, nullptr,
                                                           multiLengths, 2);
    std::vector<double> m1, m2;
    referenceP(tree, otherRates, C, 0.3, m1);
    referenceP(other, rates, C, 0.8, m2);
    double multiDelta = std::max(maxAbsDiff(getDense(instance, 4, S, C), m1), maxAbsDiff(getDense(instance, 5, S, C), m2));
    report(tag + "updateTransitionMatricesWithMultipleModels",
           rc == BEAGLE_SUCCESS && multiDelta < (dp ? 1e-14 : 1e-6), fmt("code %.0f, max |delta| %.2e", rc, multiDelta));

    const int nullRates[1] = { 1 };
    const int modelZero[1] = { 0 };
    int fresh = createInstance(S, C, dp, 8, 2);
    setModel(fresh, 0, good);
    setFrequencies(fresh, tree);
    beagleSetCategoryRates(fresh, rates);
    code = beagleUpdateTransitionMatricesWithMultipleModels(fresh, modelZero, nullRates, one, nullptr, nullptr, t, 1);
    report(tag + "WithMultipleModels with a NULL category-rate buffer", code == BEAGLE_ERROR_OUT_OF_RANGE,
           "code " + std::to_string(code));
    beagleFinalizeInstance(fresh);

    std::vector<double> zeroClade(tree.pi.begin(), tree.pi.begin() + S);
    zeroClade[tree.left[S]] = 0.0;
    zeroClade[tree.right[S]] = 0.0;
    beagleSetStateFrequencies(instance, 0, zeroClade.data());
    code = beagleUpdateTransitionMatrices(instance, 0, one, nullptr, nullptr, t, 1);
    setFrequencies(instance, tree);
    report(tag + "rejects a state-tree node with zero frequency", code == BEAGLE_ERROR_OUT_OF_RANGE && slotUnchanged(0, P),
           "code " + std::to_string(code));

    int eigenIndices[1] = { 0 };
    code = beagleUpdateTransitionMatricesWithModelCategories(instance, eigenIndices, one, nullptr, nullptr, t, 1);
    report(tag + "WithModelCategories is not implemented", code == BEAGLE_ERROR_NO_IMPLEMENTATION && slotUnchanged(0, P),
           "code " + std::to_string(code));

    beagleFinalizeInstance(instance);
}

static void statesColumn(bool dp, int S, std::mt19937_64& rng) {
    std::string tag = std::string(dp ? "double " : "single ") + "S=" + std::to_string(S) + " ";
    const int C = 4, K = S + 1;
    const double rates[C] = { 0.03338775, 0.25191592, 0.82026848, 2.89442785 };
    StateTree tree = makeRandomTree(S, rng, 2.0);
    long prec = dp ? BEAGLE_FLAG_PRECISION_DOUBLE : BEAGLE_FLAG_PRECISION_SINGLE;
    BeagleInstanceDetails det;
    int instance = beagleCreateInstance(1, 1, 1, S, K, 1, 1, C, 0, nullptr, 0, 0,
                                        BEAGLE_FLAG_HIERARCHICAL_REPRESENTATION | BEAGLE_FLAG_PROCESSOR_CPU | prec, &det);
    if (instance < 0) {
        report(tag + "states single child: create", false, "no hierarchical instance");
        return;
    }
    std::vector<int> states(K);
    for (int k = 0; k < K; ++k)
        states[k] = k;
    int rc = beagleSetTipStates(instance, 0, states.data());
    rc |= setFrequencies(instance, tree);
    std::vector<double> bundle;
    referenceBundle(tree, rates, C, 0.37, bundle);
    rc |= beagleSetTransitionMatrix(instance, 0, bundle.data(), 0.0);
    BeagleOperation op = { 1, BEAGLE_OP_NONE, BEAGLE_OP_NONE, 0, 0, BEAGLE_OP_NONE, BEAGLE_OP_NONE };
    rc |= beagleUpdatePartials(instance, &op, 1, BEAGLE_OP_NONE);
    std::vector<double> parent((size_t) C * K * S);
    rc |= beagleGetPartials(instance, 1, BEAGLE_OP_NONE, parent.data());
    std::vector<double> ref;
    referenceP(tree, rates, C, 0.37, ref);
    double worst = 0.0;
    for (int l = 0; l < C; ++l)
        for (int k = 0; k < K; ++k)
            for (int x = 0; x < S; ++x) {
                double want = 0.0;
                for (int y = 0; y < S; ++y)
                    if (k == S || y == k)
                        want += ref[((size_t) l * S + x) * S + y];
                const double got = parent[((size_t) l * K + k) * S + x];
                worst = std::max(worst, std::fabs(got - want) / std::fabs(want));
            }
    report(tag + "states single child: columns and ambiguity row sums", rc == 0 && worst < (dp ? 1e-14 : 1e-6),
           fmt("code %.0f, max relative delta %.2e", rc, worst));
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
    for (bool dp : { true, false }) {
        for (int S : { 4, 5, 8, 17, 20, 61, 128 })
            for (int C : { 1, 4 })
                for (int variant = 0; variant < 3; ++variant)
                    modelIdentity(dp, S, C, variant, rng);
        modelErrors(dp, rng);
        for (int S : { 2, 3, 4, 17 })
            statesColumn(dp, S, rng);
    }
    printf("%d of %d checks passed\n", checks - failures, checks);
    return failures == 0 ? 0 : 1;
}
