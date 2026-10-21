#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "libhmsbeagle/beagle.h"
#include "hratereference.h"

static const char* kHuman =
    "AGAAATATGTCTGATAAAAGAGTTACTTTGATAGAGTAAATAATAGGAGCTTAAACCCCCTTATTTCTACTA"
    "GGACTATGAGAATCGAACCCATCCCTGAGAATCCAAAATTCTCCGTGCCACCTATCACACCCCATCCTAAGT"
    "AAGGTCAGCTAAATAAGCTATCGGGCCCATACCCCGAAAATGTTGGTTATACCCTTCCCGTACTAAGAAATT"
    "TAGGTTAAATACAGACCAAGAGCCTTCAAAGCCCTCAGTAAGTTG-CAATACTTAATTTCTGTAAGGACTGC"
    "AAAACCCCACTCTGCATCAACTGAACGCAAATCAGCCACTTTAATTAAGCTAAGCCCTTCTAGACCAATGGG"
    "ACTTAAACCCACAAACACTTAGTTAACAGCTAAGCACCCTAATCAAC-TGGCTTCAATCTAAAGCCCCGGCA"
    "GG-TTTGAAGCTGCTTCTTCGAATTTGCAATTCAATATGAAAA-TCACCTCGGAGCTTGGTAAAAAGAGGC"
    "CTAACCCCTGTCTTTAGATTTACAGTCCAATGCTTCA-CTCAGCCATTTTACCACAAAAAAGGAAGGAATCG"
    "AACCCCCCAAAGCTGGTTTCAAGCCAACCCCATGGCCTCCATGACTTTTTCAAAAGGTATTAGAAAAACCAT"
    "TTCATAACTTTGTCAAAGTTAAATTATAGGCT-AAATCCTATATATCTTA-CACTGTAAAGCTAACTTAGCA"
    "TTAACCTTTTAAGTTAAAGATTAAGAGAACCAACACCTCTTTACAGTGA";

static const char* kChimp =
    "AGAAATATGTCTGATAAAAGAATTACTTTGATAGAGTAAATAATAGGAGTTCAAATCCCCTTATTTCTACTA"
    "GGACTATAAGAATCGAACTCATCCCTGAGAATCCAAAATTCTCCGTGCCACCTATCACACCCCATCCTAAGT"
    "AAGGTCAGCTAAATAAGCTATCGGGCCCATACCCCGAAAATGTTGGTTACACCCTTCCCGTACTAAGAAATT"
    "TAGGTTAAGCACAGACCAAGAGCCTTCAAAGCCCTCAGCAAGTTA-CAATACTTAATTTCTGTAAGGACTGC"
    "AAAACCCCACTCTGCATCAACTGAACGCAAATCAGCCACTTTAATTAAGCTAAGCCCTTCTAGATTAATGGG"
    "ACTTAAACCCACAAACATTTAGTTAACAGCTAAACACCCTAATCAAC-TGGCTTCAATCTAAAGCCCCGGCA"
    "GG-TTTGAAGCTGCTTCTTCGAATTTGCAATTCAATATGAAAA-TCACCTCAGAGCTTGGTAAAAAGAGGC"
    "TTAACCCCTGTCTTTAGATTTACAGTCCAATGCTTCA-CTCAGCCATTTTACCACAAAAAAGGAAGGAATCG"
    "AACCCCCTAAAGCTGGTTTCAAGCCAACCCCATGACCTCCATGACTTTTTCAAAAGATATTAGAAAAACTAT"
    "TTCATAACTTTGTCAAAGTTAAATTACAGGTT-AACCCCCGTATATCTTA-CACTGTAAAGCTAACCTAGCA"
    "TTAACCTTTTAAGTTAAAGATTAAGAGGACCGACACCTCTTTACAGTGA";

static const char* kGorilla =
    "AGAAATATGTCTGATAAAAGAGTTACTTTGATAGAGTAAATAATAGAGGTTTAAACCCCCTTATTTCTACTA"
    "GGACTATGAGAATTGAACCCATCCCTGAGAATCCAAAATTCTCCGTGCCACCTGTCACACCCCATCCTAAGT"
    "AAGGTCAGCTAAATAAGCTATCGGGCCCATACCCCGAAAATGTTGGTCACATCCTTCCCGTACTAAGAAATT"
    "TAGGTTAAACATAGACCAAGAGCCTTCAAAGCCCTTAGTAAGTTA-CAACACTTAATTTCTGTAAGGACTGC"
    "AAAACCCTACTCTGCATCAACTGAACGCAAATCAGCCACTTTAATTAAGCTAAGCCCTTCTAGATCAATGGG"
    "ACTCAAACCCACAAACATTTAGTTAACAGCTAAACACCCTAGTCAAC-TGGCTTCAATCTAAAGCCCCGGCA"
    "GG-TTTGAAGCTGCTTCTTCGAATTTGCAATTCAATATGAAAT-TCACCTCGGAGCTTGGTAAAAAGAGGC"
    "CCAGCCTCTGTCTTTAGATTTACAGTCCAATGCCTTA-CTCAGCCATTTTACCACAAAAAAGGAAGGAATCG"
    "AACCCCCCAAAGCTGGTTTCAAGCCAACCCCATGACCTTCATGACTTTTTCAAAAGATATTAGAAAAACTAT"
    "TTCATAACTTTGTCAAGGTTAAATTACGGGTT-AAACCCCGTATATCTTA-CACTGTAAAGCTAACCTAGCG"
    "TTAACCTTTTAAGTTAAAGATTAAGAGTATCGGCACCTCTTTGCAGTGA";

static const double kPaupLogL = -1498.89812;
static const int kCategoryCount = 4;
static const double kRates[kCategoryCount] = { 0.03338775, 0.25191592, 0.82026848, 2.89442785 };
static const double kWeights[kCategoryCount] = { 0.25, 0.25, 0.25, 0.25 };
static const int kBranchCount = 4;
static const double kEdgeLengths[kBranchCount] = { 0.1, 0.1, 0.2, 0.1 };

static int stateCode(char c) {
    switch (c) {
        case 'A': return 0;
        case 'C': return 1;
        case 'G': return 2;
        case 'T': return 3;
    }
    return -1;
}

static std::vector<int> makeStates(const char* seq, int nStates) {
    int n = (int) strlen(seq);
    std::vector<int> s(n);
    for (int i = 0; i < n; ++i) {
        int c = stateCode(seq[i]);
        s[i] = (c >= 0) ? c : nStates;
    }
    return s;
}

static std::vector<double> makePartials(const char* seq, int nStates) {
    int n = (int) strlen(seq);
    std::vector<double> p(n * nStates, 0.0);
    for (int i = 0; i < n; ++i) {
        int c = stateCode(seq[i]);
        if (c >= 0) {
            p[i * nStates + c] = 1.0;
        } else {
            for (int k = 0; k < std::min(nStates, 4); ++k)
                p[i * nStates + k] = 1.0;
        }
    }
    return p;
}

static void printFlags(long f) {
    if (f & BEAGLE_FLAG_PROCESSOR_CPU)                printf(" CPU");
    if (f & BEAGLE_FLAG_PROCESSOR_GPU)                printf(" GPU");
    if (f & BEAGLE_FLAG_PRECISION_DOUBLE)             printf(" DOUBLE");
    if (f & BEAGLE_FLAG_PRECISION_SINGLE)             printf(" SINGLE");
    if (f & BEAGLE_FLAG_FRAMEWORK_CUDA)               printf(" CUDA");
    if (f & BEAGLE_FLAG_FRAMEWORK_OPENCL)             printf(" OPENCL");
    if (f & BEAGLE_FLAG_FRAMEWORK_CPU)                printf(" CPU_FW");
    if (f & BEAGLE_FLAG_SPECTRAL_REPRESENTATION)      printf(" SPECTRAL");
    if (f & BEAGLE_FLAG_HIERARCHICAL_REPRESENTATION)  printf(" HIERARCHICAL");
}

enum class Scaling { None, Manual, Always, Auto };

struct Options {
    int forceResource = -1;
    bool doublePrecision = false;
    int reps = 1;
    long prefFlags = 0;
    bool hierarchical = true;
    int stateCount = 4;
    int threads = 1;
    bool useStates = false;
    bool partialsTips = false;
    Scaling scaling = Scaling::None;
    double kappa = -1.0;
};

struct Run {
    int instance = -1;
    BeagleInstanceDetails details;
    bool hierarchical = false;
    int failedCode = BEAGLE_SUCCESS;
    std::string failedCall;
    double logL = 0.0;
    std::vector<double> internalPartials;
    double ms = 0.0;
};

static int checks = 0;
static int failures = 0;

static void report(const char* name, bool pass, const char* detail) {
    ++checks;
    if (!pass)
        ++failures;
    printf("%-34s %s  (%s)\n", name, pass ? "PASS" : "FAIL", detail);
}

static void reportDelta(const char* name, double delta, double tol) {
    char detail[96];
    snprintf(detail, sizeof(detail), "max |delta| %.2e, tolerance %.0e", delta, tol);
    report(name, delta < tol, detail);
}

static void reportFailure(const char* name, const Run& run) {
    ++checks;
    ++failures;
    printf("%-34s FAIL  (%s returned %d)\n", name, run.failedCall.c_str(), run.failedCode);
}

static void skip(const char* name, const char* why) {
    printf("%-34s SKIP  (%s)\n", name, why);
}

static bool call(Run& run, const char* name, int code) {
    if (code < 0 && run.failedCode == BEAGLE_SUCCESS) {
        run.failedCode = code;
        run.failedCall = name;
    }
    return code >= 0;
}

static int createInstance(const Options& opts, bool hierarchical, bool oracle, long extraPrefs,
                          BeagleInstanceDetails* det) {
    int resourceBuf = opts.forceResource;
    int* resources = (!oracle && opts.forceResource >= 0) ? &resourceBuf : nullptr;
    int nResources = (!oracle && opts.forceResource >= 0) ? 1 : 0;
    long prec = opts.doublePrecision ? BEAGLE_FLAG_PRECISION_DOUBLE : BEAGLE_FLAG_PRECISION_SINGLE;
    long requirements = BEAGLE_FLAG_PROCESSOR_CPU | prec;
    if (hierarchical)
        requirements |= BEAGLE_FLAG_HIERARCHICAL_REPRESENTATION;
    long prefs = prec | extraPrefs;
    bool states = opts.useStates && !opts.partialsTips && !oracle;
    int scaleBuffers = (oracle || opts.scaling == Scaling::None) ? 0 : 3;
    if (!oracle && opts.threads > 1)
        prefs |= BEAGLE_FLAG_THREADING_CPP;
    return beagleCreateInstance(3, states ? 2 : 5, states ? 3 : 0, opts.stateCount,
                                (int) strlen(kHuman), 1, kBranchCount, kCategoryCount, scaleBuffers,
                                resources, nResources, prefs, requirements, det);
}

static long scalingFlags(Scaling s) {
    switch (s) {
        case Scaling::Manual: return BEAGLE_FLAG_SCALING_MANUAL | BEAGLE_FLAG_SCALERS_LOG;
        case Scaling::Always: return BEAGLE_FLAG_SCALING_ALWAYS;
        case Scaling::Auto:   return BEAGLE_FLAG_SCALING_AUTO;
        default:              return 0;
    }
}

static void evaluate(const Options& opts, const StateTree& tree, Run& run, bool oracle) {
    int S = opts.stateCount;
    int nPatterns = (int) strlen(kHuman);
    int instance = run.instance;
    bool states = opts.useStates && !opts.partialsTips && !oracle;
    const char* seqs[3] = { kHuman, kChimp, kGorilla };

    for (int t = 0; t < 3; ++t) {
        if (states) {
            std::vector<int> s = makeStates(seqs[t], S);
            if (!call(run, "beagleSetTipStates", beagleSetTipStates(instance, t, s.data())))
                return;
        } else {
            std::vector<double> p = opts.useStates ? partialsFromStates(makeStates(seqs[t], S), S)
                                                   : makePartials(seqs[t], S);
            if (!call(run, "beagleSetTipPartials", beagleSetTipPartials(instance, t, p.data())))
                return;
        }
    }

    std::vector<double> freqs(tree.pi.begin(), tree.pi.begin() + S);
    std::vector<double> patternWeights(nPatterns, 1.0);
    if (!call(run, "beagleSetStateFrequencies", beagleSetStateFrequencies(instance, 0, freqs.data())) ||
        !call(run, "beagleSetCategoryRates", beagleSetCategoryRates(instance, kRates)) ||
        !call(run, "beagleSetCategoryWeights", beagleSetCategoryWeights(instance, 0, kWeights)) ||
        !call(run, "beagleSetPatternWeights", beagleSetPatternWeights(instance, patternWeights.data())))
        return;

    if (!oracle && opts.threads > 1 &&
        !call(run, "beagleSetCPUThreadCount", beagleSetCPUThreadCount(instance, opts.threads)))
        return;

    Scaling scaling = oracle ? Scaling::None : opts.scaling;
    int cumulative = (scaling == Scaling::None) ? BEAGLE_OP_NONE : 2;
    int write3 = (scaling == Scaling::Manual) ? 0 : BEAGLE_OP_NONE;
    int write4 = (scaling == Scaling::Manual) ? 1 : BEAGLE_OP_NONE;
    BeagleOperation ops[2] = {
        { 3, write3, BEAGLE_OP_NONE, 0, 0, 1, 1 },
        { 4, write4, BEAGLE_OP_NONE, 2, 2, 3, 3 }
    };
    int scaleWrites[2] = { 0, 1 };
    int matrixIndices[kBranchCount] = { 0, 1, 2, 3 };
    int rootBuffer = 4, weightIndex = 0, freqIndex = 0;
    std::vector<double> payload;

    int reps = oracle ? 1 : std::max(1, opts.reps);
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r) {
        for (int b = 0; b < kBranchCount; ++b) {
            if (run.hierarchical)
                referenceBundle(tree, kRates, kCategoryCount, kEdgeLengths[b], payload);
            else
                referenceP(tree, kRates, kCategoryCount, kEdgeLengths[b], payload);
            if (!call(run, "beagleSetTransitionMatrix",
                      beagleSetTransitionMatrix(instance, matrixIndices[b], payload.data(), 1.0)))
                return;
        }
        if (scaling != Scaling::None &&
            !call(run, "beagleResetScaleFactors", beagleResetScaleFactors(instance, cumulative)))
            return;
        if (!call(run, "beagleUpdatePartials",
                  beagleUpdatePartials(instance, ops, 2,
                                       scaling == Scaling::Always ? cumulative : BEAGLE_OP_NONE)))
            return;
        if (scaling == Scaling::Manual &&
            !call(run, "beagleAccumulateScaleFactors",
                  beagleAccumulateScaleFactors(instance, scaleWrites, 2, cumulative)))
            return;
        if (!call(run, "beagleCalculateRootLogLikelihoods",
                  beagleCalculateRootLogLikelihoods(instance, &rootBuffer, &weightIndex, &freqIndex,
                                                    &cumulative, 1, &run.logL)))
            return;
    }
    auto t1 = std::chrono::steady_clock::now();
    run.ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;

    if (scaling == Scaling::None) {
        int buffers[2] = { 3, 4 };
        int scales[2] = { BEAGLE_OP_NONE, BEAGLE_OP_NONE };
        run.internalPartials.assign(2 * kCategoryCount * nPatterns * S, 0.0);
        call(run, "beagleGetPartialsBatch",
             beagleGetPartialsBatch(instance, buffers, scales, 2, run.internalPartials.data()));
    }
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

int main(int argc, char** argv) {
    Options opts;
    bool forceCPU = false, forcePlain = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--cpu")        forceCPU = true;
        else if (a == "--plain")      forcePlain = true;
        else if (a == "--double")     opts.doublePrecision = true;
        else if (a == "--highstates") opts.stateCount = 17;
        else if (a == "--states")     opts.useStates = true;
        else if (a == "--resource" && i + 1 < argc) opts.forceResource = atoi(argv[++i]);
        else if (a == "--reps" && i + 1 < argc)     opts.reps = atoi(argv[++i]);
        else if (a == "--threads" && i + 1 < argc)  opts.threads = atoi(argv[++i]);
        else if (a == "--kappa" && i + 1 < argc)    opts.kappa = atof(argv[++i]);
        else if (a == "--scaling" && i + 1 < argc) {
            std::string s = argv[++i];
            if      (s == "none")   opts.scaling = Scaling::None;
            else if (s == "manual") opts.scaling = Scaling::Manual;
            else if (s == "always") opts.scaling = Scaling::Always;
            else if (s == "auto")   opts.scaling = Scaling::Auto;
            else { fprintf(stderr, "Unknown scaling: %s\n", s.c_str()); return 1; }
        }
        else {
            fprintf(stderr, "Usage: hratetest [--cpu | --plain] [--double] [--highstates] [--states]\n"
                            "                 [--resource N] [--reps R] [--threads T] [--kappa K]\n"
                            "                 [--scaling none|manual|always|auto]\n");
            return 1;
        }
    }

    BeagleResourceList* rList = beagleGetResourceList();
    printf("BEAGLE %s  available resources:\n", beagleGetVersion());
    for (int i = 0; i < rList->length; ++i) {
        printf("  [%d] %-30s  flags:", i, rList->list[i].name);
        printFlags(rList->list[i].supportFlags);
        printf("\n");
    }
    printf("\n");

    int S = opts.stateCount;
    std::vector<double> tipPi(S, 1.0 / S);
    const bool hky = opts.kappa >= 0.0;
    if (hky && (S != 4 || opts.kappa < 1.0)) {
        fprintf(stderr, "--kappa needs 4 states and K >= 1 (negative hierarchical rates are not supported)\n");
        return 1;
    }
    const double beta = hky ? 4.0 / (opts.kappa + 2.0) : 0.0;
    const double lambdaTransition = hky ? (opts.kappa - 1.0) * beta / 2.0 : 0.0;
    StateTree tree = hky ? makeAGCTTree(beta, lambdaTransition, lambdaTransition, tipPi) : makeStarTree(S, tipPi);

    long extraPrefs = scalingFlags(opts.scaling);
    std::vector<bool> cascade;
    if      (forceCPU)   cascade = { true };
    else if (forcePlain) cascade = { false };
    else                 cascade = { true, false };

    Run run;
    for (bool h : cascade) {
        run.instance = createInstance(opts, h, false, extraPrefs, &run.details);
        if (run.instance >= 0) {
            run.hierarchical = h;
            break;
        }
    }
    if (run.instance < 0) {
        fprintf(stderr, "Failed to create BEAGLE instance (error %d)\n", run.instance);
        return 1;
    }

    printf("Selected resource %d: %s\n", run.details.resourceNumber, run.details.resourceName);
    printf("  Implementation : %s\n", run.details.implName);
    printf("  Active flags   :");
    printFlags(run.details.flags);
    printf("\n");
    printf("  Model          : %s\n", hky ? "HKY on ((A,G),(C,T)), uniform pi" : "star state tree (JC), uniform pi");
    if (hky)
        printf("                   kappa %.4f, lambda root %.6f, lambda AG = CT %.6f\n",
               opts.kappa, beta, lambdaTransition);
    printf("  Mode           : %s, %d states, %s tips, %d thread(s), scaling %s\n\n",
           run.hierarchical ? "hierarchical" : "plain CPU (oracle)", S,
           opts.useStates ? "state" : "partials", opts.threads,
           opts.scaling == Scaling::None ? "none" : opts.scaling == Scaling::Manual ? "manual" :
           opts.scaling == Scaling::Always ? "always" : "auto");

    if (opts.scaling == Scaling::Auto) {
        skip("scaling auto", "not yet implemented for hierarchical representation");
        beagleFinalizeInstance(run.instance);
        return 0;
    }

    evaluate(opts, tree, run, false);

    BeagleInstanceDetails plainDetails;
    int plain = createInstance(opts, false, true, 0, &plainDetails);
    bool plainOK = plain >= 0 && std::string(plainDetails.implName).find("Hierarchical") == std::string::npos;
    report("selection without the flag", plainOK, plain >= 0 ? plainDetails.implName : "no instance");

    bool dp = opts.doublePrecision;

    if (run.failedCode != BEAGLE_SUCCESS) {
        reportFailure("log likelihood", run);
    } else if (S == 4 && (!hky || opts.kappa == 1.0)) {
        double tol = dp ? 1e-4 : 0.5;
        double d = std::fabs(run.logL - kPaupLogL);
        char detail[96];
        snprintf(detail, sizeof(detail), "logL %.5f, |delta| %.2e, tolerance %.0e", run.logL, d, tol);
        report("log likelihood vs PAUP", d < tol, detail);
    } else {
        printf("%-34s DONE  (logL %.5f, no PAUP reference for this model)\n", "log likelihood", run.logL);
    }

    if (run.hierarchical) {
        std::vector<double> got(kCategoryCount * S * S), want;
        double worst = 0.0;
        int code = BEAGLE_SUCCESS;
        for (int b = 0; b < kBranchCount && code >= 0; ++b) {
            code = beagleGetTransitionMatrix(run.instance, b, got.data());
            referenceP(tree, kRates, kCategoryCount, kEdgeLengths[b], want);
            worst = std::max(worst, maxAbsDiff(got, want));
        }
        if (code < 0) {
            Run r;
            r.failedCall = "beagleGetTransitionMatrix";
            r.failedCode = code;
            reportFailure("transition matrices vs reference", r);
        } else {
            reportDelta("transition matrices vs reference", worst, dp ? 1e-10 : 1e-5);
        }

        if (plain >= 0) {
            Run oracle;
            oracle.instance = plain;
            oracle.details = plainDetails;
            evaluate(opts, tree, oracle, true);
            if (oracle.failedCode != BEAGLE_SUCCESS) {
                reportFailure("oracle run", oracle);
            } else if (run.failedCode != BEAGLE_SUCCESS) {
                reportFailure("log likelihood vs oracle", run);
            } else {
                reportDelta("log likelihood vs oracle", std::fabs(run.logL - oracle.logL),
                            dp ? 1e-10 : 1e-7 * std::fabs(oracle.logL));
                if (!run.internalPartials.empty())
                    reportDelta("partials vs oracle", maxAbsDiff(run.internalPartials, oracle.internalPartials),
                                dp ? 1e-12 : 1e-6);
            }
        }
    } else {
        skip("oracle comparison", "plain CPU instance is the oracle");
    }

    if (hky) {
        std::vector<double> reference, closed;
        double worst = 0.0;
        for (int b = 0; b < kBranchCount; ++b) {
            referenceP(tree, kRates, kCategoryCount, kEdgeLengths[b], reference);
            tn93P(tipPi, beta, lambdaTransition, lambdaTransition, kRates, kCategoryCount, kEdgeLengths[b], closed);
            worst = std::max(worst, maxAbsDiff(reference, closed));
        }
        reportDelta("reference P vs TN93", worst, 1e-12);
        if (run.hierarchical) {
            std::vector<double> got(kCategoryCount * S * S);
            double worstBeagle = 0.0;
            int code = BEAGLE_SUCCESS;
            for (int b = 0; b < kBranchCount && code >= 0; ++b) {
                code = beagleGetTransitionMatrix(run.instance, b, got.data());
                tn93P(tipPi, beta, lambdaTransition, lambdaTransition, kRates, kCategoryCount, kEdgeLengths[b], closed);
                worstBeagle = std::max(worstBeagle, maxAbsDiff(got, closed));
            }
            reportDelta("transition matrices vs TN93", code < 0 ? INFINITY : worstBeagle, dp ? 1e-10 : 1e-5);
        }
    }

    const bool variant = opts.useStates || opts.threads > 1 || opts.scaling != Scaling::None;
    if (run.hierarchical && variant && run.failedCode == BEAGLE_SUCCESS) {
        Options defaults = opts;
        defaults.partialsTips = true;
        defaults.threads = 1;
        defaults.scaling = Scaling::None;
        defaults.reps = 1;
        Run base;
        base.instance = createInstance(defaults, true, false, 0, &base.details);
        base.hierarchical = true;
        if (base.instance < 0) {
            base.failedCall = "beagleCreateInstance";
            base.failedCode = base.instance;
            reportFailure("default hierarchical run", base);
        } else {
            evaluate(defaults, tree, base, false);
            if (base.failedCode != BEAGLE_SUCCESS) {
                reportFailure("default hierarchical run", base);
            } else {
                reportDelta("log likelihood vs default run", std::fabs(run.logL - base.logL),
                            dp ? 1e-10 : 1e-7 * std::fabs(base.logL));
                if (!run.internalPartials.empty() && opts.useStates)
                    reportDelta("partials vs default run (state tips)",
                                maxAbsDiff(run.internalPartials, base.internalPartials), dp ? 1e-12 : 1e-6);
                else if (!run.internalPartials.empty())
                    reportDelta("partials vs default run (identical)",
                                maxAbsDiff(run.internalPartials, base.internalPartials), 1e-300);
            }
            beagleFinalizeInstance(base.instance);
        }
    }

    if (opts.scaling == Scaling::Manual)
        skip("scale factors read back", "beagleGetScaleFactors is a no-op in the CPU plugins");

    if (plain >= 0)
        beagleFinalizeInstance(plain);

    if (opts.reps > 1)
        printf("\nTiming = %.3f ms / rep  (%d reps)\n", run.ms, opts.reps);

    printf("\n%d of %d checks passed\n", checks - failures, checks);
    beagleFinalizeInstance(run.instance);
    return failures == 0 ? 0 : 1;
}
