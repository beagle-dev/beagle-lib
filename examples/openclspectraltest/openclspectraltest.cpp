/*
 * openclspectraltest.cpp
 *
 * Exercises the BEAGLE OpenCL-spectral backend with automatic fallback to
 * spectral CPU, then plain CPU.
 *
 * Dataset : human / chimp / gorilla (same as tinytest / tinygputest)
 * Model   : JC69 + 4-category discrete Gamma rate heterogeneity
 * Tree    : ((H,C),G) unrooted, two peeling operations
 * Reference logL: -1498.89812 (PAUP, verified with BEAGLE CPU backend)
 *
 * Usage:
 *   openclspectraltest [--opencl] [--cpu] [--resource N]
 *                      [--double] [--reps R]
 *
 *   --opencl     Require OpenCL-spectral backend (exit if unavailable)
 *   --cpu        Prefer spectral CPU backend
 *   --resource N Use BEAGLE resource number N explicitly
 *   --double     Use double precision (default: single)
 *   --reps R     Repeat the partials+likelihood evaluation R times (timing)
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <string>
#include <vector>

#include "libhmsbeagle/beagle.h"

// ── Reference sequences (same as tinytest.cpp / tinygputest.cpp) ─────────────

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

// ── Utilities ────────────────────────────────────────────────────────────────

static std::vector<double> makePartials(const char* seq) {
    int n = (int)strlen(seq);
    std::vector<double> p(n * 4, 0.0);
    for (int i = 0; i < n; ++i) {
        int s = -1;
        switch (seq[i]) {
            case 'A': s = 0; break;
            case 'C': s = 1; break;
            case 'G': s = 2; break;
            case 'T': s = 3; break;
        }
        if (s >= 0)
            p[i * 4 + s] = 1.0;
        else
            p[i*4]=p[i*4+1]=p[i*4+2]=p[i*4+3] = 1.0;
    }
    return p;
}

static void printFlags(long f) {
    if (f & BEAGLE_FLAG_PROCESSOR_CPU)            printf(" CPU");
    if (f & BEAGLE_FLAG_PROCESSOR_GPU)            printf(" GPU");
    if (f & BEAGLE_FLAG_PRECISION_DOUBLE)         printf(" DOUBLE");
    if (f & BEAGLE_FLAG_PRECISION_SINGLE)         printf(" SINGLE");
    if (f & BEAGLE_FLAG_FRAMEWORK_CUDA)           printf(" CUDA");
    if (f & BEAGLE_FLAG_FRAMEWORK_OPENCL)         printf(" OPENCL");
    if (f & BEAGLE_FLAG_FRAMEWORK_CPU)            printf(" CPU_FW");
    if (f & BEAGLE_FLAG_SPECTRAL_REPRESENTATION)  printf(" SPECTRAL");
}

// ── Options ───────────────────────────────────────────────────────────────────

struct Options {
    int  forceResource   = -1;
    bool doublePrecision = false;
    int  reps            = 1;
    long prefFlags       = 0;
};

static int createInstance(const Options& opts, int nTips, int nPartBufs,
                          int nPatterns, BeagleInstanceDetails* det) {
    int  resourceBuf = opts.forceResource;
    int* resources   = (opts.forceResource >= 0) ? &resourceBuf : nullptr;
    int  nResources  = (opts.forceResource >= 0) ? 1 : 0;
    long prec = opts.doublePrecision ? BEAGLE_FLAG_PRECISION_DOUBLE
                                     : BEAGLE_FLAG_PRECISION_SINGLE;
    return beagleCreateInstance(
        nTips,
        nPartBufs,
        /*compactBufs=*/0,
        /*stateCount=*/4,
        nPatterns,
        /*nEigen=*/1,
        /*nMatrices=*/4,
        /*nRateCats=*/4,
        /*scalers=*/0,
        resources, nResources,
        opts.prefFlags | prec,
        // /*reqFlags=*/0,
        BEAGLE_FLAG_SPECTRAL_REPRESENTATION,
        det);
}

// ── Main ─────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    Options opts;
    bool forceOpenCL = false, forceCPU = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--opencl") forceOpenCL = true;
        else if (a == "--cpu")    forceCPU    = true;
        else if (a == "--double") opts.doublePrecision = true;
        else if (a == "--resource" && i+1 < argc)
            opts.forceResource = atoi(argv[++i]);
        else if (a == "--reps" && i+1 < argc)
            opts.reps = atoi(argv[++i]);
        else {
            fprintf(stderr, "Unknown option: %s\n", a.c_str());
            return 1;
        }
    }

    // ── Enumerate resources ───────────────────────────────────────────────────
    BeagleResourceList* rList = beagleGetResourceList();
    printf("BEAGLE %s  available resources:\n", beagleGetVersion());
    for (int i = 0; i < rList->length; ++i) {
        printf("  [%d] %-30s  flags:", i, rList->list[i].name);
        printFlags(rList->list[i].supportFlags);
        printf("\n");
        if (rList->list[i].description && rList->list[i].description[0])
            printf("       %s\n", rList->list[i].description);
    }
    printf("\n");

    // ── Dataset ───────────────────────────────────────────────────────────────
    int nTips     = 3;
    int nPatterns = (int)strlen(kHuman);
    int nPartBufs = 5;  // 0=human, 1=chimp, 2=gorilla, 3=internal, 4=root

    // ── Backend cascade ───────────────────────────────────────────────────────
    const long kOpenCLSpectral = BEAGLE_FLAG_FRAMEWORK_OPENCL |
                                 BEAGLE_FLAG_SPECTRAL_REPRESENTATION |
                                 BEAGLE_FLAG_PROCESSOR_GPU;
    const long kCPUSpectral    = BEAGLE_FLAG_FRAMEWORK_CPU |
                                 BEAGLE_FLAG_SPECTRAL_REPRESENTATION |
                                 BEAGLE_FLAG_PROCESSOR_CPU;
    const long kCPU            = BEAGLE_FLAG_PROCESSOR_CPU;

    std::vector<long> cascade;
    if      (forceOpenCL)              cascade = { kOpenCLSpectral };
    else if (forceCPU)                 cascade = { kCPUSpectral };
    else if (opts.forceResource >= 0)  cascade = { 0 };
    else    cascade = { kOpenCLSpectral, kCPUSpectral, kCPU };

    BeagleInstanceDetails det;
    int instance = -1;
    for (long pref : cascade) {
        opts.prefFlags = pref;
        instance = createInstance(opts, nTips, nPartBufs, nPatterns, &det);
        if (instance >= 0) break;
    }

    if (instance < 0) {
        fprintf(stderr, "Failed to create BEAGLE instance (error %d)\n", instance);
        return 1;
    }

    printf("Selected resource %d: %s\n", det.resourceNumber, det.resourceName);
    printf("  Implementation : %s\n", det.implName);
    if (det.implDescription && det.implDescription[0])
        printf("  Description    : %s\n", det.implDescription);
    printf("  Active flags   :");
    printFlags(det.flags);
    printf("\n\n");

    // ── Load tip partials ─────────────────────────────────────────────────────
    auto hp = makePartials(kHuman);
    auto cp = makePartials(kChimp);
    auto gp = makePartials(kGorilla);
    beagleSetTipPartials(instance, 0, hp.data());
    beagleSetTipPartials(instance, 1, cp.data());
    beagleSetTipPartials(instance, 2, gp.data());

    // ── JC69 eigen decomposition ──────────────────────────────────────────────
    double evec[16] = {
         1.0,  2.0,  0.0,  0.5,
         1.0, -2.0,  0.5,  0.0,
         1.0,  2.0,  0.0, -0.5,
         1.0, -2.0, -0.5,  0.0
    };
    double ivec[16] = {
         0.25,  0.25,  0.25,  0.25,
         0.125,-0.125, 0.125,-0.125,
         0.0,   1.0,   0.0,  -1.0,
         1.0,   0.0,  -1.0,   0.0
    };
    double eval[4] = { 0.0, -4.0/3.0, -4.0/3.0, -4.0/3.0 };
    beagleSetEigenDecomposition(instance, 0, evec, ivec, eval);

    double freqs[16] = { 0.25,0.25,0.25,0.25, 0.25,0.25,0.25,0.25,
                         0.25,0.25,0.25,0.25, 0.25,0.25,0.25,0.25 };
    beagleSetStateFrequencies(instance, 0, freqs);

    // ── Rate heterogeneity: 4-category discrete Gamma (alpha≈0.5) ────────────
    double rates[4]   = { 0.03338775, 0.25191592, 0.82026848, 2.89442785 };
    double weights[4] = { 0.25, 0.25, 0.25, 0.25 };
    beagleSetCategoryRates(instance, rates);
    beagleSetCategoryWeights(instance, 0, weights);

    std::vector<double> patW(nPatterns, 1.0);
    beagleSetPatternWeights(instance, patW.data());

    // ── Tree: ((H:0.1, C:0.1):0.1, G:0.2) ───────────────────────────────────
    int    nodeIdx[4]  = { 0, 1, 2, 3 };
    double edgeLens[4] = { 0.1, 0.1, 0.2, 0.1 };
    beagleUpdateTransitionMatrices(instance, 0, nodeIdx, nullptr, nullptr,
                                   edgeLens, 4);

    BeagleOperation ops[2] = {
        { 3, BEAGLE_OP_NONE, BEAGLE_OP_NONE, 0, 0, 1, 1 },
        { 4, BEAGLE_OP_NONE, BEAGLE_OP_NONE, 2, 2, 3, 3 }
    };
    beagleUpdatePartials(instance, ops, 2, BEAGLE_OP_NONE);

    // ── Evaluate log-likelihood ───────────────────────────────────────────────
    int rootBuf = 4, wBuf = 0, fBuf = 0, sBuf = BEAGLE_OP_NONE;
    double logL = 0.0;

    // Warm-up (avoids counting JIT / driver init in timing)
    beagleCalculateRootLogLikelihoods(instance, &rootBuf, &wBuf, &fBuf,
                                      &sBuf, 1, &logL);

    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < opts.reps; ++r) {
        if (r > 0) {
            beagleUpdateTransitionMatrices(instance, 0, nodeIdx, nullptr, nullptr,
                                           edgeLens, 4);
            beagleUpdatePartials(instance, ops, 2, BEAGLE_OP_NONE);
        }
        int rc = beagleCalculateRootLogLikelihoods(instance, &rootBuf, &wBuf, &fBuf,
                                                    &sBuf, 1, &logL);
        if (rc < 0) {
            fprintf(stderr, "calculateRootLogLikelihoods failed: %d\n", rc);
            beagleFinalizeInstance(instance);
            return 1;
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // ── Report ────────────────────────────────────────────────────────────────
    printf("log L  = %.5f  (PAUP ref = -1498.89812)\n", logL);
    if (opts.reps > 1)
        printf("Timing = %.3f ms / rep  (%d reps, %.3f ms total)\n",
               ms / opts.reps, opts.reps, ms);
    else
        printf("Timing = %.3f ms\n", ms);

    const double kRef = -1498.89812;
    const double kTol = opts.doublePrecision ? 1e-4 : 0.5;
    double delta = std::fabs(logL - kRef);
    if (delta < kTol)
        printf("PASS   (|delta| = %.5f < %g)\n", delta, kTol);
    else
        printf("FAIL   (|delta| = %.5f >= %g)\n", delta, kTol);

    beagleFinalizeInstance(instance);
    return (delta < kTol) ? 0 : 1;
}
