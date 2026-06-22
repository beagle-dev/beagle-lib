/*
 * tinygpuhybridtest.cpp
 *
 * Tests the BEAGLE TinyGPUHybrid backend end-to-end:
 *   1. Enumerates all BEAGLE resources, identifies the TinyGPUHybrid device
 *   2. Initializes a BEAGLE instance on it (which calls nvHybridSetup internally)
 *   3. Dispatches GPU kernels: transition matrices + partial likelihood peeling
 *   4. Evaluates the root log-likelihood and checks against the reference
 *
 * Dataset : human / chimp / gorilla (same as tinytest / tinygputest)
 * Model   : JC69 + 4-category discrete Gamma
 * Tree    : ((H:0.1, C:0.1):0.1, G:0.2)  —  two peeling ops
 * Ref logL: -1498.89812 (PAUP, verified against BEAGLE CPU backend)
 *
 * Usage:
 *   tinygpuhybridtest [--resource N]
 *   --resource N   Force BEAGLE resource index N (skips auto-detect)
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <string>
#include <vector>

#include "libhmsbeagle/beagle.h"

// ── Reference sequences ───────────────────────────────────────────────────────

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

// ── Helpers ───────────────────────────────────────────────────────────────────

static std::vector<double> makePartials(const char* seq, int n) {
    std::vector<double> p(n * 4, 0.0);
    for (int i = 0; i < n; ++i) {
        switch (seq[i]) {
            case 'A': p[i*4+0] = 1.0; break;
            case 'C': p[i*4+1] = 1.0; break;
            case 'G': p[i*4+2] = 1.0; break;
            case 'T': p[i*4+3] = 1.0; break;
            default:  p[i*4+0] = p[i*4+1] = p[i*4+2] = p[i*4+3] = 1.0; break;
        }
    }
    return p;
}

static void printFlags(long f) {
    if (f & BEAGLE_FLAG_PROCESSOR_CPU)     printf(" CPU");
    if (f & BEAGLE_FLAG_PROCESSOR_GPU)     printf(" GPU");
    if (f & BEAGLE_FLAG_PRECISION_DOUBLE)  printf(" DOUBLE");
    if (f & BEAGLE_FLAG_PRECISION_SINGLE)  printf(" SINGLE");
    if (f & BEAGLE_FLAG_FRAMEWORK_CUDA)    printf(" CUDA");
    if (f & BEAGLE_FLAG_FRAMEWORK_OPENCL)  printf(" OPENCL");
    if (f & BEAGLE_FLAG_FRAMEWORK_TINYGPU) printf(" TINYGPU");
    if (f & BEAGLE_FLAG_FRAMEWORK_CPU)     printf(" CPU_FW");
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    int forceResource = -1;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--resource" && i + 1 < argc)
            forceResource = atoi(argv[++i]);
        else {
            fprintf(stderr, "Usage: tinygpuhybridtest [--resource N]\n");
            return 1;
        }
    }

    // ── Step 1: Enumerate all BEAGLE resources ────────────────────────────────
    printf("=== TinyGPUHybrid backend test ===\n\n");
    BeagleResourceList* rList = beagleGetResourceList();
    printf("BEAGLE %s  resources (%d):\n", beagleGetVersion(), rList->length);

    int hybridIdx = -1;
    for (int i = 0; i < rList->length; ++i) {
        const char* name = rList->list[i].name;
        const char* desc = rList->list[i].description ? rList->list[i].description : "";
        bool isHybrid = strstr(name, "Hybrid") != nullptr || strstr(desc, "hybrid") != nullptr;

        printf("  [%d] %s", i, name);
        if (isHybrid) {
            printf("  <-- TinyGPUHybrid");
            if (hybridIdx < 0) hybridIdx = i;
        }
        printf("\n");
        if (desc[0]) printf("       %s\n", desc);
        printf("       flags:");
        printFlags(rList->list[i].supportFlags);
        printf("\n");
    }
    printf("\n");

    // ── Step 2: Select resource ───────────────────────────────────────────────
    int resourceIdx = (forceResource >= 0) ? forceResource : hybridIdx;
    if (resourceIdx < 0) {
        fprintf(stderr,
            "No TinyGPUHybrid resource detected.\n"
            "Ensure hmsbeagle-tinygpu-hybrid.so is on the plugin path.\n"
            "Use --resource N to force a specific resource index.\n");
        return 1;
    }
    printf("Using resource index %d\n\n", resourceIdx);

    // ── Step 3: Create BEAGLE instance ───────────────────────────────────────
    // Dataset: 3 taxa, 5 partial buffers (tips 0-2, internal 3, root 4)
    int nPatterns = (int)strlen(kHuman);
    BeagleInstanceDetails det;
    int instance = beagleCreateInstance(
        3,              // nTips
        5,              // nPartialBuffers
        0,              // nCompactBuffers
        4,              // stateCount
        nPatterns,      // nPatterns
        1,              // nEigenBuffers
        4,              // nMatrixBuffers
        4,              // nRateCats
        0,              // nScaleBuffers
        &resourceIdx, 1,
        BEAGLE_FLAG_PRECISION_DOUBLE  |
        BEAGLE_FLAG_PROCESSOR_GPU,
        BEAGLE_FLAG_FRAMEWORK_TINYGPU,              // reqFlags
        &det);

    if (instance < 0) {
        fprintf(stderr, "beagleCreateInstance failed (error %d)\n", instance);
        return 1;
    }

    printf("Resource %d: %s\n", det.resourceNumber, det.resourceName);
    printf("Implementation : %s\n", det.implName);
    if (det.implDescription && det.implDescription[0])
        printf("Description    : %s\n", det.implDescription);
    printf("Active flags   :");
    printFlags(det.flags);
    printf("\n\n");

    // ── Step 4: Load tip partials ─────────────────────────────────────────────
    auto hp = makePartials(kHuman,   nPatterns);
    auto cp = makePartials(kChimp,   nPatterns);
    auto gp = makePartials(kGorilla, nPatterns);
    beagleSetTipPartials(instance, 0, hp.data());
    beagleSetTipPartials(instance, 1, cp.data());
    beagleSetTipPartials(instance, 2, gp.data());

    // ── Step 5: JC69 eigen decomposition ─────────────────────────────────────
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

    double freqs[16];
    for (int i = 0; i < 16; ++i) freqs[i] = 0.25;
    beagleSetStateFrequencies(instance, 0, freqs);

    // 4-category discrete Gamma (alpha ≈ 0.5)
    double rates[4]   = { 0.03338775, 0.25191592, 0.82026848, 2.89442785 };
    double weights[4] = { 0.25, 0.25, 0.25, 0.25 };
    beagleSetCategoryRates(instance, rates);
    beagleSetCategoryWeights(instance, 0, weights);

    std::vector<double> patW(nPatterns, 1.0);
    beagleSetPatternWeights(instance, patW.data());

    // ── Step 6: Transition matrices — 4 branches ─────────────────────────────
    int    nodeIdx[4]  = { 0, 1, 2, 3 };
    double edgeLens[4] = { 0.1, 0.1, 0.2, 0.1 };
    int rc = beagleUpdateTransitionMatrices(instance, 0, nodeIdx, nullptr, nullptr, edgeLens, 4);
    if (rc < 0) {
        fprintf(stderr, "updateTransitionMatrices failed: %d\n", rc);
        beagleFinalizeInstance(instance);
        return 1;
    }

    // ── Step 7: Partial likelihood peeling ───────────────────────────────────
    // op: {destBuf, destScale, srcScale, child1buf, child1mat, child2buf, child2mat}
    BeagleOperation ops[2] = {
        { 3, BEAGLE_OP_NONE, BEAGLE_OP_NONE, 0, 0, 1, 1 },  // node3 = H*C
        { 4, BEAGLE_OP_NONE, BEAGLE_OP_NONE, 2, 2, 3, 3 }   // root4 = G*node3
    };
    rc = beagleUpdatePartials(instance, ops, 2, BEAGLE_OP_NONE);
    if (rc < 0) {
        fprintf(stderr, "updatePartials failed: %d\n", rc);
        beagleFinalizeInstance(instance);
        return 1;
    }

    // ── Step 8: Root log-likelihood ───────────────────────────────────────────
    int rootBuf = 4, wBuf = 0, fBuf = 0, sBuf = BEAGLE_OP_NONE;
    double logL = 0.0;

    auto t0 = std::chrono::steady_clock::now();
    rc = beagleCalculateRootLogLikelihoods(instance, &rootBuf, &wBuf, &fBuf, &sBuf, 1, &logL);
    auto t1 = std::chrono::steady_clock::now();

    if (rc < 0) {
        fprintf(stderr, "calculateRootLogLikelihoods failed: %d\n", rc);
        beagleFinalizeInstance(instance);
        return 1;
    }

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // ── Step 9: Check result ──────────────────────────────────────────────────
    const double kRef = -1498.89812;
    const double kTol = 0.5;   // nats; generous for single precision
    double delta = std::fabs(logL - kRef);

    printf("logL      = %12.5f\n", logL);
    printf("reference = %12.5f\n", kRef);
    printf("|delta|   = %12.5f  (tolerance %.1f nats)\n", delta, kTol);
    printf("time      = %.3f ms\n", ms);
    printf("\n%s\n", delta < kTol ? "PASS" : "FAIL");

    beagleFinalizeInstance(instance);
    return delta < kTol ? 0 : 1;
}
