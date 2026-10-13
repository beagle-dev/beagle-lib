/*
 * spectral_probe (TODO.md plan step S3.4; PLAN-spectral-tinygpu-degree2.md): beagleCreateInstance requests against the
 * TinyGPU plugin, for the TinyGPU-only checks that the shared examples must not carry (they are also OpenCL and CUDA
 * ctests): the standard factory's refusal of a request that mentions SPECTRAL, the routing of BEAST's pre-order
 * requests, standard and spectral modules in one process, out of memory in the spectral pools, and a GPU lost during the
 * adjoint gradient.
 *
 * usage: spectral_probe [--resource r] [--pref F] [--req F] [--eigen E] [--adjoint-reps R] item ...
 *   item    S:{s|d}:{std|spec}: S states; single or double precision preferred; spec also requires SPECTRAL and
 *           PREORDER_TRANSPOSE_AUTO, as every spectral caller does
 *   F       flags added to every item's preference (--pref) or requirement (--req): a hex number, or '|'-separated names
 *           (GPU CPU TINYGPU SINGLE DOUBLE SPECTRAL TRANSPOSE_AUTO TRANSPOSE_MANUAL)
 *   --resource r       the resource list {r}; without it, none (BEAGLE chooses)
 *   --eigen E          eigen buffers in each instance (1); a spectral instance's pools grow with them
 *   --adjoint-reps R   then, on each spectral instance, a 3-tip adjoint gradient R times, printing each change of its rc
 * One line per item, "<item> rc=<rc> impl=<implementation>" (rc: the instance, or the error); every instance stays alive
 * until the end. Exit 0 once every line is printed, 2 on a usage error.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "libhmsbeagle/beagle.h"

namespace {

long parseFlags(const char* s) {
    if (!strncmp(s, "0x", 2)) return strtol(s, NULL, 16);
    static const struct { const char* name; long flag; } names[] = {
        {"GPU", BEAGLE_FLAG_PROCESSOR_GPU}, {"CPU", BEAGLE_FLAG_PROCESSOR_CPU},
        {"TINYGPU", BEAGLE_FLAG_FRAMEWORK_TINYGPU}, {"SINGLE", BEAGLE_FLAG_PRECISION_SINGLE},
        {"DOUBLE", BEAGLE_FLAG_PRECISION_DOUBLE}, {"SPECTRAL", BEAGLE_FLAG_SPECTRAL_REPRESENTATION},
        {"TRANSPOSE_AUTO", BEAGLE_FLAG_PREORDER_TRANSPOSE_AUTO}, {"TRANSPOSE_MANUAL", BEAGLE_FLAG_PREORDER_TRANSPOSE_MANUAL}};
    long flags = 0;
    std::string rest(s);
    while (!rest.empty()) {
        const size_t bar = rest.find('|');
        const std::string name = rest.substr(0, bar);
        bool found = false;
        for (const auto& n : names) if (name == n.name) { flags |= n.flag; found = true; }
        if (!found) { fprintf(stderr, "spectral_probe: unknown flag %s\n", name.c_str()); exit(2); }
        rest = (bar == std::string::npos) ? "" : rest.substr(bar + 1);
    }
    return flags;
}

// A 3-tip tree ((0,1)3,2)4: post-order 3 and 4, TOP pre-order 5 + n for node n, matrix n above node n; P = e^{Dt} with
// V = I (any decomposition does: the probe checks return codes, and the fakes compute nothing)
void setUpAdjoint(int instance, int S, int C) {
    std::vector<int> states(C);
    for (int t = 0; t < 3; t++) {
        for (int k = 0; k < C; k++) states[k] = (k * 7 + t * 3) % S;
        beagleSetTipStates(instance, t, states.data());
    }
    std::vector<double> ones(C, 1.0), freqs(S, 1.0 / S), identity(S * S, 0.0), values(S, -1.0), prior((size_t) C * S, 1.0);
    for (int i = 0; i < S; i++) identity[i * S + i] = 1.0;
    values[0] = 0.0;
    const double rate = 1.0, weight = 1.0;
    beagleSetPatternWeights(instance, ones.data());
    beagleSetStateFrequencies(instance, 0, freqs.data());
    beagleSetCategoryRates(instance, &rate);
    beagleSetCategoryWeights(instance, 0, &weight);
    beagleSetEigenDecomposition(instance, 0, identity.data(), identity.data(), values.data());
    const int matrices[4] = {0, 1, 2, 3};
    const double lengths[4] = {0.1, 0.2, 0.3, 0.4};
    beagleUpdateTransitionMatrices(instance, 0, matrices, NULL, NULL, lengths, 4);
    const BeagleOperation post[2] = {{3, BEAGLE_OP_NONE, BEAGLE_OP_NONE, 0, 0, 1, 1},
                                     {4, BEAGLE_OP_NONE, BEAGLE_OP_NONE, 3, 3, 2, 2}};
    beagleUpdatePartials(instance, post, 2, BEAGLE_OP_NONE);
    beagleSetPartials(instance, 9, prior.data());
    const BeagleOperation top[4] = {{8, BEAGLE_OP_NONE, BEAGLE_OP_NONE, 9, BEAGLE_OP_NONE, 2, 2},
                                    {7, BEAGLE_OP_NONE, BEAGLE_OP_NONE, 9, BEAGLE_OP_NONE, 3, 3},
                                    {5, BEAGLE_OP_NONE, BEAGLE_OP_NONE, 8, 3, 1, 1},
                                    {6, BEAGLE_OP_NONE, BEAGLE_OP_NONE, 8, 3, 0, 0}};
    beagleUpdatePrePartials_v5(instance, top, 4, BEAGLE_OP_NONE, BEAGLE_PARTIALS_TOP);
}

} // namespace

int main(int argc, char** argv) {
    int resource = -1, eigenCount = 1, adjointReps = 0;
    long pref = 0, req = 0;
    std::vector<std::string> items;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--resource") && i + 1 < argc) resource = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--pref") && i + 1 < argc) pref |= parseFlags(argv[++i]);
        else if (!strcmp(argv[i], "--req") && i + 1 < argc) req |= parseFlags(argv[++i]);
        else if (!strcmp(argv[i], "--eigen") && i + 1 < argc) eigenCount = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--adjoint-reps") && i + 1 < argc) adjointReps = atoi(argv[++i]);
        else if (argv[i][0] != '-') items.push_back(argv[i]);
        else {
            fprintf(stderr, "usage: spectral_probe [--resource r] [--pref F] [--req F] [--eigen E] [--adjoint-reps R] "
                            "S:{s|d}:{std|spec} ...\n");
            return 2;
        }
    }
    const int C = 10;
    struct Instance { std::string item; int id, S; bool spectral; };
    std::vector<Instance> alive;
    for (const std::string& item : items) {
        int S = 0;
        char p = 0, kind[8] = {0};
        if (sscanf(item.c_str(), "%d:%c:%7s", &S, &p, kind) != 3 || S < 2 || (p != 's' && p != 'd') ||
            (strcmp(kind, "std") && strcmp(kind, "spec"))) {
            fprintf(stderr, "spectral_probe: bad item %s\n", item.c_str());
            return 2;
        }
        const bool spectral = !strcmp(kind, "spec");
        const long itemPref = pref | (p == 's' ? BEAGLE_FLAG_PRECISION_SINGLE : BEAGLE_FLAG_PRECISION_DOUBLE);
        const long itemReq = req | (spectral ? BEAGLE_FLAG_SPECTRAL_REPRESENTATION | BEAGLE_FLAG_PREORDER_TRANSPOSE_AUTO : 0);
        BeagleInstanceDetails details;
        const int rc = beagleCreateInstance(3, 10, 3, S, C, eigenCount, 5, 1, 0, resource < 0 ? NULL : &resource,
                                            resource < 0 ? 0 : 1, itemPref, itemReq, &details);
        printf("%s rc=%d impl=%s\n", item.c_str(), rc, rc >= 0 ? details.implName : "none");
        fflush(stdout);
        if (rc >= 0) alive.push_back({item, rc, S, spectral});
    }
    for (const Instance& in : alive) {
        if (!in.spectral || adjointReps <= 0) continue;
        setUpAdjoint(in.id, in.S, C);
        const BeagleBranchOperation branches[4] = {{0, 5, 0, 0}, {1, 6, 1, 0}, {2, 7, 2, 0}, {3, 8, 3, 0}};
        std::vector<double> gradient((size_t) in.S * in.S);
        int last = 1;   // no return code has this value
        for (int r = 0; r < adjointReps; r++) {
            const int rc = beagleCalculateAdjointDerivative(in.id, branches, 0, 0, 4, 0, 4, gradient.data(), NULL);
            if (rc != last) { printf("%s adjoint %d: rc=%d\n", in.item.c_str(), r, rc); fflush(stdout); last = rc; }
        }
    }
    for (const Instance& in : alive) beagleFinalizeInstance(in.id);
    return 0;
}
