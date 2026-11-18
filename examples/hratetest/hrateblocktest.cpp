#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <set>
#include <string>
#include <vector>
#include "libhmsbeagle/CPU/HierarchicalMethods.h"

using namespace beagle::cpu;

static int checks = 0;
static int failures = 0;

static void report(const std::string& name, bool pass, const std::string& detail) {
    ++checks;
    if (!pass)
        ++failures;
    if (!pass || getenv("HRATE_VERBOSE"))
        printf("%-70s %s  %s\n", name.c_str(), pass ? "PASS" : "FAIL", detail.c_str());
}

struct Upload {
    std::string name;
    bool caterpillar;
    HierarchicalOps ops;
};

static Upload makeUpload(bool caterpillar, const int* leaf, int swaps, bool cherryOrder) {
    Upload u;
    u.caterpillar = caterpillar;
    std::vector<int> post;
    auto triple = [&](int m, int a, int b, int bit) {
        if (swaps & (1 << bit))
            std::swap(a, b);
        post.insert(post.end(), { m, a, b });
    };
    if (caterpillar) {
        triple(4, leaf[0], leaf[1], 0);
        triple(5, 4, leaf[2], 1);
        triple(6, 5, leaf[3], 2);
    } else if (cherryOrder) {
        triple(4, leaf[0], leaf[1], 0);
        triple(5, leaf[2], leaf[3], 1);
        triple(6, 4, 5, 2);
    } else {
        triple(4, leaf[2], leaf[3], 0);
        triple(5, leaf[0], leaf[1], 1);
        triple(6, 5, 4, 2);
    }
    u.ops.postOps = post;
    u.ops.preOps.resize(9);
    for (int t = 0; t < 3; t++)
        for (int c = 0; c < 3; c++)
            u.ops.preOps[3 * t + c] = post[3 * (2 - t) + c];
    hierarchicalParents(u.ops, 4);
    hierarchicalDetectShape(u.ops, 4);
    char buf[96];
    snprintf(buf, sizeof(buf), "%s leaves %d%d%d%d swaps %d order %d", caterpillar ? "caterpillar" : "balanced",
             leaf[0], leaf[1], leaf[2], leaf[3], swaps, cherryOrder ? 1 : 0);
    u.name = buf;
    return u;
}

static std::vector<Upload> allUploads() {
    std::vector<Upload> out;
    int leaf[4] = { 0, 1, 2, 3 };
    do {
        for (int swaps = 0; swaps < 8; swaps++) {
            out.push_back(makeUpload(true, leaf, swaps, true));
            out.push_back(makeUpload(false, leaf, swaps, true));
            out.push_back(makeUpload(false, leaf, swaps, false));
        }
    } while (std::next_permutation(leaf, leaf + 4));
    return out;
}

static std::vector<int> clades(const HierarchicalOps& ops) {
    std::vector<int> clade(7, 0);
    for (int x = 0; x < 4; x++)
        clade[x] = 1 << x;
    for (int t = 0; t < 3; t++)
        clade[ops.postOps[3 * t]] = clade[ops.postOps[3 * t + 1]] | clade[ops.postOps[3 * t + 2]];
    return clade;
}

static void descriptorCase(const Upload& u) {
    HierarchicalBlockTable table;
    const bool built = hierarchicalBlockTable(u.ops, table);
    const std::vector<int> clade = clades(u.ops);
    std::string why;
    bool ok = built && table.valid && table.blockCount == 2;
    if (!ok)
        why = "table not built";
    int keptOf[2] = { -1, -1 };
    std::set<int> kinds;
    for (int mask = 0; mask < 16 && ok; mask++) {
        int kept = 0;
        for (int m = 4; m < 7; m++)
            if (clade[m] & mask)
                kept |= 1 << (m - 4);
        const int b = table.blockOfMask[mask];
        if (mask == 0 || kept == 7) {
            if (b != -1) {
                ok = false;
                why = "mask " + std::to_string(mask) + " should run M0";
            }
            continue;
        }
        if (b < 0 || b > 1) {
            ok = false;
            why = "mask " + std::to_string(mask) + " has no block";
            break;
        }
        if (keptOf[b] < 0)
            keptOf[b] = kept;
        if (keptOf[b] != kept) {
            ok = false;
            why = "block " + std::to_string(b) + " mixes kept sets";
            break;
        }
    }
    for (int b = 0; b < 2 && ok; b++) {
        const HierarchicalBlock& blk = table.blocks[b];
        const int kept = keptOf[b];
        std::set<int> expected;
        for (int t = 0; t < 3; t++) {
            const int m = u.ops.postOps[3 * t];
            if (!(kept & (1 << (m - 4))))
                continue;
            for (int c = 1; c <= 2; c++) {
                const int ch = u.ops.postOps[3 * t + c];
                if (ch < 4 || !(kept & (1 << (ch - 4))))
                    expected.insert(clade[ch]);
            }
        }
        std::set<int> got;
        int cover = 0;
        for (int i = 0; i < blk.stateCount; i++) {
            const int node = blk.zNode[i];
            int set = clade[node];
            if (!(set & (1 << blk.row[i])))
                why = "row " + std::to_string(i) + " is not in its set";
            got.insert(set);
            if (cover & set)
                why = "rows overlap";
            cover |= set;
            if (i == blk.agg) {
                int extras = 0;
                for (int e = 0; e < blk.extraCount; e++)
                    extras |= 1 << blk.extra[e];
                if (extras != (set & ~(1 << blk.row[i])))
                    why = "extra rows differ from the clade";
                if (blk.top != node)
                    why = "aggregate top differs";
            } else if (node >= 4) {
                why = "a single row points at an internal node";
            }
        }
        int heights = 0;
        for (int j = 0; j < blk.stateCount - 1; j++)
            heights |= 1 << blk.height[j];
        if (got != expected)
            why = "row sets differ from the brute-force rule";
        if (cover != 15)
            why = "rows do not cover the states";
        if (heights != kept)
            why = "kept nodes differ";
        if (blk.stateCount != 1 + __builtin_popcount(kept))
            why = "reduced size differs";
        kinds.insert(blk.kind);
        if (!why.empty())
            ok = false;
    }
    if (ok) {
        const std::set<int> want = u.caterpillar ? std::set<int>{ 2, 3 } : std::set<int>{ 1 };
        if (kinds != want) {
            ok = false;
            why = "unexpected kinds";
        }
        int lump[2] = { clade[u.ops.postOps[0]], clade[u.ops.postOps[3]] };
        if (lump[0] > lump[1])
            std::swap(lump[0], lump[1]);
        if (table.lumpKey != lump[0] * 16 + lump[1]) {
            ok = false;
            why = "lump key";
        }
    }
    report("descriptor " + u.name, ok, why);
}

static void runsCase() {
    signed char blockOfMask[16];
    for (int m = 0; m < 16; m++)
        blockOfMask[m] = (signed char) (m == 1 ? 0 : m == 2 ? 1 : -1);
    std::mt19937_64 rng(20261118);
    int bad = 0, cases = 0;
    for (int trial = 0; trial < 2000; trial++) {
        const int K = 1 + (int) (rng() % 600);
        std::vector<unsigned char> mask(K);
        if (trial % 2 == 0) {
            const int a = (int) (rng() % (K + 1)), b = a + (int) (rng() % (K - a + 1));
            for (int k = 0; k < K; k++)
                mask[k] = k < a ? 1 : k < b ? 15 : 2;
        } else {
            for (int k = 0; k < K; k++)
                mask[k] = (unsigned char) ((rng() % 3 == 0) ? 1 + rng() % 2 : 15);
        }
        const int start = (int) (rng() % (K + 1)) / 2, end = start + (int) (rng() % (K - start + 1));
        if (end <= start)
            continue;
        cases++;
        const int align = 8, minRun = 64;
        std::vector<HierarchicalRun> out(1, { -7, -7, -7 });
        hierarchicalBlockRuns(blockOfMask, mask.data(), start, end, align, minRun, out);
        bool ok = out[0].block == -7;
        int at = start, reduced = 0;
        for (size_t i = 1; i < out.size() && ok; i++) {
            const HierarchicalRun& r = out[i];
            ok = r.start == at && r.end > r.start;
            at = r.end;
            if (r.block >= 0) {
                reduced++;
                ok = ok && r.end - r.start >= minRun && (r.start == start || r.start % align == 0) &&
                     (r.end == end || r.end % align == 0) && (i + 1 == out.size() || out[i + 1].block != r.block);
                for (int k = r.start; k < r.end && ok; k++)
                    ok = blockOfMask[mask[k]] == r.block;
            } else {
                ok = ok && (i + 1 == out.size() || out[i + 1].block >= 0);
            }
        }
        ok = ok && at == end;
        if (trial % 2 == 0 && ok) {
            int runs = (int) out.size() - 1;
            ok = runs <= 3;
        }
        if (!ok)
            bad++;
    }
    report("run builder: coverage, aligned inner boundaries, no reduced run under n_min, at most 3 runs when "
           "sorted [A][full][B]",
           bad == 0, std::to_string(bad) + " of " + std::to_string(cases) + " bad");
}

template <typename REALTYPE>
struct KernelCase {
    Upload u;
    HierarchicalBlockTable table;
    std::vector<double> lambda, piNode;
    std::vector<REALTYPE> pi;
    HierarchicalSlot<REALTYPE> slot1, slot2;
};

template <typename REALTYPE>
static double maxRel(const REALTYPE* a, const REALTYPE* b, int n) {
    double worst = 0.0;
    for (int i = 0; i < n; i++) {
        const double x = a[i], y = b[i];
        if (std::isnan(x) || std::isnan(y))
            return std::numeric_limits<double>::infinity();
        const double scale = std::max(std::fabs(y), 1e-300);
        worst = std::max(worst, std::fabs(x - y) / scale);
    }
    return worst;
}

template <typename REALTYPE, int Kind, bool Scaled, bool Broadcast>
static void kernelCase(const char* label, const Upload& u, int which, std::mt19937_64& rng) {
    typedef HierarchicalBlockKind<Kind> K_;
    constexpr int Sc = K_::Sc, Agg = K_::Agg, Extra = Broadcast ? K_::Extra : 0;
    const int S = 4, C = 2, K = 37, W = 5;
    HierarchicalBlockTable table;
    hierarchicalBlockTable(u.ops, table);
    int b = -1;
    for (int i = 0; i < 2; i++)
        if (table.blocks[i].kind == Kind)
            b = i;
    if (b < 0)
        return;
    const HierarchicalBlock& blk = table.blocks[b];
    std::uniform_real_distribution<double> unit(0.05, 1.0);
    std::vector<double> lambda(3), piD(4), piNode(7), rates = { 0.4, 1.7 }, prZ0Tmp(7), sumH(7);
    for (double& x : lambda)
        x = unit(rng) * 2.0;
    double tot = 0.0;
    for (double& x : piD)
        tot += (x = unit(rng));
    for (double& x : piD)
        x /= tot;
    hierarchicalNodeFrequencies(u.ops, piD.data(), S, piNode.data());
    std::vector<REALTYPE> pi(piD.begin(), piD.end());
    HierarchicalSlot<REALTYPE> s1, s2;
    hierarchicalBuildBundle(u.ops, lambda.data(), S, piNode.data(), rates.data(), C, 0.3, prZ0Tmp.data(), true, s1);
    hierarchicalBuildBundle(u.ops, lambda.data(), S, piNode.data(), rates.data(), C, 0.7, prZ0Tmp.data(), true, s2);
    hierarchicalColumnTable(s1, pi.data(), S, C, sumH.data());
    hierarchicalColumnTable(s2, pi.data(), S, C, sumH.data());
    int aggMask = 0;
    aggMask |= 1 << blk.row[blk.agg];
    for (int e = 0; e < blk.extraCount; e++)
        aggMask |= 1 << blk.extra[e];
    std::vector<REALTYPE> P1((size_t) S * K), P2((size_t) S * K), invScale(K);
    for (int k = 0; k < K; k++) {
        const double shared1 = unit(rng), shared2 = unit(rng);
        for (int x = 0; x < S; x++) {
            P1[(size_t) x * K + k] = (REALTYPE) ((aggMask >> x & 1) ? shared1 : unit(rng));
            P2[(size_t) x * K + k] = (REALTYPE) ((aggMask >> x & 1) ? shared2 : unit(rng));
        }
        invScale[k] = (REALTYPE) (0.5 + unit(rng));
    }
    std::vector<int> allowed;
    for (int x = 0; x < S; x++)
        if (!(aggMask >> x & 1))
            allowed.push_back(x);
    allowed.push_back(S);
    std::vector<int> st1(K), st2(K);
    for (int k = 0; k < K; k++) {
        st1[k] = allowed[rng() % allowed.size()];
        st2[k] = allowed[rng() % allowed.size()];
    }
    const REALTYPE nan = std::numeric_limits<REALTYPE>::quiet_NaN();
    double worst = 0.0;
    bool extrasUntouched = true;
    for (int l = 0; l < C; l++) {
        HierarchicalLaneCoef<REALTYPE, 4> f1, f2;
        const int blockLength = hierarchicalBlockLength(S);
        auto fullCoef = [&](const HierarchicalSlot<REALTYPE>& s, HierarchicalLaneCoef<REALTYPE, 4>& c) {
            const REALTYPE* H = s.values.data() + (size_t) l * blockLength;
            for (int i = 0; i < S; i++) {
                c.pi[i] = pi[u.ops.tipOfSlot[i]];
                c.z[i] = H[S - 1 + u.ops.tipOfSlot[i]];
            }
            for (int i = 0; i < S - 1; i++)
                c.h[i] = H[u.ops.intOfSlot[i] - S];
        };
        fullCoef(s1, f1);
        fullCoef(s2, f2);
        const REALTYPE* L1[4];
        const REALTYPE* L2[4];
        std::vector<REALTYPE> ref((size_t) S * K, nan), got((size_t) S * K, nan);
        REALTYPE* D[4];
        for (int i = 0; i < S; i++) {
            L1[i] = P1.data() + (size_t) u.ops.tipOfSlot[i] * K;
            L2[i] = P2.data() + (size_t) u.ops.tipOfSlot[i] * K;
            D[i] = ref.data() + (size_t) u.ops.tipOfSlot[i] * K;
        }
        const REALTYPE* T1 = s1.columnTable.data() + (size_t) l * S * W;
        const REALTYPE* T2 = s2.columnTable.data() + (size_t) l * S * W;
        REALTYPE Tp[4 * 5];
        for (int x = 0; x < S; x++)
            for (int y = 0; y < W; y++)
                Tp[x * W + y] = T2[u.ops.tipOfSlot[x] * W + y];
        typedef HierarchicalShapeDispatch<REALTYPE, 4, Scaled> Dispatch;
        if (which == 0)
            Dispatch::pair(u.ops.shapeId, L1, f1, L2, f2, D, invScale.data(), 0, K);
        else if (which == 1)
            Dispatch::column(u.ops.shapeId, L1, f1, st2.data(), Tp, D, invScale.data(), 0, K);
        else {
            REALTYPE* Dx[4];
            for (int x = 0; x < S; x++)
                Dx[x] = ref.data() + (size_t) x * K;
            hierarchicalColColPair<REALTYPE, 4, Scaled>(st1.data(), T1, st2.data(), T2, Dx, invScale.data(), 0, K);
        }
        HierarchicalLaneCoef<REALTYPE, Sc> c1, c2;
        hierarchicalBlockCoef<REALTYPE, Sc>(s1, l, blk, pi.data(), (REALTYPE) piNode[blk.top], c1);
        hierarchicalBlockCoef<REALTYPE, Sc>(s2, l, blk, pi.data(), (REALTYPE) piNode[blk.top], c2);
        const REALTYPE* B1[3];
        const REALTYPE* B2[3];
        REALTYPE* E[5];
        for (int i = 0; i < Sc; i++) {
            B1[i] = P1.data() + (size_t) blk.row[i] * K;
            B2[i] = P2.data() + (size_t) blk.row[i] * K;
            E[i] = got.data() + (size_t) blk.row[i] * K;
        }
        for (int e = 0; e < blk.extraCount; e++)
            E[Sc + e] = got.data() + (size_t) blk.extra[e] * K;
        REALTYPE R1[15], R2[15];
        hierarchicalBlockTableRows<REALTYPE, Sc>(T1, blk, R1);
        hierarchicalBlockTableRows<REALTYPE, Sc>(T2, blk, R2);
        if (which == 0)
            hierarchicalBlockPair<REALTYPE, Sc, Agg, Extra, Scaled>(B1, c1, B2, c2, E, invScale.data(), 0, K);
        else if (which == 1)
            hierarchicalBlockColPair<REALTYPE, Sc, Agg, Extra, Scaled>(B1, c1, st2.data(), R2, E, invScale.data(), 0, K);
        else
            hierarchicalBlockColColPair<REALTYPE, Sc, Agg, Extra, Scaled>(st1.data(), R1, st2.data(), R2, E,
                                                                         invScale.data(), 0, K);
        for (int x = 0; x < S; x++) {
            bool written = false;
            for (int i = 0; i < Sc; i++)
                written = written || blk.row[i] == x;
            for (int e = 0; e < Extra; e++)
                written = written || blk.extra[e] == x;
            const REALTYPE* g = got.data() + (size_t) x * K;
            if (written)
                worst = std::max(worst, maxRel(g, ref.data() + (size_t) x * K, K));
            else
                for (int k = 0; k < K; k++)
                    extrasUntouched = extrasUntouched && std::isnan(g[k]);
        }
    }
    const double tol = sizeof(REALTYPE) == 8 ? 1e-14 : 2e-6;
    const char* kernels[3] = { "BlockPair", "BlockColPair", "BlockColColPair" };
    char buf[200];
    snprintf(buf, sizeof(buf), "%s %s kind %d %s %s, %s", label, kernels[which], Kind, Scaled ? "scaled" : "unscaled",
             Broadcast ? "broadcast" : "compact", u.name.c_str());
    report(buf, worst <= tol && extrasUntouched,
           "max relative difference " + std::to_string(worst) + (extrasUntouched ? "" : ", unwritten row touched"));
}

template <typename REALTYPE, int Kind>
static void kernelKind(const char* label, const Upload& u, std::mt19937_64& rng) {
    for (int which = 0; which < 3; which++) {
        kernelCase<REALTYPE, Kind, false, true>(label, u, which, rng);
        kernelCase<REALTYPE, Kind, true, true>(label, u, which, rng);
        kernelCase<REALTYPE, Kind, false, false>(label, u, which, rng);
        kernelCase<REALTYPE, Kind, true, false>(label, u, which, rng);
    }
}

int main() {
    const std::vector<Upload> uploads = allUploads();
    for (const Upload& u : uploads)
        descriptorCase(u);
    runsCase();
    std::mt19937_64 rng(20261118);
    for (size_t i = 0; i < uploads.size(); i += 7) {
        const Upload& u = uploads[i];
        kernelKind<double, 1>("double", u, rng);
        kernelKind<double, 2>("double", u, rng);
        kernelKind<double, 3>("double", u, rng);
        kernelKind<float, 1>("single", u, rng);
        kernelKind<float, 2>("single", u, rng);
        kernelKind<float, 3>("single", u, rng);
    }
    printf("%d of %d checks passed\n", checks - failures, checks);
    return failures == 0 ? 0 : 1;
}
