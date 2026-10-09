// TODO.md plan step C14, offline: TinyGPUPool.h's free list under both vendors' pool allocators (nvd_pool_alloc with a
// TGPoolFree, AMDRuntime::alloc and release), on pools placed where the GPUs put them:
//   1. no frees: every address the bump allocator gave before C14 (its formulas, copied here), for random sizes until full;
//   2. random allocations and frees: each placement is the lowest free range that holds the block at its alignment, else the
//      bump allocator's, and a refusal means neither holds it; the pool below its fill level tiled exactly by the blocks
//      (aligned), the free ranges (coalesced, none ending at the fill level) and the gaps the bump allocator's alignment left,
//      so nothing overlaps and nothing is lost; the allocator's record of its blocks the test's; a free of an address no
//      allocation returned, or of one freed already, changes nothing; with everything freed, nothing is left allocated;
//   3. instance cycles (programs that stay, NV's per-instance programs, buffers; all of an instance freed at its end): every
//      cycle gets the first cycle's addresses, and the pool's fill level never passes the first cycle's.
// One PASS or FAIL line per check; exit 0 only if all pass.
#include "libhmsbeagle/GPU/TinyGPUNVProgram.h"
#include "libhmsbeagle/GPU/TinyGPUAMDRuntime.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <random>
#include <string>
#include <vector>

using namespace tinygpu_device;

static int fails = 0;
static void check(bool ok, const std::string& what) {
    printf("%s %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++fails;
}
static uint64_t up(uint64_t x, uint64_t a) { return (x + a - 1) / a * a; }

// One vendor's pool: the allocator under test, and its rounding and alignment, as the test predicts placements from them
struct Pool {
    virtual ~Pool() {}
    virtual uint64_t alloc(uint64_t n) = 0;   // 0: refused
    virtual void release(uint64_t va) = 0;
    virtual void shape(uint64_t n, uint64_t& size, uint64_t& align) const = 0;   // align: of base() + offset
    virtual uint64_t base() const = 0;   // what an offset's alignment is reckoned from
    virtual uint64_t va() const = 0;     // the pool's first address
    virtual uint64_t size() const = 0;
    virtual uint64_t& pos() = 0;
    virtual TGPoolFree& freed() = 0;
};
struct NVPool : Pool {
    NVDBuffer pool; uint64_t at = 0; TGPoolFree f;
    NVPool(uint64_t va, uint64_t n) { pool.va = va; pool.size = n; }
    uint64_t alloc(uint64_t n) override { return nvd_pool_alloc(pool, at, n, &f); }
    void release(uint64_t va) override { f.release(va - pool.va, at); }   // NvFreeMemory's
    void shape(uint64_t n, uint64_t& s, uint64_t& a) const override {
        s = up(std::max<uint64_t>(n, 1), n >= (8ull << 20) ? (2ull << 20) : 0x1000);
        a = std::max<uint64_t>(0x1000, 1ull << (63 - __builtin_clzll(s)));
    }
    uint64_t base() const override { return pool.va; }
    uint64_t va() const override { return pool.va; }
    uint64_t size() const override { return pool.size; }
    uint64_t& pos() override { return at; }
    TGPoolFree& freed() override { return f; }
};
struct AMDPool : Pool {
    AMDRuntime rt;
    AMDPool(uint64_t va, uint64_t n) { rt.h.pool_va = va; rt.h.pool_size = n; }
    uint64_t alloc(uint64_t n) override { uint64_t va = 0; return rt.alloc(n, va) ? va : 0; }
    void release(uint64_t va) override { rt.release(va); }
    void shape(uint64_t n, uint64_t& s, uint64_t& a) const override {
        a = n >= (8ull << 20) ? (2ull << 20) : 0x1000;
        s = up(n, a);
    }
    uint64_t base() const override { return 0; }   // AMD aligns the pool offset (its pool starts on a 2 MiB boundary anyway)
    uint64_t va() const override { return rt.h.pool_va; }
    uint64_t size() const override { return rt.h.pool_size; }
    uint64_t& pos() override { return rt.pool_used; }
    TGPoolFree& freed() override { return rt.pool_free; }
};
// the bump allocators before C14
static uint64_t old_nv(uint64_t va, uint64_t n, uint64_t& pos, uint64_t size) {
    size = up(std::max<uint64_t>(size, 1), size >= (8ull << 20) ? (2ull << 20) : 0x1000);
    uint64_t align = std::max<uint64_t>(0x1000, 1ull << (63 - __builtin_clzll(size)));
    uint64_t at = up(va + pos, align);
    if (at + size > va + n) return 0;
    pos = at + size - va;
    return at;
}
static uint64_t old_amd(uint64_t va, uint64_t n, uint64_t& used, uint64_t size) {
    const uint64_t page = size >= (8ull << 20) ? (2ull << 20) : 0x1000, sz = up(size, page), at = up(used, page);
    if (sz == 0 || at + sz > n) return 0;
    used = at + sz;
    return va + at;
}

static uint64_t rand_size(std::mt19937_64& g, double max_log) {   // log-uniform from 1 byte, and the 8 MiB rounding edge
    std::uniform_real_distribution<double> u(0, max_log);
    if (g() % 16 == 0) return (8ull << 20) - 1 + g() % 3;
    return std::max<uint64_t>(1, (uint64_t)std::exp(u(g)));
}

// 1.
static void no_frees(const char* name, bool nv, uint64_t va, uint64_t n) {
    std::mt19937_64 g(14);
    NVPool np(va, n); AMDPool ap(va, n);
    Pool& p = nv ? (Pool&)np : (Pool&)ap;
    uint64_t pos = 0, allocs = 0, refused = 0;
    bool same = true;
    for (int i = 0; i < 20000 && same; ++i) {
        uint64_t s = rand_size(g, std::log(64.0 * (1 << 20)));
        uint64_t want = nv ? old_nv(va, n, pos, s) : old_amd(va, n, pos, s), got = p.alloc(s);
        same = got == want && p.pos() == pos;
        if (got) ++allocs; else ++refused;
    }
    check(same && refused > 0 && p.freed().free.empty(), std::string(name) + ": with no frees, " + std::to_string(allocs) +
          " allocations and " + std::to_string(refused) + " refusals, each where the allocator before C14 put it");
}

struct Block { uint64_t va, n, size, align; };

// [0, the fill level) tiled exactly by the blocks, the free ranges (never two in a row, nor one at the end) and the gaps the
// bump allocator's alignment left; the allocator's record of its blocks the test's, each aligned
static bool invariants(Pool& p, const std::vector<Block>& live, const std::map<uint64_t, uint64_t>& gaps, std::string& why) {
    TGPoolFree& f = p.freed();
    std::map<uint64_t, uint64_t> want;
    for (const Block& b : live) {
        if ((p.base() + (b.va - p.va())) % b.align) { why = "a block misaligned"; return false; }
        want[b.va - p.va()] = b.size;
    }
    if (want != f.live) { why = "the allocator's blocks are not the test's"; return false; }
    std::multimap<uint64_t, std::pair<uint64_t, char>> tiles;
    for (const auto& kv : want) tiles.insert({kv.first, {kv.second, 'b'}});
    for (const auto& kv : f.free) tiles.insert({kv.first, {kv.second, 'f'}});
    for (const auto& kv : gaps) tiles.insert({kv.first, {kv.second, 'g'}});
    uint64_t at = 0;
    char prev = 0;
    for (const auto& t : tiles) {
        if (t.first != at || t.second.first == 0) { why = t.first < at ? "two pieces of the pool overlap" : "a piece of the pool lost"; return false; }
        if (t.second.second == 'f' && prev == 'f') { why = "free ranges not coalesced"; return false; }
        at += t.second.first;
        prev = t.second.second;
    }
    if (at != p.pos()) { why = at > p.pos() ? "a piece above the fill level" : "a piece of the pool lost"; return false; }
    if (prev == 'f') { why = "a free range ends at the fill level"; return false; }
    return true;
}

// 2.
static void random_ops(const char* name, bool nv, uint64_t va, uint64_t n) {
    std::mt19937_64 g(1402);
    NVPool np(va, n); AMDPool ap(va, n);
    Pool& p = nv ? (Pool&)np : (Pool&)ap;
    std::vector<Block> live;
    std::vector<uint64_t> gone;          // freed addresses
    std::map<uint64_t, uint64_t> gaps;   // where the bump allocator's alignment skipped part of the pool
    uint64_t allocs = 0, from_free = 0, refused = 0, frees = 0, ignored = 0, peak = 0;
    std::string why;
    bool ok = true;
    for (int i = 0; i < 60000 && ok; ++i) {
        const int op = (int)(g() % 100);
        if (op < 55 || live.empty()) {
            Block b;
            b.n = rand_size(g, std::log(48.0 * (1 << 20)));
            p.shape(b.n, b.size, b.align);
            // the placement predicted: the lowest free range that holds it, else the bump allocator's, else refused
            uint64_t want = 0;
            for (const auto& r : p.freed().free) {
                const uint64_t at = up(p.base() + r.first, b.align) - p.base();
                if (at + b.size <= r.first + r.second) { want = p.va() + at; break; }
            }
            const bool bump = !want;
            if (bump) {
                const uint64_t at = up(p.base() + p.pos(), b.align) - p.base();
                if (b.size && at + b.size <= p.size()) want = p.va() + at;
            }
            const uint64_t pos = p.pos();
            b.va = p.alloc(b.n);
            if (b.va != want) { ok = false; why = "an allocation placed where first fit does not put it"; break; }
            if (b.va) { ++allocs; from_free += !bump; live.push_back(b); }
            else ++refused;
            if (b.va && bump && b.va - p.va() > pos) gaps[pos] = b.va - p.va() - pos;
        } else if (op < 95) {
            const size_t k = g() % live.size();
            p.release(live[k].va);
            gone.push_back(live[k].va);
            live.erase(live.begin() + k);
            ++frees;
        } else {   // an address no live allocation returned: freed already, inside a block, or never one
            const TGPoolFree before = p.freed();
            const uint64_t pos = p.pos();
            uint64_t bad = (op % 3 == 0 && !gone.empty()) ? gone[g() % gone.size()] : live[g() % live.size()].va + 0x1000 * (1 + g() % 4);
            if (op % 3 == 2) bad = p.va() + p.size() + 0x1000;
            bool is_live = false;
            for (const Block& b : live) is_live |= b.va == bad;
            if (is_live) continue;
            p.release(bad);
            if (p.pos() != pos || p.freed().live != before.live || p.freed().free != before.free) { ok = false; why = "a free of an address no live allocation returned changed the pool"; break; }
            ++ignored;
        }
        peak = std::max(peak, p.pos());
        while (!gaps.empty() && std::prev(gaps.end())->first >= p.pos()) gaps.erase(std::prev(gaps.end()));   // a lowered fill level
        ok = invariants(p, live, gaps, why);
    }
    for (const Block& b : live) p.release(b.va);
    live.clear();
    while (!gaps.empty() && std::prev(gaps.end())->first >= p.pos()) gaps.erase(std::prev(gaps.end()));
    if (ok && !p.freed().live.empty()) { ok = false; why = "blocks left after everything was freed"; }
    if (ok) ok = invariants(p, live, gaps, why);
    check(ok && from_free > 1000 && refused > 0, std::string(name) + ": " + std::to_string(allocs) + " allocations (" +
          std::to_string(from_free) + " in freed blocks), " + std::to_string(refused) + " refused, " + std::to_string(frees) +
          " frees, " + std::to_string(ignored) + " ignored, the fill level at most " + std::to_string(peak >> 20) +
          " MiB: first fit and the pool's invariants after each" + (ok ? "" : " -- " + why));
}

// 3. An instance: its programs (NV: at each instance; AMD: a variant's, once, with its scratch), then its buffers
static void cycles(const char* name, bool nv, uint64_t va, uint64_t n) {
    std::mt19937_64 g(31);
    NVPool np(va, n); AMDPool ap(va, n);
    Pool& p = nv ? (Pool&)np : (Pool&)ap;
    std::vector<uint64_t> sizes[2];   // each instance's buffers, as BeagleGPUImpl allocates them
    for (auto& s : sizes) for (int i = 0; i < 40; ++i) s.push_back(rand_size(g, std::log(12.0 * (1 << 20))));
    const uint64_t image[2] = {380952 + 0x1000, 355992 + 0x1000}, scratch[2] = {24ull << 20, 129ull << 20};
    std::vector<uint64_t> first;
    uint64_t first_peak = 0;
    bool same = true, ok = true;
    for (int c = 0; c < 50 && ok; ++c) {
        std::vector<uint64_t> got, mine[2], prog(2, 0);
        for (int k = 0; k < 2 && ok; ++k) {
            if (nv) got.push_back(prog[k] = p.alloc(image[k]));                        // nvdLoadPrograms
            else if (c == 0) { got.push_back(p.alloc(image[k])); got.push_back(p.alloc(scratch[k])); }   // amd_runtime_load_programs
            for (uint64_t s : sizes[k]) { mine[k].push_back(p.alloc(s)); got.push_back(mine[k].back()); }
        }
        for (uint64_t a : got) ok &= a != 0;
        const uint64_t peak = p.pos();
        for (int k = 0; k < 2; ++k) {   // the destructor's frees, in an order of its own, then NvFini's
            std::vector<uint64_t> order = mine[k];
            std::shuffle(order.begin(), order.end(), g);
            for (uint64_t a : order) p.release(a);
            if (nv) p.release(prog[k]);
        }
        if (c == 0) first = got, first_peak = peak;
        else if (!nv && c == 1) first.erase(first.begin(), first.begin() + 2), first.erase(first.begin() + 40, first.begin() + 42);
        if (c > 0) same &= got == first && peak <= first_peak;
    }
    check(ok && same, std::string(name) + ": 50 cycles of two instances (" + std::to_string(2 * 40) + " buffers" +
          (nv ? " and its programs each" : ", after the variants' programs and scratch") + "), every cycle at the first cycle's addresses, the fill level never past its " +
          std::to_string(first_peak >> 20) + " MiB");
}

int main() {
    const uint64_t nv_va = 0x1080000000ull, amd_va = 0x20000c000000ull, n = 1ull << 30;   // where the GPUs put their pools
    no_frees("NV", true, nv_va, n);
    no_frees("AMD", false, amd_va, n);
    random_ops("NV", true, nv_va, n);
    random_ops("AMD", false, amd_va, n);
    cycles("NV", true, nv_va, n);
    cycles("AMD", false, amd_va, n);
    printf("\ntest_c14_pool: %s\n", fails ? (std::to_string(fails) + " FAILED").c_str() : "PASS");
    return fails ? 1 : 0;
}
