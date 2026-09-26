/*
 * TinyGPUMemory.h
 *
 * TODO.md plan step C6: tinygrad's GPU memory manager (tinygrad/runtime/support/memory.py at a9830e2b4), vendor-neutral,
 * ported statement by statement: TLSFAllocator (:23-115), PageTableTraverseContext (:124-179) and MemoryManager
 * (:181-290), with BEAGLE's palloc patch (nv_init_helper.py patch 2: palloc zeroes only allocations of at most 64 KiB).
 * The page-table type (NV's is TinyGPUHybridNVMemory.h's) reads and writes its entries where tinygrad's does, so the
 * manager sends TinyGPU.app the requests tinygrad sends, in the same order, reads included. tinygrad's exceptions become
 * TGPyError with the Python type's name; MemoryError is the one valloc recovers from.
 *
 * A manager is either built as tinygrad's constructor builds it, or restored from the state a running tinygrad exported
 * (TLSFAllocator::save's words, the root page table): the fork point at which the plugin takes the memory manager over
 * from the daemon. Not ported: GMMU=0's identity mapping (identity_va and the GMMU branches of valloc and vfree; the plugin
 * refuses GMMU=0). page_tables, which the golden image uses, came with plan step C8.
 */

#ifndef LIBHMSBEAGLE_GPU_TINYGPUMEMORY_H
#define LIBHMSBEAGLE_GPU_TINYGPUMEMORY_H

#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <cstdio>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace tinygpu_device {

// A Python exception as tinygrad's memory manager raises it: type(e).__name__ and str(e).
struct TGPyError : std::runtime_error {
    std::string type;
    TGPyError(std::string t, const std::string& msg) : std::runtime_error(msg), type(std::move(t)) {}
    std::string py() const { return type + ": " + what(); }
};

inline int tg_bit_length(uint64_t x) { return x ? 64 - __builtin_clzll(x) : 0; }   // int.bit_length()
inline uint64_t tg_round_up(uint64_t num, uint64_t amt) { return (num + amt - 1) / amt * amt; }   // helpers.round_up
inline std::string tg_hex(unsigned __int128 x) {   // f"{x:#x}"
    if (x == 0) return "0x0";
    std::string s;
    for (; x; x >>= 4) s.insert(s.begin(), "0123456789abcdef"[(unsigned)(x & 0xf)]);
    return "0x" + s;
}

// memory.py:23-115. blocks maps a block's start (relative to base) to (size, next, prev, is_free); storage[lv1][lv2] lists
// the free blocks of a bucket, oldest first, and alloc takes the first.
class TLSFAllocator {
public:
    static constexpr uint64_t kNone = ~0ull;   // Python's None: the first block's prev (and the next of a size-0 allocator's)
    struct Block { uint64_t size, next, prev; bool free; };

    uint64_t size, base, block_size;
    int l2_cnt;
    std::vector<std::vector<std::vector<uint64_t>>> storage;
    std::vector<uint64_t> lv1_entries;
    std::map<uint64_t, Block> blocks;

    TLSFAllocator(uint64_t size_ = 0, uint64_t base_ = 0, uint64_t block_size_ = 16, uint64_t lv2_cnt = 16)
        : size(size_), base(base_), block_size(block_size_), l2_cnt(tg_bit_length(lv2_cnt)) {
        storage.assign(tg_bit_length(size) + 1, std::vector<std::vector<uint64_t>>(1ull << l2_cnt));
        lv1_entries.assign(storage.size(), 0);
        blocks[0] = {size, kNone, kNone, true};
        if (size > 0) insert_block(0, size);
    }

    int lv1(uint64_t sz) const { return tg_bit_length(sz); }
    uint64_t lv2(uint64_t sz) const {
        int bl = tg_bit_length(sz);
        return (sz - (1ull << (bl - 1))) / (1ull << std::max(0, bl - l2_cnt));
    }

    uint64_t alloc(uint64_t req_size, uint64_t align = 1) {
        req_size = std::max(block_size, req_size);   // at least block size
        uint64_t sz = std::max(block_size, req_size + align - 1);
        // Round up the allocation size to the next bucket, so any entry there can fit the requested size.
        sz = tg_round_up(sz, 1ull << (tg_bit_length(sz) - l2_cnt));
        // Search for the smallest block that can fit the requested size. Start with its bucket and go up until any block is found.
        for (size_t l1 = lv1(sz); l1 < storage.size(); ++l1) {
            if (lv1_entries[l1] == 0) continue;
            for (uint64_t l2 = l1 == (size_t)tg_bit_length(sz) ? lv2(sz) : 0; l2 < (1ull << l2_cnt); ++l2) {
                if (!storage[l1][l2].empty()) {
                    uint64_t start = storage[l1][l2][0], nsize = blocks.at(start).size;
                    if (!(nsize >= sz)) throw TGPyError("AssertionError", "block must be larger");
                    // If request contains alignment, split the block into two parts.
                    uint64_t new_start = tg_round_up(start, align);
                    if (new_start != start) {
                        split_block(start, nsize, new_start - start);
                        start = new_start;
                        nsize = blocks.at(new_start).size;
                    }
                    // If the block is larger than the requested size, split it into two parts.
                    if (nsize > req_size) split_block(start, nsize, req_size);
                    remove_block(start, req_size);   // Mark the block as allocated.
                    return start + base;
                }
            }
        }
        throw TGPyError("MemoryError", "Can't allocate " + std::to_string(req_size) + " bytes");
    }

    void free(uint64_t start) {
        auto it = blocks.find(start - base);
        if (it == blocks.end()) throw TGPyError("KeyError", std::to_string(start - base));
        insert_block(start - base, it->second.size);
        merge_block(start - base);
    }

    // The state as a list of words: size, base, block_size, l2_cnt; the blocks by start (start, size, next, prev, free);
    // the non-empty buckets (lv1, lv2, count, starts oldest first); lv1_entries. kNone stands for None. The daemon exports
    // tinygrad's allocators in this form (nv_dispatch_daemon.py _tlsf_save).
    std::vector<uint64_t> save() const {
        std::vector<uint64_t> w = {size, base, block_size, (uint64_t)l2_cnt, blocks.size()};
        for (auto& [start, b] : blocks) w.insert(w.end(), {start, b.size, b.next, b.prev, (uint64_t)b.free});
        size_t nb = w.size();
        w.push_back(0);
        for (size_t l1 = 0; l1 < storage.size(); ++l1)
            for (size_t l2 = 0; l2 < storage[l1].size(); ++l2)
                if (!storage[l1][l2].empty()) {
                    w.insert(w.end(), {l1, l2, storage[l1][l2].size()});
                    w.insert(w.end(), storage[l1][l2].begin(), storage[l1][l2].end());
                    ++w[nb];
                }
        w.push_back(lv1_entries.size());
        w.insert(w.end(), lv1_entries.begin(), lv1_entries.end());
        return w;
    }
    // The inverse of save; false if the words are not a whole state.
    bool restore(const std::vector<uint64_t>& w) {
        size_t i = 0;
        auto take = [&](uint64_t& v) { if (i >= w.size()) return false; v = w[i++]; return true; };
        uint64_t l2c = 0, n = 0;
        if (!take(size) || !take(base) || !take(block_size) || !take(l2c) || !take(n) || l2c > 16) return false;
        l2_cnt = (int)l2c;
        blocks.clear();
        for (uint64_t k = 0; k < n; ++k) {
            uint64_t start, bsz, nxt, prev, fr;
            if (!take(start) || !take(bsz) || !take(nxt) || !take(prev) || !take(fr)) return false;
            blocks[start] = {bsz, nxt, prev, fr != 0};
        }
        storage.assign(tg_bit_length(size) + 1, std::vector<std::vector<uint64_t>>(1ull << l2_cnt));
        if (!take(n)) return false;
        for (uint64_t k = 0; k < n; ++k) {
            uint64_t l1, l2, cnt;
            if (!take(l1) || !take(l2) || !take(cnt) || l1 >= storage.size() || l2 >= storage[l1].size() || cnt > w.size() - i) return false;
            storage[l1][l2].assign(w.begin() + i, w.begin() + i + cnt);
            i += cnt;
        }
        if (!take(n) || n != storage.size() || n > w.size() - i) return false;
        lv1_entries.assign(w.begin() + i, w.begin() + i + n);
        return i + n == w.size();
    }

private:
    void insert_block(uint64_t start, uint64_t sz, std::optional<uint64_t> prev = std::nullopt) {
        uint64_t p = prev ? *prev : blocks.at(start).prev;
        storage[lv1(sz)][lv2(sz)].push_back(start);
        lv1_entries[lv1(sz)] += 1;
        blocks[start] = {sz, start + sz, p, true};
    }
    void remove_block(uint64_t start, uint64_t sz) {
        uint64_t p = blocks.at(start).prev;
        auto& b = storage[lv1(sz)][lv2(sz)];
        auto it = std::find(b.begin(), b.end(), start);
        if (it == b.end()) throw TGPyError("ValueError", "list.remove(x): x not in list");
        b.erase(it);
        lv1_entries[lv1(sz)] -= 1;
        blocks[start] = {sz, start + sz, p, false};
    }
    void split_block(uint64_t start, uint64_t sz, uint64_t new_size) {
        uint64_t nxt = blocks.at(start).next;
        if (!blocks.at(start).free) throw TGPyError("AssertionError", "block must be free");
        remove_block(start, sz);
        insert_block(start, new_size);
        insert_block(start + new_size, sz - new_size, start);
        auto it = blocks.find(nxt);
        if (it != blocks.end()) it->second.prev = start + new_size;
    }
    void merge_right(uint64_t start) {
        Block b = blocks.at(start);
        uint64_t sz = b.size, nxt = b.next;
        if (!b.free) throw TGPyError("AssertionError", "block must be free");
        while (b.free && blocks.count(nxt)) {
            Block blk = blocks.at(nxt);
            if (!blk.free) break;
            remove_block(start, sz);
            remove_block(nxt, blk.size);
            insert_block(start, sz = sz + blk.size);
            if (blocks.at(start).next != blk.next) throw TGPyError("AssertionError", "");
            uint64_t popped = nxt;
            nxt = blocks.at(popped).next;
            blocks.erase(popped);
        }
        auto it = blocks.find(nxt);
        if (it != blocks.end()) it->second.prev = start;
    }
    void merge_block(uint64_t start) {
        // Go left while blocks are free. Then merge all them right.
        for (uint64_t x; (x = blocks.at(start).prev) != kNone && blocks.at(x).free;) start = x;
        merge_right(start);
    }
};

enum class TGAddrSpace { PHYS = 1, SYS, PEER };   // memory.py:119

struct TGVirtMapping {   // memory.py:121-122
    uint64_t va_addr = 0, size = 0;
    std::vector<std::pair<uint64_t, uint64_t>> paddrs;
    TGAddrSpace aspace = TGAddrSpace::PHYS;
    bool uncached = false, snooped = false;
};

// memory.py:124-179, over MM, a TGMemoryManager<PT>. PT is a value handle for one page table (tinygrad's page-table entry
// class): PT(dev, paddr, lv); lv, paddr; valid, is_page, entry, address, set_entry and supports_huge_page as tinygrad's. The
// root is the stack's first table (tinygrad compares with mm.root_page_table by identity).
template <class MM> class TGPageTableTraverseContext {
public:
    using PT = typename MM::PT;
    struct Level { PT pt; uint64_t pte_idx, pte_covers; };

    MM& mm;
    uint64_t vaddr;
    bool create_pts, free_pts, inspect, boot;
    std::vector<Level> pt_stack;

    TGPageTableTraverseContext(MM& mm_, const PT& pt, uint64_t vaddr_, bool create_pts_ = false, bool free_pts_ = false,
                               bool inspect_ = false, bool boot_ = false)
        : mm(mm_), vaddr(vaddr_ - mm_.va_base), create_pts(create_pts_), free_pts(free_pts_), inspect(inspect_), boot(boot_) {
        pt_stack.push_back({pt, pte_idx(pt, vaddr), pte_size(pt)});
    }

    uint64_t pte_cnt(int lv) const { return mm.pte_cnt[lv]; }
    uint64_t pte_size(const PT& pt) const { return mm.pte_covers[pt.lv]; }
    uint64_t pte_idx(const PT& pt, uint64_t va) const { return (va / pte_size(pt)) % pte_cnt(pt.lv); }

    Level& top() {
        if (pt_stack.empty()) throw TGPyError("IndexError", "list index out of range");
        return pt_stack.back();
    }

    Level level_down() {
        Level cur = top();
        PT& pt = cur.pt;
        if (!pt.valid(cur.pte_idx)) {
            if (!create_pts) throw TGPyError("AssertionError", "Not allowed to create new page table");
            pt.set_entry(cur.pte_idx, mm.palloc(0x1000, 0x1000, true, boot, true), true, false, TGAddrSpace::PHYS, false, 0, true);
        }
        if (pt.is_page(cur.pte_idx))
            throw TGPyError("AssertionError", "Must be table pt=" + tg_hex(pt.paddr) + ", pt.lv=" + std::to_string(pt.lv) + " pte_idx=" +
                            std::to_string(cur.pte_idx) + " pt.entry(pte_idx)=" + tg_hex(pt.entry(cur.pte_idx)));
        PT child(mm.dev, pt.address(cur.pte_idx), pt.lv + 1);
        pt_stack.push_back({child, pte_idx(child, vaddr), pte_size(child)});
        return pt_stack.back();
    }

    bool try_free_pt() {
        Level cur = top();
        if (free_pts && pt_stack.size() > 1) {
            bool all_invalid = true;
            for (uint64_t i = 0; i < pte_cnt(cur.pt.lv); ++i)
                if (cur.pt.valid(i)) { all_invalid = false; break; }
            if (all_invalid) {
                mm.pfree(cur.pt.paddr, true);
                Level parent = pt_stack[pt_stack.size() - 2];
                parent.pt.set_entry(parent.pte_idx, 0x0, false, false, TGAddrSpace::PHYS, false, 0, false);
                return true;
            }
        }
        return false;
    }

    void level_up() {
        while (try_free_pt() || top().pte_idx == pte_cnt(top().pt.lv)) {
            Level popped = top();
            pt_stack.pop_back();
            if (popped.pte_idx == pte_cnt(popped.pt.lv)) top().pte_idx += 1;
        }
    }

    // The generator next(size, paddr, off): visit(off, pt, pte_idx, entries, pte_covers) runs where tinygrad's consumer runs,
    // between the yield and the advance. paddr is nullptr for None. A visit returning bool true leaves the generator there, as a
    // consumer that returns from inside its for loop does (page_tables).
    template <class F> void next(int64_t size, const uint64_t* paddr, uint64_t off, F&& visit) {
        while (size > 0) {
            Level cur = top();
            PT pt = cur.pt;
            uint64_t pte_idx_ = cur.pte_idx, pte_covers = cur.pte_covers;
            // create_pts goes down until the page covers the request.
            // free_pts goes down to the table, it assumses all entries are valid on the range (and validates that)
            // inspect just visits any valid ranges and yields them.
            if (create_pts) {
                if (!paddr) throw TGPyError("AssertionError", "paddr must be provided when allocating new page tables");
                while ((int64_t)pte_covers > size || !pt.supports_huge_page(*paddr + off) || (vaddr & (pte_covers - 1)) != 0) {
                    Level l = level_down();
                    pt = l.pt; pte_idx_ = l.pte_idx; pte_covers = l.pte_covers;
                }
            } else {
                while (!pt.is_page(pte_idx_) && (free_pts || pt.valid(pte_idx_))) {
                    Level l = level_down();
                    pt = l.pt; pte_idx_ = l.pte_idx; pte_covers = l.pte_covers;
                }
            }
            int64_t entries = std::max<int64_t>(std::min<int64_t>(size / (int64_t)pte_covers, (int64_t)(pte_cnt(pt.lv) - pte_idx_)), inspect ? 1 : 0);
            if (!(entries > 0)) throw TGPyError("AssertionError", "Invalid entries size=" + tg_hex(size) + ", pte_covers=" + tg_hex(pte_covers));
            if constexpr (std::is_same_v<std::invoke_result_t<F&, uint64_t, PT&, uint64_t, uint64_t, uint64_t>, bool>) {
                if (visit(off, pt, pte_idx_, (uint64_t)entries, pte_covers)) return;
            } else visit(off, pt, pte_idx_, (uint64_t)entries, pte_covers);
            size -= entries * (int64_t)pte_covers;
            off += entries * pte_covers;
            vaddr += entries * pte_covers;
            top() = {pt, pte_idx_ + entries, pte_covers};
            level_up();
        }
    }
};

// memory.py:181-290 with BEAGLE's palloc patch. Dev: what MemoryManager uses of its device (is_booting, smi_dev, and
// vram_zero(paddr, size) for dev.vram[paddr:paddr+size] = bytes(size)).
template <class PT_> class TGMemoryManager {
public:
    using PT = PT_;
    using Dev = typename PT::Dev;
    using Ctx = TGPageTableTraverseContext<TGMemoryManager>;
    static constexpr uint64_t kPallocZeroLimit = 64 << 10;   // nv_init_helper.py _PALLOC_ZERO_LIMIT

    Dev* dev;
    uint64_t vram_size, va_base, va_bits;
    std::vector<uint64_t> va_shifts, pte_covers, pte_cnt;
    std::vector<std::pair<uint64_t, uint64_t>> palloc_ranges;
    int level_cnt;
    bool reserve_ptable;
    TLSFAllocator boot_allocator, ptable_allocator, pa_allocator;
    TLSFAllocator* va_allocator = nullptr;   // MemoryManager.va_allocator: a class variable, one for all of a vendor's devices
    PT root_page_table;

    // tinygrad's constructor
    TGMemoryManager(Dev* dev_, uint64_t vram_size_, uint64_t boot_size, uint64_t va_bits_, const std::vector<uint64_t>& va_shifts_,
                    uint64_t va_base_, const std::vector<std::pair<uint64_t, uint64_t>>& palloc_ranges_, int first_lv = 0,
                    bool reserve_ptable_ = false)
        : TGMemoryManager(dev_, vram_size_, va_bits_, va_shifts_, va_base_, palloc_ranges_, reserve_ptable_) {
        boot_allocator = TLSFAllocator(boot_size, 0);
        ptable_allocator = TLSFAllocator(reserve_ptable ? tg_round_up(vram_size / 512, 1 << 20) : 0, boot_allocator.size);
        uint64_t off_sz = boot_allocator.size + ptable_allocator.size;
        pa_allocator = TLSFAllocator(vram_size - off_sz, off_sz);
        root_page_table = PT(dev, palloc(0x1000, 0x1000, !dev->smi_dev, true), first_lv);
    }

    // Restored at a fork point: tinygrad's allocators as TLSFAllocator::save wrote them, and its root page table. False (and
    // the manager unusable) if a state is not whole.
    struct Restored { const std::vector<uint64_t>& boot; const std::vector<uint64_t>& ptable; const std::vector<uint64_t>& pa;
                      uint64_t root_paddr; int root_lv; };
    TGMemoryManager(Dev* dev_, uint64_t vram_size_, uint64_t va_bits_, const std::vector<uint64_t>& va_shifts_, uint64_t va_base_,
                    const std::vector<std::pair<uint64_t, uint64_t>>& palloc_ranges_, bool reserve_ptable_, const Restored& r, bool& ok)
        : TGMemoryManager(dev_, vram_size_, va_bits_, va_shifts_, va_base_, palloc_ranges_, reserve_ptable_) {
        ok = boot_allocator.restore(r.boot) && ptable_allocator.restore(r.ptable) && pa_allocator.restore(r.pa);
        root_page_table = PT(dev, r.root_paddr, r.root_lv);
    }

    virtual ~TGMemoryManager() = default;
    virtual void on_range_mapped() {}
    // BEAGLE's, before map_range sends anything: a vendor refuses a mapping here (NV: the IOVA fence)
    virtual void check_mapping(const std::vector<std::pair<uint64_t, uint64_t>>& paddrs, TGAddrSpace aspace) { (void)paddrs; (void)aspace; }

    // _frag_size: the TLB fragment of a range (fragment 0 is 4 KiB, 1 is 8 KiB, ...)
    static int64_t frag_size(uint64_t va, uint64_t sz, bool must_cover = true) {
        uint64_t va_pwr2_div = va > 0 ? (va & (~va + 1)) : (1ull << 63), sz_pwr2_div = sz & (~sz + 1);
        uint64_t sz_pwr2_max = 1ull << (tg_bit_length(sz) - 1);
        return (int64_t)tg_bit_length(must_cover ? std::min(va_pwr2_div, sz_pwr2_div) : std::min(va_pwr2_div, sz_pwr2_max)) - 1 - 12;
    }

    TGVirtMapping map_range(uint64_t vaddr, uint64_t size, const std::vector<std::pair<uint64_t, uint64_t>>& paddrs, TGAddrSpace aspace,
                            bool uncached = false, bool snooped = false, bool boot = false) {
        uint64_t total = 0;
        for (auto& p : paddrs) total += p.second;
        if (size != total)
            throw TGPyError("AssertionError", "Size mismatch size=" + std::to_string(size) + " sum(p[1] for p in paddrs)=" + std::to_string(total));
        check_mapping(paddrs, aspace);
        {
            Ctx ctx(*this, root_page_table, vaddr, false, false, true, boot);
            ctx.next((int64_t)size, nullptr, 0, [&](uint64_t, PT& pt, uint64_t pte_idx, uint64_t pte_cnt_, uint64_t) {
                for (uint64_t pte_off = 0; pte_off < pte_cnt_; ++pte_off)
                    if (pt.valid(pte_idx + pte_off)) throw TGPyError("AssertionError", "PTE already mapped: " + tg_hex(pt.entry(pte_idx + pte_off)));
            });
        }
        Ctx ctx(*this, root_page_table, vaddr, true, false, false, boot);
        for (const auto& chunk : paddrs) {
            const uint64_t paddr = chunk.first;
            ctx.next((int64_t)chunk.second, &paddr, 0, [&](uint64_t off, PT& pt, uint64_t pte_idx, uint64_t pte_cnt_, uint64_t pte_covers_) {
                for (uint64_t pte_off = 0; pte_off < pte_cnt_; ++pte_off)
                    pt.set_entry(pte_idx + pte_off, paddr + off + pte_off * pte_covers_, false, uncached, aspace, snooped,
                                 frag_size(ctx.vaddr + off, pte_cnt_ * pte_covers_), true);
            });
        }
        on_range_mapped();
        return TGVirtMapping{vaddr, size, paddrs, aspace, uncached, snooped};
    }

    void unmap_range(uint64_t vaddr, uint64_t size) {
        Ctx ctx(*this, root_page_table, vaddr, false, true, false, false);
        ctx.next((int64_t)size, nullptr, 0, [&](uint64_t, PT& pt, uint64_t pte_idx, uint64_t pte_cnt_, uint64_t) {
            for (uint64_t pte_id = pte_idx; pte_id < pte_idx + pte_cnt_; ++pte_id) {
                if (!pt.valid(pte_id)) throw TGPyError("AssertionError", "PTE not mapped: " + tg_hex(pt.entry(pte_id)));
                pt.set_entry(pte_id, 0x0, false, false, TGAddrSpace::PHYS, false, 0, false);
            }
        });
    }

    // page_tables (memory.py:204-206): the tables from the root down to the level whose entries cover size at vaddr, created
    // where missing; the generator's state at its first yield, where tinygrad returns
    std::vector<PT> page_tables(uint64_t vaddr, uint64_t size) {
        Ctx ctx(*this, root_page_table, vaddr, true);
        std::vector<PT> out;
        const uint64_t paddr = 0;
        ctx.next((int64_t)size, &paddr, 0, [&](uint64_t, PT&, uint64_t, uint64_t, uint64_t) {
            for (const auto& l : ctx.pt_stack) out.push_back(l.pt);
            return true;
        });
        return out;
    }

    uint64_t alloc_vaddr(uint64_t size, uint64_t align = 0x1000) {
        if (!va_allocator) throw TGPyError("AssertionError", "must be set");
        if (size == 0) throw TGPyError("ValueError", "negative shift count");   // 1 << (0).bit_length() - 1
        return va_allocator->alloc(size, std::max<uint64_t>(1ull << (tg_bit_length(size) - 1), align));
    }

    TGVirtMapping valloc(uint64_t size, uint64_t align = 0x1000, bool uncached = false, bool contiguous = false, bool zero = false) {
        // Alloc physical memory and map it to the virtual address
        size = tg_round_up(size, 0x1000);
        uint64_t va = alloc_vaddr(size, align);
        std::vector<std::pair<uint64_t, uint64_t>> paddrs;
        if (contiguous) paddrs = {{palloc(size, 0x1000, true), size}};
        else {
            // Traverse the PT to find the largest contiguous sizes we need to allocate. Try to allocate the longest segment to reduce TLB pressure.
            size_t nxt_range = 0;
            uint64_t rem_size = size;
            while (rem_size > 0) {
                while (palloc_ranges[nxt_range].first > rem_size) nxt_range += 1;
                uint64_t try_sz = palloc_ranges[nxt_range].first;
                try { paddrs.push_back({palloc(try_sz, palloc_ranges[nxt_range].second, zero), try_sz}); }
                catch (const TGPyError& e) {
                    if (e.type != "MemoryError") throw;
                    // Move to a smaller size and try again.
                    nxt_range += 1;
                    if (nxt_range == palloc_ranges.size()) {
                        for (auto& p : paddrs) pfree(p.first);
                        throw TGPyError("MemoryError", "Failed to allocate memory (OOM). Request size=" + tg_hex(size) + " ((" +   // the tuple's repr
                                        std::to_string(palloc_ranges[nxt_range - 1].first) + ", " + std::to_string(palloc_ranges[nxt_range - 1].second) + "))");
                    }
                    continue;
                }
                rem_size -= palloc_ranges[nxt_range].first;
            }
        }
        return map_range(va, size, paddrs, TGAddrSpace::PHYS, uncached);
    }

    void vfree(const TGVirtMapping& vm) {
        if (!va_allocator) throw TGPyError("AssertionError", "must be set");
        unmap_range(vm.va_addr, vm.size);
        va_allocator->free(vm.va_addr);
        for (auto& p : vm.paddrs) pfree(p.first);
    }

    uint64_t palloc(uint64_t size, uint64_t align = 0x1000, bool zero = true, bool boot = false, bool ptable = false) {
        if (dev->is_booting != boot) throw TGPyError("AssertionError", "During booting, only boot memory can be allocated");
        if (zero && size > kPallocZeroLimit) zero = false;   // BEAGLE's patch
        TLSFAllocator& allocator = boot ? boot_allocator : (reserve_ptable && ptable ? ptable_allocator : pa_allocator);
        uint64_t paddr = allocator.alloc(tg_round_up(size, 0x1000), align);
        if (zero) dev->vram_zero(paddr, size);
        return paddr;
    }

    void pfree(uint64_t paddr, bool ptable = false) { (reserve_ptable && ptable ? ptable_allocator : pa_allocator).free(paddr); }

private:
    TGMemoryManager(Dev* dev_, uint64_t vram_size_, uint64_t va_bits_, const std::vector<uint64_t>& va_shifts_, uint64_t va_base_,
                    const std::vector<std::pair<uint64_t, uint64_t>>& palloc_ranges_, bool reserve_ptable_)
        : dev(dev_), vram_size(vram_size_), va_base(va_base_), va_bits(va_bits_), va_shifts(va_shifts_), palloc_ranges(palloc_ranges_),
          level_cnt((int)va_shifts_.size()), reserve_ptable(reserve_ptable_) {
        std::vector<uint64_t> lvl_msb = va_shifts;
        lvl_msb.push_back(va_bits + 1);
        for (size_t i = va_shifts.size(); i-- > 0;) {
            pte_covers.push_back(1ull << va_shifts[i]);
            pte_cnt.push_back(1ull << (lvl_msb[i + 1] - lvl_msb[i]));
        }
    }
};

} // namespace tinygpu_device

#endif // LIBHMSBEAGLE_GPU_TINYGPUMEMORY_H
