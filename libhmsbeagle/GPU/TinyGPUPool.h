/*
 * TinyGPUPool.h
 *
 * TODO.md plan step C14: GPUInterface::FreeMemory on both vendors. Each runtime carves BEAGLE's buffers, its program images
 * and its scratch from one VRAM pool with a bump allocator (nvd_pool_alloc on NV, AMDRuntime::alloc on AMD), which took
 * nothing back. TGPoolFree adds the frees: it records the blocks allocated and keeps the free ranges below the bump
 * allocator's fill level, address-ordered and coalesced. An allocation tries the free ranges first, the lowest that holds
 * the block at its alignment, and else bumps as before, so a run that frees nothing gets the addresses it always got. A free
 * that leaves the top of the used pool free lowers the fill level instead.
 * There is no wait for the GPU before a free, as tinygrad's HCQAllocator._free has (hcq.py:545): it unmaps the buffer, while
 * the pool stays mapped, and every launch and copy here waits on the GPU for everything submitted before it. So it is enough
 * that the freeing instance's queued launches are submitted first (NvFreeMemory, AmdFreeMemory, and NvFini, which frees an
 * instance's programs): whatever reuses the block reaches it only after them.
 */

#ifndef LIBHMSBEAGLE_GPU_TINYGPUPOOL_H
#define LIBHMSBEAGLE_GPU_TINYGPUPOOL_H

#include <cstdint>
#include <iterator>
#include <map>

namespace tinygpu_device {

struct TGPoolFree {
    std::map<uint64_t, uint64_t> live, free;   // a block's or a free range's offset in the pool -> its size

    // The lowest free range that holds size bytes at an offset off with base + off a multiple of align: carved, what is left
    // on either side stays free, and the block is recorded. False if none holds it.
    bool take(uint64_t base, uint64_t size, uint64_t align, uint64_t& off) {
        for (auto it = free.begin(); it != free.end(); ++it) {
            const uint64_t start = it->first, end = start + it->second, at = (base + start + align - 1) / align * align - base;
            if (at + size > end) continue;
            free.erase(it);
            if (at > start) free[start] = at - start;
            if (at + size < end) free[at + size] = end - at - size;
            live[off = at] = size;
            return true;
        }
        return false;
    }

    // Frees the block at off, merged with the free ranges either side of it; when that range ends at pos (the bump allocator's
    // fill level), pos drops to its start instead. False, and nothing freed, if no block starts at off.
    bool release(uint64_t off, uint64_t& pos) {
        auto b = live.find(off);
        if (b == live.end()) return false;
        uint64_t start = off, end = off + b->second;
        live.erase(b);
        auto next = free.lower_bound(start);
        if (next != free.end() && next->first == end) { end += next->second; next = free.erase(next); }
        if (next != free.begin() && std::prev(next)->first + std::prev(next)->second == start) {
            start = std::prev(next)->first;
            free.erase(std::prev(next));
        }
        if (end == pos) pos = start; else free[start] = end - start;
        return true;
    }

    uint64_t bytes() const {   // the free ranges' total
        uint64_t n = 0;
        for (const auto& r : free) n += r.second;
        return n;
    }
};

} // namespace tinygpu_device

#endif // LIBHMSBEAGLE_GPU_TINYGPUPOOL_H
