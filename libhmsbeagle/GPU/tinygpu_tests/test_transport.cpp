// TODO.md plan step C3: TinyGPUTransport.h against TinyGPU.app's real server.c on an IOKit stub (server_stub.c), run by
// test_c3_transport.sh. Prints one line per check, "PASS <what>" or "FAIL <what>: <detail>", and exits 1 on any FAIL.
//   test_transport scenario <server pid>   the protocol, the client limits, error replies, sysmem, and a lost server
//   test_transport paths                   tinygrad's temp() for the socket and lock paths
//   test_transport app                     the TinyGPU.app check (BEAGLE_TINYGPU_APP names the bundle)
//   test_transport open [hold seconds]     open() alone: its result, then optionally a hold (the lock test)
#include "libhmsbeagle/GPU/TinyGPUTransport.h"
#include <csignal>
#include <vector>
using namespace tinygpu_device;

static int g_fail = 0;
static void check(bool ok, const char* what, const std::string& detail = "") {
    printf("%s %s%s%s\n", ok ? "PASS" : "FAIL", what, ok || detail.empty() ? "" : ": ", ok ? "" : detail.c_str());
    if (!ok) g_fail = 1;
}

static bool cfg_ok(TGTransport& t) {   // the stream is still in step: a round trip answers as it should
    uint64_t v = 0;
    std::string e;
    return t.read_config(0, 4, v, e) && v == 0x288210de;
}

static int scenario(pid_t server) {
    TGTransport t;
    std::string e = t.open(), e2;
    check(e.empty(), "open", e);
    if (!e.empty()) return 1;
    check(t.open() == "a TinyGPU.app connection is already open in this process", "a second open of the same transport is refused");
    uint64_t v = 0, addr = 0, size0 = 0, size1 = 0;
    check(t.read_config(0, 4, v, e) && v == 0x288210de, "CFG_READ answers 10de:2882", e);
    check(t.write_config(4, 0x6, 2, e) && t.write_config_flush(4, 0x7, 2, e) && cfg_ok(t), "CFG_WRITE and write_config_flush", e);
    check(t.bar_info(0, addr, size0, e) && size0 == (16 << 20), "MAP_BAR 0: 16 MiB", e);
    check(t.bar_info(1, addr, size1, e) && size1 == (64 << 20), "MAP_BAR 1: 64 MiB", e);
    check(t.resize_bar(1, e) && cfg_ok(t), "RESIZE_BAR", e);

    std::vector<uint8_t> out(4096), in(4096);
    for (size_t i = 0; i < out.size(); ++i) out[i] = (uint8_t)(i * 7 + 3);
    check(t.bulk_write(1, 0x1000, out.data(), out.size(), e) && t.bulk_read(1, 0x1000, in.data(), in.size(), e) && in == out,
          "MMIO_WRITE then MMIO_READ round-trips 4 KiB", e);
    TGWrite frame[3] = {{1, 0x3000, out.data(), 8}, {1, 0x3010, out.data() + 8, 4}, {0, 0x10, out.data() + 12, 4}};
    uint8_t back[16] = {};
    check(t.bulk_write_frame(frame, 3, e) && t.bulk_read(1, 0x3000, back, 8, e) && !memcmp(back, out.data(), 8) &&
          t.bulk_read(0, 0x10, back, 4, e) && !memcmp(back, out.data() + 12, 4), "three posted writes in one frame", e);

    // the client's limits: each refusal sends nothing, leaves the connection usable and the stream in step
    struct Refusal { const char* what; bool (*run)(TGTransport&, std::string&); } refusals[] = {
        {"MMIO on a BAR the session never mapped (TinyGPU.app would drop it)",
         [](TGTransport& t, std::string& e) { uint32_t x = 1; return t.bulk_write(2, 0, &x, 4, e); }},
        {"a write past the BAR's end (TinyGPU.app would drop it)",
         [](TGTransport& t, std::string& e) { uint8_t x[16] = {}; return t.bulk_write(1, (64 << 20) - 8, x, 16, e); }},
        {"a read past the BAR's end",
         [](TGTransport& t, std::string& e) { uint8_t x[16]; return t.bulk_read(1, (64 << 20) - 8, x, 16, e); }},
        {"a write over 64 MB (TinyGPU.app would overflow its buffer)",   // the length is refused before any byte is read
         [](TGTransport& t, std::string& e) { uint8_t x[1] = {}; return t.bulk_write(0, 0, x, (64ull << 20) + 1, e); }},
        {"an offset whose sum with the length wraps",
         [](TGTransport& t, std::string& e) { uint8_t x[16] = {}; return t.bulk_write(1, ~0ull - 4, x, 16, e); }},
        {"one bad write refuses the whole frame",
         [](TGTransport& t, std::string& e) { uint32_t x = 1; TGWrite f[2] = {{1, 0, &x, 4}, {3, 0, &x, 4}}; return t.bulk_write_frame(f, 2, e); }},
    };
    for (auto& r : refusals) {
        e.clear();
        bool sent = r.run(t, e);
        check(!sent && !e.empty() && !t.lost() && cfg_ok(t), r.what, e);
    }
    uint32_t mark = 0;   // the refused frame's first write must not have gone out
    check(t.bulk_read(1, 0, &mark, 4, e) && mark == 0, "nothing of a refused frame reached the BAR", e);

    // an error reply keeps the stream in step: an invalid MMIO_READ, sent raw, is answered with status 1 and no data
    uint64_t r0, r1;
    uint8_t junk[16];
    e.clear();
    bool ok = t.rpc(TGC_MMIO_READ, 64 << 20, 16, 0, 1, r0, r1, e, junk, 16);
    check(!ok && e == "RPC failed: unknown error" && !t.lost() && cfg_ok(t), "an MMIO_READ error reply carries no data", e);
    e.clear();
    ok = t.rpc(TGC_PING, 0, 0, 0, 0, r0, r1, e);
    check(!ok && e == "RPC failed: unknown error" && !t.lost() && cfg_ok(t), "an unimplemented command's error reply", e);

    // sysmem: the fd by SCM_RIGHTS, the segment list, the IOVA whitelist
    TGSysmem m;
    check(t.alloc_sysmem(0x5000, false, m, e) && m.mapped_size == 0x5000 && m.paddrs.size() == 5 &&
          m.paddrs[0] == 0x100000000ull && m.paddrs[4] == 0x100004000ull, "MAP_SYSMEM_FD: 5 pages from 2 segments", e);
    if (m.view) { memset(m.view, 0xab, m.mapped_size); check(m.view[0x4fff] == 0xab, "the sysmem mapping is writable"); }
    bool pages_known = !m.paddrs.empty();
    for (uint64_t p : m.paddrs) pages_known = pages_known && t.iova_known(p, 0x1000);
    check(pages_known && t.iova_known(0x100000000ull, 0x4000) && !t.iova_known(0x100004000ull, 0x1001) &&
          !t.iova_known(0x100003000ull, 0x2000) && !t.iova_known(0xdead000, 1),
          "the IOVA whitelist knows each page given, and no range beyond one segment");
    e.clear();
    check(!t.alloc_sysmem(0x777000, false, m, e) && e == "RPC failed: unknown error" && !t.lost() && cfg_ok(t),
          "a failed MAP_SYSMEM_FD carries no fd, and the stream stays in step", e);
    int allocated = 1;
    for (; allocated < 128 && t.alloc_sysmem(0x4000, false, m, e); ++allocated) {}
    check(allocated == 128, "128 allocations", e);
    e.clear();
    check(!t.alloc_sysmem(0x4000, false, m, e) && e.find("129th") != std::string::npos && !t.lost() && cfg_ok(t),
          "the 129th allocation is refused before it is sent", e);

    // a lost server: the connection is marked lost, and nothing more is sent on it
    kill(server, SIGKILL);
    usleep(300000);
    uint32_t x = 1;
    for (int i = 0; i < 3 && !t.lost(); ++i) { t.bulk_write(1, 0, &x, 4, e); t.read_config(0, 4, v, e); }
    e.clear();
    check(t.lost() && !t.read_config(0, 4, v, e) && e == "the TinyGPU.app connection was lost; nothing more is sent on it",
          "a lost server: the connection is marked lost and refuses everything", e);
    t.close();
    // the killed server skipped its cleanup: its shared memory names (TinyGPU.app's own scheme, which it unlinks before
    // every create, so a live TinyGPU.app is not affected)
    for (int i = 0; i < 128; ++i) { char name[32]; snprintf(name, sizeof(name), "/tinygpu_%d", i); shm_unlink(name); }
    return g_fail;
}

int main(int argc, char** argv) {
    signal(SIGPIPE, SIG_IGN);   // belt and braces: the transport sets SO_NOSIGPIPE
    std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "scenario" && argc > 2) return scenario((pid_t)atoi(argv[2]));
    if (mode == "paths") {
        struct { const char* tmpdir; const char* want; } cases[] = {
            {"/a/b/", "/a/b/tinygpu.sock"}, {"/a/b", "/a/b/tinygpu.sock"}, {"/a/b//", "/a/b/tinygpu.sock"}, {"", "/tmp/tinygpu.sock"},
            {nullptr, "/tmp/tinygpu.sock"}};
        for (auto& c : cases) {
            if (c.tmpdir) setenv("TMPDIR", c.tmpdir, 1); else unsetenv("TMPDIR");
            std::string got = tg_temp_path("tinygpu.sock");
            check(got == c.want, (std::string("temp(\"tinygpu.sock\") with TMPDIR=") + (c.tmpdir ? c.tmpdir : "(unset)")).c_str(), got);
        }
        return g_fail;
    }
    if (mode == "app") { printf("%s\n", TGTransport::check_app().c_str()); return 0; }
    if (mode == "open") {
        TGTransport t;
        std::string e = t.open();
        printf("open: %s\n", e.empty() ? "ok" : e.c_str());
        fflush(stdout);
        if (e.empty() && argc > 2) sleep((unsigned)atoi(argv[2]));
        return e.empty() ? 0 : 1;
    }
    fprintf(stderr, "usage: test_transport scenario <server pid> | paths | app | open [hold seconds]\n");
    return 2;
}
