/*
 * TinyGPUFirmware.h
 *
 * TODO.md plan step C4: finds the NVIDIA firmware a C++ boot needs (TinyGPUFirmwareManifest.h) and checks it the way
 * tinygrad's fetch does before it uses a cached file (helpers.py:469-474: the file exists and its sha256 matches). Three
 * places, in order:
 *   1. $BEAGLE_TINYGPU_FW/<subdir>/<name>
 *   2. <the directory holding this code>/../share/beagle/firmware/<subdir>/<name> (an installed plugin's share/)
 *   3. BEAGLE's download cache, ${XDG_CACHE_HOME:-~/Library/Caches}/beagle/firmware/<subdir>/<name>
 * tinygrad's download cache (${XDG_CACHE_HOME:-~/Library/Caches}/tinygrad/downloads/fw/<md5(url)>) is not searched (since
 * 2026-10-05, the user's request). The file is mapped read-only (the GSP image is 63.5 MB) and its SHA-256 computed with
 * CommonCrypto before it is returned. When no place has it, it is downloaded (since 2026-10-01, the user's request; plan
 * decision 5 had kept BEAGLE off the network) from the manifest's pinned linux-firmware URL, as fetch_fw would, by
 * /usr/bin/curl into a temporary file in BEAGLE's cache, which is renamed into place only once its SHA-256 matches. The NV boot fetches its chip
 * family's files this way before it writes anything to the GPU (GPUInterfaceTinyGPUHybridNV.cpp nv_fw_prefetch).
 * BEAGLE_TINYGPU_NO_DOWNLOAD=1 turns downloading off (the offline tests set it), and BEAGLE_TINYGPU_FW_BASE_URL replaces
 * the linux-firmware URL (a mirror, or a file:// copy for the tests). Without the file, the error says what each place
 * held and how to fetch it by hand (curl and shasum, or tinygpu_fetch_firmware.sh). fetch_fw's
 * /lib/firmware/<path>/<name>.zst branch (helpers.py:506-508, Linux with Python 3.14) has no macOS counterpart.
 */

#ifndef LIBHMSBEAGLE_GPU_TINYGPUFIRMWARE_H
#define LIBHMSBEAGLE_GPU_TINYGPUFIRMWARE_H

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include <CommonCrypto/CommonDigest.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <spawn.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

extern char** environ;

#include "libhmsbeagle/GPU/TinyGPUFirmwareManifest.h"

namespace tinygpu_device {

// A firmware file, mapped read-only for as long as this lives.
class TGFirmwareFile {
public:
    TGFirmwareFile() = default;
    TGFirmwareFile(const TGFirmwareFile&) = delete;
    TGFirmwareFile& operator=(const TGFirmwareFile&) = delete;
    TGFirmwareFile(TGFirmwareFile&& o) noexcept { *this = std::move(o); }
    TGFirmwareFile& operator=(TGFirmwareFile&& o) noexcept {
        if (this != &o) {
            reset();
            std::swap(data_, o.data_);
            std::swap(size_, o.size_);
            path_ = std::move(o.path_);
        }
        return *this;
    }
    ~TGFirmwareFile() { reset(); }
    const uint8_t* data() const { return data_; }
    size_t size() const { return size_; }
    const std::string& path() const { return path_; }
    void reset() {
        if (data_ && size_) munmap(const_cast<uint8_t*>(data_), size_);
        data_ = nullptr;
        size_ = 0;
        path_.clear();
    }
    // Maps path; false with why when it cannot (why is "not found" when it does not exist).
    bool map(const std::string& path, std::string& why) {
        reset();
        int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) { why = errno == ENOENT ? "not found" : std::string("cannot open: ") + strerror(errno); return false; }
        struct stat st;
        if (fstat(fd, &st) != 0 || !S_ISREG(st.st_mode)) { close(fd); why = "not a file"; return false; }
        void* m = st.st_size ? mmap(nullptr, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0) : nullptr;
        close(fd);
        if (m == MAP_FAILED) { why = std::string("cannot map: ") + strerror(errno); return false; }
        data_ = (const uint8_t*)m;
        size_ = (size_t)st.st_size;
        path_ = path;
        return true;
    }
private:
    const uint8_t* data_ = nullptr;
    size_t size_ = 0;
    std::string path_;
};

inline std::string tg_sha256_hex(const uint8_t* p, size_t n) {
    CC_SHA256_CTX ctx;
    CC_SHA256_Init(&ctx);
    for (size_t off = 0; off < n;) {   // CC_LONG is 32 bits
        CC_LONG len = (CC_LONG)std::min<size_t>(n - off, 1u << 30);
        CC_SHA256_Update(&ctx, p + off, len);
        off += len;
    }
    unsigned char md[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256_Final(md, &ctx);
    char hex[2 * CC_SHA256_DIGEST_LENGTH + 1];
    for (int i = 0; i < CC_SHA256_DIGEST_LENGTH; ++i) snprintf(hex + 2 * i, 3, "%02x", md[i]);
    return hex;
}

// The manifest entry a chip family's boot fetches for a role, or nullptr.
// whose firmware an entry is, for the messages: AMD's live under amdgpu/ (TinyGPUAMDBootTables.h's am::fw, plan step A2d)
inline const char* tg_fw_vendor(const nvfw::TGFirmware& fw) { return strncmp(fw.subdir, "amdgpu", 6) == 0 ? "AMD" : "NVIDIA"; }

inline const nvfw::TGFirmware* tg_fw_entry(const std::string& chip, const std::string& role) {
    for (const nvfw::TGFirmware& f : nvfw::kFirmware)
        if (chip == f.chip && role == f.role) return &f;
    return nullptr;
}

inline std::string tg_fw_url(const nvfw::TGFirmware& fw) {   // fetch_fw's URL (helpers.py:509), or BEAGLE_TINYGPU_FW_BASE_URL's
    const char* base = getenv("BEAGLE_TINYGPU_FW_BASE_URL");
    return std::string(base && base[0] ? base : nvfw::kLinuxFirmware) + "/" + fw.subdir + "/" + fw.name;
}

inline std::string tg_fw_cache_root() {   // ${XDG_CACHE_HOME:-~/Library/Caches}
    const char* xdg = getenv("XDG_CACHE_HOME");
    const char* home = getenv("HOME");
    return xdg ? xdg : std::string(home ? home : "") + "/Library/Caches";
}

// BEAGLE's own download cache
inline std::string tg_fw_beagle_cache(const nvfw::TGFirmware& fw) {
    return tg_fw_cache_root() + "/beagle/firmware/" + fw.subdir + "/" + fw.name;
}

// The places searched, in order, as (label, path); a place with no path (BEAGLE_TINYGPU_FW unset) is listed empty.
inline std::vector<std::pair<std::string, std::string>> tg_fw_candidates(const nvfw::TGFirmware& fw) {
    std::vector<std::pair<std::string, std::string>> out;
    const std::string rel = std::string(fw.subdir) + "/" + fw.name;
    const char* env = getenv("BEAGLE_TINYGPU_FW");
    out.push_back({"$BEAGLE_TINYGPU_FW", env && env[0] ? std::string(env) + "/" + rel : ""});
    static const int anchor = 0;   // an address inside the image that holds this code: the plugin, or a test program
    Dl_info info;
    std::string share;
    if (dladdr(&anchor, &info) && info.dli_fname) {
        std::string dir = info.dli_fname;
        dir = dir.substr(0, dir.find_last_of('/') == std::string::npos ? 0 : dir.find_last_of('/'));
        share = (dir.empty() ? "." : dir) + "/../share/beagle/firmware/" + rel;
    }
    out.push_back({"share/beagle/firmware", share});
    out.push_back({"BEAGLE's download cache", tg_fw_beagle_cache(fw)});
    return out;
}

// Downloads fw into BEAGLE's cache: /usr/bin/curl writes a temporary file beside the destination (curl -f: an HTTP error
// writes nothing usable; at most 256 MiB; it gives up if under 1 KiB/s for a minute), whose SHA-256 must match before it
// is renamed into place. "" or why not.
inline std::string tg_fw_download(const nvfw::TGFirmware& fw) {
    const std::string dest = tg_fw_beagle_cache(fw), url = tg_fw_url(fw);
    std::string dir = dest.substr(0, dest.find_last_of('/'));
    for (size_t p = 1; p != std::string::npos; p = dir.find('/', p + 1)) {   // mkdir -p
        std::string d = dir.substr(0, dir.find('/', p + 1));
        if (mkdir(d.c_str(), 0755) != 0 && errno != EEXIST) return "cannot create " + d + ": " + strerror(errno);
    }
    const std::string tmp = dest + ".part." + std::to_string((long)getpid());
    fprintf(stderr, "TinyGPU: downloading %s firmware %s/%s from %s ...\n", tg_fw_vendor(fw), fw.subdir, fw.name, url.c_str());
    fflush(stderr);
    const char* argv[] = {"/usr/bin/curl", "-fsSL", "--proto", "=https,file", "--connect-timeout", "30", "--speed-limit", "1024",
                          "--speed-time", "60", "--max-filesize", "268435456", "-o", tmp.c_str(), url.c_str(), nullptr};
    posix_spawn_file_actions_t fa;   // as the crash guard's spawn: no fd of the host but stdin and stdout on /dev/null, and stderr
    posix_spawnattr_t at;
    posix_spawn_file_actions_init(&fa);
    posix_spawnattr_init(&at);
    posix_spawn_file_actions_addopen(&fa, 0, "/dev/null", O_RDONLY, 0);
    posix_spawn_file_actions_addopen(&fa, 1, "/dev/null", O_WRONLY, 0);
    posix_spawn_file_actions_addinherit_np(&fa, 2);
    posix_spawnattr_setflags(&at, POSIX_SPAWN_CLOEXEC_DEFAULT);
    pid_t pid = 0;
    int rc = posix_spawn(&pid, argv[0], &fa, &at, const_cast<char* const*>(argv), environ);
    posix_spawn_file_actions_destroy(&fa);
    posix_spawnattr_destroy(&at);
    if (rc != 0) return std::string("cannot run /usr/bin/curl: ") + strerror(rc);
    int st = 0;
    while (waitpid(pid, &st, 0) < 0 && errno == EINTR) {}
    std::string why;
    if (!WIFEXITED(st) || WEXITSTATUS(st) != 0) why = "curl failed (exit " + std::to_string(WIFEXITED(st) ? WEXITSTATUS(st) : -1) + ") for " + url;
    if (why.empty()) {
        TGFirmwareFile f;
        if (!f.map(tmp, why)) why = "the download: " + why;
        else {
            const std::string sha = tg_sha256_hex(f.data(), f.size());
            if (sha != fw.sha256) why = "the download from " + url + " is " + std::to_string(f.size()) + " bytes with sha256 " + sha + ", not this file";
        }
    }
    if (why.empty() && rename(tmp.c_str(), dest.c_str()) != 0) why = "cannot rename the download to " + dest + ": " + strerror(errno);
    if (!why.empty()) { unlink(tmp.c_str()); return why; }
    fprintf(stderr, "TinyGPU: downloaded %s/%s into %s (sha256 checked)\n", fw.subdir, fw.name, dest.c_str());
    return "";
}

// Finds fw, maps it and checks its sha256. Returns an empty string on success, otherwise what each place held and how to
// fetch the file.
inline std::string tg_fw_locate(const nvfw::TGFirmware& fw, TGFirmwareFile& out) {
    std::string report;
    for (const auto& c : tg_fw_candidates(fw)) {
        if (c.second.empty()) { report += "  " + c.first + ": not set\n"; continue; }
        std::string why;
        if (out.map(c.second, why)) {
            std::string sha = tg_sha256_hex(out.data(), out.size());
            if (sha == fw.sha256) return "";
            why = std::to_string(out.size()) + " bytes with sha256 " + sha + ", not this file";
            out.reset();
        }
        report += "  " + c.first + " " + c.second + ": " + why + "\n";
    }
    const char* nodl = getenv("BEAGLE_TINYGPU_NO_DOWNLOAD");
    std::string dl = nodl && strcmp(nodl, "0") != 0 ? "not tried (BEAGLE_TINYGPU_NO_DOWNLOAD is set)" : tg_fw_download(fw);
    if (dl.empty()) {
        std::string why;
        if (out.map(tg_fw_beagle_cache(fw), why)) return "";
        dl = "the downloaded file: " + why;
    }
    report += "  the download: " + dl + "\n";
    const std::string dest = tg_fw_beagle_cache(fw), url = tg_fw_url(fw);
    return "TinyGPU: " + std::string(tg_fw_vendor(fw)) + " firmware " + std::string(fw.subdir) + "/" + fw.name + " (sha256 " + fw.sha256 +
           ") is missing or damaged:\n" +
           report + "To fetch it by hand into BEAGLE's cache:\n" +
           "  mkdir -p '" + dest.substr(0, dest.find_last_of('/')) + "' && curl -fL -o '" + dest + "' '" + url + "' && shasum -a 256 '" +
           dest + "'\n  (shasum must print " + fw.sha256 + "), or run libhmsbeagle/GPU/tinygpu_fetch_firmware.sh DIR and set "
           "BEAGLE_TINYGPU_FW=DIR.";
}

} // namespace tinygpu_device

#endif // LIBHMSBEAGLE_GPU_TINYGPUFIRMWARE_H
