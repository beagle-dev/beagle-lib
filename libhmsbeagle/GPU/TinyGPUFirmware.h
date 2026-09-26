/*
 * TinyGPUFirmware.h
 *
 * TODO.md plan step C4: finds the NVIDIA firmware a C++ boot needs (TinyGPUFirmwareManifest.h) and checks it the way
 * tinygrad's fetch does before it uses a cached file (helpers.py:469-474: the file exists and its sha256 matches), with no
 * network access (plan decision 5). Three places, in order:
 *   1. $BEAGLE_TINYGPU_FW/<subdir>/<name>
 *   2. <the directory holding this code>/../share/beagle/firmware/<subdir>/<name> (an installed plugin's share/)
 *   3. tinygrad's download cache, where its fetch_fw leaves the file: ${XDG_CACHE_HOME:-~/Library/Caches}/tinygrad/
 *      downloads/fw/<md5(url)> (helpers.py:396, 454-472)
 * The file is mapped read-only (the GSP image is 63.5 MB) and its SHA-256 computed with CommonCrypto before it is
 * returned, so a boot can check all of its firmware before it opens the TinyGPU.app socket. When no place has the file,
 * the error says what each place held and how to fetch it (curl and shasum, or tinygpu_fetch_firmware.sh). fetch_fw's
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
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

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
inline const nvfw::TGFirmware* tg_fw_entry(const std::string& chip, const std::string& role) {
    for (const nvfw::TGFirmware& f : nvfw::kFirmware)
        if (chip == f.chip && role == f.role) return &f;
    return nullptr;
}

inline std::string tg_fw_url(const nvfw::TGFirmware& fw) {   // fetch_fw's URL (helpers.py:509)
    return std::string(nvfw::kLinuxFirmware) + "/" + fw.subdir + "/" + fw.name;
}

// tinygrad's cache_dir/downloads/fw (helpers.py:396, 454-472, fetch_fw's subdir "fw"); not the tinybox /raid path
inline std::string tg_fw_tinygrad_cache(const nvfw::TGFirmware& fw) {
    const char* xdg = getenv("XDG_CACHE_HOME");
    const char* home = getenv("HOME");
    std::string base = xdg ? xdg : std::string(home ? home : "") + "/Library/Caches";
    return base + "/tinygrad/downloads/fw/" + fw.url_md5;
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
    out.push_back({"tinygrad's download cache", tg_fw_tinygrad_cache(fw)});
    return out;
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
    const std::string dest = tg_fw_tinygrad_cache(fw), url = tg_fw_url(fw);
    return "TinyGPU: NVIDIA firmware " + std::string(fw.subdir) + "/" + fw.name + " (sha256 " + fw.sha256 + ") is missing or damaged:\n" +
           report + "BEAGLE does not download firmware. To fetch it where tinygrad keeps it:\n" +
           "  mkdir -p '" + dest.substr(0, dest.find_last_of('/')) + "' && curl -fL -o '" + dest + "' '" + url + "' && shasum -a 256 '" +
           dest + "'\n  (shasum must print " + fw.sha256 + "), or run libhmsbeagle/GPU/tinygpu_fetch_firmware.sh DIR and set "
           "BEAGLE_TINYGPU_FW=DIR.";
}

} // namespace tinygpu_device

#endif // LIBHMSBEAGLE_GPU_TINYGPUFIRMWARE_H
