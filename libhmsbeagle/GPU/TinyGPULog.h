/*
 * TinyGPULog.h
 *
 * The C++ side's own log of every hardware-facing step it takes (TODO.md plan step C5, inv:nv-gsp-boot#23): the
 * evidence after a panic, once the daemon's log (nv_dispatch_daemon.py:702-705, truncated at every start) is gone or
 * never written. Lines are appended to ~/Library/Logs/beagle_tinygpu.log (BEAGLE_TINYGPU_LOG names another file) through
 * a descriptor opened O_APPEND | O_SYNC, each followed by an fsync, so a line that returned is on disk even if the
 * process is killed (kill -9) or macOS panics right after. Each line carries the time and the pid. Never fatal: a log
 * that cannot be opened or written is skipped, once reported on stderr.
 */

#ifndef LIBHMSBEAGLE_GPU_TINYGPULOG_H
#define LIBHMSBEAGLE_GPU_TINYGPULOG_H

#include <cerrno>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

// Status notes on stderr (the boot's progress, the build stamps, a clean teardown's report) print only in a build with
// BEAGLE_TINYGPU_STATUS defined (CMake -DBEAGLE_TINYGPU_STATUS=ON), as the test harness needs; errors print in every build.
// TG_STATUS_OR_ERROR(is_error, ...) is a note that is an error when is_error holds (an unconfirmed unload, say).
#ifdef BEAGLE_TINYGPU_STATUS
#define TG_STATUS(...) fprintf(stderr, __VA_ARGS__)
#define TG_STATUS_OR_ERROR(is_error, ...) fprintf(stderr, __VA_ARGS__)
#else
#define TG_STATUS(...) ((void)0)
#define TG_STATUS_OR_ERROR(is_error, ...) ((is_error) ? (void)fprintf(stderr, __VA_ARGS__) : (void)0)
#endif

namespace tinygpu_device {

inline std::string tg_log_path() {
    const char* p = getenv("BEAGLE_TINYGPU_LOG");
    if (p && p[0]) return p;
    const char* home = getenv("HOME");
    return std::string(home ? home : "/tmp") + "/Library/Logs/beagle_tinygpu.log";
}

// One formatted line (no newline needed), appended and synced before this returns.
inline void tg_log(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
inline void tg_log(const char* fmt, ...) {
    static int fd = -2;
    if (fd == -2) {
        std::string path = tg_log_path();
        std::string dir = path.substr(0, path.rfind('/'));
        if (!dir.empty()) mkdir(dir.c_str(), 0755);   // ~/Library/Logs exists on macOS; a test's directory may not
        fd = open(path.c_str(), O_WRONLY | O_CREAT | O_APPEND | O_SYNC | O_CLOEXEC, 0644);
        if (fd < 0) fprintf(stderr, "TinyGPU: cannot open the log %s (%s); not logging\n", path.c_str(), strerror(errno));
    }
    if (fd < 0) return;
    char line[2048];
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    struct tm tm;
    localtime_r(&tv.tv_sec, &tm);
    int n = (int)strftime(line, sizeof(line), "%Y-%m-%d %H:%M:%S", &tm);
    n += snprintf(line + n, sizeof(line) - n, ".%03d [%d] ", (int)(tv.tv_usec / 1000), (int)getpid());
    va_list ap;
    va_start(ap, fmt);
    int m = vsnprintf(line + n, sizeof(line) - n - 1, fmt, ap);
    va_end(ap);
    n = m < 0 ? n : (n + m < (int)sizeof(line) - 1 ? n + m : (int)sizeof(line) - 2);
    line[n++] = '\n';
    if (write(fd, line, (size_t)n) == n) fsync(fd);
}

} // namespace tinygpu_device

#endif // LIBHMSBEAGLE_GPU_TINYGPULOG_H
