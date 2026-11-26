/*
 *  RemoteChannel.cpp
 *  BEAGLE
 *
 * The TCP transport: each message is an 8-byte length followed by its bytes.
 */

#include "libhmsbeagle/Remote/RemoteChannel.h"

#include <cerrno>
#include <cstdio>
#include <cstring>

#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>

namespace beagle {
namespace remote {

namespace {

#ifdef MSG_NOSIGNAL
const int kSendFlags = MSG_NOSIGNAL;
#else
const int kSendFlags = 0;
#endif

void configureSocket(int fd) {
    int on = 1;
    // batches are small and each waits for its answer, so Nagle's algorithm would only add delay
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &on, sizeof(on));
#ifdef SO_NOSIGPIPE
    // a peer that has gone away is reported by send(), not by a signal that would kill the client
    setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &on, sizeof(on));
#endif
}

class TcpChannel : public Channel {
public:
    explicit TcpChannel(int fd) : fd_(fd) {}

    virtual ~TcpChannel() {
        if (fd_ >= 0) close(fd_);
    }

    virtual bool send(const Buffer& message) {
        uint64_t length = message.size();
        struct iovec parts[2];
        parts[0].iov_base = &length;
        parts[0].iov_len = sizeof(length);
        parts[1].iov_base = (void*) message.data();
        parts[1].iov_len = message.size();

        struct msghdr header;
        memset(&header, 0, sizeof(header));
        header.msg_iov = parts;
        header.msg_iovlen = message.size() ? 2 : 1;

        while (header.msg_iovlen > 0) {
            ssize_t sent = sendmsg(fd_, &header, kSendFlags);
            if (sent < 0) {
                if (errno == EINTR) continue;
                return false;
            }
            size_t remaining = (size_t) sent;
            while (header.msg_iovlen > 0 && remaining >= header.msg_iov[0].iov_len) {
                remaining -= header.msg_iov[0].iov_len;
                header.msg_iov++;
                header.msg_iovlen--;
            }
            if (header.msg_iovlen > 0) {
                header.msg_iov[0].iov_base = (char*) header.msg_iov[0].iov_base + remaining;
                header.msg_iov[0].iov_len -= remaining;
            }
        }
        return true;
    }

    virtual bool receive(Buffer& message) {
        uint64_t length = 0;
        if (!receiveBytes(&length, sizeof(length))) return false;
        if (length > receiveLimit_) return false;
        message.clear();
        message.resize((size_t) length);
        return receiveBytes(message.data(), (size_t) length);
    }

private:
    bool receiveBytes(void* bytes, size_t length) {
        char* p = (char*) bytes;
        while (length > 0) {
            ssize_t received = recv(fd_, p, length, 0);
            if (received < 0) {
                if (errno == EINTR) continue;
                return false;
            }
            if (received == 0) return false; // closed by the peer
            p += received;
            length -= (size_t) received;
        }
        return true;
    }

    int fd_;
};

std::string describeAddress(const struct sockaddr* address, socklen_t length) {
    char host[NI_MAXHOST];
    char service[NI_MAXSERV];
    if (getnameinfo(address, length, host, sizeof(host), service, sizeof(service),
                    NI_NUMERICHOST | NI_NUMERICSERV) != 0)
        return "unknown";
    return std::string(host) + ":" + service;
}

// connect() bounded by a timeout, so that a worker that is switched off does not stall the client's start-up
bool connectWithTimeout(int fd, const struct sockaddr* address, socklen_t length, int timeoutMilliseconds) {
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0 || fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0) return false;

    bool connected = false;
    if (connect(fd, address, length) == 0) {
        connected = true;
    } else if (errno == EINPROGRESS) {
        struct pollfd waiting;
        waiting.fd = fd;
        waiting.events = POLLOUT;
        waiting.revents = 0;
        int ready;
        do {
            ready = poll(&waiting, 1, timeoutMilliseconds);
        } while (ready < 0 && errno == EINTR);
        if (ready > 0) {
            int socketError = 0;
            socklen_t errorLength = sizeof(socketError);
            if (getsockopt(fd, SOL_SOCKET, SO_ERROR, &socketError, &errorLength) == 0 && socketError == 0)
                connected = true;
            else
                errno = socketError;
        } else if (ready == 0) {
            errno = ETIMEDOUT;
        }
    }

    if (connected && fcntl(fd, F_SETFL, flags) < 0) connected = false;
    return connected;
}

} // anonymous namespace

Channel* connectTcp(const std::string& host, int port, int timeoutMilliseconds, std::string* error) {
    char service[16];
    snprintf(service, sizeof(service), "%d", port);

    struct addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    struct addrinfo* candidates = NULL;
    int lookup = getaddrinfo(host.c_str(), service, &hints, &candidates);
    if (lookup != 0) {
        if (error) *error = gai_strerror(lookup);
        return NULL;
    }

    std::string lastError = "no address";
    Channel* channel = NULL;
    for (struct addrinfo* candidate = candidates; candidate != NULL && channel == NULL; candidate = candidate->ai_next) {
        int fd = socket(candidate->ai_family, candidate->ai_socktype, candidate->ai_protocol);
        if (fd < 0) {
            lastError = strerror(errno);
            continue;
        }
        if (connectWithTimeout(fd, candidate->ai_addr, candidate->ai_addrlen, timeoutMilliseconds)) {
            configureSocket(fd);
            channel = new TcpChannel(fd);
        } else {
            lastError = strerror(errno);
            close(fd);
        }
    }
    freeaddrinfo(candidates);

    if (channel == NULL && error) *error = lastError;
    return channel;
}

namespace {

// A non-blocking socket listening on the address, or -1 with the reason in error
int openListeningSocket(const std::string& address, int port, std::string* error) {
    char service[16];
    snprintf(service, sizeof(service), "%d", port);

    struct addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = address.empty() ? AF_INET6 : AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;

    struct addrinfo* candidates = NULL;
    int lookup = getaddrinfo(address.empty() ? NULL : address.c_str(), service, &hints, &candidates);
    if (lookup != 0) {
        if (error) *error = gai_strerror(lookup);
        return -1;
    }

    std::string lastError = "no address";
    int listening = -1;
    for (struct addrinfo* candidate = candidates; candidate != NULL && listening < 0; candidate = candidate->ai_next) {
        int fd = socket(candidate->ai_family, candidate->ai_socktype, candidate->ai_protocol);
        if (fd < 0) {
            lastError = strerror(errno);
            continue;
        }
        int on = 1;
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));
        if (candidate->ai_family == AF_INET6) {
            // every interface: one socket for IPv4 and IPv6 clients. One address: that address and no other
            int v6Only = address.empty() ? 0 : 1;
            setsockopt(fd, IPPROTO_IPV6, IPV6_V6ONLY, &v6Only, sizeof(v6Only));
        }
        int flags = fcntl(fd, F_GETFL, 0);
        if (bind(fd, candidate->ai_addr, candidate->ai_addrlen) == 0 && ::listen(fd, 64) == 0 &&
            flags >= 0 && fcntl(fd, F_SETFL, flags | O_NONBLOCK) == 0) {
            listening = fd;
        } else {
            lastError = strerror(errno);
            close(fd);
        }
    }
    freeaddrinfo(candidates);

    if (listening < 0 && error) *error = lastError;
    return listening;
}

} // anonymous namespace

TcpListener::~TcpListener() {
    for (std::map<std::string, int>::iterator it = sockets_.begin(); it != sockets_.end(); ++it)
        close(it->second);
}

void TcpListener::listenOn(const std::vector<std::string>& addresses, int port, std::string* error) {
    for (std::map<std::string, int>::iterator it = sockets_.begin(); it != sockets_.end();) {
        bool wanted = false;
        for (size_t i = 0; i < addresses.size() && !wanted; i++) wanted = (addresses[i] == it->first);
        if (wanted) {
            ++it;
        } else {
            close(it->second);
            sockets_.erase(it++);
        }
    }

    for (size_t i = 0; i < addresses.size(); i++) {
        if (sockets_.count(addresses[i])) continue;
        std::string reason;
        int fd = openListeningSocket(addresses[i], port, &reason);
        if (fd >= 0)
            sockets_[addresses[i]] = fd;
        else if (error)
            *error = (addresses[i].empty() ? std::string("*") : addresses[i]) + ": " + reason;
    }
}

std::vector<std::string> TcpListener::addresses() const {
    std::vector<std::string> listening;
    for (std::map<std::string, int>::const_iterator it = sockets_.begin(); it != sockets_.end(); ++it)
        listening.push_back(it->first);
    return listening;
}

Channel* TcpListener::accept(std::string* peer, int timeoutMilliseconds) {
    std::vector<struct pollfd> waiting;
    for (std::map<std::string, int>::const_iterator it = sockets_.begin(); it != sockets_.end(); ++it) {
        struct pollfd entry;
        entry.fd = it->second;
        entry.events = POLLIN;
        entry.revents = 0;
        waiting.push_back(entry);
    }

    int ready = poll(waiting.empty() ? NULL : &waiting[0], (nfds_t) waiting.size(), timeoutMilliseconds);
    if (ready <= 0) return NULL; // nobody in time, or a signal: the caller comes round again

    for (size_t i = 0; i < waiting.size(); i++) {
        if (!(waiting[i].revents & POLLIN)) continue;
        struct sockaddr_storage address;
        socklen_t length = sizeof(address);
        int fd = ::accept(waiting[i].fd, (struct sockaddr*) &address, &length);
        if (fd < 0) continue; // the client gave up between poll and accept
        // an accepted socket can inherit the listening one's non-blocking mode; the channel wants to block
        int flags = fcntl(fd, F_GETFL, 0);
        if (flags < 0 || fcntl(fd, F_SETFL, flags & ~O_NONBLOCK) < 0) {
            close(fd);
            continue;
        }
        configureSocket(fd);
        if (peer) *peer = describeAddress((struct sockaddr*) &address, length);
        return new TcpChannel(fd);
    }
    return NULL;
}

} // namespace remote
} // namespace beagle
