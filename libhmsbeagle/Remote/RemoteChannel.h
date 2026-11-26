/*
 *  RemoteChannel.h
 *  BEAGLE
 *
 * The link between the remote plugin and a worker: reliable, ordered, and message-oriented (a receive returns
 * exactly what one send was given). The protocol needs nothing else of it, so a transport is one subclass.
 *
 * TCP is the transport here. The interface is shaped for RDMA over Thunderbolt as the second: that transport
 * (Apple TN3205) offers only two-sided send and receive on a queue pair, at most 10 queue pairs, messages of
 * whole 4 KB frames that the receiver must have posted a buffer of the same frame count for, and completions
 * by polling. A channel per BEAGLE instance and one message per batch fit within that; the queue pair's
 * addresses are exchanged over TCP first, which is why a worker is always reached by TCP.
 */

#ifndef __BEAGLE_REMOTE_CHANNEL_H__
#define __BEAGLE_REMOTE_CHANNEL_H__

#include <string>

#include "libhmsbeagle/Remote/RemoteProtocol.h"

namespace beagle {
namespace remote {

class Channel {
public:
    virtual ~Channel() {}

    /** Sends one message. False if the link has failed; the channel is then unusable. */
    virtual bool send(const Buffer& message) = 0;

    /** Receives one message, replacing the buffer's contents. False if the link has failed or was closed. */
    virtual bool receive(Buffer& message) = 0;
};

/**
 * Connects to a worker by TCP. Returns NULL, with the reason in error, if it cannot within the timeout.
 */
Channel* connectTcp(const std::string& host, int port, int timeoutMilliseconds, std::string* error);

/** Accepts TCP connections from clients. */
class TcpListener {
public:
    TcpListener() : fd_(-1) {}
    ~TcpListener();

    /** Listens on an address ("" for every interface) and port. False, with the reason in error, on failure. */
    bool listen(const std::string& address, int port, std::string* error);

    /** Waits for a client. Returns its channel and, in peer, its address; NULL if the listener has failed. */
    Channel* accept(std::string* peer);

private:
    int fd_;
};

} // namespace remote
} // namespace beagle

#endif // __BEAGLE_REMOTE_CHANNEL_H__
