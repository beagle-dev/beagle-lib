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

#include <map>
#include <string>
#include <vector>

#include "libhmsbeagle/Remote/RemoteProtocol.h"

namespace beagle {
namespace remote {

class Channel {
public:
    Channel() : receiveLimit_(kMaxMessageBytes) {}
    virtual ~Channel() {}

    /** A message larger than this is taken to be a corrupt length rather than allocated. */
    static const uint64_t kMaxMessageBytes = 1ULL << 36;

    /**
     * Makes receive() fail on a message longer than this. A worker keeps the limit small until its client has
     * shown that it knows the secret.
     */
    void limitReceive(uint64_t bytes) { receiveLimit_ = bytes; }

    /** Sends one message. False if the link has failed; the channel is then unusable. */
    virtual bool send(const Buffer& message) = 0;

    /** Receives one message, replacing the buffer's contents. False if the link has failed or was closed. */
    virtual bool receive(Buffer& message) = 0;

protected:
    uint64_t receiveLimit_;
};

/**
 * Connects to a worker by TCP. Returns NULL, with the reason in error, if it cannot within the timeout.
 */
Channel* connectTcp(const std::string& host, int port, int timeoutMilliseconds, std::string* error);

/**
 * Accepts TCP connections from clients, on a set of addresses that can change while it runs (the address of a
 * Thunderbolt link exists only while its cable is connected).
 */
class TcpListener {
public:
    TcpListener() {}
    ~TcpListener();

    /**
     * Makes the addresses listened on those given: opens the ones that are new and closes the ones no longer
     * listed. An address is numeric or a host name; "" is every interface. One that cannot be listened on is
     * left out, with the reason in error, and is tried again at the next call.
     */
    void listenOn(const std::vector<std::string>& addresses, int port, std::string* error);

    /** The addresses being listened on. */
    std::vector<std::string> addresses() const;

    /**
     * Waits for a client for up to the timeout (-1 for ever). Returns its channel and, in peer, its address;
     * NULL if none came in time.
     */
    Channel* accept(std::string* peer, int timeoutMilliseconds);

private:
    TcpListener(const TcpListener&);
    TcpListener& operator=(const TcpListener&);

    std::map<std::string, int> sockets_; // by address
};

} // namespace remote
} // namespace beagle

#endif // __BEAGLE_REMOTE_CHANNEL_H__
