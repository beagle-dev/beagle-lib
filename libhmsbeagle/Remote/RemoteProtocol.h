/*
 *  RemoteProtocol.h
 *  BEAGLE
 *
 * Wire protocol between the remote plugin (hmsbeagle-remote, in the client process) and beagle-worker (on the
 * machine that does the computing).
 *
 * A connection carries messages; the channel (RemoteChannel.h) keeps their boundaries. The first exchange is a
 * hello: the client sends kMagic and kProtocolVersion, the worker answers with the same two and its resource
 * list. Every later request is a batch of commands, each an opcode followed by its arguments, and every batch
 * gets one response:
 *
 *     i64 deferredError   first non-zero return code of the commands before the last one (0 if none)
 *     i64 status          return code of the last command
 *     ...                 the outputs of the last command, present only when status is BEAGLE_SUCCESS
 *
 * Only the last command of a batch returns outputs. The client queues the calls that return nothing and sends
 * the batch when a call needs an answer, so a likelihood evaluation costs one round trip.
 *
 * Everything is written in 8-byte units in host byte order (the hello detects a mismatch, as kMagic reads
 * differently), so that the arrays of a received message can be handed to BEAGLE in place. An array is an i64
 * element count, -1 for a NULL pointer, then its elements padded to 8 bytes.
 */

#ifndef __BEAGLE_REMOTE_PROTOCOL_H__
#define __BEAGLE_REMOTE_PROTOCOL_H__

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>
#include <stdexcept>
#include <string>

namespace beagle {
namespace remote {

static const int64_t kMagic = 0x42474c52454d4f54LL; // "BGLREMOT"
static const int64_t kProtocolVersion = 1;
static const int kDefaultPort = 4780;

/** Environment variable listing the workers: host[:port],host[:port],... */
static const char* const kHostsVariable = "BEAGLE_REMOTE_HOSTS";

enum Opcode {
    OP_SYNC = 1,            // no operation; ends a batch that was sent only because it grew large
    OP_CREATE_INSTANCE,
    OP_SET_CPU_THREAD_COUNT,
    OP_ENSURE_BUFFER_COUNTS,
    OP_SET_TIP_STATES,
    OP_SET_TIP_PARTIALS,
    OP_SET_PARTIALS,
    OP_SET_ROOT_PRE_PARTIALS,
    OP_GET_PARTIALS,
    OP_GET_PARTIALS_BATCH,
    OP_SET_EIGEN_DECOMPOSITION,
    OP_SET_STATE_FREQUENCIES,
    OP_SET_CATEGORY_WEIGHTS,
    OP_SET_PATTERN_WEIGHTS,
    OP_SET_PATTERN_PARTITIONS,
    OP_SET_CATEGORY_RATES,
    OP_SET_CATEGORY_RATES_WITH_INDEX,
    OP_SET_TRANSITION_MATRIX,
    OP_SET_DIFFERENTIAL_MATRIX,
    OP_SET_TRANSITION_MATRICES,
    OP_GET_TRANSITION_MATRIX,
    OP_CONVOLVE_TRANSITION_MATRICES,
    OP_ADD_TRANSITION_MATRICES,
    OP_TRANSPOSE_TRANSITION_MATRICES,
    OP_UPDATE_TRANSITION_MATRICES,
    OP_UPDATE_TRANSITION_MATRICES_WITH_MODEL_CATEGORIES,
    OP_UPDATE_TRANSITION_MATRICES_WITH_MULTIPLE_MODELS,
    OP_UPDATE_PARTIALS,
    OP_UPDATE_PRE_PARTIALS,
    OP_UPDATE_PARTIALS_BY_PARTITION,
    OP_UPDATE_PRE_PARTIALS_BY_PARTITION,
    OP_WAIT_FOR_PARTIALS,
    OP_ACCUMULATE_SCALE_FACTORS,
    OP_ACCUMULATE_SCALE_FACTORS_BY_PARTITION,
    OP_REMOVE_SCALE_FACTORS,
    OP_REMOVE_SCALE_FACTORS_BY_PARTITION,
    OP_RESET_SCALE_FACTORS,
    OP_RESET_SCALE_FACTORS_BY_PARTITION,
    OP_COPY_SCALE_FACTORS,
    OP_GET_SCALE_FACTORS,
    OP_CALCULATE_ROOT_LOG_LIKELIHOODS,
    OP_CALCULATE_ROOT_LOG_LIKELIHOODS_BY_PARTITION,
    OP_CALCULATE_EDGE_LOG_LIKELIHOODS,
    OP_CALCULATE_EDGE_LOG_LIKELIHOODS_BY_PARTITION,
    OP_GET_LOG_LIKELIHOOD,
    OP_GET_DERIVATIVES,
    OP_GET_SITE_LOG_LIKELIHOODS,
    OP_GET_SITE_DERIVATIVES
};

/** A malformed message: it ends early, or an array does not have the length the receiver requires. */
class ProtocolError : public std::runtime_error {
public:
    explicit ProtocolError(const std::string& what) : std::runtime_error(what) {}
};

/**
 * A growable message buffer whose storage is aligned, so that the int and double arrays inside it can be used
 * in place.
 */
class Buffer {
public:
    Buffer() : data_(NULL), size_(0), capacity_(0) {}
    ~Buffer() { free(data_); }

    const uint8_t* data() const { return data_; }
    uint8_t* data() { return data_; }
    size_t size() const { return size_; }
    void clear() { size_ = 0; }

    /** Sets the size, keeping the contents up to the smaller of the old and new sizes. */
    void resize(size_t size) {
        if (size > capacity_) {
            size_t capacity = capacity_ ? capacity_ : 4096;
            while (capacity < size) capacity *= 2;
            void* data = NULL;
            if (posix_memalign(&data, 4096, capacity) != 0) throw std::bad_alloc();
            if (size_) memcpy(data, data_, size_);
            free(data_);
            data_ = (uint8_t*) data;
            capacity_ = capacity;
        }
        size_ = size;
    }

    void putI64(int64_t value) { append(&value, sizeof(value)); }
    void putF64(double value) { append(&value, sizeof(value)); }

    /** An array of count ints; a NULL pointer is written as such and read back as NULL. */
    void putInts(const int* values, size_t count) { putArray(values, count, sizeof(int)); }
    void putDoubles(const double* values, size_t count) { putArray(values, count, sizeof(double)); }
    void putString(const std::string& value) { putArray(value.data(), value.size(), 1); }

private:
    Buffer(const Buffer&);
    Buffer& operator=(const Buffer&);

    void append(const void* bytes, size_t length) {
        size_t offset = size_;
        size_t padded = (length + 7) & ~(size_t) 7;
        resize(offset + padded);
        if (length) memcpy(data_ + offset, bytes, length);
        if (padded > length) memset(data_ + offset + length, 0, padded - length);
    }

    void putArray(const void* values, size_t count, size_t elementSize) {
        if (values == NULL) {
            putI64(-1);
        } else {
            putI64((int64_t) count);
            append(values, count * elementSize);
        }
    }

    uint8_t* data_;
    size_t size_;
    size_t capacity_;
};

/** Reads a message in the order it was written. Throws ProtocolError rather than read past the end. */
class Reader {
public:
    Reader(const uint8_t* data, size_t size) : p_(data), end_(data + size) {}
    explicit Reader(const Buffer& buffer) : p_(buffer.data()), end_(buffer.data() + buffer.size()) {}

    bool atEnd() const { return p_ == end_; }

    int64_t i64() { int64_t value; memcpy(&value, take(sizeof(value)), sizeof(value)); return value; }
    int i32() { return (int) i64(); }
    double f64() { double value; memcpy(&value, take(sizeof(value)), sizeof(value)); return value; }

    /** A count that sizes later arrays: negative is malformed. */
    size_t count() {
        int64_t value = i64();
        if (value < 0) throw ProtocolError("negative count");
        return (size_t) value;
    }

    /** An array that must be present and have exactly this many elements. */
    const int* ints(size_t expected) { return (const int*) array(expected, sizeof(int), false); }
    const double* doubles(size_t expected) { return (const double*) array(expected, sizeof(double), false); }

    /** As above, but the writer may have passed NULL, which is returned as NULL. */
    const int* optionalInts(size_t expected) { return (const int*) array(expected, sizeof(int), true); }
    const double* optionalDoubles(size_t expected) { return (const double*) array(expected, sizeof(double), true); }

    /** Copies an array of exactly this many doubles to a destination, which may be NULL to discard it. */
    void copyDoubles(double* destination, size_t expected) {
        const double* values = doubles(expected);
        if (destination != NULL && expected) memcpy(destination, values, expected * sizeof(double));
    }

    std::string string() {
        int64_t length = i64();
        if (length < 0) throw ProtocolError("missing string");
        const uint8_t* bytes = take((size_t) length);
        return std::string((const char*) bytes, (size_t) length);
    }

private:
    const uint8_t* take(size_t length) {
        if (length > (size_t) (end_ - p_)) throw ProtocolError("message is shorter than its contents");
        size_t padded = (length + 7) & ~(size_t) 7;
        if (padded > (size_t) (end_ - p_)) throw ProtocolError("message is shorter than its contents");
        const uint8_t* bytes = p_;
        p_ += padded;
        return bytes;
    }

    const void* array(size_t expected, size_t elementSize, bool optional) {
        int64_t length = i64();
        if (length == -1) {
            if (!optional) throw ProtocolError("missing array");
            return NULL;
        }
        if (length < 0 || (size_t) length != expected) throw ProtocolError("array has the wrong length");
        if (expected > (size_t) (end_ - p_) / elementSize) throw ProtocolError("message is shorter than its contents");
        return take(expected * elementSize);
    }

    const uint8_t* p_;
    const uint8_t* end_;
};

} // namespace remote
} // namespace beagle

#endif // __BEAGLE_REMOTE_PROTOCOL_H__
