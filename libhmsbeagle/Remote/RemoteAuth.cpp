/*
 *  RemoteAuth.cpp
 *  BEAGLE
 */

#include "libhmsbeagle/Remote/RemoteAuth.h"

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include "libhmsbeagle/Remote/RemoteProtocol.h"

namespace beagle {
namespace remote {

namespace {

const uint32_t kRoundConstants[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

inline uint32_t rotateRight(uint32_t value, int bits) {
    return (value >> bits) | (value << (32 - bits));
}

void compress(uint32_t state[8], const uint8_t block[64]) {
    uint32_t w[64];
    for (int i = 0; i < 16; i++)
        w[i] = ((uint32_t) block[4 * i] << 24) | ((uint32_t) block[4 * i + 1] << 16) |
               ((uint32_t) block[4 * i + 2] << 8) | (uint32_t) block[4 * i + 3];
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = rotateRight(w[i - 15], 7) ^ rotateRight(w[i - 15], 18) ^ (w[i - 15] >> 3);
        uint32_t s1 = rotateRight(w[i - 2], 17) ^ rotateRight(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
    for (int i = 0; i < 64; i++) {
        uint32_t s1 = rotateRight(e, 6) ^ rotateRight(e, 11) ^ rotateRight(e, 25);
        uint32_t choose = (e & f) ^ (~e & g);
        uint32_t t1 = h + s1 + choose + kRoundConstants[i] + w[i];
        uint32_t s0 = rotateRight(a, 2) ^ rotateRight(a, 13) ^ rotateRight(a, 22);
        uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2 = s0 + majority;
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

} // anonymous namespace

std::string sha256(const std::string& message) {
    uint32_t state[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };

    // the message, a 1 bit, zeros, and the length in bits, to a whole number of 64-byte blocks
    std::string padded = message;
    padded.push_back((char) 0x80);
    while (padded.size() % 64 != 56) padded.push_back('\0');
    uint64_t bits = (uint64_t) message.size() * 8;
    for (int shift = 56; shift >= 0; shift -= 8) padded.push_back((char) (bits >> shift));

    for (size_t offset = 0; offset < padded.size(); offset += 64)
        compress(state, (const uint8_t*) padded.data() + offset);

    std::string digest(32, '\0');
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 4; j++)
            digest[4 * i + j] = (char) (state[i] >> (24 - 8 * j));
    return digest;
}

std::string hmacSha256(const std::string& key, const std::string& message) {
    std::string block = key.size() > 64 ? sha256(key) : key;
    block.resize(64, '\0');

    std::string inner(64, '\0'), outer(64, '\0');
    for (int i = 0; i < 64; i++) {
        inner[i] = (char) (block[i] ^ 0x36);
        outer[i] = (char) (block[i] ^ 0x5c);
    }
    return sha256(outer + sha256(inner + message));
}

bool sameBytes(const std::string& a, const std::string& b) {
    if (a.size() != b.size()) return false;
    unsigned char difference = 0;
    for (size_t i = 0; i < a.size(); i++) difference |= (unsigned char) (a[i] ^ b[i]);
    return difference == 0;
}

std::string proofOfSecret(const std::string& secret, const char* role, const std::string& clientNonce,
                          const std::string& workerNonce) {
    // the nonces have a fixed length, so the three parts cannot be divided in another way
    std::string text = std::string("BEAGLE remote ") + role;
    text.push_back('\0');
    return hmacSha256(secret, text + clientNonce + workerNonce);
}

std::string randomNonce() {
    std::string nonce(kNonceBytes, '\0');
#if defined(__APPLE__) || defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__NetBSD__)
    arc4random_buf(&nonce[0], nonce.size());
#else
    int fd = open("/dev/urandom", O_RDONLY);
    if (fd < 0) throw ProtocolError("no random source");
    size_t filled = 0;
    while (filled < nonce.size()) {
        ssize_t got = read(fd, &nonce[filled], nonce.size() - filled);
        if (got < 0 && errno == EINTR) continue;
        if (got <= 0) {
            close(fd);
            throw ProtocolError("no random source");
        }
        filled += (size_t) got;
    }
    close(fd);
#endif
    return nonce;
}

bool readSecretFile(const std::string& path, std::string* secret, std::string* error) {
    FILE* file = fopen(path.c_str(), "rb");
    if (file == NULL) {
        if (error) *error = "cannot read the secret file " + path + " (" + strerror(errno) + ")";
        return false;
    }

    struct stat status;
    if (fstat(fileno(file), &status) != 0 || (status.st_mode & (S_IRWXG | S_IRWXO)) != 0) {
        fclose(file);
        if (error) *error = "the secret file " + path + " can be read by other users (chmod 600 it)";
        return false;
    }

    std::string contents;
    char chunk[4096];
    size_t got;
    while ((got = fread(chunk, 1, sizeof(chunk), file)) > 0) contents.append(chunk, got);
    fclose(file);

    const char* space = " \t\r\n";
    size_t first = contents.find_first_not_of(space);
    if (first == std::string::npos) {
        if (error) *error = "the secret file " + path + " is empty";
        return false;
    }
    size_t last = contents.find_last_not_of(space);
    *secret = contents.substr(first, last - first + 1);
    return true;
}

bool secretFromEnvironment(std::string* secret, std::string* error) {
    secret->clear();
    const char* path = getenv(kSecretFileVariable);
    if (path != NULL && *path != '\0') return readSecretFile(path, secret, error);
    const char* value = getenv(kSecretVariable);
    if (value != NULL) *secret = value;
    return true;
}

} // namespace remote
} // namespace beagle
