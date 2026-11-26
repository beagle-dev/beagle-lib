/*
 *  RemoteAuth.h
 *  BEAGLE
 *
 * The shared secret by which a client and a worker recognise each other. Neither sends the secret: each sends
 * a nonce, and proves it knows the secret by returning an HMAC-SHA256 over both nonces. The client's proof and
 * the worker's are over different text, so that one cannot be played back as the other.
 *
 * The secret settles who is at the other end when the connection opens. It does not encrypt what follows, or
 * protect it from being altered on the way.
 */

#ifndef __BEAGLE_REMOTE_AUTH_H__
#define __BEAGLE_REMOTE_AUTH_H__

#include <string>

namespace beagle {
namespace remote {

/** Environment variables naming the secret, or a file that holds it (the file wins if both are set). */
static const char* const kSecretVariable = "BEAGLE_REMOTE_SECRET";
static const char* const kSecretFileVariable = "BEAGLE_REMOTE_SECRET_FILE";

static const size_t kNonceBytes = 32;

/**
 * Reads a secret from a file, without the white space around it. False, with the reason in error, if the file
 * cannot be read, is empty, or can be read by other users.
 */
bool readSecretFile(const std::string& path, std::string* secret, std::string* error);

/**
 * The secret the environment names; empty, and true, if it names none. False, with the reason in error, if it
 * names a file that cannot be used.
 */
bool secretFromEnvironment(std::string* secret, std::string* error);

/** kNonceBytes from the system's random source. Throws ProtocolError if there is none. */
std::string randomNonce();

/** What the party in this role ("client" or "worker") sends to show it knows the secret. */
std::string proofOfSecret(const std::string& secret, const char* role, const std::string& clientNonce,
                          const std::string& workerNonce);

/** Compares in time that does not depend on where the two differ. */
bool sameBytes(const std::string& a, const std::string& b);

/** SHA-256 and HMAC-SHA256 as 32 raw bytes (FIPS 180-4, RFC 2104). */
std::string sha256(const std::string& message);
std::string hmacSha256(const std::string& key, const std::string& message);

} // namespace remote
} // namespace beagle

#endif // __BEAGLE_REMOTE_AUTH_H__
