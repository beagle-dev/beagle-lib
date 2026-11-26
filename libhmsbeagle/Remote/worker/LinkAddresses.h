/*
 *  LinkAddresses.h
 *  BEAGLE
 *
 * The addresses a worker listens on when it is not told otherwise: those of this machine's Thunderbolt links,
 * which reach only a machine at the other end of a cable, and the loopback addresses.
 */

#ifndef __BEAGLE_REMOTE_LINK_ADDRESSES_H__
#define __BEAGLE_REMOTE_LINK_ADDRESSES_H__

#include <set>
#include <string>
#include <vector>

namespace beagle {
namespace remote {

/** Whether this build can tell which network interfaces are Thunderbolt links (macOS only). */
bool canFindThunderboltInterfaces();

/**
 * The names of the network interfaces that are Thunderbolt links, and of the bridges whose members are all
 * Thunderbolt links (macOS's "Thunderbolt Bridge").
 */
std::set<std::string> thunderboltInterfaces();

/**
 * The numeric addresses, at this moment, of the loopback interfaces and of the interfaces named. An address
 * of a Thunderbolt link exists only while its cable is connected.
 */
std::vector<std::string> linkAddresses(const std::set<std::string>& interfaces);

} // namespace remote
} // namespace beagle

#endif // __BEAGLE_REMOTE_LINK_ADDRESSES_H__
