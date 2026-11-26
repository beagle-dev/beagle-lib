/*
 *  LinkAddresses.cpp
 *  BEAGLE
 */

#include "libhmsbeagle/Remote/worker/LinkAddresses.h"

#include <ifaddrs.h>
#include <net/if.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>

#ifdef __APPLE__
#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>
#include <SystemConfiguration/SystemConfiguration.h>
#endif

namespace beagle {
namespace remote {

#ifdef __APPLE__

namespace {

// The text of a property that should be a string; empty if it is not one
std::string asString(CFTypeRef value) {
    char bytes[64];
    if (value == NULL || CFGetTypeID(value) != CFStringGetTypeID() ||
        !CFStringGetCString((CFStringRef) value, bytes, sizeof(bytes), kCFStringEncodingUTF8))
        return "";
    return bytes;
}

CFDictionaryRef asDictionary(CFTypeRef value) {
    return (value != NULL && CFGetTypeID(value) == CFDictionaryGetTypeID()) ? (CFDictionaryRef) value : NULL;
}

// Whether a network interface in the I/O registry is driven by the Thunderbolt networking driver
bool isBelowThunderboltPort(io_registry_entry_t interface) {
    io_registry_entry_t entry = interface;
    IOObjectRetain(entry);
    bool found = false;
    while (!found) {
        io_registry_entry_t parent = IO_OBJECT_NULL;
        kern_return_t result = IORegistryEntryGetParentEntry(entry, kIOServicePlane, &parent);
        IOObjectRelease(entry);
        if (result != KERN_SUCCESS) return false;
        entry = parent;
        found = IOObjectConformsTo(entry, "AppleThunderboltIPPort");
    }
    IOObjectRelease(entry);
    return true;
}

} // anonymous namespace

bool canFindThunderboltInterfaces() {
    return true;
}

std::set<std::string> thunderboltInterfaces() {
    std::set<std::string> names;

    io_iterator_t interfaces = IO_OBJECT_NULL;
    if (IOServiceGetMatchingServices(MACH_PORT_NULL, IOServiceMatching("IOEthernetInterface"), &interfaces) ==
        KERN_SUCCESS) {
        io_registry_entry_t interface;
        while ((interface = IOIteratorNext(interfaces)) != IO_OBJECT_NULL) {
            if (isBelowThunderboltPort(interface)) {
                CFTypeRef name = IORegistryEntryCreateCFProperty(interface, CFSTR("BSD Name"),
                                                                 kCFAllocatorDefault, 0);
                if (name != NULL) {
                    if (!asString(name).empty()) names.insert(asString(name));
                    CFRelease(name);
                }
            }
            IOObjectRelease(interface);
        }
        IOObjectRelease(interfaces);
    }

    // A bridge carries the addresses of its members. It counts only if every member is a Thunderbolt link: a
    // bridge that also had an Ethernet port in it would let the rest of the network in. The members are read
    // from the network configuration (VirtualNetworkInterfaces / Bridge / <name> / Interfaces).
    SCPreferencesRef preferences = SCPreferencesCreate(NULL, CFSTR("beagle-worker"), NULL);
    if (preferences != NULL) {
        CFDictionaryRef bridges = NULL;
        CFDictionaryRef virtualInterfaces =
            asDictionary(SCPreferencesGetValue(preferences, CFSTR("VirtualNetworkInterfaces")));
        if (virtualInterfaces != NULL)
            bridges = asDictionary(CFDictionaryGetValue(virtualInterfaces, CFSTR("Bridge")));

        CFIndex bridgeCount = bridges ? CFDictionaryGetCount(bridges) : 0;
        std::vector<const void*> keys(bridgeCount), values(bridgeCount);
        if (bridgeCount > 0) CFDictionaryGetKeysAndValues(bridges, &keys[0], &values[0]);
        for (CFIndex i = 0; i < bridgeCount; i++) {
            CFDictionaryRef bridge = asDictionary(values[i]);
            CFTypeRef members = bridge ? CFDictionaryGetValue(bridge, CFSTR("Interfaces")) : NULL;
            if (members == NULL || CFGetTypeID(members) != CFArrayGetTypeID()) continue;

            CFIndex memberCount = CFArrayGetCount((CFArrayRef) members);
            bool allThunderbolt = memberCount > 0;
            for (CFIndex j = 0; j < memberCount && allThunderbolt; j++)
                allThunderbolt = names.count(asString(CFArrayGetValueAtIndex((CFArrayRef) members, j))) > 0;
            std::string name = asString(keys[i]);
            if (allThunderbolt && !name.empty()) names.insert(name);
        }
        CFRelease(preferences);
    }
    return names;
}

#else // not __APPLE__

bool canFindThunderboltInterfaces() {
    return false;
}

std::set<std::string> thunderboltInterfaces() {
    return std::set<std::string>();
}

#endif

std::vector<std::string> linkAddresses(const std::set<std::string>& interfaces) {
    std::vector<std::string> addresses;
    struct ifaddrs* list = NULL;
    if (getifaddrs(&list) != 0) return addresses;

    for (struct ifaddrs* entry = list; entry != NULL; entry = entry->ifa_next) {
        if (entry->ifa_addr == NULL || entry->ifa_name == NULL || !(entry->ifa_flags & IFF_UP)) continue;
        if (!(entry->ifa_flags & IFF_LOOPBACK) && !interfaces.count(entry->ifa_name)) continue;

        socklen_t length;
        if (entry->ifa_addr->sa_family == AF_INET)
            length = sizeof(struct sockaddr_in);
        else if (entry->ifa_addr->sa_family == AF_INET6)
            length = sizeof(struct sockaddr_in6);
        else
            continue;

        // a link-local IPv6 address comes out with its interface ("fe80::1%en2"), which listening on it needs
        char host[NI_MAXHOST];
        if (getnameinfo(entry->ifa_addr, length, host, sizeof(host), NULL, 0, NI_NUMERICHOST) == 0)
            addresses.push_back(host);
    }
    freeifaddrs(list);
    return addresses;
}

} // namespace remote
} // namespace beagle
