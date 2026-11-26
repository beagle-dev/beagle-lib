/*
 *  BeagleRemotePlugin.cpp
 *  BEAGLE
 */

#include "libhmsbeagle/Remote/BeagleRemotePlugin.h"

#include <cstdlib>
#include <iostream>
#include <sstream>

namespace beagle {
namespace remote {

namespace {

// How long to wait for each listed worker when the resource list is first asked for
const int kDiscoveryTimeoutMilliseconds = 1000;

// Splits "host", "host:port" or "[v6-address]:port"; a bare IPv6 address is taken whole
void parseEndpoint(const std::string& text, std::string* host, int* port) {
    *host = text;
    *port = kDefaultPort;
    std::string portText;
    if (!text.empty() && text[0] == '[') {
        size_t close = text.find(']');
        if (close == std::string::npos) return;
        *host = text.substr(1, close - 1);
        if (close + 1 < text.size() && text[close + 1] == ':') portText = text.substr(close + 2);
    } else {
        size_t colon = text.find(':');
        if (colon == std::string::npos || text.find(':', colon + 1) != std::string::npos) return;
        *host = text.substr(0, colon);
        portText = text.substr(colon + 1);
    }
    if (!portText.empty()) *port = atoi(portText.c_str());
}

std::string trim(const std::string& text) {
    size_t first = text.find_first_not_of(" \t");
    if (first == std::string::npos) return "";
    size_t last = text.find_last_not_of(" \t");
    return text.substr(first, last - first + 1);
}

} // anonymous namespace

BeagleRemotePlugin::BeagleRemotePlugin() :
Plugin("Remote", "Remote")
{
    const char* hosts = getenv(kHostsVariable);
    if (hosts == NULL || *hosts == '\0') return;

    // The workers in the order they are listed, so that resource numbers are the same from run to run
    std::stringstream list(hosts);
    std::string entry;
    while (std::getline(list, entry, ',')) {
        entry = trim(entry);
        if (entry.empty()) continue;

        std::string host;
        int port;
        parseEndpoint(entry, &host, &port);

        std::vector<RemoteResource> found;
        std::string error;
        Channel* channel = openWorker(host, port, kDiscoveryTimeoutMilliseconds, &found, &error);
        if (channel == NULL) {
            std::cerr << "BEAGLE remote: no worker at " << entry << " (" << error << ")" << std::endl;
            continue;
        }
        delete channel;

        for (size_t i = 0; i < found.size(); i++) {
            // resources of the same name are merged into one, so the name has to say which worker it is on
            found[i].name += " on " + entry;
            resources_.push_back(found[i]);
        }
    }
    if (resources_.empty()) return;

    long long allFlags = BEAGLE_FLAG_FRAMEWORK_REMOTE;
    for (size_t i = 0; i < resources_.size(); i++) {
        BeagleResource resource;
        resource.name = (char*) resources_[i].name.c_str();
        resource.description = (char*) resources_[i].description.c_str();
        resource.supportFlags = resources_[i].supportFlags | BEAGLE_FLAG_FRAMEWORK_REMOTE;
        // what a local implementation lacks, so that only the remote one is tried on this resource
        resource.requiredFlags = BEAGLE_FLAG_FRAMEWORK_REMOTE;
        beagleResources.push_back(resource);
        allFlags |= resource.supportFlags;
    }

    // The factory stands for whatever implementation each worker picks, so it claims every flag a worker
    // supports; whether an instance with the requested flags can be made is decided by the worker.
    // A factory with the spectral-representation flag is only a candidate when that is asked for, so the flag
    // has a factory of its own.
    beagleFactories.push_back(new BeagleRemoteImplFactory(resources_, allFlags & ~BEAGLE_FLAG_SPECTRAL_REPRESENTATION));
    if (allFlags & BEAGLE_FLAG_SPECTRAL_REPRESENTATION)
        beagleFactories.push_back(new BeagleRemoteImplFactory(resources_, allFlags));
}

} // namespace remote
} // namespace beagle

extern "C" {
BEAGLE_PLUGIN_EXPORT void* plugin_init(void){
    return new beagle::remote::BeagleRemotePlugin();
}
}
