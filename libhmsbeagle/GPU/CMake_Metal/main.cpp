
// main.cpp

#include <cstdio>

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

int main() {
    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    std::printf("%s\n", device->description()->cString(NS::UTF8StringEncoding));
    device->release();
}
