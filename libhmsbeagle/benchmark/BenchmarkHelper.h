//#define BEAGLE_DEBUG_TIME
#ifdef BEAGLE_DEBUG_TIME
#include <sys/time.h>
double debugTimeTotal;
double debugGetTime() {
    struct timeval tim;
    gettimeofday(&tim, NULL);
    return (tim.tv_sec+(tim.tv_usec/1000000.0));
}
#define DEBUG_CREATE_TIME() debugTimeTotal=0; fprintf(stderr,"\n*** BEAGLE instance created ***\n");
#define DEBUG_START_TIME() double debugInitialTime=debugGetTime();
#define DEBUG_END_TIME() debugTimeTotal+=debugGetTime()-debugInitialTime;
#define DEBUG_FINALIZE_TIME() fprintf(stderr,"\n*** Total time used by BEAGLE instance: %f seconds ***\n", debugTimeTotal);
#elif BEAGLE_BENCHMARK
#include <chrono>
#include <string>
#include <vector>
#include <iostream>
#include <map>

std::map<std::string,std::pair<std::chrono::duration<double>, int>> benchmarkDuration;

#define DEBUG_CREATE_TIME()
// Note: Calling DEBUG_FINALIZE_TIME() in beagle_library_finalize() will lead to memory errors
#define DEBUG_FINALIZE_TIME() std::map<std::string,std::pair<std::chrono::duration<double>, int>>::iterator it;\
std::cerr << "\nBENCHMARKS: " << __FILE__ << "\n" << "Function\tTime" << std::endl;\
for (it = benchmarkDuration.begin(); it != benchmarkDuration.end(); it++) {\
    std::cerr << it->first << "\t" << std::chrono::duration_cast<std::chrono::nanoseconds>(it->second.first).count() << " ns" << "\t" << it->second.second << std::endl;\
}
#define DEBUG_START_TIME() auto start = std::chrono::high_resolution_clock::now();
#define DEBUG_END_TIME() auto end = std::chrono::high_resolution_clock::now();\
std::string key = __func__;\
std::map<std::string,std::pair<std::chrono::duration<double>, int>>::iterator it = benchmarkDuration.find(key);\
if(it != benchmarkDuration.end()){\
    it->second.second += 1;\
    it->second.first += end-start;\
} else {\
    benchmarkDuration.insert(benchmarkDuration.end(), std::pair<std::string,std::pair<std::chrono::duration<double>, int>>(key, std::pair<std::chrono::duration<double>, int>(end - start, 1)));\
}
#else
#define DEBUG_CREATE_TIME()
#define DEBUG_START_TIME()
#define DEBUG_END_TIME()
#define DEBUG_FINALIZE_TIME()
#endif

#ifdef BEAGLE_BENCHMARK_ENERGY
#include <string>
#include <vector>
#include <iostream>
#include <map>
#include <nvml.h>

std::map<std::string, std::pair<unsigned long long int, int>> energyConsumption;

//TODO: Add checks on nvmlReturn_t objects

#define DEBUG_CREATE_ENERGY() nvmlReturn_t statusInit = nvmlInit();
#define DEBUG_FINALIZE_ENERGY() nvmlReturn_t statusShutdown = nvmlShutdown();\
std::map<std::string, std::pair<unsigned long long int, int>>::iterator it2;\
std::cerr << "\nBENCHMARKS ENERGY: " << __FILE__ << "\n" << "Function\tEnergy (mJ)\tnTimes" << std::endl;\
for (it2 = energyConsumption.begin(); it2 != energyConsumption.end(); it2++) {\
    std::cerr << it2->first << "\t" << it2->second.first << "\t" << it2->second.second << std::endl;\
}
#define DEBUG_START_ENERGY() nvmlDevice_t handle;\
nvmlReturn_t status = nvmlDeviceGetHandleByIndex(0, &handle);\
unsigned long long int energyStart;\
status = nvmlDeviceGetTotalEnergyConsumption(handle, &energyStart);
#define DEBUG_END_ENERGY() unsigned long long int energyEnd;\
status = nvmlDeviceGetTotalEnergyConsumption(handle, &energyEnd);\
std::string energyKey = __func__;\
std::map<std::string, std::pair<unsigned long long int, int>>::iterator it2 = energyConsumption.find(energyKey);\
if(it2 != energyConsumption.end()){\
    it2->second.first += energyEnd - energyStart;\
    it2->second.second += 1;\
} else {\
    energyConsumption.insert(energyConsumption.end(), std::pair<std::string, std::pair<unsigned long long int, int>>(energyKey, std::pair<unsigned long long int, int>(energyEnd - energyStart, 1)));\
}
#else
#define DEBUG_CREATE_ENERGY()
#define DEBUG_FINALIZE_ENERGY()
#define DEBUG_START_ENERGY()
#define DEBUG_END_ENERGY()
#endif