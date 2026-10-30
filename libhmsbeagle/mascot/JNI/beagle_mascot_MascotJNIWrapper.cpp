/*
 * beagle_mascot_MascotJNIWrapper.cpp -- JNI entry points for BEAST's beagle.mascot.MascotJNIWrapper.
 *
 * Java arrays are copied in and out with Get/Set<Type>ArrayRegion through staging buffers owned per instance, so a
 * call allocates nothing once warm and never holds the GC lock (no GetPrimitiveArrayCritical). Return values are
 * MascotStatus codes; Java maps them to exceptions.
 */
#include <jni.h>

#include "../mascot.h"

#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace {

struct Staging {
    int stateCount = 0, lineageCount = 0, maxEpochCount = 0;
    std::vector<double> intervalLengths, boundaries, migration, inversePopulation, branchRates;
    std::vector<int> eventTypes, sampleNodes, child1, child2, parent;
    std::vector<double> tipPartials, thetaGradient, clockGradient, ancestralScores;
};

std::mutex stagingMutex;
std::vector<std::unique_ptr<Staging>> stagingTable;

Staging* staging(int instance) {
    std::lock_guard<std::mutex> lock(stagingMutex);
    if (instance < 0 || instance >= (int) stagingTable.size()) return nullptr;
    return stagingTable[instance].get();
}

/* Copies a Java array of at least n elements into v (grown, never shrunk); false if it is too short. */
bool copyIn(JNIEnv* env, jdoubleArray array, int n, std::vector<double>& v) {
    if (!array || env->GetArrayLength(array) < n) return false;
    if ((int) v.size() < n) v.resize(n);
    if (n > 0) env->GetDoubleArrayRegion(array, 0, n, v.data());
    return true;
}

bool copyIn(JNIEnv* env, jintArray array, int n, std::vector<int>& v) {
    if (!array || env->GetArrayLength(array) < n) return false;
    if ((int) v.size() < n) v.resize(n);
    if (n > 0) env->GetIntArrayRegion(array, 0, n, v.data());
    return true;
}

bool copyOut(JNIEnv* env, jdoubleArray array, int n, const std::vector<double>& v) {
    if (!array || env->GetArrayLength(array) < n) return false;
    if (n > 0) env->SetDoubleArrayRegion(array, 0, n, v.data());
    return true;
}

} // namespace

extern "C" {

JNIEXPORT jint JNICALL Java_beagle_mascot_MascotJNIWrapper_getApiVersion(JNIEnv* env, jclass, jintArray majorMinor) {
    int major = 0, minor = 0;
    int status = mascotGetApiVersion(&major, &minor);
    if (!majorMinor || env->GetArrayLength(majorMinor) < 2) return MASCOT_ERROR_ARGUMENT;
    jint out[2] = {major, minor};
    env->SetIntArrayRegion(majorMinor, 0, 2, out);
    return status;
}

JNIEXPORT jint JNICALL Java_beagle_mascot_MascotJNIWrapper_createInstance(JNIEnv*, jclass, jint stateCount,
                                                                           jint lineageCount, jint maxEpochCount,
                                                                           jlong flags) {
    int instance = mascotCreateInstance(stateCount, lineageCount, maxEpochCount, (long) flags);
    if (instance < 0) return instance;
    std::unique_ptr<Staging> s(new Staging());
    s->stateCount = stateCount;
    s->lineageCount = lineageCount;
    s->maxEpochCount = maxEpochCount;
    std::lock_guard<std::mutex> lock(stagingMutex);
    if ((int) stagingTable.size() <= instance) stagingTable.resize(instance + 1);
    stagingTable[instance] = std::move(s);
    return instance;
}

JNIEXPORT jint JNICALL Java_beagle_mascot_MascotJNIWrapper_finalizeInstance(JNIEnv*, jclass, jint instance) {
    int status = mascotFinalizeInstance(instance);
    if (status == MASCOT_OK) {
        std::lock_guard<std::mutex> lock(stagingMutex);
        stagingTable[instance].reset();
    }
    return status;
}

/* "kernel simdWidth fpMode isa", e.g. "A 0 fma neon"; null for an unknown instance. */
JNIEXPORT jstring JNICALL Java_beagle_mascot_MascotJNIWrapper_getInstanceDetails(JNIEnv* env, jclass, jint instance) {
    MascotInstanceDetails d;
    if (mascotGetInstanceDetails(instance, &d) != MASCOT_OK) return nullptr;
    std::string text = std::string(1, d.kernel) + " " + std::to_string(d.simdWidth) + " " +
                       (d.fpMode == MASCOT_FP_FMA ? "fma" : "strict") + " " + d.isa;
    return env->NewStringUTF(text.c_str());
}

JNIEXPORT jint JNICALL Java_beagle_mascot_MascotJNIWrapper_setTipPartials(JNIEnv* env, jclass, jint instance,
                                                                           jdoubleArray tipPartials) {
    Staging* s = staging(instance);
    if (!s || !copyIn(env, tipPartials, s->lineageCount * s->stateCount, s->tipPartials)) return MASCOT_ERROR_ARGUMENT;
    return mascotSetTipPartials(instance, s->tipPartials.data());
}

JNIEXPORT jint JNICALL Java_beagle_mascot_MascotJNIWrapper_evaluate(
        JNIEnv* env, jclass, jint instance,
        jint intervalCount, jdouble initialTime, jint initialSampleNode,
        jdoubleArray intervalLengths, jintArray eventTypes, jintArray sampleNodes,
        jintArray child1, jintArray child2, jintArray parent,
        jint epochCount, jdoubleArray boundaries, jdoubleArray migration, jdoubleArray inversePopulation,
        jdoubleArray branchRates,
        jint stepRuleKind, jdouble maxStep, jdouble tau, jdouble hMin, jint maxStepsPerInterval,
        jboolean checkProbabilities, jint checkpointReadSet, jint checkpointWriteSet, jint outputs,
        jdoubleArray logLikelihood, jdoubleArray thetaGradient, jdoubleArray clockGradient,
        jdoubleArray ancestralScores) {
    Staging* s = staging(instance);
    if (!s || intervalCount < 0 || epochCount < 1 || epochCount > s->maxEpochCount) return MASCOT_ERROR_ARGUMENT;
    const int K = s->stateCount, L = s->lineageCount;
    if (!copyIn(env, intervalLengths, intervalCount, s->intervalLengths) ||
        !copyIn(env, eventTypes, intervalCount, s->eventTypes) ||
        !copyIn(env, sampleNodes, intervalCount, s->sampleNodes) ||
        !copyIn(env, child1, intervalCount, s->child1) ||
        !copyIn(env, child2, intervalCount, s->child2) ||
        !copyIn(env, parent, intervalCount, s->parent) ||
        !copyIn(env, boundaries, epochCount + 1, s->boundaries) ||
        !copyIn(env, migration, epochCount * K * K, s->migration) ||
        !copyIn(env, inversePopulation, epochCount * K, s->inversePopulation) ||
        (branchRates && !copyIn(env, branchRates, L, s->branchRates))) {
        return MASCOT_ERROR_ARGUMENT;
    }
    const int thetaCount = epochCount * (K * (K - 1) + K);
    const bool wantTheta = (outputs & MASCOT_OUT_THETA_GRADIENT) != 0;
    const bool wantClock = (outputs & MASCOT_OUT_CLOCK_GRADIENT) != 0;
    const bool wantAncestral = (outputs & MASCOT_OUT_ANCESTRAL_SCORES) != 0;
    if (wantTheta && (int) s->thetaGradient.size() < thetaCount) s->thetaGradient.resize(thetaCount);
    if (wantClock && (int) s->clockGradient.size() < L) s->clockGradient.resize(L);
    if (wantAncestral && (int) s->ancestralScores.size() < L * K) s->ancestralScores.resize(L * K);

    MascotStepRule rule = {stepRuleKind, maxStep, tau, hMin, maxStepsPerInterval};
    double logL = 0.0;
    int status = mascotEvaluate(instance, intervalCount, initialTime, initialSampleNode,
            s->intervalLengths.data(), s->eventTypes.data(), s->sampleNodes.data(), s->child1.data(),
            s->child2.data(), s->parent.data(), epochCount, s->boundaries.data(), s->migration.data(),
            s->inversePopulation.data(), branchRates ? s->branchRates.data() : nullptr, &rule,
            checkProbabilities ? 1 : 0, checkpointReadSet, checkpointWriteSet, outputs, &logL,
            wantTheta ? s->thetaGradient.data() : nullptr, wantClock ? s->clockGradient.data() : nullptr,
            wantAncestral ? s->ancestralScores.data() : nullptr);
    if (status != MASCOT_OK) return status;

    if (!logLikelihood || env->GetArrayLength(logLikelihood) < 1) return MASCOT_ERROR_ARGUMENT;
    env->SetDoubleArrayRegion(logLikelihood, 0, 1, &logL);
    if ((wantTheta && !copyOut(env, thetaGradient, thetaCount, s->thetaGradient)) ||
        (wantClock && !copyOut(env, clockGradient, L, s->clockGradient)) ||
        (wantAncestral && !copyOut(env, ancestralScores, L * K, s->ancestralScores))) {
        return MASCOT_ERROR_ARGUMENT;
    }
    return MASCOT_OK;
}

JNIEXPORT jint JNICALL Java_beagle_mascot_MascotJNIWrapper_getLastStepCounts(JNIEnv* env, jclass, jint instance,
                                                                              jintArray steps, jintArray count) {
    if (!steps || !count || env->GetArrayLength(count) < 1) return MASCOT_ERROR_ARGUMENT;
    jint capacity = env->GetArrayLength(steps);
    std::vector<int> buffer(capacity > 0 ? capacity : 0);
    int n = 0;
    int status = mascotGetLastStepCounts(instance, buffer.data(), capacity, &n);
    if (status != MASCOT_OK) return status;
    if (n > 0) env->SetIntArrayRegion(steps, 0, n, reinterpret_cast<const jint*>(buffer.data()));
    jint out = n;
    env->SetIntArrayRegion(count, 0, 1, &out);
    return MASCOT_OK;
}

JNIEXPORT jint JNICALL Java_beagle_mascot_MascotJNIWrapper_getLastRestartInterval(JNIEnv* env, jclass, jint instance,
                                                                                  jintArray interval) {
    if (!interval || env->GetArrayLength(interval) < 1) return MASCOT_ERROR_ARGUMENT;
    int value = 0;
    int status = mascotGetLastRestartInterval(instance, &value);
    if (status != MASCOT_OK) return status;
    jint out = value;
    env->SetIntArrayRegion(interval, 0, 1, &out);
    return MASCOT_OK;
}

} // extern "C"
