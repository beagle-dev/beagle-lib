KW_GLOBAL_KERNEL void kernelPartialsPartialsNoScale(KW_GLOBAL_VAR REAL* KW_RESTRICT partials1,
                                                    KW_GLOBAL_VAR REAL* KW_RESTRICT partials2,
                                                    KW_GLOBAL_VAR REAL* KW_RESTRICT partials3,
                                                    KW_GLOBAL_VAR REAL* KW_RESTRICT matrices1,
                                                    KW_GLOBAL_VAR REAL* KW_RESTRICT matrices2,
                                                    int totalPatterns) {
#ifdef FW_OPENCL_CPU // CPU/MIC implementation
    DETERMINE_INDICES_X_CPU();
    SUM_PARTIALS_PARTIALS_X_CPU();
    partials3[u] = sum1 * sum2;
#else // GPU implementation
    DETERMINE_INDICES_X_GPU();
    SUM_PARTIALS_PARTIALS_X_GPU();
    if (pattern < totalPatterns)
        partials3[u] = sum1 * sum2;
#endif // FW_OPENCL_CPU
}