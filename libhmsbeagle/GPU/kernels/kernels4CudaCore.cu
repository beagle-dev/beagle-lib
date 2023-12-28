
KW_GLOBAL_KERNEL void kernelPartialsPartialsNoScale(KW_GLOBAL_VAR REAL* KW_RESTRICT partials1,
                                                    KW_GLOBAL_VAR REAL* KW_RESTRICT partials2,
                                                    KW_GLOBAL_VAR REAL* KW_RESTRICT partials3,
                                                    KW_GLOBAL_VAR REAL* KW_RESTRICT matrices1,
                                                    KW_GLOBAL_VAR REAL* KW_RESTRICT matrices2,
                                                    int endPattern) {
#ifdef FW_OPENCL_CPU // CPU/MIC implementation
    DETERMINE_INDICES_4_CPU();
    SUM_PARTIALS_PARTIALS_4_CPU();
    for(int i = 0; i < PADDED_STATE_COUNT; i++) {
        partials3[deltaPartials + i] = sum1[i] * sum2[i];
    }
#else // GPU implementation
    DETERMINE_INDICES_4_GPU();
    LOAD_PARTIALS_PARTIALS_4_GPU();
    LOAD_MATRIX_4_GPU();
    if (pattern < endPattern) { // Remove padded threads!
        SUM_PARTIALS_PARTIALS_4_GPU();
        partials3[u] = sum1 * sum2;
    }
#endif // FW_OPENCL_CPU

#ifdef KERNEL_PRINT_ENABLED
    printf("matrix = %d, pat = %d for tx = %d and state = %d :  u = %d\n",
           matrix, pattern, tx, state, u);
#endif
}