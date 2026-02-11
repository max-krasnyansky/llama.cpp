// Helper inline functions for computation

static inline HVX_Vector load_d_shuff(const uint8_t * ptr) {
    return Q6_Vh_vshuff_Vh(*(const HVX_UVector *) ptr);
}

static inline HVX_Vector load_d_direct(const uint8_t * ptr) {
    return *(const HVX_UVector *) ptr;
}

static inline HVX_Vector compute_qx_q8_block(HVX_Vector_x8 r_q, HVX_Vector_x8 y_q, HVX_Vector r_d, HVX_Vector y_d) {
    HVX_Vector r_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_full(r_q, y_q));
    HVX_Vector r_dd = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(r_d, y_d)));
    return Q6_Vqf32_vmpy_VsfVsf(r_ia, r_dd);
}

static inline HVX_Vector compute_mxfp4_q8_block(HVX_Vector_x8 r_q, HVX_Vector_x8 y_q, HVX_Vector r_d, HVX_Vector y_d) {
    HVX_Vector r_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_full(r_q, y_q));

    // Convert vy_d from fp16 to fp32 while applying 0.5 scaling which is used for e8m0 halving
    HVX_Vector half = Q6_Vh_vsplat_R(0x3800);  // 0.5 in fp16
    y_d             = Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(y_d), half));
    y_d             = Q6_Vsf_equals_Vqf32(y_d);

    // Convert rX_d scales from e8m0 to fp32
    // Expand and zero-pad 32x uint8 e8m0 values to uint32s : 0 0 0 0, 0 0 0 1, 0 0 0 2, ...
    // Left shift with zero fill to create FP32
    // FIXME: might need to handle zero as a special case (see ggml-cpu code)
    HVX_Vector expand    = *(const HVX_Vector *) expand_x32_e8m0;
    HVX_Vector e8m0_mask = Q6_V_vsplat_R(0x000000ff);
    r_d                  = Q6_V_vdelta_VV(r_d, expand);
    r_d                  = Q6_V_vand_VV(r_d, e8m0_mask);
    r_d                  = Q6_Vw_vasl_VwR(r_d, 23);

    HVX_Vector r_dd = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(r_d, y_d));

    return Q6_Vqf32_vmpy_VsfVsf(r_ia, r_dd);
}

static inline HVX_Vector compute_qx_q8_nloe(HVX_Vector_x8 r_q, HVX_Vector_x8 y_q, HVX_Vector r_d, HVX_Vector y_d, int32_t nloe) {
    HVX_Vector r_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_nloe(r_q, y_q, nloe));
    HVX_Vector r_dd = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(r_d, y_d)));

    // Zero out unused scales
    HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe / 8);
    r_dd                 = Q6_V_vand_QV(bmask, r_dd);
    r_ia                 = Q6_V_vand_QV(bmask, r_ia);

    return Q6_Vqf32_vmpy_VsfVsf(r_ia, r_dd);
}

static inline HVX_Vector compute_mxfp4_q8_nloe(HVX_Vector_x8 r_q, HVX_Vector_x8 y_q, HVX_Vector r_d, HVX_Vector y_d, int32_t nloe) {
    HVX_Vector r_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_nloe(r_q, y_q, nloe));

    HVX_Vector half = Q6_Vh_vsplat_R(0x3800);
    y_d             = Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(y_d), half));
    y_d             = Q6_Vsf_equals_Vqf32(y_d);

    HVX_Vector expand    = *(const HVX_Vector *) expand_x32_e8m0;
    HVX_Vector e8m0_mask = Q6_V_vsplat_R(0x000000ff);
    r_d                  = Q6_V_vdelta_VV(r_d, expand);
    r_d                  = Q6_V_vand_VV(r_d, e8m0_mask);
    r_d                  = Q6_Vw_vasl_VwR(r_d, 23);

    HVX_Vector r_dd = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(r_d, y_d));

    // Zero-out unused values
    HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe / 8);
    r_dd                 = Q6_V_vand_QV(bmask, r_dd);
    r_ia                 = Q6_V_vand_QV(bmask, r_ia);

    return Q6_Vqf32_vmpy_VsfVsf(r_ia, r_dd);
}

#define VEC_DOT_COMMON_SETUP(QK, QROW_SHIFT) \
    assert(n % 32 == 0); \
    assert((unsigned long) vx % 128 == 0); \
    assert((unsigned long) vy % 128 == 0); \
    const uint32_t nb   = n / QK; \
    int32_t        nloe = n % QK; \
    const uint32_t y_qrow_size = n; \
    const uint8_t * restrict y_q = ((const uint8_t *) vy + 0); \
    const uint8_t * restrict y_d = ((const uint8_t *) vy + y_qrow_size);

#define DEFINE_VEC_DOT_Qx_Q8_VARIANTS(NAME, QK, X_DBLK, X_QBLK, X_QROW_SHIFT, Y_DBLK, Y_QBLK, LOAD_X_Q, LOAD_X_D, LOAD_Y_D, COMPUTE_BLOCK, COMPUTE_NLOE) \
static void vec_dot_##NAME##_1x1(const int n, float * restrict s, const void * restrict vx, const void * restrict vy) { \
    const uint32_t x_qrow_size = n >> X_QROW_SHIFT; \
    VEC_DOT_COMMON_SETUP(QK, 0) \
    const uint8_t * restrict r0_x_q = ((const uint8_t *) vx + 0); \
    const uint8_t * restrict r0_x_d = ((const uint8_t *) vx + x_qrow_size); \
    HVX_Vector r0_sum = Q6_V_vsplat_R(0); \
    uint32_t i = 0; \
    for (; i < nb; i++) { \
        HVX_Vector_x8 vy_q = hvx_vec_load_q8x4x8(y_q + i * Y_QBLK); \
        HVX_Vector_x8 r0_q = LOAD_X_Q(r0_x_q + i * X_QBLK); \
        HVX_Vector vy_d = LOAD_Y_D(y_d + i * Y_DBLK); \
        HVX_Vector r0_d = LOAD_X_D(r0_x_d + i * X_DBLK); \
        HVX_Vector r0_acc = COMPUTE_BLOCK(r0_q, vy_q, r0_d, vy_d); \
        r0_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(r0_acc, r0_sum)); \
    } \
    if (nloe) { \
        HVX_Vector_x8 vy_q = hvx_vec_load_q8x4x8(y_q + i * Y_QBLK); \
        HVX_Vector_x8 r0_q = LOAD_X_Q(r0_x_q + i * X_QBLK); \
        HVX_Vector vy_d = LOAD_Y_D(y_d + i * Y_DBLK); \
        HVX_Vector r0_d = LOAD_X_D(r0_x_d + i * X_DBLK); \
        HVX_Vector r0_acc = COMPUTE_NLOE(r0_q, vy_q, r0_d, vy_d, nloe); \
        r0_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(r0_acc, r0_sum)); \
    } \
    r0_sum = hvx_vec_reduce_sum_f32(r0_sum); \
    hvx_vec_store_u(&s[0], 4, r0_sum); \
} \
static void vec_dot_##NAME##_2x1(const int n, float * restrict s, const void * restrict vx, uint32_t vx_row_size, const void * restrict vy) { \
    const uint32_t x_qrow_size = n >> X_QROW_SHIFT; \
    VEC_DOT_COMMON_SETUP(QK, 0) \
    const uint8_t * restrict r0_x_q = ((const uint8_t *) (vx + (0 * vx_row_size)) + 0); \
    const uint8_t * restrict r0_x_d = ((const uint8_t *) (vx + (0 * vx_row_size)) + x_qrow_size); \
    const uint8_t * restrict r1_x_q = ((const uint8_t *) (vx + (1 * vx_row_size)) + 0); \
    const uint8_t * restrict r1_x_d = ((const uint8_t *) (vx + (1 * vx_row_size)) + x_qrow_size); \
    HVX_Vector r0_sum = Q6_V_vsplat_R(0); \
    HVX_Vector r1_sum = Q6_V_vsplat_R(0); \
    uint32_t i = 0; \
    for (; i < nb; i++) { \
        HVX_Vector_x8 vy_q = hvx_vec_load_q8x4x8(y_q + i * Y_QBLK); \
        HVX_Vector_x8 r0_q = LOAD_X_Q(r0_x_q + i * X_QBLK); \
        HVX_Vector_x8 r1_q = LOAD_X_Q(r1_x_q + i * X_QBLK); \
        HVX_Vector vy_d = LOAD_Y_D(y_d + i * Y_DBLK); \
        HVX_Vector r0_d = LOAD_X_D(r0_x_d + i * X_DBLK); \
        HVX_Vector r1_d = LOAD_X_D(r1_x_d + i * X_DBLK); \
        HVX_Vector r0_acc = COMPUTE_BLOCK(r0_q, vy_q, r0_d, vy_d); \
        HVX_Vector r1_acc = COMPUTE_BLOCK(r1_q, vy_q, r1_d, vy_d); \
        r0_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(r0_acc, r0_sum)); \
        r1_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(r1_acc, r1_sum)); \
    } \
    if (nloe) { \
        HVX_Vector_x8 vy_q = hvx_vec_load_q8x4x8(y_q + i * Y_QBLK); \
        HVX_Vector_x8 r0_q = LOAD_X_Q(r0_x_q + i * X_QBLK); \
        HVX_Vector_x8 r1_q = LOAD_X_Q(r1_x_q + i * X_QBLK); \
        HVX_Vector vy_d = LOAD_Y_D(y_d + i * Y_DBLK); \
        HVX_Vector r0_d = LOAD_X_D(r0_x_d + i * X_DBLK); \
        HVX_Vector r1_d = LOAD_X_D(r1_x_d + i * X_DBLK); \
        HVX_Vector r0_acc = COMPUTE_NLOE(r0_q, vy_q, r0_d, vy_d, nloe); \
        HVX_Vector r1_acc = COMPUTE_NLOE(r1_q, vy_q, r1_d, vy_d, nloe); \
        r0_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(r0_acc, r0_sum)); \
        r1_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(r1_acc, r1_sum)); \
    } \
    HVX_Vector rsum = hvx_vec_reduce_sum_f32x2(r0_sum, r1_sum); \
    hvx_vec_store_u(&s[0], 8, rsum); \
} \
static void vec_dot_##NAME##_2x2(const int n, float * restrict s, const void * restrict vx, uint32_t vx_row_size, const void * restrict vy, uint32_t vy_row_size) { \
    const uint32_t x_qrow_size = n >> X_QROW_SHIFT; \
    VEC_DOT_COMMON_SETUP(QK, 0) \
    const uint8_t * restrict y0_q = ((const uint8_t *) (vy + (0 * vy_row_size)) + 0); \
    const uint8_t * restrict y0_d = ((const uint8_t *) (vy + (0 * vy_row_size)) + y_qrow_size); \
    const uint8_t * restrict y1_q = ((const uint8_t *) (vy + (1 * vy_row_size)) + 0); \
    const uint8_t * restrict y1_d = ((const uint8_t *) (vy + (1 * vy_row_size)) + y_qrow_size); \
    const uint8_t * restrict r0_x_q = ((const uint8_t *) (vx + (0 * vx_row_size)) + 0); \
    const uint8_t * restrict r0_x_d = ((const uint8_t *) (vx + (0 * vx_row_size)) + x_qrow_size); \
    const uint8_t * restrict r1_x_q = ((const uint8_t *) (vx + (1 * vx_row_size)) + 0); \
    const uint8_t * restrict r1_x_d = ((const uint8_t *) (vx + (1 * vx_row_size)) + x_qrow_size); \
    HVX_Vector r00_sum = Q6_V_vsplat_R(0); \
    HVX_Vector r01_sum = Q6_V_vsplat_R(0); \
    HVX_Vector r10_sum = Q6_V_vsplat_R(0); \
    HVX_Vector r11_sum = Q6_V_vsplat_R(0); \
    uint32_t i = 0; \
    for (; i < nb; i++) { \
        HVX_Vector_x8 vy0_q = hvx_vec_load_q8x4x8(y0_q + i * Y_QBLK); \
        HVX_Vector_x8 vy1_q = hvx_vec_load_q8x4x8(y1_q + i * Y_QBLK); \
        HVX_Vector_x8 r0_q = LOAD_X_Q(r0_x_q + i * X_QBLK); \
        HVX_Vector_x8 r1_q = LOAD_X_Q(r1_x_q + i * X_QBLK); \
        HVX_Vector vy0_d = LOAD_Y_D(y0_d + i * Y_DBLK); \
        HVX_Vector vy1_d = LOAD_Y_D(y1_d + i * Y_DBLK); \
        HVX_Vector r0_d = LOAD_X_D(r0_x_d + i * X_DBLK); \
        HVX_Vector r1_d = LOAD_X_D(r1_x_d + i * X_DBLK); \
        HVX_Vector r00_acc = COMPUTE_BLOCK(r0_q, vy0_q, r0_d, vy0_d); \
        HVX_Vector r01_acc = COMPUTE_BLOCK(r0_q, vy1_q, r0_d, vy1_d); \
        HVX_Vector r10_acc = COMPUTE_BLOCK(r1_q, vy0_q, r1_d, vy0_d); \
        HVX_Vector r11_acc = COMPUTE_BLOCK(r1_q, vy1_q, r1_d, vy1_d); \
        r00_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(r00_acc, r00_sum)); \
        r01_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(r01_acc, r01_sum)); \
        r10_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(r10_acc, r10_sum)); \
        r11_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(r11_acc, r11_sum)); \
    } \
    if (nloe) { \
        HVX_Vector_x8 vy0_q = hvx_vec_load_q8x4x8(y0_q + i * Y_QBLK); \
        HVX_Vector_x8 vy1_q = hvx_vec_load_q8x4x8(y1_q + i * Y_QBLK); \
        HVX_Vector_x8 r0_q = LOAD_X_Q(r0_x_q + i * X_QBLK); \
        HVX_Vector_x8 r1_q = LOAD_X_Q(r1_x_q + i * X_QBLK); \
        HVX_Vector vy0_d = LOAD_Y_D(y0_d + i * Y_DBLK); \
        HVX_Vector vy1_d = LOAD_Y_D(y1_d + i * Y_DBLK); \
        HVX_Vector r0_d = LOAD_X_D(r0_x_d + i * X_DBLK); \
        HVX_Vector r1_d = LOAD_X_D(r1_x_d + i * X_DBLK); \
        HVX_Vector r00_acc = COMPUTE_NLOE(r0_q, vy0_q, r0_d, vy0_d, nloe); \
        HVX_Vector r01_acc = COMPUTE_NLOE(r0_q, vy1_q, r0_d, vy1_d, nloe); \
        HVX_Vector r10_acc = COMPUTE_NLOE(r1_q, vy0_q, r1_d, vy0_d, nloe); \
        HVX_Vector r11_acc = COMPUTE_NLOE(r1_q, vy1_q, r1_d, vy1_d, nloe); \
        r00_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(r00_acc, r00_sum)); \
        r01_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(r01_acc, r01_sum)); \
        r10_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(r10_acc, r10_sum)); \
        r11_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(r11_acc, r11_sum)); \
    } \
    HVX_Vector r0_sum = hvx_vec_reduce_sum_f32x2(r00_sum, r01_sum); \
    HVX_Vector r1_sum = hvx_vec_reduce_sum_f32x2(r10_sum, r11_sum); \
    hvx_vec_store_u(&s[0], 8, r0_sum); \
    hvx_vec_store_u(&s[2], 8, r1_sum); \
}

DEFINE_VEC_DOT_Qx_Q8_VARIANTS(q4x4x2_q8x4x2,    (QK_Q4_0x4x2 * 4),   (8 * 4 * 2), (QK_Q4_0x4x2 * 4) / 2, 1, (8 * 4 * 2), (QK_Q4_0x4x2 * 4), hvx_vec_load_q4x4x8,   load_d_shuff,  load_d_shuff,  compute_qx_q8_block,    compute_qx_q8_nloe)
DEFINE_VEC_DOT_Qx_Q8_VARIANTS(q8x4x2_q8x4x2,    (QK_Q8_0x4x2 * 4),   (8 * 4 * 2), (QK_Q8_0x4x2 * 4),     0, (8 * 4 * 2), (QK_Q8_0x4x2 * 4), hvx_vec_load_q8x4x8,   load_d_shuff,  load_d_shuff,  compute_qx_q8_block,    compute_qx_q8_nloe)
DEFINE_VEC_DOT_Qx_Q8_VARIANTS(mxfp4x4x2_q8x4x2, (QK_MXFP4x4x2 * 4),  (8 * 4 * 1), (QK_MXFP4x4x2 * 4) / 2,1, (8 * 4 * 2), (QK_MXFP4x4x2 * 4),hvx_vec_load_mxfp4x4x8,load_d_direct, load_d_direct, compute_mxfp4_q8_block, compute_mxfp4_q8_nloe)
