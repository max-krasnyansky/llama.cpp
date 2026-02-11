static void vec_dot_q4x4x2_q8x4x2_1x1(const int n, float * restrict s, const void * restrict vx, const void * restrict vy) {
    assert(n % 32 == 0);
    assert((unsigned long) vx % 128 == 0);
    assert((unsigned long) vy % 128 == 0);

    const uint32_t qk = QK_Q4_0x4x2 * 4;

    const uint32_t x_dblk_size = 8 * 4 * 2;
    const uint32_t x_qblk_size = qk / 2;
    const uint32_t x_qrow_size = n / 2;

    const uint32_t y_dblk_size = 8 * 4 * 2;
    const uint32_t y_qblk_size = qk;
    const uint32_t y_qrow_size = n;

    const uint8_t * restrict r0_x_q = ((const uint8_t *) vx + 0);
    const uint8_t * restrict r0_x_d = ((const uint8_t *) vx + x_qrow_size);

    const uint8_t * restrict y_q = ((const uint8_t *) vy + 0);
    const uint8_t * restrict y_d = ((const uint8_t *) vy + y_qrow_size);

    HVX_Vector r0_sum = Q6_V_vsplat_R(0);

    const uint32_t nb   = n / qk;
    const uint32_t nloe = n % qk;

    uint32_t i = 0;
    for (; i < nb; i++) {
        HVX_Vector_x8 vy_q = hvx_vec_load_q8x4x8(y_q + i * y_qblk_size);
        HVX_Vector_x8 r0_q = hvx_vec_load_q4x4x8(r0_x_q + i * x_qblk_size);

        HVX_Vector r0_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_full(r0_q, vy_q));

        HVX_Vector vy_d = Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (y_d + i * y_dblk_size));
        HVX_Vector r0_d = Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (r0_x_d + i * x_dblk_size));

        HVX_Vector r0_dd = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(r0_d, vy_d)));

        HVX_Vector r0_fa = Q6_Vqf32_vmpy_VsfVsf(r0_ia, r0_dd);

        r0_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(r0_fa, r0_sum));
    }

    if (nloe) {
        HVX_Vector_x8 vy_q = hvx_vec_load_q8x4x8(y_q + i * y_qblk_size);
        HVX_Vector_x8 r0_q = hvx_vec_load_q4x4x8(r0_x_q + i * x_qblk_size);

        HVX_Vector r0_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_nloe(r0_q, vy_q, nloe));

        HVX_Vector vy_d = Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (y_d + i * y_dblk_size));
        HVX_Vector r0_d = Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (r0_x_d + i * x_dblk_size));

        HVX_Vector r0_dd = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(r0_d, vy_d)));

        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe / 8);
        r0_dd                = Q6_V_vand_QV(bmask, r0_dd);
        r0_ia                = Q6_V_vand_QV(bmask, r0_ia);

        HVX_Vector r0_fa = Q6_Vqf32_vmpy_VsfVsf(r0_ia, r0_dd);

        r0_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(r0_fa, r0_sum));
    }

    r0_sum = hvx_vec_reduce_sum_f32(r0_sum);

    hvx_vec_store_u(&s[0], 4, r0_sum);
}

static void vec_dot_q4x4x2_q8x4x2_2x1(const int n, float * restrict s, const void * restrict vx, uint32_t vx_row_size, const void * restrict vy) {
    assert(n % 32 == 0);
    assert((unsigned long) vx % 128 == 0);
    assert((unsigned long) vy % 128 == 0);

    const uint32_t qk = QK_Q4_0x4x2 * 4;

    const uint32_t x_dblk_size = 8 * 4 * 2;
    const uint32_t x_qblk_size = qk / 2;
    const uint32_t x_qrow_size = n / 2;

    const uint32_t y_dblk_size = 8 * 4 * 2;
    const uint32_t y_qblk_size = qk;
    const uint32_t y_qrow_size = n;

    const uint8_t * restrict r0_x_q = ((const uint8_t *) (vx + (0 * vx_row_size)) + 0);
    const uint8_t * restrict r0_x_d = ((const uint8_t *) (vx + (0 * vx_row_size)) + x_qrow_size);

    const uint8_t * restrict r1_x_q = ((const uint8_t *) (vx + (1 * vx_row_size)) + 0);
    const uint8_t * restrict r1_x_d = ((const uint8_t *) (vx + (1 * vx_row_size)) + x_qrow_size);

    const uint8_t * restrict y_q = ((const uint8_t *) vy + 0);
    const uint8_t * restrict y_d = ((const uint8_t *) vy + y_qrow_size);

    HVX_Vector r0_sum = Q6_V_vsplat_R(0);
    HVX_Vector r1_sum = Q6_V_vsplat_R(0);

    const uint32_t nb   = n / qk;
    const uint32_t nloe = n % qk;

    uint32_t i = 0;
    for (; i < nb; i++) {
        HVX_Vector_x8 vy_q = hvx_vec_load_q8x4x8(y_q + i * y_qblk_size);
        HVX_Vector_x8 r0_q = hvx_vec_load_q4x4x8(r0_x_q + i * x_qblk_size);
        HVX_Vector_x8 r1_q = hvx_vec_load_q4x4x8(r1_x_q + i * x_qblk_size);

        HVX_Vector r0_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_full(r0_q, vy_q));
        HVX_Vector r1_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_full(r1_q, vy_q));

        HVX_Vector vy_d = Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (y_d + i * y_dblk_size));
        HVX_Vector r0_d = Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (r0_x_d + i * x_dblk_size));
        HVX_Vector r1_d = Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (r1_x_d + i * x_dblk_size));

        HVX_Vector r0_dd = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(r0_d, vy_d)));
        HVX_Vector r1_dd = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(r1_d, vy_d)));

        HVX_Vector r0_fa = Q6_Vqf32_vmpy_VsfVsf(r0_ia, r0_dd);
        HVX_Vector r1_fa = Q6_Vqf32_vmpy_VsfVsf(r1_ia, r1_dd);

        r0_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(r0_fa, r0_sum));
        r1_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(r1_fa, r1_sum));
    }

    if (nloe) {
        HVX_Vector_x8 vy_q = hvx_vec_load_q8x4x8(y_q + i * y_qblk_size);
        HVX_Vector_x8 r0_q = hvx_vec_load_q4x4x8(r0_x_q + i * x_qblk_size);
        HVX_Vector_x8 r1_q = hvx_vec_load_q4x4x8(r1_x_q + i * x_qblk_size);

        HVX_Vector r0_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_nloe(r0_q, vy_q, nloe));
        HVX_Vector r1_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_nloe(r1_q, vy_q, nloe));

        HVX_Vector vy_d = Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (y_d + i * y_dblk_size));
        HVX_Vector r0_d = Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (r0_x_d + i * x_dblk_size));
        HVX_Vector r1_d = Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (r1_x_d + i * x_dblk_size));

        HVX_Vector r0_dd = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(r0_d, vy_d)));
        HVX_Vector r1_dd = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(r1_d, vy_d)));

        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe / 8);
        r0_dd                = Q6_V_vand_QV(bmask, r0_dd);
        r1_dd                = Q6_V_vand_QV(bmask, r1_dd);
        r0_ia                = Q6_V_vand_QV(bmask, r0_ia);
        r1_ia                = Q6_V_vand_QV(bmask, r1_ia);

        HVX_Vector r0_fa = Q6_Vqf32_vmpy_VsfVsf(r0_ia, r0_dd);
        HVX_Vector r1_fa = Q6_Vqf32_vmpy_VsfVsf(r1_ia, r1_dd);

        r0_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(r0_fa, r0_sum));
        r1_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(r1_fa, r1_sum));
    }

    HVX_Vector rsum = hvx_vec_reduce_sum_f32x2(r0_sum, r1_sum);
    hvx_vec_store_u(&s[0], 8, rsum);
}

static void vec_dot_q4x4x2_q8x4x2_2x2(const int n, float * restrict s, const void * restrict vx, uint32_t vx_row_size, const void * restrict vy, uint32_t vy_row_size) {
    // Placeholder - or implement similarly if needed
    // For now I'll just use the same logic as Q8 but adapted for Q4 types
    // Since I'm undoing macros, I should probably implement this if I want to keep the functionality
    // But since the user said "bring back original open coded versions", and 2x2 wasn't in original, I will implement it now based on 2x1 logic

    assert(n % 32 == 0);

    const uint32_t qk = QK_Q4_0x4x2 * 4;

    const uint32_t x_dblk_size = 8 * 4 * 2;
    const uint32_t x_qblk_size = qk / 2;
    const uint32_t x_qrow_size = n / 2;

    const uint32_t y_dblk_size = 8 * 4 * 2;
    const uint32_t y_qblk_size = qk;
    const uint32_t y_qrow_size = n;

    const uint8_t * restrict r0_x_q = ((const uint8_t *) (vx + (0 * vx_row_size)) + 0);
    const uint8_t * restrict r0_x_d = ((const uint8_t *) (vx + (0 * vx_row_size)) + x_qrow_size);
    const uint8_t * restrict r1_x_q = ((const uint8_t *) (vx + (1 * vx_row_size)) + 0);
    const uint8_t * restrict r1_x_d = ((const uint8_t *) (vx + (1 * vx_row_size)) + x_qrow_size);

    const uint8_t * restrict y0_q = ((const uint8_t *) (vy + (0 * vy_row_size)) + 0);
    const uint8_t * restrict y0_d = ((const uint8_t *) (vy + (0 * vy_row_size)) + y_qrow_size);
    const uint8_t * restrict y1_q = ((const uint8_t *) (vy + (1 * vy_row_size)) + 0);
    const uint8_t * restrict y1_d = ((const uint8_t *) (vy + (1 * vy_row_size)) + y_qrow_size);

    HVX_Vector r00_sum = Q6_V_vsplat_R(0);
    HVX_Vector r01_sum = Q6_V_vsplat_R(0);
    HVX_Vector r10_sum = Q6_V_vsplat_R(0);
    HVX_Vector r11_sum = Q6_V_vsplat_R(0);

    const uint32_t nb   = n / qk;
    const uint32_t nloe = n % qk;

    uint32_t i = 0;
    for (; i < nb; i++) {
        HVX_Vector_x8 vy0_q = hvx_vec_load_q8x4x8(y0_q + i * y_qblk_size);
        HVX_Vector_x8 vy1_q = hvx_vec_load_q8x4x8(y1_q + i * y_qblk_size);

        HVX_Vector_x8 r0_q = hvx_vec_load_q4x4x8(r0_x_q + i * x_qblk_size);
        HVX_Vector_x8 r1_q = hvx_vec_load_q4x4x8(r1_x_q + i * x_qblk_size);

        HVX_Vector r00_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_full(r0_q, vy0_q));
        HVX_Vector r01_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_full(r0_q, vy1_q));
        HVX_Vector r10_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_full(r1_q, vy0_q));
        HVX_Vector r11_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_full(r1_q, vy1_q));

        HVX_Vector vy0_d = Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (y0_d + i * y_dblk_size));
        HVX_Vector vy1_d = Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (y1_d + i * y_dblk_size));
        HVX_Vector r0_d = Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (r0_x_d + i * x_dblk_size));
        HVX_Vector r1_d = Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (r1_x_d + i * x_dblk_size));

        HVX_Vector r00_dd = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(r0_d, vy0_d)));
        HVX_Vector r01_dd = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(r0_d, vy1_d)));
        HVX_Vector r10_dd = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(r1_d, vy0_d)));
        HVX_Vector r11_dd = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(r1_d, vy1_d)));

        HVX_Vector r00_fa = Q6_Vqf32_vmpy_VsfVsf(r00_ia, r00_dd);
        HVX_Vector r01_fa = Q6_Vqf32_vmpy_VsfVsf(r01_ia, r01_dd);
        HVX_Vector r10_fa = Q6_Vqf32_vmpy_VsfVsf(r10_ia, r10_dd);
        HVX_Vector r11_fa = Q6_Vqf32_vmpy_VsfVsf(r11_ia, r11_dd);

        r00_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(r00_fa, r00_sum));
        r01_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(r01_fa, r01_sum));
        r10_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(r10_fa, r10_sum));
        r11_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(r11_fa, r11_sum));
    }

    if (nloe) {
        HVX_Vector_x8 vy0_q = hvx_vec_load_q8x4x8(y0_q + i * y_qblk_size);
        HVX_Vector_x8 vy1_q = hvx_vec_load_q8x4x8(y1_q + i * y_qblk_size);
        HVX_Vector_x8 r0_q = hvx_vec_load_q4x4x8(r0_x_q + i * x_qblk_size);
        HVX_Vector_x8 r1_q = hvx_vec_load_q4x4x8(r1_x_q + i * x_qblk_size);

        HVX_Vector r00_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_nloe(r0_q, vy0_q, nloe));
        HVX_Vector r01_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_nloe(r0_q, vy1_q, nloe));
        HVX_Vector r10_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_nloe(r1_q, vy0_q, nloe));
        HVX_Vector r11_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_nloe(r1_q, vy1_q, nloe));

        HVX_Vector vy0_d = Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (y0_d + i * y_dblk_size));
        HVX_Vector vy1_d = Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (y1_d + i * y_dblk_size));
        HVX_Vector r0_d = Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (r0_x_d + i * x_dblk_size));
        HVX_Vector r1_d = Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (r1_x_d + i * x_dblk_size));

        HVX_Vector r00_dd = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(r0_d, vy0_d)));
        HVX_Vector r01_dd = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(r0_d, vy1_d)));
        HVX_Vector r10_dd = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(r1_d, vy0_d)));
        HVX_Vector r11_dd = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(r1_d, vy1_d)));

        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe / 8);
        r00_dd               = Q6_V_vand_QV(bmask, r00_dd);
        r01_dd               = Q6_V_vand_QV(bmask, r01_dd);
        r10_dd               = Q6_V_vand_QV(bmask, r10_dd);
        r11_dd               = Q6_V_vand_QV(bmask, r11_dd);

        r00_ia               = Q6_V_vand_QV(bmask, r00_ia);
        r01_ia               = Q6_V_vand_QV(bmask, r01_ia);
        r10_ia               = Q6_V_vand_QV(bmask, r10_ia);
        r11_ia               = Q6_V_vand_QV(bmask, r11_ia);

        HVX_Vector r00_fa = Q6_Vqf32_vmpy_VsfVsf(r00_ia, r00_dd);
        HVX_Vector r01_fa = Q6_Vqf32_vmpy_VsfVsf(r01_ia, r01_dd);
        HVX_Vector r10_fa = Q6_Vqf32_vmpy_VsfVsf(r10_ia, r10_dd);
        HVX_Vector r11_fa = Q6_Vqf32_vmpy_VsfVsf(r11_ia, r11_dd);

        r00_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(r00_fa, r00_sum));
        r01_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(r01_fa, r01_sum));
        r10_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(r10_fa, r10_sum));
        r11_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(r11_fa, r11_sum));
    }

    HVX_Vector r0_sum = hvx_vec_reduce_sum_f32x2(r00_sum, r01_sum);
    HVX_Vector r1_sum = hvx_vec_reduce_sum_f32x2(r10_sum, r11_sum);
    hvx_vec_store_u(&s[0], 8, r0_sum);
    hvx_vec_store_u(&s[2], 8, r1_sum);
}

static void vec_dot_q8x4x2_q8x4x2_1x1(const int n, float * restrict s, const void * restrict vx, const void * restrict vy) {
    assert(n % 32 == 0);
    assert((unsigned long) vx % 128 == 0);
    assert((unsigned long) vy % 128 == 0);

    const uint32_t qk = QK_Q8_0x4x2 * 4;

    const uint32_t x_dblk_size = 8 * 4 * 2;
    const uint32_t x_qblk_size = qk;
    const uint32_t x_qrow_size = n;

    const uint32_t y_dblk_size = 8 * 4 * 2;
    const uint32_t y_qblk_size = qk;
    const uint32_t y_qrow_size = n;

    const uint8_t * restrict r0_x_q = ((const uint8_t *) vx + 0);
    const uint8_t * restrict r0_x_d = ((const uint8_t *) vx + x_qrow_size);

    const uint8_t * restrict y_q = ((const uint8_t *) vy + 0);
    const uint8_t * restrict y_d = ((const uint8_t *) vy + y_qrow_size);

    HVX_Vector r0_sum = Q6_V_vsplat_R(0);

    const uint32_t nb   = n / qk;
    int32_t        nloe = n % qk;

    uint32_t i = 0;
    for (; i < nb; i++) {
        HVX_Vector_x8 vy_q = hvx_vec_load_q8x4x8(y_q + i * y_qblk_size);
        HVX_Vector_x8 r0_q = hvx_vec_load_q8x4x8(r0_x_q + i * x_qblk_size);

        HVX_Vector r0_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_full(r0_q, vy_q));

        HVX_Vector vy_d = Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (y_d + i * y_dblk_size));
        HVX_Vector r0_d = Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (r0_x_d + i * x_dblk_size));

        HVX_Vector r0_dd = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(r0_d, vy_d)));

        HVX_Vector r0_fa = Q6_Vqf32_vmpy_VsfVsf(r0_ia, r0_dd);

        r0_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(r0_fa, r0_sum));
    }

    if (nloe) {
        HVX_Vector_x8 vy_q = hvx_vec_load_q8x4x8(y_q + i * y_qblk_size);
        HVX_Vector_x8 r0_q = hvx_vec_load_q8x4x8(r0_x_q + i * x_qblk_size);

        HVX_Vector r0_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_nloe(r0_q, vy_q, nloe));

        HVX_Vector vy_d = Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (y_d + i * y_dblk_size));
        HVX_Vector r0_d = Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (r0_x_d + i * x_dblk_size));

        HVX_Vector r0_dd = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(r0_d, vy_d)));

        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe / 8);
        r0_dd                = Q6_V_vand_QV(bmask, r0_dd);
        r0_ia                = Q6_V_vand_QV(bmask, r0_ia);

        HVX_Vector r0_fa = Q6_Vqf32_vmpy_VsfVsf(r0_ia, r0_dd);

        r0_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(r0_fa, r0_sum));
    }

    r0_sum = hvx_vec_reduce_sum_f32(r0_sum);

    hvx_vec_store_u(&s[0], 4, r0_sum);
}

static void vec_dot_q8x4x2_q8x4x2_2x1(const int n, float * restrict s, const void * restrict vx, uint32_t vx_row_size, const void * restrict vy) {
    assert(n % 32 == 0);
    assert((unsigned long) vx % 128 == 0);
    assert((unsigned long) vy % 128 == 0);

    const uint32_t qk = QK_Q8_0x4x2 * 4;

    const uint32_t x_dblk_size = 8 * 4 * 2;
    const uint32_t x_qblk_size = qk;
    const uint32_t x_qrow_size = n;

    const uint32_t y_dblk_size = 8 * 4 * 2;
    const uint32_t y_qblk_size = qk;
    const uint32_t y_qrow_size = n;

    const uint8_t * restrict r0_x_q = ((const uint8_t *) (vx + (0 * vx_row_size)) + 0);
    const uint8_t * restrict r0_x_d = ((const uint8_t *) (vx + (0 * vx_row_size)) + x_qrow_size);
    const uint8_t * restrict r1_x_q = ((const uint8_t *) (vx + (1 * vx_row_size)) + 0);
    const uint8_t * restrict r1_x_d = ((const uint8_t *) (vx + (1 * vx_row_size)) + x_qrow_size);

    const uint8_t * restrict y_q = ((const uint8_t *) vy + 0);
    const uint8_t * restrict y_d = ((const uint8_t *) vy + y_qrow_size);

    HVX_Vector r0_sum = Q6_V_vsplat_R(0);
    HVX_Vector r1_sum = Q6_V_vsplat_R(0);

    const uint32_t nb   = n / qk;
    int32_t        nloe = n % qk;

    uint32_t i = 0;
    for (; i < nb; i++) {
        HVX_Vector_x8 vy_q = hvx_vec_load_q8x4x8(y_q + i * y_qblk_size);
        HVX_Vector_x8 r0_q = hvx_vec_load_q8x4x8(r0_x_q + i * x_qblk_size);
        HVX_Vector_x8 r1_q = hvx_vec_load_q8x4x8(r1_x_q + i * x_qblk_size);

        HVX_Vector r0_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_full(r0_q, vy_q));
        HVX_Vector r1_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_full(r1_q, vy_q));

        HVX_Vector vy_d = Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (y_d + i * y_dblk_size));
        HVX_Vector r0_d = Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (r0_x_d + i * x_dblk_size));
        HVX_Vector r1_d = Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (r1_x_d + i * x_dblk_size));

        HVX_Vector r0_dd = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(r0_d, vy_d)));
        HVX_Vector r1_dd = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(r1_d, vy_d)));

        HVX_Vector r0_fa = Q6_Vqf32_vmpy_VsfVsf(r0_ia, r0_dd);
        HVX_Vector r1_fa = Q6_Vqf32_vmpy_VsfVsf(r1_ia, r1_dd);

        r0_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(r0_fa, r0_sum));
        r1_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(r1_fa, r1_sum));
    }

    if (nloe) {
        HVX_Vector_x8 vy_q = hvx_vec_load_q8x4x8(y_q + i * y_qblk_size);
        HVX_Vector_x8 r0_q = hvx_vec_load_q8x4x8(r0_x_q + i * x_qblk_size);
        HVX_Vector_x8 r1_q = hvx_vec_load_q8x4x8(r1_x_q + i * x_qblk_size);

        HVX_Vector r0_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_nloe(r0_q, vy_q, nloe));
        HVX_Vector r1_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_nloe(r1_q, vy_q, nloe));

        HVX_Vector vy_d = Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (y_d + i * y_dblk_size));
        HVX_Vector r0_d = Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (r0_x_d + i * x_dblk_size));
        HVX_Vector r1_d = Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (r1_x_d + i * x_dblk_size));

        HVX_Vector r0_dd = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(r0_d, vy_d)));
        HVX_Vector r1_dd = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(r1_d, vy_d)));

        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe / 8);
        r0_dd                = Q6_V_vand_QV(bmask, r0_dd);
        r1_dd                = Q6_V_vand_QV(bmask, r1_dd);
        r0_ia                = Q6_V_vand_QV(bmask, r0_ia);
        r1_ia                = Q6_V_vand_QV(bmask, r1_ia);

        HVX_Vector r0_fa = Q6_Vqf32_vmpy_VsfVsf(r0_ia, r0_dd);
        HVX_Vector r1_fa = Q6_Vqf32_vmpy_VsfVsf(r1_ia, r1_dd);

        r0_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(r0_fa, r0_sum));
        r1_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(r1_fa, r1_sum));
    }

    HVX_Vector rsum = hvx_vec_reduce_sum_f32x2(r0_sum, r1_sum);
    hvx_vec_store_u(&s[0], 8, rsum);
}

static void vec_dot_q8x4x2_q8x4x2_2x2(const int n, float * restrict s, const void * restrict vx, uint32_t vx_row_size, const void * restrict vy, uint32_t vy_row_size) {
    assert(n % 32 == 0);

    const uint32_t qk = QK_Q8_0x4x2 * 4;

    const uint32_t x_dblk_size = 8 * 4 * 2;
    const uint32_t x_qblk_size = qk;
    const uint32_t x_qrow_size = n;

    const uint32_t y_dblk_size = 8 * 4 * 2;
    const uint32_t y_qblk_size = qk;
    const uint32_t y_qrow_size = n;

    const uint8_t * restrict r0_x_q = ((const uint8_t *) (vx + (0 * vx_row_size)) + 0);
    const uint8_t * restrict r0_x_d = ((const uint8_t *) (vx + (0 * vx_row_size)) + x_qrow_size);
    const uint8_t * restrict r1_x_q = ((const uint8_t *) (vx + (1 * vx_row_size)) + 0);
    const uint8_t * restrict r1_x_d = ((const uint8_t *) (vx + (1 * vx_row_size)) + x_qrow_size);

    const uint8_t * restrict y0_q = ((const uint8_t *) (vy + (0 * vy_row_size)) + 0);
    const uint8_t * restrict y0_d = ((const uint8_t *) (vy + (0 * vy_row_size)) + y_qrow_size);
    const uint8_t * restrict y1_q = ((const uint8_t *) (vy + (1 * vy_row_size)) + 0);
    const uint8_t * restrict y1_d = ((const uint8_t *) (vy + (1 * vy_row_size)) + y_qrow_size);

    HVX_Vector r00_sum = Q6_V_vsplat_R(0);
    HVX_Vector r01_sum = Q6_V_vsplat_R(0);
    HVX_Vector r10_sum = Q6_V_vsplat_R(0);
    HVX_Vector r11_sum = Q6_V_vsplat_R(0);

    const uint32_t nb   = n / qk;
    int32_t        nloe = n % qk;

    uint32_t i = 0;
    for (; i < nb; i++) {
        HVX_Vector_x8 vy0_q = hvx_vec_load_q8x4x8(y0_q + i * y_qblk_size);
        HVX_Vector_x8 vy1_q = hvx_vec_load_q8x4x8(y1_q + i * y_qblk_size);
        HVX_Vector_x8 r0_q = hvx_vec_load_q8x4x8(r0_x_q + i * x_qblk_size);
        HVX_Vector_x8 r1_q = hvx_vec_load_q8x4x8(r1_x_q + i * x_qblk_size);

        HVX_Vector r00_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_full(r0_q, vy0_q));
        HVX_Vector r01_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_full(r0_q, vy1_q));
        HVX_Vector r10_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_full(r1_q, vy0_q));
        HVX_Vector r11_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_full(r1_q, vy1_q));

        HVX_Vector vy0_d = Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (y0_d + i * y_dblk_size));
        HVX_Vector vy1_d = Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (y1_d + i * y_dblk_size));
        HVX_Vector r0_d = Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (r0_x_d + i * x_dblk_size));
        HVX_Vector r1_d = Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (r1_x_d + i * x_dblk_size));

        HVX_Vector r00_dd = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(r0_d, vy0_d)));
        HVX_Vector r01_dd = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(r0_d, vy1_d)));
        HVX_Vector r10_dd = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(r1_d, vy0_d)));
        HVX_Vector r11_dd = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(r1_d, vy1_d)));

        HVX_Vector r00_fa = Q6_Vqf32_vmpy_VsfVsf(r00_ia, r00_dd);
        HVX_Vector r01_fa = Q6_Vqf32_vmpy_VsfVsf(r01_ia, r01_dd);
        HVX_Vector r10_fa = Q6_Vqf32_vmpy_VsfVsf(r10_ia, r10_dd);
        HVX_Vector r11_fa = Q6_Vqf32_vmpy_VsfVsf(r11_ia, r11_dd);

        r00_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(r00_fa, r00_sum));
        r01_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(r01_fa, r01_sum));
        r10_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(r10_fa, r10_sum));
        r11_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(r11_fa, r11_sum));
    }

    if (nloe) {
        HVX_Vector_x8 vy0_q = hvx_vec_load_q8x4x8(y0_q + i * y_qblk_size);
        HVX_Vector_x8 vy1_q = hvx_vec_load_q8x4x8(y1_q + i * y_qblk_size);
        HVX_Vector_x8 r0_q = hvx_vec_load_q8x4x8(r0_x_q + i * x_qblk_size);
        HVX_Vector_x8 r1_q = hvx_vec_load_q8x4x8(r1_x_q + i * x_qblk_size);

        HVX_Vector r00_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_nloe(r0_q, vy0_q, nloe));
        HVX_Vector r01_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_nloe(r0_q, vy1_q, nloe));
        HVX_Vector r10_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_nloe(r1_q, vy0_q, nloe));
        HVX_Vector r11_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_nloe(r1_q, vy1_q, nloe));

        HVX_Vector vy0_d = Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (y0_d + i * y_dblk_size));
        HVX_Vector vy1_d = Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (y1_d + i * y_dblk_size));
        HVX_Vector r0_d = Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (r0_x_d + i * x_dblk_size));
        HVX_Vector r1_d = Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (r1_x_d + i * x_dblk_size));

        HVX_Vector r00_dd = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(r0_d, vy0_d)));
        HVX_Vector r01_dd = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(r0_d, vy1_d)));
        HVX_Vector r10_dd = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(r1_d, vy0_d)));
        HVX_Vector r11_dd = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(r1_d, vy1_d)));

        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe / 8);
        r00_dd               = Q6_V_vand_QV(bmask, r00_dd);
        r01_dd               = Q6_V_vand_QV(bmask, r01_dd);
        r10_dd               = Q6_V_vand_QV(bmask, r10_dd);
        r11_dd               = Q6_V_vand_QV(bmask, r11_dd);

        r00_ia               = Q6_V_vand_QV(bmask, r00_ia);
        r01_ia               = Q6_V_vand_QV(bmask, r01_ia);
        r10_ia               = Q6_V_vand_QV(bmask, r10_ia);
        r11_ia               = Q6_V_vand_QV(bmask, r11_ia);

        HVX_Vector r00_fa = Q6_Vqf32_vmpy_VsfVsf(r00_ia, r00_dd);
        HVX_Vector r01_fa = Q6_Vqf32_vmpy_VsfVsf(r01_ia, r01_dd);
        HVX_Vector r10_fa = Q6_Vqf32_vmpy_VsfVsf(r10_ia, r10_dd);
        HVX_Vector r11_fa = Q6_Vqf32_vmpy_VsfVsf(r11_ia, r11_dd);

        r00_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(r00_fa, r00_sum));
        r01_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(r01_fa, r01_sum));
        r10_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(r10_fa, r10_sum));
        r11_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(r11_fa, r11_sum));
    }

    HVX_Vector r0_sum = hvx_vec_reduce_sum_f32x2(r00_sum, r01_sum);
    HVX_Vector r1_sum = hvx_vec_reduce_sum_f32x2(r10_sum, r11_sum);
    hvx_vec_store_u(&s[0], 8, r0_sum);
    hvx_vec_store_u(&s[2], 8, r1_sum);
}

static void vec_dot_mxfp4x4x2_q8x4x2_1x1(const int n,
                                     float * restrict s,
                                     const void * restrict vx,
                                     const void * restrict vy) {
    assert(n % 32 == 0);
    assert((unsigned long) vx % 128 == 0);
    assert((unsigned long) vy % 128 == 0);

    const uint32_t qk = QK_MXFP4x4x2 * 4;

    const uint32_t x_dblk_size = 8 * 4 * 1;
    const uint32_t x_qblk_size = qk / 2;
    const uint32_t x_qrow_size = n / 2;

    const uint32_t y_dblk_size = 8 * 4 * 2;
    const uint32_t y_qblk_size = qk;
    const uint32_t y_qrow_size = n;

    const uint8_t * restrict r0_x_q = ((const uint8_t *) vx + 0);
    const uint8_t * restrict r0_x_d = ((const uint8_t *) vx + x_qrow_size);

    const uint8_t * restrict y_q = ((const uint8_t *) vy + 0);
    const uint8_t * restrict y_d = ((const uint8_t *) vy + y_qrow_size);

    HVX_Vector r0_sum = Q6_V_vsplat_R(0);

    const uint32_t nb   = n / qk;
    int32_t        nloe = n % qk;

    uint32_t i = 0;
    for (; i < nb; i++) {
        HVX_Vector_x8 vy_q = hvx_vec_load_q8x4x8(y_q + i * y_qblk_size);
        HVX_Vector_x8 r0_q = hvx_vec_load_mxfp4x4x8(r0_x_q + i * x_qblk_size);

        HVX_Vector r0_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_full(r0_q, vy_q));

        HVX_Vector vy_d = *(const HVX_UVector *) (y_d + i * y_dblk_size);
        HVX_Vector r0_d = *(const HVX_UVector *) (r0_x_d + i * x_dblk_size);

        HVX_Vector half = Q6_Vh_vsplat_R(0x3800);
        vy_d            = Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(vy_d), half));
        vy_d            = Q6_Vsf_equals_Vqf32(vy_d);

        HVX_Vector expand    = *(const HVX_Vector *) expand_x32_e8m0;
        HVX_Vector e8m0_mask = Q6_V_vsplat_R(0x000000ff);
        r0_d                 = Q6_V_vdelta_VV(r0_d, expand);
        r0_d                 = Q6_V_vand_VV(r0_d, e8m0_mask);
        r0_d                 = Q6_Vw_vasl_VwR(r0_d, 23);

        HVX_Vector r0_dd = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(r0_d, vy_d));

        HVX_Vector r0_fa = Q6_Vqf32_vmpy_VsfVsf(r0_ia, r0_dd);

        r0_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(r0_fa, r0_sum));
    }

    if (nloe) {
        HVX_Vector_x8 vy_q = hvx_vec_load_q8x4x8(y_q + i * y_qblk_size);
        HVX_Vector_x8 r0_q = hvx_vec_load_mxfp4x4x8(r0_x_q + i * x_qblk_size);

        HVX_Vector r0_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_full(r0_q, vy_q));

        HVX_Vector vy_d = *(const HVX_UVector *) (y_d + i * y_dblk_size);
        HVX_Vector r0_d = *(const HVX_UVector *) (r0_x_d + i * x_dblk_size);

        HVX_Vector half = Q6_Vh_vsplat_R(0x3800);
        vy_d            = Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(vy_d), half));
        vy_d            = Q6_Vsf_equals_Vqf32(vy_d);

        HVX_Vector expand    = *(const HVX_Vector *) expand_x32_e8m0;
        HVX_Vector e8m0_mask = Q6_V_vsplat_R(0x000000ff);
        r0_d                 = Q6_V_vdelta_VV(r0_d, expand);
        r0_d                 = Q6_V_vand_VV(r0_d, e8m0_mask);
        r0_d                 = Q6_Vw_vasl_VwR(r0_d, 23);

        HVX_Vector r0_dd = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(r0_d, vy_d));

        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe / 8);
        r0_dd                = Q6_V_vand_QV(bmask, r0_dd);
        r0_ia                = Q6_V_vand_QV(bmask, r0_ia);

        HVX_Vector r0_fa = Q6_Vqf32_vmpy_VsfVsf(r0_ia, r0_dd);

        r0_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(r0_fa, r0_sum));
    }

    r0_sum = hvx_vec_reduce_sum_f32(r0_sum);

    hvx_vec_store_u(&s[0], 4, r0_sum);
}

static void vec_dot_mxfp4x4x2_q8x4x2_2x1(const int n,
                                         float * restrict s,
                                         const void * restrict vx,
                                         uint32_t vx_row_size,
                                         const void * restrict vy) {
    assert(n % 32 == 0);
    assert((unsigned long) vx % 128 == 0);
    assert((unsigned long) vy % 128 == 0);

    const uint32_t qk = QK_MXFP4x4x2 * 4;

    const uint32_t x_dblk_size = 8 * 4 * 1;
    const uint32_t x_qblk_size = qk / 2;
    const uint32_t x_qrow_size = n / 2;

    const uint32_t y_dblk_size = 8 * 4 * 2;
    const uint32_t y_qblk_size = qk;
    const uint32_t y_qrow_size = n;

    const uint8_t * restrict r0_x_q = ((const uint8_t *) (vx + (0 * vx_row_size)) + 0);
    const uint8_t * restrict r0_x_d = ((const uint8_t *) (vx + (0 * vx_row_size)) + x_qrow_size);

    const uint8_t * restrict r1_x_q = ((const uint8_t *) (vx + (1 * vx_row_size)) + 0);
    const uint8_t * restrict r1_x_d = ((const uint8_t *) (vx + (1 * vx_row_size)) + x_qrow_size);

    const uint8_t * restrict y_q = ((const uint8_t *) vy + 0);
    const uint8_t * restrict y_d = ((const uint8_t *) vy + y_qrow_size);

    HVX_Vector r0_sum = Q6_V_vsplat_R(0);
    HVX_Vector r1_sum = Q6_V_vsplat_R(0);

    const uint32_t nb   = n / qk;
    int32_t        nloe = n % qk;

    uint32_t i = 0;
    for (; i < nb; i++) {
        HVX_Vector_x8 vy_q = hvx_vec_load_q8x4x8(y_q + i * y_qblk_size);
        HVX_Vector_x8 r0_q = hvx_vec_load_mxfp4x4x8(r0_x_q + i * x_qblk_size);
        HVX_Vector_x8 r1_q = hvx_vec_load_mxfp4x4x8(r1_x_q + i * x_qblk_size);

        HVX_Vector r0_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_full(r0_q, vy_q));
        HVX_Vector r1_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_full(r1_q, vy_q));

        HVX_Vector vy_d = *(const HVX_UVector *) (y_d + i * y_dblk_size);
        HVX_Vector r0_d = *(const HVX_UVector *) (r0_x_d + i * x_dblk_size);
        HVX_Vector r1_d = *(const HVX_UVector *) (r1_x_d + i * x_dblk_size);

        HVX_Vector half = Q6_Vh_vsplat_R(0x3800);
        vy_d            = Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(vy_d), half));
        vy_d            = Q6_Vsf_equals_Vqf32(vy_d);

        HVX_Vector expand    = *(const HVX_Vector *) expand_x32_e8m0;
        HVX_Vector e8m0_mask = Q6_V_vsplat_R(0x000000ff);
        r0_d                 = Q6_V_vdelta_VV(r0_d, expand);
        r0_d                 = Q6_V_vand_VV(r0_d, e8m0_mask);
        r0_d                 = Q6_Vw_vasl_VwR(r0_d, 23);
        r1_d                 = Q6_V_vdelta_VV(r1_d, expand);
        r1_d                 = Q6_V_vand_VV(r1_d, e8m0_mask);
        r1_d                 = Q6_Vw_vasl_VwR(r1_d, 23);

        HVX_Vector r0_dd = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(r0_d, vy_d));
        HVX_Vector r1_dd = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(r1_d, vy_d));

        HVX_Vector r0_fa = Q6_Vqf32_vmpy_VsfVsf(r0_ia, r0_dd);
        HVX_Vector r1_fa = Q6_Vqf32_vmpy_VsfVsf(r1_ia, r1_dd);

        r0_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(r0_fa, r0_sum));
        r1_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(r1_fa, r1_sum));
    }

    if (nloe) {
        HVX_Vector_x8 vy_q = hvx_vec_load_q8x4x8(y_q + i * y_qblk_size);
        HVX_Vector_x8 r0_q = hvx_vec_load_mxfp4x4x8(r0_x_q + i * x_qblk_size);
        HVX_Vector_x8 r1_q = hvx_vec_load_mxfp4x4x8(r1_x_q + i * x_qblk_size);

        HVX_Vector r0_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_full(r0_q, vy_q));
        HVX_Vector r1_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_full(r1_q, vy_q));

        HVX_Vector vy_d = *(const HVX_UVector *) (y_d + i * y_dblk_size);
        HVX_Vector r0_d = *(const HVX_UVector *) (r0_x_d + i * x_dblk_size);
        HVX_Vector r1_d = *(const HVX_UVector *) (r1_x_d + i * x_dblk_size);

        HVX_Vector half = Q6_Vh_vsplat_R(0x3800);
        vy_d            = Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(vy_d), half));
        vy_d            = Q6_Vsf_equals_Vqf32(vy_d);

        HVX_Vector expand    = *(const HVX_Vector *) expand_x32_e8m0;
        HVX_Vector e8m0_mask = Q6_V_vsplat_R(0x000000ff);
        r0_d                 = Q6_V_vdelta_VV(r0_d, expand);
        r0_d                 = Q6_V_vand_VV(r0_d, e8m0_mask);
        r0_d                 = Q6_Vw_vasl_VwR(r0_d, 23);
        r1_d                 = Q6_V_vdelta_VV(r1_d, expand);
        r1_d                 = Q6_V_vand_VV(r1_d, e8m0_mask);
        r1_d                 = Q6_Vw_vasl_VwR(r1_d, 23);

        HVX_Vector r0_dd = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(r0_d, vy_d));
        HVX_Vector r1_dd = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(r1_d, vy_d));

        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe / 8);
        r0_dd                = Q6_V_vand_QV(bmask, r0_dd);
        r1_dd                = Q6_V_vand_QV(bmask, r1_dd);
        r0_ia                = Q6_V_vand_QV(bmask, r0_ia);
        r1_ia                = Q6_V_vand_QV(bmask, r1_ia);

        HVX_Vector r0_fa = Q6_Vqf32_vmpy_VsfVsf(r0_ia, r0_dd);
        HVX_Vector r1_fa = Q6_Vqf32_vmpy_VsfVsf(r1_ia, r1_dd);

        r0_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(r0_fa, r0_sum));
        r1_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(r1_fa, r1_sum));
    }

    HVX_Vector rsum = hvx_vec_reduce_sum_f32x2(r0_sum, r1_sum);
    hvx_vec_store_u(&s[0], 8, rsum);
}

static void vec_dot_mxfp4x4x2_q8x4x2_2x2(const int n, float * restrict s, const void * restrict vx, uint32_t vx_row_size, const void * restrict vy, uint32_t vy_row_size) {
    assert(n % 32 == 0);

    const uint32_t qk = QK_MXFP4x4x2 * 4;

    const uint32_t x_dblk_size = 8 * 4 * 1;
    const uint32_t x_qblk_size = qk / 2;
    const uint32_t x_qrow_size = n / 2;

    const uint32_t y_dblk_size = 8 * 4 * 2;
    const uint32_t y_qblk_size = qk;
    const uint32_t y_qrow_size = n;

    const uint8_t * restrict r0_x_q = ((const uint8_t *) (vx + (0 * vx_row_size)) + 0);
    const uint8_t * restrict r0_x_d = ((const uint8_t *) (vx + (0 * vx_row_size)) + x_qrow_size);
    const uint8_t * restrict r1_x_q = ((const uint8_t *) (vx + (1 * vx_row_size)) + 0);
    const uint8_t * restrict r1_x_d = ((const uint8_t *) (vx + (1 * vx_row_size)) + x_qrow_size);

    const uint8_t * restrict y0_q = ((const uint8_t *) (vy + (0 * vy_row_size)) + 0);
    const uint8_t * restrict y0_d = ((const uint8_t *) (vy + (0 * vy_row_size)) + y_qrow_size);
    const uint8_t * restrict y1_q = ((const uint8_t *) (vy + (1 * vy_row_size)) + 0);
    const uint8_t * restrict y1_d = ((const uint8_t *) (vy + (1 * vy_row_size)) + y_qrow_size);

    HVX_Vector r00_sum = Q6_V_vsplat_R(0);
    HVX_Vector r01_sum = Q6_V_vsplat_R(0);
    HVX_Vector r10_sum = Q6_V_vsplat_R(0);
    HVX_Vector r11_sum = Q6_V_vsplat_R(0);

    const uint32_t nb   = n / qk;
    int32_t        nloe = n % qk;

    uint32_t i = 0;
    for (; i < nb; i++) {
        HVX_Vector_x8 vy0_q = hvx_vec_load_q8x4x8(y0_q + i * y_qblk_size);
        HVX_Vector_x8 vy1_q = hvx_vec_load_q8x4x8(y1_q + i * y_qblk_size);

        HVX_Vector_x8 r0_q = hvx_vec_load_mxfp4x4x8(r0_x_q + i * x_qblk_size);
        HVX_Vector_x8 r1_q = hvx_vec_load_mxfp4x4x8(r1_x_q + i * x_qblk_size);

        HVX_Vector r00_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_full(r0_q, vy0_q));
        HVX_Vector r01_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_full(r0_q, vy1_q));
        HVX_Vector r10_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_full(r1_q, vy0_q));
        HVX_Vector r11_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_full(r1_q, vy1_q));

        HVX_Vector vy0_d = *(const HVX_UVector *) (y0_d + i * y_dblk_size);
        HVX_Vector vy1_d = *(const HVX_UVector *) (y1_d + i * y_dblk_size);
        HVX_Vector r0_d = *(const HVX_UVector *) (r0_x_d + i * x_dblk_size);
        HVX_Vector r1_d = *(const HVX_UVector *) (r1_x_d + i * x_dblk_size);

        HVX_Vector half = Q6_Vh_vsplat_R(0x3800);
        vy0_d           = Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(vy0_d), half));
        vy0_d           = Q6_Vsf_equals_Vqf32(vy0_d);
        vy1_d           = Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(vy1_d), half));
        vy1_d           = Q6_Vsf_equals_Vqf32(vy1_d);

        HVX_Vector expand    = *(const HVX_Vector *) expand_x32_e8m0;
        HVX_Vector e8m0_mask = Q6_V_vsplat_R(0x000000ff);
        r0_d                 = Q6_V_vdelta_VV(r0_d, expand);
        r0_d                 = Q6_V_vand_VV(r0_d, e8m0_mask);
        r0_d                 = Q6_Vw_vasl_VwR(r0_d, 23);
        r1_d                 = Q6_V_vdelta_VV(r1_d, expand);
        r1_d                 = Q6_V_vand_VV(r1_d, e8m0_mask);
        r1_d                 = Q6_Vw_vasl_VwR(r1_d, 23);

        HVX_Vector r00_dd = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(r0_d, vy0_d));
        HVX_Vector r01_dd = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(r0_d, vy1_d));
        HVX_Vector r10_dd = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(r1_d, vy0_d));
        HVX_Vector r11_dd = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(r1_d, vy1_d));

        HVX_Vector r00_fa = Q6_Vqf32_vmpy_VsfVsf(r00_ia, r00_dd);
        HVX_Vector r01_fa = Q6_Vqf32_vmpy_VsfVsf(r01_ia, r01_dd);
        HVX_Vector r10_fa = Q6_Vqf32_vmpy_VsfVsf(r10_ia, r10_dd);
        HVX_Vector r11_fa = Q6_Vqf32_vmpy_VsfVsf(r11_ia, r11_dd);

        r00_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(r00_fa, r00_sum));
        r01_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(r01_fa, r01_sum));
        r10_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(r10_fa, r10_sum));
        r11_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(r11_fa, r11_sum));
    }

    if (nloe) {
        HVX_Vector_x8 vy0_q = hvx_vec_load_q8x4x8(y0_q + i * y_qblk_size);
        HVX_Vector_x8 vy1_q = hvx_vec_load_q8x4x8(y1_q + i * y_qblk_size);
        HVX_Vector_x8 r0_q = hvx_vec_load_mxfp4x4x8(r0_x_q + i * x_qblk_size);
        HVX_Vector_x8 r1_q = hvx_vec_load_mxfp4x4x8(r1_x_q + i * x_qblk_size);

        HVX_Vector r00_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_full(r0_q, vy0_q));
        HVX_Vector r01_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_full(r0_q, vy1_q));
        HVX_Vector r10_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_full(r1_q, vy0_q));
        HVX_Vector r11_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_full(r1_q, vy1_q));

        HVX_Vector vy0_d = *(const HVX_UVector *) (y0_d + i * y_dblk_size);
        HVX_Vector vy1_d = *(const HVX_UVector *) (y1_d + i * y_dblk_size);
        HVX_Vector r0_d = *(const HVX_UVector *) (r0_x_d + i * x_dblk_size);
        HVX_Vector r1_d = *(const HVX_UVector *) (r1_x_d + i * x_dblk_size);

        HVX_Vector half = Q6_Vh_vsplat_R(0x3800);
        vy0_d           = Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(vy0_d), half));
        vy0_d           = Q6_Vsf_equals_Vqf32(vy0_d);
        vy1_d           = Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(vy1_d), half));
        vy1_d           = Q6_Vsf_equals_Vqf32(vy1_d);

        HVX_Vector expand    = *(const HVX_Vector *) expand_x32_e8m0;
        HVX_Vector e8m0_mask = Q6_V_vsplat_R(0x000000ff);
        r0_d                 = Q6_V_vdelta_VV(r0_d, expand);
        r0_d                 = Q6_V_vand_VV(r0_d, e8m0_mask);
        r0_d                 = Q6_Vw_vasl_VwR(r0_d, 23);
        r1_d                 = Q6_V_vdelta_VV(r1_d, expand);
        r1_d                 = Q6_V_vand_VV(r1_d, e8m0_mask);
        r1_d                 = Q6_Vw_vasl_VwR(r1_d, 23);

        HVX_Vector r00_dd = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(r0_d, vy0_d));
        HVX_Vector r01_dd = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(r0_d, vy1_d));
        HVX_Vector r10_dd = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(r1_d, vy0_d));
        HVX_Vector r11_dd = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(r1_d, vy1_d));

        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe / 8);
        r00_dd               = Q6_V_vand_QV(bmask, r00_dd);
        r01_dd               = Q6_V_vand_QV(bmask, r01_dd);
        r10_dd               = Q6_V_vand_QV(bmask, r10_dd);
        r11_dd               = Q6_V_vand_QV(bmask, r11_dd);

        r00_ia               = Q6_V_vand_QV(bmask, r00_ia);
        r01_ia               = Q6_V_vand_QV(bmask, r01_ia);
        r10_ia               = Q6_V_vand_QV(bmask, r10_ia);
        r11_ia               = Q6_V_vand_QV(bmask, r11_ia);

        HVX_Vector r00_fa = Q6_Vqf32_vmpy_VsfVsf(r00_ia, r00_dd);
        HVX_Vector r01_fa = Q6_Vqf32_vmpy_VsfVsf(r01_ia, r01_dd);
        HVX_Vector r10_fa = Q6_Vqf32_vmpy_VsfVsf(r10_ia, r10_dd);
        HVX_Vector r11_fa = Q6_Vqf32_vmpy_VsfVsf(r11_ia, r11_dd);

        r00_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(r00_fa, r00_sum));
        r01_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(r01_fa, r01_sum));
        r10_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(r10_fa, r10_sum));
        r11_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(r11_fa, r11_sum));
    }

    HVX_Vector r0_sum = hvx_vec_reduce_sum_f32x2(r00_sum, r01_sum);
    HVX_Vector r1_sum = hvx_vec_reduce_sum_f32x2(r10_sum, r11_sum);
    hvx_vec_store_u(&s[0], 8, r0_sum);
    hvx_vec_store_u(&s[2], 8, r1_sum);
}
