#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <HAP_perf.h>

#include <math.h>
#include <string.h>

#include "hvx-utils.h"

float hvx_self_sum_f32(const uint8_t * restrict src, const int num_elems) {
    int left_over       = num_elems & (VLEN_FP32 - 1);
    int num_elems_whole = num_elems - left_over;

    int unaligned_addr = 0;
    int unaligned_loop = 0;
    if (0 == hex_is_aligned((void *) src, VLEN)) {
        FARF(HIGH, "hvx_self_sum_f32: unaligned address in hvx op, possibly slower execution\n");
        unaligned_addr = 1;
    }

    if ((1 == unaligned_addr) && (num_elems_whole != 0)) {
        unaligned_loop = 1;
        FARF(HIGH, "hvx_self_sum_f32: unaligned loop in hvx op, possibly slower execution\n");
    }

    HVX_Vector sum_vec  = Q6_V_vsplat_R(0x00000000);
    HVX_Vector zero_vec = Q6_V_vsplat_R(0x00000000);

    if (0 == unaligned_loop) {
        HVX_Vector * vec_in = (HVX_Vector *) src;

        #pragma unroll(4)
        for (int i = 0; i < num_elems_whole; i += VLEN_FP32) {
            // sum_vec = Q6_Vqf32_vadd_Vqf32Vsf(sum_vec, *vec_in++);
            sum_vec = Q6_Vqf32_vadd_VsfVsf(Q6_Vsf_equals_Vqf32(sum_vec), *vec_in++);
        }
    } else {
        #pragma unroll(4)
        for (int i = 0; i < num_elems_whole; i += VLEN_FP32) {
            HVX_Vector in = *(HVX_UVector *) (src + i * SIZEOF_FP32);

            sum_vec = Q6_Vqf32_vadd_VsfVsf(Q6_Vsf_equals_Vqf32(sum_vec), in);
        }
    }

    if (left_over > 0) {
        const float * srcf = (const float *) src + num_elems_whole;

        HVX_Vector vec_left = *(HVX_UVector *) srcf;
        HVX_Vector vec_tmp  = Q6_V_valign_VVR(vec_left, zero_vec, left_over * SIZEOF_FP32);
        // sum_vec = Q6_Vqf32_vadd_Vqf32Vsf(sum_vec, vec_tmp);
        sum_vec             = Q6_Vqf32_vadd_VsfVsf(Q6_Vsf_equals_Vqf32(sum_vec), vec_tmp);
    }

    HVX_Vector v = hvx_vec_qf32_reduce_sum(sum_vec);
    return hvx_vec_get_fp32(Q6_Vsf_equals_Vqf32(v));
}

float hvx_self_max_f32(const uint8_t * restrict src, const int num_elems) {
    int left_over       = num_elems & (VLEN_FP32 - 1);
    int num_elems_whole = num_elems - left_over;

    int unaligned_addr = 0;
    int unaligned_loop = 0;
    if (0 == hex_is_aligned((void *) src, VLEN)) {
        FARF(HIGH, "hvx_self_max_f32: unaligned address in hvx op, possibly slower execution\n");
        unaligned_addr = 1;
    }

    if ((1 == unaligned_addr) && (num_elems_whole != 0)) {
        unaligned_loop = 1;
        FARF(HIGH, "hvx_self_max_f32: unaligned loop in hvx op, possibly slower execution\n");
    }

    HVX_Vector vec_max   = hvx_vec_splat_fp32(((const float *) src)[0]);
    HVX_Vector vec_first = hvx_vec_splat_fp32(((const float *) src)[0]);

    if (0 == unaligned_loop) {
        HVX_Vector * restrict vec_in = (HVX_Vector *) src;

        #pragma unroll(4)
        for (int i = 0; i < num_elems_whole; i += VLEN_FP32) {
            vec_max = Q6_Vsf_vmax_VsfVsf(vec_max, *vec_in++);
        }
    } else {
        #pragma unroll(4)
        for (int i = 0; i < num_elems_whole; i += VLEN_FP32) {
            HVX_Vector in = *(HVX_UVector *) (src + i * SIZEOF_FP32);

            vec_max = Q6_Vsf_vmax_VsfVsf(vec_max, in);
        }
    }

    if (left_over > 0) {
        const float * srcf = (const float *) src + num_elems_whole;

        HVX_Vector in = *(HVX_UVector *) srcf;

        HVX_Vector temp = Q6_V_valign_VVR(in, vec_first, left_over * SIZEOF_FP32);
        vec_max         = Q6_Vsf_vmax_VsfVsf(vec_max, temp);
    }

    HVX_Vector v = hvx_vec_reduce_max_fp32(vec_max);
    return hvx_vec_get_fp32(v);
}
