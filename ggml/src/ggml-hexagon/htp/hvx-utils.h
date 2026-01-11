#ifndef HVX_UTILS_H
#define HVX_UTILS_H

#include <stdbool.h>
#include <stdint.h>

#include "hex-utils.h"

#include "hvx-types.h"
#include "hvx-copy.h"
#include "hvx-scale.h"
#include "hvx-exp.h"
#include "hvx-base.h"

static inline HVX_Vector hvx_vec_int32_reduce_sum_n(HVX_Vector in, unsigned int n) {
    unsigned int total = n * 4;  // total vec nbytes
    unsigned int width = 4;      // int32

    HVX_Vector sum = in, sum_t;
    while (width < total) {
        sum_t = Q6_V_vror_VR(sum, width);     // rotate right
        sum   = Q6_Vw_vadd_VwVw(sum_t, sum);  // elementwise sum
        width = width << 1;
    }
    return sum;
}

static inline HVX_Vector hvx_vec_int32_reduce_sum(HVX_Vector in) {
    return hvx_vec_int32_reduce_sum_n(in, 32);
}

static inline HVX_Vector hvx_vec_qf32_reduce_sum_n(HVX_Vector in, unsigned int n) {
    unsigned int total = n * 4;  // total vec nbytes
    unsigned int width = 4;      // fp32 nbytes

    HVX_Vector sum = in, sum_t;
    while (width < total) {
        sum_t = Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(sum), width);  // rotate right
        sum   = Q6_Vqf32_vadd_Vqf32Vsf(sum, sum_t);             // elementwise sum
        width = width << 1;
    }
    return sum;
}

static inline HVX_Vector hvx_vec_qf32_reduce_sum(HVX_Vector in) {
    return hvx_vec_qf32_reduce_sum_n(in, 32);
}

static inline HVX_Vector hvx_vec_fp32_reduce_sum_n(HVX_Vector in, unsigned int n) {
    unsigned int total = n * 4;  // total vec nbytes
    unsigned int width = 4;      // fp32 nbytes

    HVX_Vector sum = in, sum_t;
    while (width < total) {
        sum_t = Q6_V_vror_VR(sum, width);                               // rotate right
        sum   = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(sum, sum_t));  // elementwise sum
        width = width << 1;
    }
    return sum;
}

static inline HVX_Vector hvx_vec_fp32_reduce_sum(HVX_Vector in) {
    return hvx_vec_fp32_reduce_sum_n(in, 32);
}

static inline HVX_Vector hvx_vec_reduce_max_fp16(HVX_Vector in) {
    unsigned total = 128;  // total vec nbytes
    unsigned width = 2;    // fp16 nbytes

    HVX_Vector _max = in, _max_t;
    while (width < total) {
        _max_t = Q6_V_vror_VR(_max, width);         // rotate right
        _max   = Q6_Vhf_vmax_VhfVhf(_max_t, _max);  // elementwise max
        width  = width << 1;
    }

    return _max;
}

static inline HVX_Vector hvx_vec_reduce_max2_fp16(HVX_Vector in, HVX_Vector _max) {
    unsigned total = 128;  // total vec nbytes
    unsigned width = 2;    // fp32 nbytes

    HVX_Vector _max_t;

    _max = Q6_Vhf_vmax_VhfVhf(in, _max);
    while (width < total) {
        _max_t = Q6_V_vror_VR(_max, width);         // rotate right
        _max   = Q6_Vhf_vmax_VhfVhf(_max_t, _max);  // elementwise max
        width  = width << 1;
    }

    return _max;
}

static inline HVX_Vector hvx_vec_reduce_max_fp32(HVX_Vector in) {
    unsigned total = 128;  // total vec nbytes
    unsigned width = 4;    // fp32 nbytes

    HVX_Vector _max = in, _max_t;
    while (width < total) {
        _max_t = Q6_V_vror_VR(_max, width);         // rotate right
        _max   = Q6_Vsf_vmax_VsfVsf(_max_t, _max);  // elementwise max
        width  = width << 1;
    }

    return _max;
}

static inline HVX_Vector hvx_vec_reduce_max2_fp32(HVX_Vector in, HVX_Vector _max) {
    unsigned total = 128;  // total vec nbytes
    unsigned width = 4;    // fp32 nbytes

    HVX_Vector _max_t;

    _max = Q6_Vsf_vmax_VsfVsf(in, _max);
    while (width < total) {
        _max_t = Q6_V_vror_VR(_max, width);         // rotate right
        _max   = Q6_Vsf_vmax_VsfVsf(_max_t, _max);  // elementwise max
        width  = width << 1;
    }

    return _max;
}

// ====================================================
// FUNCTION: 1/(x+1)     y(0) = 1,  y(0.5) = 0.6667, y(1) = 0.5
// Order:3; continuity: True; Ends forced: True
// Mode: unsigned;   Result fractional bits: 14
// Peak Error: 1.1295e-04  Rms Error: 2.8410e-05   Mean Error: 1.1370e-05
//      32769  -32706   31252  -10589
//      32590  -30635   22793   -4493
//      32066  -27505   16481   -2348
//      31205  -24054   11849   -1306

static inline HVX_Vector hvx_vec_recip_xp1_O3_unsigned(HVX_Vector vx) {
    // input is 0..0xffff representing 0.0  .. 1.0
    HVX_Vector p;
    p = Q6_Vh_vlut4_VuhPh(vx, 0xFAE6F6D4EE73D6A3ull);
    p = Q6_Vh_vmpa_VhVhVuhPuh_sat(p, vx, 0x2E49406159097A14ull);
    p = Q6_Vh_vmps_VhVhVuhPuh_sat(p, vx, 0x5DF66B7177AB7FC2ull);
    p = Q6_Vh_vmpa_VhVhVuhPuh_sat(p, vx, 0x79E57D427F4E8001ull);
    return p;  // signed result, 14 fractional bits
}

// Find reciprocal of fp16.
// (1) first, convert to fp32, multiplying by 1.0; this is done to
//    handle denormals. Ignoring sign and zero, result should be at
//    least 5.9604645e-08 (32-bit code 0x33800000) and at most 131008 (0x47ffe000)
//    (exponent in range [103,143])
// (2) extract the mantissa into 16-bit unsigned; find reciprocal using a fitted poly
// (3) put this, along with '253-exp' (exp from (1)) together to make an qf32
// (4) convert that to fp16
// (5) put sign back in. Also, if the original value (w/o sign) was <0x81, replace
//     the result with the max value.
static inline HVX_Vector hvx_vec_inverse_fp16(HVX_Vector vals) {
    HVX_Vector     em_mask  = Q6_Vh_vsplat_R(0x7FFF);
    HVX_Vector     avals    = Q6_V_vand_VV(vals, em_mask);
    HVX_VectorPred is_neg   = Q6_Q_vcmp_gt_VhVh(avals, vals);
    // is too small to 1/x ? for 'standard' fp16, this would be 0x101
    HVX_VectorPred is_small = Q6_Q_vcmp_gt_VhVh(Q6_Vh_vsplat_R(0x101), avals);

    HVX_VectorPair to_qf32  = Q6_Wqf32_vmpy_VhfVhf(avals, Q6_Vh_vsplat_R(0x3C00));  // *1.0
    HVX_Vector     to_f32_0 = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(to_qf32));
    HVX_Vector     to_f32_1 = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(to_qf32));

    // bits 22..13 contain the mantissa now (w/o hidden bit); move to bit 14..5 of a 16-bit vector
    HVX_Vector mant_u16 = Q6_Vh_vshuffo_VhVh(Q6_Vw_vasl_VwR(to_f32_1, 9), Q6_Vw_vasl_VwR(to_f32_0, 9));
    // likewise extract the upper 16 from each, containing the exponents in range 103..142
    HVX_Vector exp_u16  = Q6_Vh_vshuffo_VhVh(to_f32_1, to_f32_0);
    //Get exponent in IEEE 32-bit representation
    exp_u16             = Q6_Vuh_vlsr_VuhR(exp_u16, 7);

    // so, mant_u16 contains an unbiased mantissa in upper 10 bits of each u16 lane
    // We can consider it to be x-1.0, with 16 fractional bits, where 'x' is in range [1.0,2.0)
    // Use poly to transform to 1/x, with 14 fractional bits
    //
    HVX_Vector rm = hvx_vec_recip_xp1_O3_unsigned(mant_u16);

    HVX_Vector vcl0 = Q6_Vuh_vcl0_Vuh(rm);  //count leading zeros

    // Get mantissa for 16-bit represenation
    HVX_Vector mant_recip = Q6_V_vand_VV(Q6_Vh_vasr_VhR(Q6_Vh_vasl_VhVh(rm, vcl0), 5), Q6_Vh_vsplat_R(0x03FF));

    //Compute Reciprocal Exponent
    HVX_Vector exp_recip =
        Q6_Vh_vsub_VhVh(Q6_Vh_vsub_VhVh(Q6_Vh_vsplat_R(254), exp_u16), Q6_Vh_vsub_VhVh(vcl0, Q6_Vh_vsplat_R(1)));
    //Convert it for 16-bit representation
    exp_recip = Q6_Vh_vadd_VhVh_sat(Q6_Vh_vsub_VhVh(exp_recip, Q6_Vh_vsplat_R(127)), Q6_Vh_vsplat_R(15));
    exp_recip = Q6_Vh_vasl_VhR(exp_recip, 10);

    //Merge exponent and mantissa for reciprocal
    HVX_Vector recip = Q6_V_vor_VV(exp_recip, mant_recip);
    // map 'small' inputs to standard largest value 0x7bff
    recip            = Q6_V_vmux_QVV(is_small, Q6_Vh_vsplat_R(0x7bff), recip);
    // add sign back
    recip            = Q6_V_vandor_VQR(recip, is_neg, 0x80008000);
    return recip;
}

static inline HVX_Vector hvx_vec_i16_from_hf_rnd_sat(HVX_Vector vin) {
    // This looks complicated.
    // Ideally should just be Q6_Vh_equals_Vhf(vin)
    // but that instruction does not do proper rounding.

    // convert to qf32, multiplying by 1.0 in the process.
    HVX_VectorPair v32 = Q6_Wqf32_vmpy_VhfVhf(vin, Q6_Vh_vsplat_R(0x3C00));

    // 'in-range' values are +/32752.
    // add 192K to it, convert to sf
    HVX_Vector v192K = Q6_V_vsplat_R(0x48400000);
    HVX_Vector vsf_0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_lo_W(v32), v192K));
    HVX_Vector vsf_1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_hi_W(v32), v192K));

    // for in-range cases, result is {163858... 229360} so the exponent is always 144.
    // if we extract bits 21..0 as a signed quantity, and round 6 bits off, that will be the answer.
    // Start by <<10 to get the final 'sign' bit in bit 15...
    vsf_0 = Q6_Vw_vasl_VwR(vsf_0, 10);
    vsf_1 = Q6_Vw_vasl_VwR(vsf_1, 10);

    // now round down to 16
    return Q6_Vh_vround_VwVw_sat(vsf_1, vsf_0);
}

static inline HVX_Vector hvx_vec_inverse_fp32(HVX_Vector v_sf) {
    HVX_Vector inv_aprox_sf = Q6_V_vsplat_R(0x7EEEEBB3);
    HVX_Vector two_sf       = hvx_vec_splat_fp32(2.0);

    // First approximation
    HVX_Vector i_sf = Q6_Vw_vsub_VwVw(inv_aprox_sf, v_sf);

    HVX_Vector r_qf;

    // Refine
    r_qf = Q6_Vqf32_vmpy_VsfVsf(
        i_sf, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(two_sf, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(i_sf, v_sf)))));
    r_qf = Q6_Vqf32_vmpy_Vqf32Vqf32(
        r_qf, Q6_Vqf32_vsub_VsfVsf(two_sf, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(r_qf), v_sf))));
    r_qf = Q6_Vqf32_vmpy_Vqf32Vqf32(
        r_qf, Q6_Vqf32_vsub_VsfVsf(two_sf, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(r_qf), v_sf))));

    return Q6_Vsf_equals_Vqf32(r_qf);
}

#define FAST_SIGMOID_LOG2F (0x3fb8aa3b)  // 1.442695022
#define FAST_SIGMOID_C1    (0x3d009076)  // 0.03138777
#define FAST_SIGMOID_C2    (0x3e8d74bd)  // 0.276281267
#define FAST_SIGMOID_C3    (0x3f000000)  // 0.5

static inline HVX_Vector hvx_vec_fast_sigmoid_fp32(HVX_Vector v) {
    v = Q6_Vqf32_vmpy_VsfVsf(v, Q6_V_vsplat_R(FAST_SIGMOID_LOG2F));
    v = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(v), Q6_V_vsplat_R(FAST_SIGMOID_C3));

    HVX_Vector in_int = hvx_vec_truncate_fp32(Q6_Vsf_equals_Vqf32(v));
    HVX_Vector x      = Q6_Vqf32_vsub_Vqf32Vsf(v, Q6_Vsf_equals_Vw(in_int));
    HVX_Vector xx     = Q6_Vqf32_vmpy_Vqf32Vqf32(x, x);

    HVX_Vector v1 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(xx), Q6_V_vsplat_R(FAST_SIGMOID_C2));
    v1            = Q6_Vqf32_vadd_Vqf32Vsf(v1, Q6_V_vsplat_R(FAST_SIGMOID_LOG2F));

    HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(x), Q6_V_vsplat_R(FAST_SIGMOID_C1));
    v2            = Q6_Vqf32_vmpy_Vqf32Vqf32(v2, xx);
    v2            = Q6_Vqf32_vadd_Vqf32Vqf32(v2, x);

    HVX_Vector v3          = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(v2, v1));
    HVX_Vector v3_exponent = Q6_Vw_vasl_VwR(v3, 1);
    v3_exponent            = Q6_Vuw_vlsr_VuwR(v3_exponent, 24);
    v3_exponent            = Q6_Vw_vadd_VwVw(in_int, v3_exponent);
    v3                     = Q6_Vw_vaslacc_VwVwR(v3, in_int, 24);

    HVX_Vector v4 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(v2, v1));
    HVX_Vector v5 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(v3, v4));

    HVX_Vector res = hvx_vec_inverse_fp32(v5);
    res            = Q6_Vqf32_vmpy_VsfVsf(v3, res);

    return Q6_Vsf_equals_Vqf32(res);
}

#define RSQRT_CONST        0x5f3759df  // Constant for fast inverse square root calculation
#define RSQRT_ONE_HALF     0x3f000000  // 0.5
#define RSQRT_THREE_HALVES 0x3fc00000  // 1.5

static inline HVX_Vector hvx_vec_rsqrt_fp32(HVX_Vector in_vec) {
    //Algorithm :
    //  x2 = input*0.5
    //  y  = * (long *) &input
    //  y  = 0x5f3759df - (y>>2)
    //  y  = y*(threehalfs - x2*y*y)

    HVX_Vector rsqrtconst = Q6_V_vsplat_R(RSQRT_CONST);
    HVX_Vector onehalf    = Q6_V_vsplat_R(RSQRT_ONE_HALF);
    HVX_Vector threehalfs = Q6_V_vsplat_R(RSQRT_THREE_HALVES);

    HVX_Vector x2, y, ypower2, temp;

    x2 = Q6_Vqf32_vmpy_VsfVsf(in_vec, onehalf);
    x2 = Q6_Vqf32_vadd_Vqf32Vsf(x2, Q6_V_vzero());

    y = Q6_Vw_vasr_VwR(in_vec, 1);
    y = Q6_Vw_vsub_VwVw(rsqrtconst, y);

    // 1st iteration
    ypower2 = Q6_Vqf32_vmpy_VsfVsf(y, y);
    ypower2 = Q6_Vqf32_vadd_Vqf32Vsf(ypower2, Q6_V_vzero());
    temp    = Q6_Vqf32_vmpy_Vqf32Vqf32(x2, ypower2);
    temp    = Q6_Vqf32_vsub_VsfVsf(threehalfs, Q6_Vsf_equals_Vqf32(temp));
    temp    = Q6_Vqf32_vmpy_VsfVsf(y, Q6_Vsf_equals_Vqf32(temp));

    // 2nd iteration
    y       = Q6_Vqf32_vadd_Vqf32Vsf(temp, Q6_V_vzero());
    ypower2 = Q6_Vqf32_vmpy_Vqf32Vqf32(y, y);
    ypower2 = Q6_Vqf32_vadd_Vqf32Vsf(ypower2, Q6_V_vzero());
    temp    = Q6_Vqf32_vmpy_Vqf32Vqf32(x2, ypower2);
    temp    = Q6_Vqf32_vsub_VsfVsf(threehalfs, Q6_Vsf_equals_Vqf32(temp));
    temp    = Q6_Vqf32_vmpy_Vqf32Vqf32(y, temp);

    // 3rd iteration
    y       = Q6_Vqf32_vadd_Vqf32Vsf(temp, Q6_V_vzero());
    ypower2 = Q6_Vqf32_vmpy_Vqf32Vqf32(y, y);
    ypower2 = Q6_Vqf32_vadd_Vqf32Vsf(ypower2, Q6_V_vzero());
    temp    = Q6_Vqf32_vmpy_Vqf32Vqf32(x2, ypower2);
    temp    = Q6_Vqf32_vsub_VsfVsf(threehalfs, Q6_Vsf_equals_Vqf32(temp));
    temp    = Q6_Vqf32_vmpy_Vqf32Vqf32(y, temp);

    return Q6_Vsf_equals_Vqf32(temp);
}

static inline HVX_Vector hvx_vec_fast_sigmoid_fp32_guard(HVX_Vector v,
                                                         HVX_Vector one,
                                                         HVX_Vector max_exp,
                                                         HVX_Vector min_exp) {
    const HVX_VectorPred pred_max = Q6_Q_vcmp_gt_VsfVsf(max_exp, v);
    const HVX_VectorPred pred_min = Q6_Q_vcmp_gt_VsfVsf(v, min_exp);

    HVX_Vector out = hvx_vec_fast_sigmoid_fp32(v);
    out            = Q6_V_vmux_QVV(pred_max, out, one);
    return Q6_V_vmux_QVV(pred_min, out, Q6_V_vzero());
}

static inline HVX_Vector hvx_vec_tanh_fp32(HVX_Vector x) {
    // tanh(x) = 2 * sigmoid(2x) - 1
    HVX_Vector two = hvx_vec_splat_fp32(2.0f);
    HVX_Vector one = hvx_vec_splat_fp32(1.0f);
    HVX_Vector x2  = Q6_Vqf32_vmpy_VsfVsf(x, two);

    static const float kMinExp = -87.f;  // 0
    static const float kMaxExp = 87.f;   // 1
    HVX_Vector max_exp = hvx_vec_splat_fp32(kMaxExp);
    HVX_Vector min_exp = hvx_vec_splat_fp32(kMinExp);

    HVX_Vector sig2x = hvx_vec_fast_sigmoid_fp32_guard(Q6_Vsf_equals_Vqf32(x2), one, max_exp, min_exp);

    HVX_Vector res = Q6_Vqf32_vmpy_VsfVsf(sig2x, two);
    res = Q6_Vqf32_vsub_Vqf32Vsf(res, one);
    return Q6_Vsf_equals_Vqf32(res);
}

static inline void hvx_fast_sigmoid_f32(const uint8_t * restrict src, uint8_t * restrict dst, const int num_elems) {
    int step_of_1 = num_elems >> 5;
    int remaining = num_elems - step_of_1 * VLEN_FP32;

    const HVX_Vector * restrict v_src = (HVX_Vector *) src;
    HVX_Vector * restrict v_dst       = (HVX_Vector *) dst;

    static const float kMinExp = -87.f;  // 0
    static const float kMaxExp = 87.f;   // 1

    const HVX_Vector one     = hvx_vec_splat_fp32(1.f);
    const HVX_Vector max_exp = hvx_vec_splat_fp32(kMaxExp);
    const HVX_Vector min_exp = hvx_vec_splat_fp32(kMinExp);

    #pragma unroll(4)
    for (int i = 0; i < step_of_1; i++) {
        v_dst[i] = hvx_vec_fast_sigmoid_fp32_guard(v_src[i], one, max_exp, min_exp);
    }

    if (remaining > 0) {
        const float * srcf = ((const float *) src) + step_of_1* VLEN_FP32;
        float *       dstf = (float *) dst + step_of_1*VLEN_FP32;

        HVX_Vector in  = *(HVX_UVector *) srcf;
        HVX_Vector out = hvx_vec_fast_sigmoid_fp32_guard(in, one, max_exp, min_exp);
        hvx_vec_store_u((void *) dstf, remaining * SIZEOF_FP32, out);
    }
}

static inline void hvx_sigmoid_f32(const uint8_t * restrict src, uint8_t * restrict dst, const int num_elems){
    int step_of_1 = num_elems >> 5;  // divby 32, because 32 float = 128 bytes per HVX vector
    int leftover = num_elems - (step_of_1 * VLEN_FP32);

    int32_t leftover_size = leftover * sizeof(float);

    static const float kMinExp = -87.f;  // 0
    static const float kMaxExp = 87.f;   // 1

    const HVX_Vector one     = hvx_vec_splat_fp32(1.f);
    const HVX_Vector max_exp = hvx_vec_splat_fp32(kMaxExp);
    const HVX_Vector min_exp = hvx_vec_splat_fp32(kMinExp);

    const float *input = (float *)src;
    float *output = (float *)dst;

    HVX_Vector *  input_v_ptr  = (HVX_Vector *) input;
    HVX_UVector * output_v_ptr = (HVX_UVector *) output;

    HVX_Vector slinep;
    HVX_Vector slinec;
    HVX_Vector sline;

    slinep = *input_v_ptr++;
    #pragma unroll(4)
    for (int i = step_of_1 - 1; i > 0; i--) {
        slinec                              = *input_v_ptr++;
        sline                               = Q6_V_valign_VVR(slinec, slinep, (size_t) input);
        *((HVX_UVector *) (output_v_ptr++)) = hvx_vec_fast_sigmoid_fp32_guard(sline, one, max_exp, min_exp);
        slinep                              = slinec;
    }

    if (step_of_1 > 0) {
        slinec = hex_is_aligned(input_v_ptr, 128) && leftover == 0 ? slinep : *input_v_ptr++;
        sline  = Q6_V_valign_VVR(slinec, slinep, (size_t) input);
        *((HVX_UVector *) (output_v_ptr++)) = hvx_vec_fast_sigmoid_fp32_guard(sline, one, max_exp, min_exp);
        slinep = slinec;
    }
    if (leftover > 0) {
        slinec = (hex_is_one_chunk(input_v_ptr, leftover_size, 128) ? slinep : *input_v_ptr++);

        sline = Q6_V_valign_VVR(slinec, slinep, (size_t) input);

        HVX_Vector sout = hvx_vec_fast_sigmoid_fp32_guard(sline, one, max_exp, min_exp);
        hvx_vec_store_u(output_v_ptr, leftover_size, sout);
    }
}

float hvx_sum_of_squares_f32(const uint8_t * restrict src, const int num_elems);
void  hvx_mul_f32(const uint8_t * restrict src0,
                  const uint8_t * restrict src1,
                  uint8_t * restrict dst,
                  const int num_elems);
void  hvx_mul_f32_opt(const uint8_t * restrict src0,
                      const uint8_t * restrict src1,
                      uint8_t * restrict dst,
                      const int num_elems);
void  hvx_mul_mul_f32_opt(const uint8_t * restrict src0,
                          const uint8_t * restrict src1,
                          const uint8_t * restrict src2,
                          uint8_t * restrict dst,
                          const int num_elems);
void  hvx_mul_scalar_f32(const uint8_t * restrict src, const float val, uint8_t * restrict dst, const int num_elems);
void  hvx_add_f32(const uint8_t * restrict src0,
                  const uint8_t * restrict src1,
                  uint8_t * restrict dst,
                  const int num_elems);
void  hvx_add_f32_opt(const uint8_t * restrict src0,
                      const uint8_t * restrict src1,
                      uint8_t * restrict dst,
                      const int num_elems);
void  hvx_add_scalar_f32(const uint8_t * restrict src, const float val, uint8_t * restrict dst, const int num_elems);
void  hvx_sub_f32(const uint8_t * restrict src0,
                  const uint8_t * restrict src1,
                  uint8_t * restrict dst,
                  const int num_elems);
void  hvx_sub_f32_opt(const uint8_t * restrict src0,
                      const uint8_t * restrict src1,
                      uint8_t * restrict dst,
                      const int num_elems);
void  hvx_sub_scalar_f32(const uint8_t * restrict src, const float val, uint8_t * restrict dst, const int num_elems);
void  hvx_inverse_f32(const uint8_t * restrict src, uint8_t * restrict dst, const int num_elems);
void  hvx_sigmoid_f32(const uint8_t * restrict src, uint8_t * restrict dst, const int num_elems);
float hvx_self_max_f32(const uint8_t * restrict src, const int num_elems);
float hvx_self_sum_f32(const uint8_t * restrict src, const int num_elems);
void  hvx_min_scalar_f32(const uint8_t * restrict src, const float val, uint8_t * restrict dst, const int num_elems);
void  hvx_clamp_scalar_f32(const uint8_t * restrict src,
                           const float limit_left,
                           const float limit_right,
                           uint8_t * restrict dst,
                           const int num_elems);

#endif /* HVX_UTILS_H */
