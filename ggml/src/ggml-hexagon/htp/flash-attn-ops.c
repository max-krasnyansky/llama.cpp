#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#ifdef HTP_DEBUG
#    define FARF_HIGH 1
#endif
#include <HAP_farf.h>
#include <HAP_mem.h>
#include <HAP_perf.h>
#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <math.h>
#include <string.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-msg.h"
#include "htp-ops.h"
#include "hvx-utils.h"
#include "ops-utils.h"

// Dot product of FP32 and FP16 vectors, accumulating to float
static void hvx_dot_f32_f16_uu(float * restrict r, const void * restrict y, const void * restrict x, unsigned int n, float s) {
    const HVX_UVector * restrict vy = (const HVX_UVector * restrict) y; // fp32
    const HVX_UVector * restrict vx = (const HVX_UVector * restrict) x; // fp16

    uint32_t nvec = n / VLEN_FP16; // num full fp16 hvx vectors
    uint32_t nloe = n % VLEN_FP16; // leftover elements

    const HVX_Vector zero = Q6_V_vsplat_R(0);
    HVX_Vector       rsum = Q6_V_vsplat_R(0);

    uint32_t i = 0;

    #pragma unroll(2)
    for (i = 0; i < nvec; i++) {
        // Load y (fp32) and convert into fp16
        HVX_Vector y0_qf = Q6_Vqf32_vsub_VsfVsf(vy[i*2+0], zero);  // 32 elements
        HVX_Vector y1_qf = Q6_Vqf32_vsub_VsfVsf(vy[i*2+1], zero);  // 32 elements
        HVX_Vector y_hf  = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(y1_qf, y0_qf));

        // Load x (fp16)
        HVX_Vector x_hf = vx[i];

        HVX_VectorPair xy_qf = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(x_hf), y_hf);

        rsum = Q6_Vqf32_vadd_Vqf32Vqf32(rsum, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(xy_qf),  Q6_V_hi_W(xy_qf)));
    }

    if (nloe) {
        // Load y (fp32) and convert into fp16
        HVX_Vector y0_qf = Q6_Vqf32_vsub_VsfVsf(vy[i*2+0], zero);  // 32 elements
        HVX_Vector y1_qf = Q6_Vqf32_vsub_VsfVsf(vy[i*2+1], zero);  // 32 elements
        HVX_Vector y_hf  = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(y1_qf, y0_qf));

        // Load x (fp16) and zero-out unused elements
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 2);
        HVX_Vector      x_hf = Q6_V_vand_QV(bmask, vx[i]);

        HVX_VectorPair xy_qf = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(x_hf), y_hf);

        rsum = Q6_Vqf32_vadd_Vqf32Vqf32(rsum, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(xy_qf),  Q6_V_hi_W(xy_qf)));
    }

    rsum = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(rsum), hvx_vec_splat_fp32(s));
    rsum = Q6_Vsf_equals_Vqf32(hvx_vec_qf32_reduce_sum(rsum));

    hvx_vec_store_u(r, 4, rsum);
}

// Dot product of two F16 vectors, accumulating to float
static void hvx_dot_f16_f16_uu(float * restrict r, const void * restrict x, const void * restrict y, unsigned int n, float s) {
    const HVX_UVector * restrict vx = (const HVX_UVector * restrict) x; // fp16
    const HVX_UVector * restrict vy = (const HVX_UVector * restrict) y; // fp16

    uint32_t nvec = n / VLEN_FP16; // num full fp16 hvx vectors
    uint32_t nloe = n % VLEN_FP16; // leftover elements

    const HVX_Vector zero = Q6_V_vsplat_R(0);
    HVX_Vector       rsum = Q6_V_vsplat_R(0);

    uint32_t i = 0;

    #pragma unroll(2)
    for (i = 0; i < nvec; i++) {
        HVX_Vector y_hf = vy[i];
        HVX_Vector x_hf = vx[i];

        HVX_VectorPair xy_qf = Q6_Wqf32_vmpy_VhfVhf(x_hf, y_hf);

        rsum = Q6_Vqf32_vadd_Vqf32Vqf32(rsum, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(xy_qf),  Q6_V_hi_W(xy_qf)));
    }

    if (nloe) {
        HVX_Vector y_hf = vy[i];

        // Load x (fp16) and zero-out unused elements
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 2);
        HVX_Vector      x_hf = Q6_V_vand_QV(bmask, vx[i]);

        HVX_VectorPair xy_qf = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(x_hf), y_hf);

        rsum = Q6_Vqf32_vadd_Vqf32Vqf32(rsum, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(xy_qf),  Q6_V_hi_W(xy_qf)));
    }

    rsum = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(rsum), hvx_vec_splat_fp32(s));
    rsum = Q6_Vsf_equals_Vqf32(hvx_vec_qf32_reduce_sum(rsum));
    hvx_vec_store_u(r, 4, rsum);
}

// MAD: y (F32) += x (F16) * v (float)
static void hvx_mad_f32_f16_au(float * restrict y, const void * restrict x, int n, float s) {
    const HVX_UVector * restrict ptr_x = (const HVX_UVector *) x;
    HVX_Vector * restrict ptr_y = (HVX_Vector *) y;

    uint32_t nvec = n / VLEN_FP16; // num full fp16 hvx vectors
    uint32_t nloe = n % VLEN_FP16; // leftover elements

    HVX_Vector S = hvx_vec_splat_fp16(s);

    uint32_t i = 0;
    for (i = 0; i < nvec; ++i) {
        // Multiply x * s -> pair of F32 vectors
        HVX_VectorPair xs_p = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(ptr_x[i]), S);
        ptr_y[i*2]   = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_lo_W(xs_p), ptr_y[i*2]));
        ptr_y[i*2+1] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_hi_W(xs_p), ptr_y[i*2+1]));
    }

    if (nloe) {
        HVX_VectorPair xs_p = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(ptr_x[i]), S);

        HVX_Vector xs = Q6_V_lo_W(xs_p);
        i = 2 * i; // index for ptr_y

        if (nloe >= 32) {
            ptr_y[i] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(xs, ptr_y[i]));
            nloe -= 32; ++i; xs = Q6_V_hi_W(xs_p);
        }

        if (nloe) {
            HVX_Vector xy = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(xs, ptr_y[i]));
            hvx_vec_store_u(&ptr_y[i], nloe * 4, xy);
        }
    }
}

static void flash_attn_ext_f16_thread_scalar(struct htp_ops_context * octx, int ith, int nth) {
    const struct htp_tensor * q = &octx->src0;
    const struct htp_tensor * k = &octx->src1;
    const struct htp_tensor * v = &octx->src2;
    const struct htp_tensor * mask  = (octx->src3.data) ? &octx->src3 : NULL;
    const struct htp_tensor * sinks = (octx->src4.data) ? &octx->src4 : NULL;
    struct htp_tensor * dst = &octx->dst;

    const uint32_t neq0 = q->ne[0];
    const uint32_t neq1 = q->ne[1];
    const uint32_t neq2 = q->ne[2];
    const uint32_t neq3 = q->ne[3];

    const uint32_t nek0 = k->ne[0];
    const uint32_t nek1 = k->ne[1];
    const uint32_t nek2 = k->ne[2];
    const uint32_t nek3 = k->ne[3];

    const uint32_t nev0 = v->ne[0];
    const uint32_t nev1 = v->ne[1];
    const uint32_t nev2 = v->ne[2];
    const uint32_t nev3 = v->ne[3];

    const uint32_t nbq1 = q->nb[1];
    const uint32_t nbq2 = q->nb[2];
    const uint32_t nbq3 = q->nb[3];

    const uint32_t nbk1 = k->nb[1];
    const uint32_t nbk2 = k->nb[2];
    const uint32_t nbk3 = k->nb[3];

    const uint32_t nbv1 = v->nb[1];
    const uint32_t nbv2 = v->nb[2];
    const uint32_t nbv3 = v->nb[3];

    const uint32_t ne1 = dst->ne[1];
    const uint32_t ne2 = dst->ne[2];
    const uint32_t ne3 = dst->ne[3];

    const uint32_t nb1 = dst->nb[1];
    const uint32_t nb2 = dst->nb[2];
    const uint32_t nb3 = dst->nb[3];

    float scale         = 1.0f;
    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;

    memcpy(&scale,         (float *) octx->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (float *) octx->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (float *) octx->op_params + 2, sizeof(float));

    if (logit_softcap != 0) {
        scale /= logit_softcap;
    }

    // total rows in q
    const uint32_t nr = neq1*neq2*neq3;

    const uint32_t dr = (nr + nth - 1) / nth;
    const uint32_t ir0 = dr * ith;
    const uint32_t ir1 = MIN(ir0 + dr, nr);

    if (ir0 >= ir1) return;

    const uint32_t DK = nek0;
    const uint32_t DV = nev0;

    // Use scratchpad for the accumulator
    // VKQ32: DV * sizeof(float)

    uint8_t * spad = octx->src0_spad.data + octx->src0_spad.size_per_thread * ith;
    float * VKQ32 = (float *) spad;

    const uint32_t n_head = neq2;
    const uint32_t n_head_log2 = 1u << (uint32_t) floor(log2(n_head));
    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    for (uint32_t ir = ir0; ir < ir1; ++ir) {
        const uint32_t iq3 = fastdiv(ir, &octx->src0_div21);
        const uint32_t iq2 = fastdiv(ir - iq3*neq2*neq1, &octx->src0_div1);
        const uint32_t iq1 = (ir - iq3*neq2*neq1 - iq2 * neq1);

        const uint32_t ik3 = fastdiv(iq3, &octx->broadcast_rk3);
        const uint32_t ik2 = fastdiv(iq2, &octx->broadcast_rk2);

        const uint32_t iv3 = fastdiv(iq3, &octx->broadcast_rv3);
        const uint32_t iv2 = fastdiv(iq2, &octx->broadcast_rv2);

        const uint32_t h = iq2; // head index
        const float slope = (max_bias > 0.0f) ? (h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_head_log2) + 1)) : 1.0f;

        float S = 0.0f;      // sum
        float M = -INFINITY; // maximum KQ value

        // Clear accumulator
        memset(VKQ32, 0, DV * sizeof(float));

        const uint8_t * q_row_ptr = (const uint8_t *) q->data + (iq1*nbq1 + iq2*nbq2 + iq3*nbq3);

        const __fp16 * mp;
        if (mask) {
            // Mask offset calculation from ggml-cpu:
            // mask->data + iq1*mask->nb[1] + (iq2%mask->ne[2])*mask->nb[2] + (iq3%mask->ne[3])*mask->nb[3]
            // We need broadcasting indices.
            const uint32_t im2 = fastmodulo(iq2, mask->ne[2], &octx->src3_div2);
            const uint32_t im3 = fastmodulo(iq3, mask->ne[3], &octx->src3_div3);
            mp = (const __fp16 *) ((const uint8_t *) mask->data + iq1*mask->nb[1] + im2*mask->nb[2] + im3*mask->nb[3]);
        }

        // Loop over KV
        for (uint32_t ic = 0; ic < nek1; ++ic) {
            float s_val;

            const uint8_t * k_ptr = (const uint8_t *) k->data + (ic*nbk1 + ik2*nbk2 + ik3*nbk3);

            if (q->type == HTP_TYPE_F32) {
                hvx_dot_f32_f16_uu(&s_val, q_row_ptr, k_ptr, DK, scale);
            } else {
                hvx_dot_f16_f16_uu(&s_val, q_row_ptr, k_ptr, DK, scale);
            }

            if (logit_softcap != 0.0f) {
                s_val = logit_softcap * tanhf(s_val);
            }

            if (mask) {
                const float m_val = mp[ic];
                s_val += slope * m_val;
            }

            const float Mold = M;
            float ms = 1.0f;
            float vs = 1.0f;

            if (s_val > M) {
                M = s_val;
                ms = expf(Mold - M);
                hvx_scale_f32((const uint8_t *) VKQ32, (uint8_t *) VKQ32, DV, ms);
            } else {
                vs = expf(s_val - M);
            }

            const uint8_t * v_ptr = (const uint8_t *) v->data + (ic*nbv1 + iv2*nbv2 + iv3*nbv3);

            // Accumulate V: VKQ32 += v_ptr * vs
            hvx_mad_f32_f16_au(VKQ32, v_ptr, DV, vs);

            S = S * ms + vs;
        }

        // sinks
        if (sinks) {
            const float s = ((float *)((char *) sinks->data))[h];

            float ms = 1.0f;
            float vs = 1.0f;

            if (s > M) {
                ms = expf(M - s);
                hvx_scale_f32((const uint8_t *) VKQ32, (uint8_t *) VKQ32, DV, ms);
            } else {
                vs = expf(s - M);
            }

            S = S * ms + vs;
        }

        const float S_inv = S == 0.0f ? 0.0f : 1.0f/S;
        hvx_scale_f32((const uint8_t *) VKQ32, (uint8_t *) VKQ32, DV, S_inv);

        // Store result
        // dst indices
        const int i1 = iq1;
        const int i2 = iq2;
        const int i3 = iq3;

        // dst is permuted
        uint8_t * dst_ptr = (uint8_t *) dst->data + (i3*ne2*ne1 + i2 + i1*ne1) * nb1;

        if (dst->type == HTP_TYPE_F32) {
            hvx_copy_fp32_ua(dst_ptr, (uint8_t *) VKQ32, DV);
        } else if (dst->type == HTP_TYPE_F16) {
            __fp16 * d = (__fp16 *) dst_ptr;
            for (int i = 0; i < DV; ++i) {
                d[i] = (__fp16)VKQ32[i];
            }
        }
    }
}

static inline HVX_Vector load_32_fp16_to_fp32(const void * ptr) {
    __fp16 buf[64] __attribute__((aligned(128)));
    hvx_copy_fp16_ua((uint8_t*)buf, (const uint8_t*)ptr, 32); // Copy 32 halves (64 bytes)
    memset(buf + 32, 0, 64); // Zero upper 32 halves

    HVX_Vector v_hf = *(HVX_Vector*)buf;
    HVX_Vector one = hvx_vec_splat_fp16(1.0f);

    HVX_VectorPair pair = Q6_Wqf32_vmpy_VhfVhf(v_hf, one);
    return Q6_V_lo_W(pair);
}

#define MAX_ACC_VECS 8

static void flash_attn_ext_f16_thread_hvx(struct htp_ops_context * octx, int ith, int nth) {
    const struct htp_tensor * q = &octx->src0;
    const struct htp_tensor * k = &octx->src1;
    const struct htp_tensor * v = &octx->src2;
    const struct htp_tensor * mask  = (octx->src3.data) ? &octx->src3 : NULL;
    const struct htp_tensor * sinks = (octx->src4.data) ? &octx->src4 : NULL;
    struct htp_tensor * dst = &octx->dst;

    const uint32_t neq0 = q->ne[0];
    const uint32_t neq1 = q->ne[1];
    const uint32_t neq2 = q->ne[2];
    const uint32_t neq3 = q->ne[3];

    const uint32_t nek0 = k->ne[0];
    const uint32_t nek1 = k->ne[1];
    const uint32_t nek2 = k->ne[2];
    const uint32_t nek3 = k->ne[3];

    const uint32_t nev0 = v->ne[0];
    const uint32_t nev1 = v->ne[1];
    const uint32_t nev2 = v->ne[2];
    const uint32_t nev3 = v->ne[3];

    const uint32_t nbq1 = q->nb[1];
    const uint32_t nbq2 = q->nb[2];
    const uint32_t nbq3 = q->nb[3];

    const uint32_t nbk1 = k->nb[1];
    const uint32_t nbk2 = k->nb[2];
    const uint32_t nbk3 = k->nb[3];

    const uint32_t nbv1 = v->nb[1];
    const uint32_t nbv2 = v->nb[2];
    const uint32_t nbv3 = v->nb[3];

    const uint32_t ne1 = dst->ne[1];
    const uint32_t ne2 = dst->ne[2];
    const uint32_t ne3 = dst->ne[3];

    const uint32_t nb1 = dst->nb[1];
    const uint32_t nb2 = dst->nb[2];
    const uint32_t nb3 = dst->nb[3];

    float scale         = 1.0f;
    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;

    memcpy(&scale,         (float *) octx->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (float *) octx->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (float *) octx->op_params + 2, sizeof(float));

    if (logit_softcap != 0) {
        scale /= logit_softcap;
    }

    // total rows in q
    const uint32_t nr = neq1*neq2*neq3;

    const uint32_t dr = (nr + nth - 1) / nth;
    const uint32_t ir0 = dr * ith;
    const uint32_t ir1 = MIN(ir0 + dr, nr);

    if (ir0 >= ir1) return;

    const uint32_t DK = nek0;
    const uint32_t DV = nev0;

    const uint32_t n_head = neq2;
    const uint32_t n_head_log2 = 1u << (uint32_t) floor(log2(n_head));
    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    const bool q_is_f32 = (q->type == HTP_TYPE_F32);
    const uint32_t dv_vecs = (DV + 31) / 32;

    for (uint32_t ir = ir0; ir < ir1; ++ir) {
        const uint32_t iq3 = fastdiv(ir, &octx->src0_div21);
        const uint32_t iq2 = fastdiv(ir - iq3*neq2*neq1, &octx->src0_div1);
        const uint32_t iq1 = (ir - iq3*neq2*neq1 - iq2 * neq1);

        const uint32_t ik3 = fastdiv(iq3, &octx->broadcast_rk3);
        const uint32_t ik2 = fastdiv(iq2, &octx->broadcast_rk2);

        const uint32_t iv3 = fastdiv(iq3, &octx->broadcast_rv3);
        const uint32_t iv2 = fastdiv(iq2, &octx->broadcast_rv2);

        const uint32_t h = iq2; // head index
        const float slope = (max_bias > 0.0f) ? (h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_head_log2) + 1)) : 1.0f;

        // Accumulator in registers
        HVX_Vector acc_vec[MAX_ACC_VECS];
        for (int i = 0; i < dv_vecs; ++i) acc_vec[i] = Q6_V_vzero();

        float S = 0.0f;      // sum
        float M = -INFINITY; // maximum KQ value

        const uint8_t * q_row_ptr = (const uint8_t *) q->data + (iq1*nbq1 + iq2*nbq2 + iq3*nbq3);

        const __fp16 * mp;
        if (mask) {
            const uint32_t im2 = fastmodulo(iq2, mask->ne[2], &octx->src3_div2);
            const uint32_t im3 = fastmodulo(iq3, mask->ne[3], &octx->src3_div3);
            mp = (const __fp16 *) ((const uint8_t *) mask->data + iq1*mask->nb[1] + im2*mask->nb[2] + im3*mask->nb[3]);
        }

        // Loop over KV blocks
        for (uint32_t ic = 0; ic < nek1; ic += 32) {
            uint32_t n_block = MIN(32, nek1 - ic);

            // Use VTCM for scores (reusing spad space)
            // spad has size for DV*4 bytes (VKQ32)
            // We need 32*4 bytes for scores.
            // We can reuse the beginning of spad as we don't need VKQ32 accumulation.
            float * scores = (float *) spad;

            const uint8_t * k_base_ptr = (const uint8_t *) k->data + (ic*nbk1 + ik2*nbk2 + ik3*nbk3);

            for (uint32_t k_idx = 0; k_idx < n_block; ++k_idx) {
                const uint8_t * k_ptr = k_base_ptr + k_idx * nbk1;
                if (q_is_f32) {
                    hvx_dot_f32_f16_uu(&scores[k_idx], q_row_ptr, k_ptr, DK, scale);
                } else {
                    hvx_dot_f16_f16_uu(&scores[k_idx], q_row_ptr, k_ptr, DK, scale);
                }
            }
            for (uint32_t k_idx = n_block; k_idx < 32; ++k_idx) {
                scores[k_idx] = -INFINITY;
            }

            HVX_Vector s_vec = *(HVX_Vector*)scores;

            if (logit_softcap != 0.0f) {
                HVX_Vector two = hvx_vec_splat_fp32(2.0f);
                HVX_Vector val2 = Q6_Vqf32_vmpy_VsfVsf(s_vec, two); // 2*x

                HVX_Vector sig = hvx_vec_fast_sigmoid_fp32(Q6_Vsf_equals_Vqf32(val2));

                // 2 * sig - 1
                HVX_Vector sig2 = Q6_Vqf32_vmpy_VsfVsf(sig, two);
                HVX_Vector minus_one = hvx_vec_splat_fp32(-1.0f);
                HVX_Vector tanh_approx = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(sig2, minus_one));

                HVX_Vector cap = hvx_vec_splat_fp32(logit_softcap);
                s_vec = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(tanh_approx, cap));
            }

            if (mask) {
                // Load mask block
                const __fp16 * mp_block = mp + ic;
                HVX_Vector m_vec = load_32_fp16_to_fp32(mp_block);

                HVX_Vector slope_vec = hvx_vec_splat_fp32(slope);
                HVX_Vector added = Q6_Vqf32_vmpy_VsfVsf(m_vec, slope_vec);
                s_vec = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(s_vec, Q6_Vsf_equals_Vqf32(added)));
            }

            // Softmax
            HVX_Vector max_vec = hvx_vec_reduce_max_fp32(s_vec);
            float m_local = hvx_vec_get_fp32(max_vec);

            float M_new = (m_local > M) ? m_local : M;
            float ms = expf(M - M_new);

            HVX_Vector M_new_vec = hvx_vec_splat_fp32(M_new);
            HVX_Vector diff = Q6_Vqf32_vsub_VsfVsf(s_vec, M_new_vec);
            HVX_Vector exp_vec = hvx_vec_exp_fp32(Q6_Vsf_equals_Vqf32(diff));

            // Update Accumulator
            HVX_Vector ms_vec = hvx_vec_splat_fp32(ms);
            for (int i = 0; i < dv_vecs; ++i) {
                acc_vec[i] = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(acc_vec[i]), ms_vec);
            }

            // Accumulate V
            // Use VTCM for exp_arr (reuse spad after scores)
            float * exp_arr = (float *) spad + 32;
            hvx_vec_store_u(exp_arr, 32*4, exp_vec);

            const uint8_t * v_base_ptr = (const uint8_t *) v->data + (ic*nbv1 + iv2*nbv2 + iv3*nbv3);

            for (uint32_t k_idx = 0; k_idx < n_block; ++k_idx) {
                 float w = exp_arr[k_idx];
                 HVX_Vector w_vec = hvx_vec_splat_fp32(w);
                 const uint8_t * v_ptr = v_base_ptr + k_idx * nbv1;

                 for (int d = 0; d < dv_vecs; ++d) {
                      HVX_Vector v_val;
                      int n_rem = DV - d*32;
                      if (n_rem >= 32) {
                          v_val = load_32_fp16_to_fp32(v_ptr + d*64);
                      } else {
                          __fp16 buf[64] __attribute__((aligned(128)));
                          hvx_copy_fp16_ua((uint8_t*)buf, (const uint8_t*)(v_ptr + d*64), n_rem);
                          memset(buf + n_rem, 0, (32-n_rem)*sizeof(__fp16));

                           HVX_Vector v_hf = *(HVX_Vector*)buf;
                           HVX_Vector one = hvx_vec_splat_fp16(1.0f);
                           HVX_VectorPair pair = Q6_Wqf32_vmpy_VhfVhf(v_hf, one);
                           v_val = Q6_V_lo_W(pair);
                      }

                      acc_vec[d] = Q6_Vqf32_vadd_Vqf32Vqf32(acc_vec[d], Q6_Vqf32_vmpy_VsfVsf(v_val, w_vec));
                 }
            }

            // Update S
            HVX_Vector sum_exp = hvx_vec_fp32_reduce_sum(exp_vec);
            float sum_local = hvx_vec_get_fp32(sum_exp);
            S = S * ms + sum_local;
            M = M_new;
        }

        // sinks
        if (sinks) {
            const float s = ((float *)((char *) sinks->data))[h];

            float ms = 1.0f;
            float vs = 1.0f;

            if (s > M) {
                ms = expf(M - s);
                HVX_Vector ms_vec = hvx_vec_splat_fp32(ms);
                for (int i = 0; i < dv_vecs; ++i) {
                     acc_vec[i] = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(acc_vec[i]), ms_vec);
                }
            } else {
                vs = expf(s - M);
            }

            S = S * ms + vs;
        }

        const float S_inv_val = (S == 0.0f) ? 0.0f : 1.0f/S;
        HVX_Vector S_inv = hvx_vec_splat_fp32(S_inv_val);
        for (int i = 0; i < dv_vecs; ++i) {
             acc_vec[i] = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(acc_vec[i]), S_inv);
        }

        // Store result
        const int i1 = iq1;
        const int i2 = iq2;
        const int i3 = iq3;

        uint8_t * dst_ptr = (uint8_t *) dst->data + (i3*ne2*ne1 + i2 + i1*ne1) * nb1;

        if (dst->type == HTP_TYPE_F32) {
             for (int i = 0; i < dv_vecs; ++i) {
                 HVX_Vector res = Q6_Vsf_equals_Vqf32(acc_vec[i]);
                 int n_rem = DV - i*32;
                 if (n_rem >= 32) {
                     hvx_vec_store_u(dst_ptr + i*128, 128, res);
                 } else {
                     hvx_vec_store_u(dst_ptr + i*128, n_rem*4, res);
                 }
             }
        } else if (dst->type == HTP_TYPE_F16) {
             // Vectorized F16 store
             for (int i = 0; i < dv_vecs; i += 2) {
                 HVX_Vector res0 = Q6_Vsf_equals_Vqf32(acc_vec[i]);
                 HVX_Vector res1 = (i + 1 < dv_vecs) ? Q6_Vsf_equals_Vqf32(acc_vec[i+1]) : Q6_V_vzero();

                 // Convert two vectors of F32 to one vector of F16
                 // Q6_W_vcombine_VV(hi, lo)
                 // Layout of vcombine depends on architecture but usually puts args into hi/lo lanes.
                 // We want res0 in lower part, res1 in higher part (assuming sequential order).
                 // Actually vcombine produces a pair (double vector).
                 // Q6_Vhf_equals_Wqf32 converts a pair of qf32 to one vector of halves.
                 // Wait, we have Vsf. We need Vqf32 for conversion?
                 // Q6_Vhf_equals_Wqf32 takes Wqf32 (Vector Pair of qf32).
                 // We need to convert Vsf -> Vqf32 first?
                 // Or maybe Q6_W_vcombine_VV takes Vsf? No, it takes any vector.
                 // But Q6_Vhf_equals_Wqf32 expects Wqf32 input.
                 // Wqf32 is just 2 vectors.
                 // If we pass res1, res0 (Vsf) treated as Vqf32.
                 // The instruction normalizes/rounds/packs.
                 // We need to verify if res0/res1 (Vsf) are valid inputs for Wqf32 conversion.
                 // Yes, Vsf and Vqf32 are bitwise distinct but here we probably need to convert back to qf32?
                 // No, standard float (Vsf) to half (Vhf).
                 // hvx_vec_i16_from_hf_rnd_sat is half->i16.
                 // To convert Float -> Half:
                 // We can use Q6_Vhf_equals_Wsf(Q6_W_vcombine_VV(res1, res0)) if available.
                 // If not, we might need to use `Q6_Vhf_equals_Wqf32`.
                 // If we use `Q6_Vhf_equals_Wqf32`, we need to ensure input is in qf32 format.
                 // To convert Vsf -> Vqf32: Q6_Vqf32_vadd_VsfVsf(v, zero).

                 HVX_Vector qf0 = Q6_Vqf32_vadd_VsfVsf(res0, Q6_V_vzero());
                 HVX_Vector qf1 = Q6_Vqf32_vadd_VsfVsf(res1, Q6_V_vzero());

                 HVX_Vector vhf = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(qf1, qf0));

                 // Store
                 // vhf contains 64 halves (128 bytes).
                 // res0 had 32 floats (converted to 32 halves), res1 had 32 floats (converted to 32 halves).
                 // Total 64 halves.

                 int n_rem = DV - i*32;
                 if (n_rem >= 64) {
                     hvx_vec_store_u(dst_ptr + i*64, 128, vhf);
                 } else {
                     hvx_vec_store_u(dst_ptr + i*64, n_rem*2, vhf);
                 }
             }
        }
    }
}

static void flash_attn_ext_f16_thread(struct htp_ops_context * octx, int ith, int nth) {
    const struct htp_tensor * v = &octx->src2;
    const uint32_t DV = v->ne[0];
    if (DV <= MAX_ACC_VECS * 32) {
        flash_attn_ext_f16_thread_hvx(octx, ith, nth);
    } else {
        flash_attn_ext_f16_thread_scalar(octx, ith, nth);
    }
}

static void htp_flash_attn_ext_job(unsigned int n, unsigned int i, void * data) {
    struct htp_ops_context * octx = data;
    flash_attn_ext_f16_thread(octx, i, n);
}

int op_flash_attn_ext(struct htp_ops_context * octx) {
    const struct htp_tensor * q = &octx->src0;
    const struct htp_tensor * k = &octx->src1;
    const struct htp_tensor * v = &octx->src2;
    const struct htp_tensor * mask = (octx->src3.type != HTP_TYPE_COUNT) ? &octx->src3 : NULL;
    struct htp_tensor * dst = &octx->dst;

    // Check support
    if ((q->type != HTP_TYPE_F16 && q->type != HTP_TYPE_F32) ||
        k->type != HTP_TYPE_F16 ||
        v->type != HTP_TYPE_F16) {
        return HTP_STATUS_NO_SUPPORT;
    }

    octx->src0_div21 = init_fastdiv_values(q->ne[2] * q->ne[1]);
    octx->src0_div1  = init_fastdiv_values(q->ne[1]);

    octx->broadcast_rk2 = init_fastdiv_values(q->ne[2]/k->ne[2]);
    octx->broadcast_rk3 = init_fastdiv_values(q->ne[3]/k->ne[3]);
    octx->broadcast_rv2 = init_fastdiv_values(q->ne[2]/v->ne[2]);
    octx->broadcast_rv3 = init_fastdiv_values(q->ne[3]/v->ne[3]);

    if (mask) {
        octx->src3_div2 = init_fastdiv_values(mask->ne[2]);
        octx->src3_div3 = init_fastdiv_values(mask->ne[3]);
    }

    size_t spad_size_vkq = htp_round_up(octx->src2.ne[0] * sizeof(float), 128); // VKQ32

    octx->src0_spad.size_per_thread = spad_size_vkq;
    octx->src0_spad.size =  octx->src0_spad.size_per_thread * octx->n_threads;

    if (octx->ctx->vtcm_size < octx->src0_spad.size) {
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    octx->src0_spad.data = octx->ctx->vtcm_base;

    if (!(octx->flags & HTP_OPFLAGS_SKIP_COMPUTE)) {
        worker_pool_run_func(octx->ctx->worker_pool, htp_flash_attn_ext_job, octx, octx->n_threads);
    }

    return HTP_STATUS_OK;
}
