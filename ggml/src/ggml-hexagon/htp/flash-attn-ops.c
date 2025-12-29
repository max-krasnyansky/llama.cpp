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

// Dot product of two F16 vectors, accumulating to float
static void hvx_vec_dot_f16(int n, float * restrict s, const void * restrict vx, const void * restrict vy) {
    const HVX_UVector * restrict ptr_x = (const HVX_UVector *) vx;
    const HVX_UVector * restrict ptr_y = (const HVX_UVector *) vy;

    // Process 64 fp16 elements (1 vector) at a time
    int nvec = n / 64;
    int left = n % 64;

    HVX_Vector sum0 = Q6_V_vzero();
    HVX_Vector sum1 = Q6_V_vzero();

    for (int i = 0; i < nvec; ++i) {
        HVX_Vector v0 = ptr_x[i];
        HVX_Vector v1 = ptr_y[i];

        HVX_VectorPair pair = Q6_Wqf32_vmpy_VhfVhf(v0, v1);
        sum0 = Q6_Vqf32_vadd_Vqf32Vqf32(sum0, Q6_V_lo_W(pair));
        sum1 = Q6_Vqf32_vadd_Vqf32Vqf32(sum1, Q6_V_hi_W(pair));
    }

    HVX_Vector total = Q6_Vqf32_vadd_Vqf32Vqf32(sum0, sum1);

    // Reduce total vector
    HVX_Vector res_vec = hvx_vec_fp32_reduce_sum(total);
    float res = hvx_vec_get_fp32(res_vec);

    if (left > 0) {
        const __fp16 * px = (const __fp16 *) vx + nvec * 64;
        const __fp16 * py = (const __fp16 *) vy + nvec * 64;
        for (int i = 0; i < left; ++i) {
            res += (float)(px[i] * py[i]);
        }
    }

    *s = res;
}

// MAD F16: y += x * v (scalar)
// y (aligned) and x (unaligned) are F16 vectors. v is float scalar.
static void hvx_vec_mad_f16(int n, void * restrict y, const void * restrict x, float v) {
    const HVX_UVector * restrict ptr_x = (const HVX_UVector *) x;
    HVX_Vector * restrict ptr_y = (HVX_Vector *) y;

    int nvec = n / 64;
    int left = n % 64;

    __fp16 vf16 = (__fp16)v;
    union { __fp16 f; uint16_t i; } u16 = { .f = vf16 };

    HVX_Vector v_vec = Q6_Vh_vsplat_R(u16.i);

    for (int i = 0; i < nvec; ++i) {
        HVX_Vector vx = ptr_x[i];
        HVX_Vector vy = ptr_y[i];

        HVX_Vector prod_qf16 = Q6_Vqf16_vmpy_VhfVhf(vx, v_vec);
        HVX_Vector prod_hf = Q6_Vhf_equals_Vqf16(prod_qf16);

        ptr_y[i] = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(vy, prod_hf));
    }

    if (left > 0) {
        const __fp16 * px = (const __fp16 *) x + nvec * 64;
        __fp16 * py = (__fp16 *) y + nvec * 64;
        for (int i = 0; i < left; ++i) {
            py[i] += px[i] * vf16;
        }
    }
}

// Scale F16 vector: y *= scale
static void hvx_vec_scale_f16(int n, void * restrict y, float v) {
    HVX_Vector * restrict ptr_y = (HVX_Vector *) y;

    int nvec = n / 64;
    int left = n % 64;

    __fp16 vf16 = (__fp16)v;
    union { __fp16 f; uint16_t i; } u16 = { .f = vf16 };
    HVX_Vector v_vec = Q6_Vh_vsplat_R(u16.i);

    for (int i = 0; i < nvec; ++i) {
        HVX_Vector vy = ptr_y[i];
        HVX_Vector prod_qf16 = Q6_Vqf16_vmpy_VhfVhf(vy, v_vec);
        ptr_y[i] = Q6_Vhf_equals_Vqf16(prod_qf16);
    }

    if (left > 0) {
        __fp16 * py = (__fp16 *) y + nvec * 64;
        for (int i = 0; i < left; ++i) {
            py[i] *= vf16;
        }
    }
}

static void flash_attn_ext_f16_thread(struct htp_ops_context * octx, int ith, int nth) {
    const struct htp_tensor * q = &octx->src0;
    const struct htp_tensor * k = &octx->src1;
    const struct htp_tensor * v = &octx->src2;
    struct htp_tensor * dst = &octx->dst;

    float scale = 0.0f;
    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&scale,         (float *) octx->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (float *) octx->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (float *) octx->op_params + 2, sizeof(float));

    if (logit_softcap != 0) {
        scale /= logit_softcap;
    }

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

    const uint32_t nb1 = dst->nb[1];
    const uint32_t nb2 = dst->nb[2];
    const uint32_t nb3 = dst->nb[3];

    const uint32_t DK = nek0;
    const uint32_t DV = nev0;

    // broadcast factors
    const uint32_t rk2 = neq2/nek2;
    const uint32_t rk3 = neq3/nek3;
    const uint32_t rv2 = neq2/nev2;
    const uint32_t rv3 = neq3/nev3;

    // total rows in q
    const uint32_t nr = neq1*neq2*neq3;

    const uint32_t dr = (nr + nth - 1) / nth;
    const uint32_t ir0 = dr * ith;
    const uint32_t ir1 = MIN(ir0 + dr, nr);

    if (ir0 >= ir1) return;

    uint8_t * spad = octx->src0_spad.data + octx->src0_spad.size_per_thread * ith;
    __fp16 * VKQ16 = (__fp16 *) spad;

    // const uint32_t n_head = neq2;
    // const uint32_t n_head_log2 = 1u << (uint32_t) floor(log2(n_head));
    // const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    // const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    const uint32_t neq2_neq1 = neq2 * neq1;

    for (uint32_t ir = ir0; ir < ir1; ++ir) {
        const uint32_t iq3 = fastdiv(ir, &octx->src0_div21);
        const uint32_t rem = ir - iq3 * neq2_neq1;
        const uint32_t iq2 = fastdiv(rem, &octx->src0_div1);
        const uint32_t iq1 = rem - iq2 * neq1;

        // const uint32_t h = iq2; // head index
        // const float slope = (max_bias > 0.0f) ? (h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_head_log2) + 1)) : 1.0f;

        float S = 0.0f;      // sum
        float M = -INFINITY; // maximum KQ value

        // Clear accumulator
        memset(VKQ16, 0, DV * sizeof(__fp16));

        const uint32_t ik3 = iq3 / rk3;
        const uint32_t ik2 = iq2 / rk2;

        const uint32_t iv3 = iq3 / rv3;
        const uint32_t iv2 = iq2 / rv2;

        const uint8_t * q_ptr = (const uint8_t *) q->data + (iq1*nbq1 + iq2*nbq2 + iq3*nbq3);

        // Loop over KV
        for (uint32_t ic = 0; ic < nek1; ++ic) {
            float s_val;
            const uint8_t * k_ptr = (const uint8_t *) k->data + (ic*nbk1 + ik2*nbk2 + ik3*nbk3);

            hvx_vec_dot_f16(DK, &s_val, q_ptr, k_ptr);

            s_val *= scale;

            if (logit_softcap != 0.0f) {
                s_val = logit_softcap * tanhf(s_val);
            }

            // ALiBi slope application would go here if mask was supported

            const float Mold = M;
            float ms = 1.0f;
            float vs = 1.0f;

            if (s_val > M) {
                M = s_val;
                ms = expf(Mold - M);
                hvx_vec_scale_f16(DV, VKQ16, ms);
            } else {
                vs = expf(s_val - M);
            }

            const uint8_t * v_ptr = (const uint8_t *) v->data + (ic*nbv1 + iv2*nbv2 + iv3*nbv3);
            hvx_vec_mad_f16(DV, VKQ16, v_ptr, vs);

            S = S*ms + vs;
        }

        const float S_inv = S == 0.0f ? 0.0f : 1.0f/S;
        hvx_vec_scale_f16(DV, VKQ16, S_inv);

        // Store result
        uint8_t * dst_ptr = (uint8_t *) dst->data + iq1 * nb2 + iq2 * nb1 + iq3 * nb3;

        if (dst->type == HTP_TYPE_F32) {
            float * d = (float *) dst_ptr;
            for (int i = 0; i < DV; ++i) {
                d[i] = (float)VKQ16[i];
            }
        } else if (dst->type == HTP_TYPE_F16) {
            __fp16 * d = (__fp16 *) dst_ptr;
            for (int i = 0; i < DV; ++i) {
                d[i] = VKQ16[i];
            }
        }
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
    struct htp_tensor * dst = &octx->dst;

    if (q->type != HTP_TYPE_F16 || k->type != HTP_TYPE_F16 || v->type != HTP_TYPE_F16) {
        return HTP_STATUS_NO_SUPPORT;
    }

    octx->src0_div21 = init_fastdiv_values(q->ne[2] * q->ne[1]);
    octx->src0_div1  = init_fastdiv_values(q->ne[1]);

    size_t spad_size = octx->src2.ne[0] * sizeof(__fp16); // DV
    spad_size = htp_round_up(spad_size, 128);
    octx->src0_spad.size_per_thread = spad_size;
    octx->src0_spad.size = spad_size * octx->n_threads;

    if (octx->ctx->vtcm_size < octx->src0_spad.size) {
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    octx->src0_spad.data = octx->ctx->vtcm_base;

    worker_pool_run_func(octx->ctx->worker_pool, htp_flash_attn_ext_job, octx, octx->n_threads);

    return HTP_STATUS_OK;
}
