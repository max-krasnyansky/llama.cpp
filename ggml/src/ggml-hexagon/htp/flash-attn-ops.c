#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#ifdef HTP_DEBUG
#    define FARF_HIGH 1
#endif
#include <HAP_farf.h>
#include <HAP_mem.h>
#include <HAP_perf.h>
#include <HAP_ps.h>
#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <math.h>
#include <qurt_thread.h>
#include <string.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-dma.h"
#include "htp-msg.h"
#include "htp-ops.h"
#include "hvx-utils.h"
#include "ops-utils.h"

struct flash_attn_th_ctx {
    float scale;
    float max_bias;
    float logit_softcap;
    float m0;
    float m1;
    uint32_t n_head;
    uint32_t n_head_log2;

    struct htp_ops_context * octx;
};

#define htp_flash_attn_preamble3                           \
    const uint32_t neq0 = q->ne[0];                        \
    const uint32_t neq1 = q->ne[1];                        \
    const uint32_t neq2 = q->ne[2];                        \
    const uint32_t neq3 = q->ne[3];                        \
                                                           \
    const uint32_t nbq0 = q->nb[0];                        \
    const uint32_t nbq1 = q->nb[1];                        \
    const uint32_t nbq2 = q->nb[2];                        \
    const uint32_t nbq3 = q->nb[3];                        \
                                                           \
    const uint32_t nek0 = k->ne[0];                        \
    const uint32_t nek1 = k->ne[1];                        \
    const uint32_t nek2 = k->ne[2];                        \
    const uint32_t nek3 = k->ne[3];                        \
                                                           \
    const uint32_t nbk0 = k->nb[0];                        \
    const uint32_t nbk1 = k->nb[1];                        \
    const uint32_t nbk2 = k->nb[2];                        \
    const uint32_t nbk3 = k->nb[3];                        \
                                                           \
    const uint32_t nev0 = v->ne[0];                        \
    const uint32_t nev1 = v->ne[1];                        \
    const uint32_t nev2 = v->ne[2];                        \
    const uint32_t nev3 = v->ne[3];                        \
                                                           \
    const uint32_t nbv0 = v->nb[0];                        \
    const uint32_t nbv1 = v->nb[1];                        \
    const uint32_t nbv2 = v->nb[2];                        \
    const uint32_t nbv3 = v->nb[3];                        \
                                                           \
    const uint32_t ne0 = dst->ne[0];                       \
    const uint32_t ne1 = dst->ne[1];                       \
    const uint32_t ne2 = dst->ne[2];                       \
    const uint32_t ne3 = dst->ne[3];                       \
                                                           \
    const uint32_t nb0 = dst->nb[0];                       \
    const uint32_t nb1 = dst->nb[1];                       \
    const uint32_t nb2 = dst->nb[2];                       \
    const uint32_t nb3 = dst->nb[3];

// Based on vec_dot_f16_f32 from matmul-ops.c but for float accumulation
static void vec_dot_f32_f32(const int n, float * restrict s, const void * restrict x, const void * restrict y) {
    const HVX_UVector * restrict vx     = (const HVX_UVector * restrict) x;
    const HVX_UVector * restrict vy     = (const HVX_UVector * restrict) y;

    uint32_t nv0 = n / 32;  // num full fp32 hvx vectors
    uint32_t nv1 = n % 32;  // leftover elements

    volatile HVX_Vector rsum = Q6_V_vsplat_R(0);

    uint32_t i = 0;

    for (i = 0; i < nv0; i++) {
        HVX_Vector vx_v = vx[i];
        HVX_Vector vy_v = vy[i];
        HVX_Vector prod = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(vx_v), Q6_Vsf_equals_Vqf32(vy_v));
        rsum = Q6_Vqf32_vadd_Vqf32Vqf32(rsum, prod);
    }

    if (nv1) {
        HVX_Vector vx_v = vx[i];
        HVX_Vector vy_v = vy[i];

        HVX_Vector prod = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(vx_v), Q6_Vsf_equals_Vqf32(vy_v));

        // Mask out elements beyond n
        HVX_VectorPred mask = Q6_Q_vsetq_R(nv1 * 4);
        prod = Q6_V_vmux_QVV(mask, prod, Q6_V_vzero());

        rsum = Q6_Vqf32_vadd_Vqf32Vqf32(rsum, prod);
    }

    rsum = hvx_vec_qf32_reduce_sum(rsum);
    *s = hvx_vec_get_fp32(Q6_Vsf_equals_Vqf32(rsum));
}

// Compute dst += src * scalar for F32 arrays using HVX
static void vec_mad_f32(const int n, float * restrict dst, const void * restrict src, float scalar) {
    const HVX_UVector * restrict vsrc = (const HVX_UVector * restrict) src;
    HVX_UVector * restrict vdst       = (HVX_UVector * restrict) dst;

    HVX_Vector vscalar = hvx_vec_splat_fp32(scalar);

    uint32_t nv0 = n / 32;
    uint32_t nv1 = n % 32;

    uint32_t i = 0;
    for (i = 0; i < nv0; i++) {
        HVX_Vector v_src = vsrc[i];
        HVX_Vector v_dst = vdst[i];

        // v_dst = v_dst + v_src * scalar
        // Q6_Vqf32_vmpy_VsfVsf takes Vsf (float), so cast v_src
        HVX_Vector v_prod = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(v_src), vscalar);

        HVX_Vector v_res = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vsf_equals_Vqf32(v_dst), v_prod);

        vdst[i] = Q6_Vsf_equals_Vqf32(v_res);
    }

    if (nv1) {
        HVX_Vector v_src = vsrc[i];
        HVX_Vector v_dst = vdst[i];

        HVX_Vector v_prod = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(v_src), vscalar);
        HVX_Vector v_res = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vsf_equals_Vqf32(v_dst), v_prod);

        // Masked store
        hvx_vec_store_u(vdst + i, nv1 * 4, Q6_Vsf_equals_Vqf32(v_res));
    }
}

static inline float hvx_expf(float x) {
    // Use hvx_vec_exp_fp32 on a splatted vector
    HVX_Vector vx = hvx_vec_splat_fp32(x);
    HVX_Vector vexp = hvx_vec_exp_fp32(vx);
    return hvx_vec_get_fp32(vexp);
}

static void flash_attn_job_f32(unsigned int n, unsigned int i, void * data) {
    struct flash_attn_th_ctx * ctx = (struct flash_attn_th_ctx *) data;
    struct htp_ops_context * octx = ctx->octx;

    const struct htp_tensor * q = &octx->src0;
    const struct htp_tensor * k = &octx->src1;
    const struct htp_tensor * v = &octx->src2;
    const struct htp_tensor * mask = &octx->src3;
    struct htp_tensor * dst = &octx->dst;

    htp_flash_attn_preamble3;

    const int64_t DK = nek0;
    const int64_t DV = nev0;
    const int64_t N  = neq1;

    const int64_t rk2 = neq2/nek2;
    const int64_t rk3 = neq3/nek3;

    const int64_t rv2 = neq2/nev2;
    const int64_t rv3 = neq3/nev3;

    const int64_t nr = neq1*neq2*neq3; // total rows in q

    // rows per thread
    const int64_t dr = (nr + n - 1)/n;

    // row range for this thread
    const int64_t ir0 = dr*i;
    const int64_t ir1 = MIN(ir0 + dr, nr);

    float * wdata = (float *) (octx->src0_spad.data + i * octx->src0_spad.size_per_thread);
    // Layout: [VKQ32 (DV floats)] [Q_q (reserved for quantized Q, used as temp here, DK floats)]
    // Actually for F32 implementation we just need space for VKQ (DV floats) and Q vector (DK floats)
    // Let's align them
    float * VKQ32 = wdata;
    float * Q_vec = wdata + htp_round_up(DV, 32);

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t iq3 = ir/(neq2*neq1);
        const int64_t iq2 = (ir - iq3*neq2*neq1)/neq1;
        const int64_t iq1 = (ir - iq3*neq2*neq1 - iq2*neq1);

        const uint32_t h = iq2;
        const float slope = (ctx->max_bias > 0.0f) ?
                            (h < ctx->n_head_log2 ? powf(ctx->m0, h + 1) : powf(ctx->m1, 2*(h - ctx->n_head_log2) + 1)) :
                            1.0f;

        float S = 0.0f;
        float M = -INFINITY;

        memset(VKQ32, 0, DV * sizeof(float));

        // Copy Q to contiguous buffer for vector operations
        const float * pq = (const float *) ((char *) q->data + (iq1*nbq1 + iq2*nbq2 + iq3*nbq3));

        // Ensure Q is loaded into Q_vec aligned
        if (((uintptr_t)pq % 128) == 0 && (DK % 32 == 0)) {
             hvx_copy_fp32_aa((uint8_t*)Q_vec, (const uint8_t*)pq, DK);
        } else {
             // Unaligned copy
             memcpy(Q_vec, pq, DK * sizeof(float));
        }

        const int64_t ik3 = iq3 / rk3;
        const int64_t ik2 = iq2 / rk2;

        const int64_t iv3 = iq3 / rv3;
        const int64_t iv2 = iq2 / rv2;

        const float * mp = mask->data ? (const float *)((char *) mask->data + iq1*mask->nb[1] + (iq2%mask->ne[2])*mask->nb[2] + (iq3%mask->ne[3])*mask->nb[3]) : NULL;

        // Loop over KV heads
        for (int64_t ic = 0; ic < nek1; ++ic) {
            const float mv = mp ? slope*mp[ic] : 0.0f;
            if (mv == -INFINITY) {
                continue;
            }

            float s;
            const float * k_data = (const float *) ((const char *) k->data + (ic*nbk1 + ik2*nbk2 + ik3*nbk3));

            // Q * K^T
            vec_dot_f32_f32(DK, &s, k_data, Q_vec);

            s = s * ctx->scale;

            if (ctx->logit_softcap != 0.0f) {
                s = ctx->logit_softcap * tanhf(s);
            }

            s += mv;

            const float Mold = M;

            float ms = 1.0f;
            float vs = 1.0f;

            if (s > M) {
                M = s;
                ms = hvx_expf(Mold - M);
                hvx_scale_f32((const uint8_t *) VKQ32, (uint8_t *) VKQ32, DV, ms);
            } else {
                vs = hvx_expf(s - M);
            }

            const float * v_data = (const float *) ((const char *) v->data + (ic*nbv1 + iv2*nbv2 + iv3*nbv3));

            // V += v * exp(s - M)
            vec_mad_f32(DV, VKQ32, v_data, vs);

            S = S*ms + vs;
        }

        const float S_inv = S == 0.0f ? 0.0f : 1.0f/S;
        hvx_scale_f32((const uint8_t *) VKQ32, (uint8_t *) VKQ32, DV, S_inv);

        // Store result
        // permute(0, 2, 1, 3)
        // dst indices
        const int i1 = iq1;
        const int i2 = iq2;
        const int i3 = iq3;

        float * dst_ptr = (float *) ((char *) dst->data + (i3*ne2*ne1 + i2 + i1*ne1)*nb1);
        memcpy(dst_ptr, VKQ32, DV * sizeof(float));
    }
}

static int execute_op_flash_attn_f32(struct htp_ops_context * octx) {
    const struct htp_tensor * q = &octx->src0;
    const struct htp_tensor * k = &octx->src1;
    const struct htp_tensor * v = &octx->src2;
    struct htp_tensor * dst = &octx->dst;

    struct flash_attn_th_ctx ctx;
    memset(&ctx, 0, sizeof(ctx));

    memcpy(&ctx.scale,         (float *) octx->op_params + 0, sizeof(float));
    memcpy(&ctx.max_bias,      (float *) octx->op_params + 1, sizeof(float));
    memcpy(&ctx.logit_softcap, (float *) octx->op_params + 2, sizeof(float));

    if (ctx.logit_softcap != 0) {
        ctx.scale /= ctx.logit_softcap;
    }

    ctx.n_head = q->ne[2];
    ctx.n_head_log2 = 1u << (uint32_t) floor(log2(ctx.n_head));
    ctx.m0 = powf(2.0f, -(ctx.max_bias) / ctx.n_head_log2);
    ctx.m1 = powf(2.0f, -(ctx.max_bias / 2.0f) / ctx.n_head_log2);
    ctx.octx = octx;

    // Allocate scratchpad memory
    // Need space per thread for:
    // - VKQ32 accumulator (DV floats)
    // - Q vector (DK floats)
    // - Alignment padding

    uint32_t DV = v->ne[0];
    uint32_t DK = k->ne[0];

    size_t spad_per_thread = htp_round_up(DV * sizeof(float), 128) +
                             htp_round_up(DK * sizeof(float), 128) +
                             128; // Extra padding

    octx->src0_spad.size_per_thread = spad_per_thread;
    octx->src0_spad.size = spad_per_thread * octx->n_threads;
    octx->src0_spad.data = octx->ctx->vtcm_base;

    if (octx->ctx->vtcm_size < octx->src0_spad.size) {
        FARF(ERROR, "flash_attn: VTCM size too small (%zu < %zu)\n", octx->ctx->vtcm_size, octx->src0_spad.size);
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    worker_pool_run_func(octx->ctx->worker_pool, flash_attn_job_f32, &ctx, octx->n_threads);

    return HTP_STATUS_OK;
}

int op_flash_attn_ext(struct htp_ops_context * octx) {
    // Dispatch based on types. Currently only implementing F32.
    if (octx->src0.type == HTP_TYPE_F32 &&
        octx->src1.type == HTP_TYPE_F32 &&
        octx->src2.type == HTP_TYPE_F32) {
        return execute_op_flash_attn_f32(octx);
    }

    return HTP_STATUS_NO_SUPPORT;
}
