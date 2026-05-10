#include <math.h>
#include <stdint.h>
#include <string.h>

#include "hexagon_types.h"
#include "hvx-utils.h"
#include "hvx-copy.h"
#include "hvx-exp.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "hmx-ops.h"
#include "hmx-utils.h"
#include "vtcm-utils.h"
#include "hmx-queue.h"

static inline float gdn_mul_dot_f32_hmx(float * restrict dst, const float * restrict mul,
        const float * restrict dot, uint32_t n) {
    HVX_Vector acc = Q6_V_vzero();
    const uint32_t epv = 128 / sizeof(float);
    const uint32_t nvec = n / epv;
    for (uint32_t i = 0; i < nvec; ++i) {
        HVX_Vector vd = hvx_vmemu(dst + i * epv);
        HVX_Vector vm = hvx_vmem(mul + i * epv);
        HVX_Vector vdot = hvx_vmem(dot + i * epv);
        HVX_Vector out = hvx_vec_mul_f32_f32(vd, vm);
        hvx_vmemu(dst + i * epv) = out;
        acc = hvx_vec_add_f32_f32(acc, hvx_vec_mul_f32_f32(out, vdot));
    }
    return hvx_vec_get_f32(hvx_vec_reduce_sum_f32(acc));
}

static inline float gdn_add_scaled_dot_f32_hmx(float * restrict dst, const float * restrict src,
        float scale, const float * restrict dot, uint32_t n) {
    HVX_Vector acc = Q6_V_vzero();
    const HVX_Vector vscale = hvx_vec_splat_f32(scale);
    const uint32_t epv = 128 / sizeof(float);
    const uint32_t nvec = n / epv;
    for (uint32_t i = 0; i < nvec; ++i) {
        HVX_Vector vd = hvx_vmemu(dst + i * epv);
        HVX_Vector vs = hvx_vmem(src + i * epv);
        HVX_Vector vdot = hvx_vmem(dot + i * epv);
        HVX_Vector out = hvx_vec_add_f32_f32(vd, hvx_vec_mul_f32_f32(vs, vscale));
        hvx_vmemu(dst + i * epv) = out;
        acc = hvx_vec_add_f32_f32(acc, hvx_vec_mul_f32_f32(out, vdot));
    }
    return hvx_vec_get_f32(hvx_vec_reduce_sum_f32(acc));
}

typedef struct {
    __fp16 * state_tiles;
    __fp16 * q_tiles;
    __fp16 * attn_out_tiles;
    uint8_t * hmx_scales;
    size_t S_v;
} hmx_gdn_job_t;

static void hmx_gdn_worker(void * data) {
    hmx_gdn_job_t * job = (hmx_gdn_job_t *) data;
    const size_t n_tiles = job->S_v / HMX_FP16_TILE_N_ROWS;

    Q6_bias_mxmem2_A((void *) job->hmx_scales);

    for (size_t r = 0; r < n_tiles; ++r) {
        for (size_t c = 0; c < 1; ++c) { // We only need 1 column tile of outputs
            const __fp16 * row_tiles = job->state_tiles + r * n_tiles * HMX_FP16_TILE_N_ELMS;
            const __fp16 * col_tiles = job->q_tiles;
            __fp16 * out_tile = job->attn_out_tiles + r * HMX_FP16_TILE_N_ELMS;

            for (size_t k = 0; k < n_tiles; ++k) {
                Q6_activation_hf_mxmem_RR((unsigned int) row_tiles, 2047);
                Q6_weight_hf_mxmem_RR((unsigned int) col_tiles, 2047);
                row_tiles += HMX_FP16_TILE_N_ELMS;
                col_tiles += HMX_FP16_TILE_N_ELMS;
            }
            Q6_mxmem_AR_after_hf(out_tile, 0);
        }
    }
}

int hmx_gated_delta_net_ext(struct htp_ops_context * octx) {
    const struct htp_tensor * q     = octx->src[0];
    const struct htp_tensor * k     = octx->src[1];
    const struct htp_tensor * v     = octx->src[2];
    const struct htp_tensor * g     = octx->src[3];
    const struct htp_tensor * beta  = octx->src[4];
    const struct htp_tensor * state = octx->src[5];
    const struct htp_tensor * dst   = octx->dst;

    struct htp_context * const ctx = octx->ctx;
    if (!ctx->hmx_enabled) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const uint32_t S_v = v->ne[0];
    const uint32_t H = v->ne[1];
    const uint32_t n_tokens = v->ne[2];
    const uint32_t n_seqs = v->ne[3];

    if (S_v % 32 != 0 || S_v > 128) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const size_t n_tiles = S_v / HMX_FP16_TILE_N_ROWS;
    const size_t state_bytes = n_tiles * n_tiles * HMX_FP16_TILE_SIZE;
    const size_t vec_bytes = n_tiles * HMX_FP16_TILE_SIZE;

    uint8_t * vtcm_cur = ctx->vtcm_base;

    __fp16 * vtcm_state = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, state_bytes);
    __fp16 * vtcm_q     = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, vec_bytes);
    __fp16 * vtcm_attn  = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, vec_bytes);
    uint8_t * hmx_scales = vtcm_seq_alloc(&vtcm_cur, 256);
    __fp16 * vtcm_state_f16 = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, state_bytes);

    if ((size_t)(vtcm_cur - (uint8_t *)ctx->vtcm_base) > ctx->vtcm_size) {
        return HTP_STATUS_NO_SUPPORT;
    }

    hmx_init_column_scales(hmx_scales, Q6_V_vsplat_R(0x3c00));

    const float scale = 1.0f / sqrtf((float) S_v);
    float * dst_base = (float *) (uintptr_t) dst->data;
    float * state_out_base = dst_base + (uint64_t) S_v * H * n_tokens * n_seqs;
    const float * state_in_base = (const float *) (uintptr_t) state->data;

    const uint32_t rq3 = n_seqs / q->ne[3];
    const uint32_t rk3 = n_seqs / k->ne[3];

    float local_gate[128] __attribute__((aligned(128)));
    float local_q[128] __attribute__((aligned(128)));
    float local_k[128] __attribute__((aligned(128)));
    __fp16 local_q_f16[128 * 32] __attribute__((aligned(128)));

    for (uint32_t iv3 = 0; iv3 < n_seqs; ++iv3) {
        for (uint32_t iv1 = 0; iv1 < H; ++iv1) {
            float * s_out = state_out_base + ((uint64_t) iv3 * H + iv1) * S_v * S_v;
            const float * s_in = state_in_base + ((uint64_t) iv3 * H + iv1) * S_v * S_v;
            memcpy(s_out, s_in, S_v * S_v * sizeof(float));

            float * attn_data = dst_base + ((uint64_t) iv3 * n_tokens * H + iv1) * S_v;

            for (uint32_t t = 0; t < n_tokens; ++t) {
                const uint32_t iq3 = iv3 / rq3;
                const uint32_t ik3 = iv3 / rk3;
                const float * q_t = (const float *) ((const uint8_t *) q->data + (uint64_t) iq3 * q->nb[3] + t * q->nb[2] + iv1 * q->nb[1]);
                const float * k_t = (const float *) ((const uint8_t *) k->data + (uint64_t) ik3 * k->nb[3] + t * k->nb[2] + iv1 * k->nb[1]);
                const float * v_t = (const float *) ((const uint8_t *) v->data + (uint64_t) iv3 * v->nb[3] + t * v->nb[2] + iv1 * v->nb[1]);
                const float * g_t = (const float *) ((const uint8_t *) g->data + (uint64_t) iv3 * g->nb[3] + t * g->nb[2] + iv1 * g->nb[1]);
                const float beta_val = *(const float *) ((const uint8_t *) beta->data + (uint64_t) iv3 * beta->nb[3] + t * beta->nb[2] + iv1 * beta->nb[1]);

                memcpy(local_q, q_t, S_v * sizeof(float));
                memcpy(local_k, k_t, S_v * sizeof(float));
                hvx_exp_f32((uint8_t *) local_gate, (const uint8_t *) g_t, S_v, false);

                for (uint32_t j = 0; j < S_v; ++j) {
                    float * row = s_out + j * S_v;
                    const float sum = gdn_mul_dot_f32_hmx(row, local_gate, local_k, S_v);
                    const float dj = (v_t[j] - sum) * beta_val;
                    // Update row
                    for (uint32_t i = 0; i < S_v; ++i) {
                        row[i] += local_k[i] * dj;
                    }
                }

                // Prepare Q tiles for HMX: q is 1xS_v, we pad to 32xS_v because HMX requires 32 columns
                for (uint32_t i = 0; i < S_v; ++i) {
                    __fp16 q_val = (__fp16)(local_q[i] * scale);
                    for (uint32_t pad = 0; pad < 32; ++pad) {
                        local_q_f16[i * 32 + pad] = q_val;
                    }
                }

                HAP_compute_res_hmx_lock(ctx->vtcm_rctx);


                        // Convert state (s_out) from F32 to F16 before interleaving
                        for (uint32_t i = 0; i < S_v * S_v; ++i) {
                            vtcm_state_f16[i] = (__fp16) s_out[i];
                        }
                        hmx_interleave_rows_to_tiles(vtcm_state, vtcm_state_f16, S_v, S_v, S_v * sizeof(__fp16), 0, S_v);

                hmx_interleave_cols_to_tiles(vtcm_q, local_q_f16, S_v, 32, 32 * sizeof(__fp16), n_tiles, 0, S_v);

                hmx_gdn_job_t job = {
                    .state_tiles = vtcm_state,
                    .q_tiles = vtcm_q,
                    .attn_out_tiles = vtcm_attn,
                    .hmx_scales = hmx_scales,
                    .S_v = S_v
                };

                hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(hmx_gdn_worker, &job));
                hmx_queue_pop(ctx->hmx_queue);

                hmx_queue_suspend(ctx->hmx_queue);
                HAP_compute_res_hmx_unlock(ctx->vtcm_rctx);

                // Read output back
                // vtcm_attn is interleaved. Since it's n_tiles x 1 tile (S_v x 32), we can just read the first column.
                for (uint32_t r = 0; r < S_v; ++r) {
                    size_t tile_idx = r / 32;
                    size_t row_in_tile = r % 32;
                    attn_data[t * S_v + r] = (float) vtcm_attn[tile_idx * HMX_FP16_TILE_N_ELMS + row_in_tile];
                }
            }
        }
    }

    return HTP_STATUS_OK;
}
