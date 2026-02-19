#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <HAP_perf.h>

#include <math.h>
#include <string.h>

#include "hex-dma.h"
#include "hvx-utils.h"
#include "hex-fastdiv.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-msg.h"
#include "htp-ops.h"

#define HTP_SOFTMAX_SPAD_NROWS  16
#define HTP_SOFTMAX_SPAD_BLOCK  (HTP_SOFTMAX_SPAD_NROWS/2)

#define htp_softmax_preamble3                              \
    const uint32_t ne00 = src0->ne[0];                     \
    const uint32_t ne01 = src0->ne[1];                     \
    const uint32_t ne02 = src0->ne[2];                     \
    const uint32_t ne03 = src0->ne[3];                     \
                                                           \
    const uint32_t nb00 = src0->nb[0];                     \
    const uint32_t nb01 = src0->nb[1];                     \
    const uint32_t nb02 = src0->nb[2];                     \
    const uint32_t nb03 = src0->nb[3];                     \
                                                           \
    const uint32_t ne10 = (src1->ne[0]) ? src1->ne[0] : 1; \
    const uint32_t ne11 = (src1->ne[0]) ? src1->ne[1] : 1; \
    const uint32_t ne12 = (src1->ne[0]) ? src1->ne[2] : 1; \
    const uint32_t ne13 = (src1->ne[0]) ? src1->ne[3] : 1; \
                                                           \
    const uint32_t nb10 = (src1->ne[0]) ? src1->nb[0] : 1; \
    const uint32_t nb11 = (src1->ne[0]) ? src1->nb[1] : 1; \
    const uint32_t nb12 = (src1->ne[0]) ? src1->nb[2] : 1; \
    const uint32_t nb13 = (src1->ne[0]) ? src1->nb[3] : 1; \
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

struct htp_softmax_context {
    bool     use_f16;
    bool     use_src1;
    uint32_t n_head;
    uint32_t n_head_log2;

    float scale;
    float max_bias;
    float m0;
    float m1;

    struct htp_ops_context * octx;

    size_t src0_row_size;
    size_t dst_row_size;
    size_t src0_row_size_aligned;
    size_t dst_row_size_aligned;
    size_t spad_pad_offset;
    size_t spad_src1_offset; // only used if use_src1
    size_t src1_row_size;
    size_t src1_row_size_aligned;

    uint32_t src0_nrows;
    uint32_t src0_nrows_per_thread;
};

static void init_softmax_ctx(struct htp_softmax_context * smctx, struct htp_ops_context * octx) {
    const struct htp_tensor * src0 = &octx->src0;
    const struct htp_tensor * src1 = &octx->src1;

    memset(smctx, 0, sizeof(struct htp_softmax_context));

    memcpy(&smctx->scale, (float *) octx->op_params, sizeof(float));
    memcpy(&smctx->max_bias, (float *) octx->op_params + 1, sizeof(float));

    smctx->n_head      = src0->ne[2];
    smctx->n_head_log2 = 1u << (uint32_t) floor(log2(smctx->n_head));

    smctx->m0 = powf(2.0f, -(smctx->max_bias) / smctx->n_head_log2);
    smctx->m1 = powf(2.0f, -(smctx->max_bias / 2.0f) / smctx->n_head_log2);

    smctx->use_src1 = (src1->ne[0] != 0);
    smctx->use_f16  = (src1->ne[0] != 0) && (src1->type == HTP_TYPE_F16);

    smctx->octx = octx;

}

static void hvx_fast_softmax_prep_f32(const uint8_t * restrict src,
                                      uint8_t * restrict dst,
                                      const int num_elems,
                                      float     scale,
                                      const uint8_t * restrict mask,
                                      float slope) {
    const uint8_t * restrict src_curr  = src;
    uint8_t * restrict dst_curr        = dst;
    const uint8_t * restrict mask_curr = mask;

    HVX_Vector scale_vec = hvx_vec_splat_f32(scale);
    HVX_Vector slope_vec = hvx_vec_splat_f32(slope);

    int step_of_1 = num_elems >> 5;

    #pragma unroll(4)
    for (int i = 0; i < step_of_1; i++) {
        HVX_Vector v1 = *(HVX_Vector *) src_curr;

        HVX_Vector v3 = *(HVX_Vector *) mask_curr;

        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_vec);

        HVX_Vector v4 = Q6_Vqf32_vmpy_VsfVsf(v3, slope_vec);

        HVX_Vector v5 = Q6_Vqf32_vadd_Vqf32Vqf32(v2, v4);

        *(HVX_Vector *) dst_curr = Q6_Vsf_equals_Vqf32(v5);

        src_curr += VLEN;
        dst_curr += VLEN;
        mask_curr += VLEN;
    }
}

static void hvx_fast_softmax_f32(const uint8_t * restrict src,
                                 uint8_t * restrict dst,
                                 uint8_t * restrict pad,
                                 const int num_elems) {
    const HVX_Vector * restrict v_src = (HVX_Vector *) src;
    HVX_Vector * restrict v_pad       = (HVX_Vector *) pad;
    HVX_Vector * restrict v_dst       = (HVX_Vector *) dst;

    HVX_Vector sum_vec = Q6_V_vsplat_R(0x00000000);
    HVX_Vector max_vec = hvx_vec_splat_f32(((const float *) src)[0]);
    HVX_Vector zero_v  = Q6_V_vzero();
    HVX_Vector one_v   = hvx_vec_splat_f32(1.0);

    int step_of_1 = num_elems >> 5;

    #pragma unroll(4)
    for (int i = 0; i < step_of_1; i++) {
        HVX_Vector v1 = v_src[i];
        max_vec       = Q6_Vsf_vmax_VsfVsf(max_vec, v1);
    }

    HVX_Vector v = hvx_vec_reduce_max_f32(max_vec);
    max_vec      = hvx_vec_repl4(v);

    #pragma unroll(4)
    for (int i = 0; i < step_of_1; i++) {
        HVX_Vector v1 = v_src[i];
        HVX_Vector v2 = Q6_Vqf32_vsub_VsfVsf(v1, max_vec);

        HVX_Vector v3 = hvx_vec_exp_f32(Q6_Vsf_equals_Vqf32(v2));

        sum_vec = Q6_Vqf32_vadd_VsfVsf(Q6_Vsf_equals_Vqf32(sum_vec), v3);

        v_pad[i] = v3;
    }

    v       = hvx_vec_reduce_sum_f32(Q6_Vsf_equals_Vqf32(sum_vec));
    sum_vec = hvx_vec_repl4(v);

    HVX_VectorPred pos_sum   = Q6_Q_vcmp_gt_VwVw(sum_vec, zero_v);
    HVX_Vector     v4        = hvx_vec_inverse_f32(sum_vec);
    HVX_Vector     scale_vec = Q6_V_vmux_QVV(pos_sum, v4, one_v);

    #pragma unroll(4)
    for (int i = 0; i < step_of_1; i++) {
        HVX_Vector v1 = v_pad[i];
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_vec);
        v_dst[i]      = Q6_Vsf_equals_Vqf32(v2);
    }
}

static float hvx_softmax_f32(const uint8_t * restrict src,
                             uint8_t * restrict dst,
                             uint8_t * restrict spad,
                             const int   num_elems,
                             const float max) {
    hvx_sub_scalar_f32(spad, src, max, num_elems);

    hvx_exp_f32(spad, dst, num_elems, false);

    float sum = hvx_reduce_sum_f32(dst, num_elems);

    return sum;
}

static void softmax_job_f32(unsigned int nth, unsigned int ith, void * data) {
    struct htp_softmax_context * smctx = (struct htp_softmax_context *) data;
    struct htp_ops_context * octx = smctx->octx;

    const struct htp_tensor * src0 = &octx->src0;
    const struct htp_tensor * src1 = &octx->src1;
    struct htp_tensor *       dst  = &octx->dst;

    htp_softmax_preamble3;

    const uint32_t src0_nrows            = smctx->src0_nrows;
    const uint32_t src0_nrows_per_thread = smctx->src0_nrows_per_thread;

    const uint32_t src0_start_row = src0_nrows_per_thread * ith;
    const uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, src0_nrows);

    // no work for this thread
    if (src0_start_row >= src0_end_row) {
        return;
    }

    uint64_t t1, t2;
    t1 = HAP_perf_get_qtimer_count();

    uint8_t * src0_spad_base_ptr = octx->src0_spad.data + (ith * octx->src0_spad.size_per_thread);
    uint8_t * dst_spad_base_ptr  = octx->dst_spad.data + (ith * octx->dst_spad.size_per_thread);

    dma_queue * dma_queue = octx->ctx->dma[ith];

    uint32_t prev_i2 = (uint32_t)-1;
    float slope = 1.0f;

    const bool use_src1 = smctx->use_src1;

    // Initial indices
    uint32_t cur_r = src0_start_row;
    uint32_t cur_i3 = cur_r / (ne02 * ne01);
    uint32_t rem = cur_r % (ne02 * ne01);
    uint32_t cur_i2 = rem / ne01;
    uint32_t cur_i1 = rem % ne01;

    // Prefetch loop variables
    uint32_t pf_r = cur_r;
    uint32_t pf_i1 = cur_i1;
    uint32_t pf_i2 = cur_i2;
    uint32_t pf_i3 = cur_i3;
    uint32_t pf_rem = src0_end_row - cur_r;

    // Compute loop variables
    uint32_t cm_r = cur_r;
    uint32_t cm_i1 = cur_i1;
    uint32_t cm_i2 = cur_i2;
    uint32_t cm_i3 = cur_i3;
    uint32_t cm_rem = src0_end_row - cur_r;

    while (cm_rem > 0) {
        // PREFETCH
        // Fill up to HTP_SOFTMAX_SPAD_NROWS if queue depth is low
        while (pf_rem > 0 && dma_queue_depth(dma_queue) < (HTP_SOFTMAX_SPAD_NROWS / HTP_SOFTMAX_SPAD_BLOCK)) {
             uint32_t block = MIN(pf_rem, HTP_SOFTMAX_SPAD_BLOCK);
             uint32_t rows_in_i1 = ne01 - pf_i1;
             block = MIN(block, rows_in_i1);

             uint32_t pf_slot = (pf_r - src0_start_row) % HTP_SOFTMAX_SPAD_NROWS;
             uint32_t slots_avail = HTP_SOFTMAX_SPAD_NROWS - pf_slot;
             block = MIN(block, slots_avail);

             uint8_t * s0_spad = src0_spad_base_ptr + pf_slot * smctx->src0_row_size_aligned;
             uint8_t * d_spad  = dst_spad_base_ptr  + pf_slot * smctx->dst_row_size_aligned;

             const uint8_t * s0_addr = (const uint8_t *) src0->data + pf_i3 * nb03 + pf_i2 * nb02 + pf_i1 * nb01;

             uint8_t * s1_spad = NULL;
             const uint8_t * s1_addr = NULL;
             size_t s1_stride = 0;

             if (use_src1) {
                 uint32_t i12 = (ne12 == ne02) ? pf_i2 : (pf_i2 % ne12);
                 uint32_t i13 = (ne13 == ne03) ? pf_i3 : (pf_i3 % ne13);
                 uint32_t i11 = (ne11 == ne01) ? pf_i1 : (pf_i1 % ne11);

                 s1_addr = (const uint8_t *) src1->data + i13 * nb13 + i12 * nb12 + i11 * nb11;
                 s1_spad = src0_spad_base_ptr + smctx->spad_src1_offset + pf_slot * smctx->src1_row_size_aligned;
                 s1_stride = (ne11 == 1) ? 0 : nb11;
             }

             // Push Dummy DST (to carry pointer)
             dma_queue_push_vtcm_to_ddr(dma_queue, dma_make_ptr((void*)dst->data, d_spad), 0, 0, 0);

             // Push SRC0
             dma_queue_push_ddr_to_vtcm(dma_queue, dma_make_ptr(s0_spad, s0_addr),
                                        smctx->src0_row_size_aligned, smctx->src0_row_size, block);

             // Push SRC1
             if (use_src1) {
                 dma_queue_push(dma_queue, dma_make_ptr(s1_spad, s1_addr),
                                smctx->src1_row_size_aligned, // dst_stride
                                s1_stride, // src_stride
                                smctx->src1_row_size, // width
                                block);
             }

             pf_r += block;
             pf_i1 += block;
             if (pf_i1 >= ne01) {
                 pf_i1 = 0;
                 pf_i2++;
                 if (pf_i2 >= ne02) {
                     pf_i2 = 0;
                     pf_i3++;
                 }
             }
             pf_rem -= block;
        }

        // COMPUTE
        uint32_t cm_slot = (cm_r - src0_start_row) % HTP_SOFTMAX_SPAD_NROWS;
        uint32_t c_block = MIN(cm_rem, HTP_SOFTMAX_SPAD_BLOCK);
        uint32_t rows_in_i1 = ne01 - cm_i1;
        c_block = MIN(c_block, rows_in_i1);
        uint32_t slots_avail = HTP_SOFTMAX_SPAD_NROWS - cm_slot;
        c_block = MIN(c_block, slots_avail);

        uint8_t * d_spad = (uint8_t *) dma_queue_pop(dma_queue).src;
        uint8_t * s0_spad = (uint8_t *) dma_queue_pop(dma_queue).dst;
        uint8_t * s1_spad = NULL;
        if (use_src1) {
            s1_spad = (uint8_t *) dma_queue_pop(dma_queue).dst;
        }

        uint8_t * p_spad = src0_spad_base_ptr + smctx->spad_pad_offset + cm_slot * smctx->src0_row_size_aligned;

        for (uint32_t b = 0; b < c_block; ++b) {
            uint32_t cur_i2_local = cm_i2; // constant for block

            // ALiBi
            if (cur_i2_local != prev_i2) {
                const uint32_t h = cur_i2_local;
                slope = (smctx->max_bias > 0.0f) ?
                            h < smctx->n_head_log2 ?
                            powf(smctx->m0, h + 1) :
                            powf(smctx->m1, 2 * (h - smctx->n_head_log2) + 1) :
                            1.0f;
                prev_i2 = cur_i2_local;
            }

            uint8_t * row_s0 = s0_spad + b * smctx->src0_row_size_aligned;
            uint8_t * row_d  = d_spad  + b * smctx->dst_row_size_aligned;
            uint8_t * row_p  = p_spad  + b * smctx->src0_row_size_aligned;

            if (use_src1) {
                uint8_t * row_s1 = s1_spad + b * smctx->src1_row_size_aligned;
                if (smctx->use_f16) {
                    hvx_scale_f32(row_s0, row_s0, ne00, smctx->scale);
                    float * r_s0_f = (float *) row_s0;
                    __fp16 * r_s1_h = (__fp16 *) row_s1;
                    for (uint32_t i = 0; i < ne00; ++i) {
                        r_s0_f[i] += slope * (float) r_s1_h[i];
                    }
                } else {
                    hvx_fast_softmax_prep_f32(row_s0, row_s0, ne00, smctx->scale, row_s1, slope);
                }
            } else {
                hvx_scale_f32(row_s0, row_s0, ne00, smctx->scale);
            }

            hvx_fast_softmax_f32(row_s0, row_d, row_p, ne00);
        }

        uint8_t * dst_addr = (uint8_t *) dst->data + cm_i3 * nb3 + cm_i2 * nb2 + cm_i1 * nb1;
        dma_queue_push_vtcm_to_ddr(dma_queue, dma_make_ptr(dst_addr, d_spad),
                                   smctx->dst_row_size, smctx->dst_row_size_aligned, c_block);

        cm_r += c_block;
        cm_i1 += c_block;
        if (cm_i1 >= ne01) {
            cm_i1 = 0;
            cm_i2++;
            if (cm_i2 >= ne02) {
                cm_i2 = 0;
                cm_i3++;
            }
        }
        cm_rem -= c_block;
    }

    dma_queue_flush(dma_queue);

    t2 = HAP_perf_get_qtimer_count();

    FARF(HIGH, "softmax-f32 %d/%d/%d: %ux%ux%ux%u (%u:%u) x %ux%ux%ux%u -> %ux%ux%ux%u usec %u\n", ith, nth,
         smctx->use_f16, ne00, ne01, ne02, ne03, src0_start_row, src0_end_row, ne10, ne11, ne12, ne13,
         ne0, ne1, ne2, ne3, (unsigned) HAP_perf_qtimer_count_to_us(t2 - t1));
}

static int execute_op_softmax_f32(struct htp_ops_context * octx) {
    int err = HTP_STATUS_OK;

    const struct htp_tensor * src0 = &octx->src0;
    const struct htp_tensor * src1 = &octx->src1;
    struct htp_tensor *       dst  = &octx->dst;

    struct htp_softmax_context smctx;
    const char * op_type = "softmax-f32";

    switch (octx->op) {
        case HTP_OP_SOFTMAX:
            init_softmax_ctx(&smctx, octx);
            break;

        default:
            FARF(ERROR, "Unsupported Op %u\n", octx->op);
            return HTP_STATUS_NO_SUPPORT;
    }

    const uint32_t n_threads = octx->n_threads;

    const size_t src0_row_size = src0->nb[1];
    const size_t dst_row_size  = dst->nb[1];
    const size_t src1_row_size = (src1->ne[0]) ? src1->nb[1] : 0;

    // Aligned row sizes
    const size_t src0_row_size_aligned = hex_round_up(src0_row_size, 128);
    const size_t dst_row_size_aligned  = hex_round_up(dst_row_size, 128);
    const size_t src1_row_size_aligned = hex_round_up(src1_row_size, 128);

    // Calculate spad sizes per thread
    // src0_spad includes: src0 rows, pad rows (intermediate), and src1 rows (if used)
    size_t src0_spad_size = HTP_SOFTMAX_SPAD_NROWS * src0_row_size_aligned;
    size_t pad_spad_size  = HTP_SOFTMAX_SPAD_NROWS * src0_row_size_aligned; // same size as src0
    size_t src1_spad_size = (src1->ne[0]) ? (HTP_SOFTMAX_SPAD_NROWS * src1_row_size_aligned) : 0;

    size_t dst_spad_size  = HTP_SOFTMAX_SPAD_NROWS * dst_row_size_aligned;

    size_t src0_total_per_thread = src0_spad_size + pad_spad_size + src1_spad_size;
    size_t dst_total_per_thread  = dst_spad_size;

    size_t total_vtcm = (src0_total_per_thread + dst_total_per_thread) * n_threads;

    if (src1->ne[0]) {
        FARF(HIGH,
             "%s: %ux%ux%ux%u x %ux%ux%ux%u -> %ux%ux%ux%u : vtcm needed %u\n",
             op_type, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src1->ne[0], src1->ne[1], src1->ne[2],
             src1->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], total_vtcm);
    } else {
        FARF(HIGH, "%s: %ux%ux%ux%u -> %ux%ux%ux%u : vtcm needed %u\n", op_type,
             src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
             total_vtcm);
    }

    if (octx->ctx->vtcm_size < total_vtcm) {
        FARF(ERROR, "%s : current VTCM reservation %zu is too small, needed %zu\n", op_type, octx->ctx->vtcm_size,
             total_vtcm);
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    octx->src0_spad.size_per_thread = src0_total_per_thread;
    octx->dst_spad.size_per_thread  = dst_total_per_thread;
    octx->src1_spad.size_per_thread = 0; // Packed into src0

    octx->src0_spad.size = octx->src0_spad.size_per_thread * n_threads;
    octx->dst_spad.size  = octx->dst_spad.size_per_thread * n_threads;
    octx->src1_spad.size = 0;

    octx->src0_spad.data = octx->ctx->vtcm_base;
    octx->src1_spad.data = NULL;
    octx->dst_spad.data  = octx->src0_spad.data + octx->src0_spad.size;

    smctx.src0_row_size = src0_row_size;
    smctx.src0_row_size_aligned = src0_row_size_aligned;
    smctx.dst_row_size = dst_row_size;
    smctx.dst_row_size_aligned = dst_row_size_aligned;
    smctx.src1_row_size = src1_row_size;
    smctx.src1_row_size_aligned = src1_row_size_aligned;

    smctx.spad_pad_offset  = src0_spad_size;
    smctx.spad_src1_offset = src0_spad_size + pad_spad_size;

    uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    smctx.src0_nrows = src0_nrows;

    if (!(octx->flags & HTP_OPFLAGS_SKIP_COMPUTE)) {
        uint32_t n_jobs             = MIN(n_threads, src0_nrows);
        smctx.src0_nrows_per_thread = (src0_nrows + n_jobs - 1) / n_jobs;
        worker_pool_run_func(octx->ctx->worker_pool, softmax_job_f32, &smctx, n_jobs);
    }

    return err;
}

int op_softmax(struct htp_ops_context * octx) {
    int err = HTP_STATUS_OK;

    switch (octx->src0.type) {
        case HTP_TYPE_F32:
            err = execute_op_softmax_f32(octx);
            break;

        default:
            err = HTP_STATUS_NO_SUPPORT;
            break;
    }

    return err;
}
