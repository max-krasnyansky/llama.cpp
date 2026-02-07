#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <HAP_perf.h>

#include <math.h>
#include <string.h>

#include "hex-dma.h"
#include "hvx-utils.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-msg.h"
#include "htp-ops.h"

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

typedef void (*hvx_elemwise_f32_func)(uint8_t * data_dst, const uint8_t * src0, const uint8_t * src1, const uint32_t num_elems);

static hvx_elemwise_f32_func func_table_HVX[]     = { hvx_mul_f32, hvx_add_f32, hvx_sub_f32, hvx_div_f32 };
static hvx_elemwise_f32_func func_table_HVX_opt[] = { hvx_mul_f32_aa, hvx_add_f32_aa, hvx_sub_f32_aa, hvx_div_f32_aa };

struct htp_binary_context {
    struct htp_ops_context * octx;
    enum htp_op              op;
    uint32_t                 nrows_per_thread;

    struct fastdiv_values src0_div21; // fastdiv values for ne2 * ne1
    struct fastdiv_values src0_div1;  // fastdiv values for ne1

    // Broadcasting divisors
    struct fastdiv_values src1_div3;  // fastdiv values for ne13
    struct fastdiv_values src1_div2;  // fastdiv values for ne12
    struct fastdiv_values src1_div1;  // fastdiv values for ne11
};

#define htp_binary_preamble            \
    const struct htp_tensor * src0 = &octx->src0; \
    const struct htp_tensor * src1 = &octx->src1; \
    const struct htp_tensor * src2 = &octx->src2; \
    struct htp_tensor *       dst  = &octx->dst;  \
                                       \
    const uint32_t ne00 = src0->ne[0]; \
    const uint32_t ne01 = src0->ne[1]; \
    const uint32_t ne02 = src0->ne[2]; \
    const uint32_t ne03 = src0->ne[3]; \
                                       \
    const uint32_t ne10 = src1->ne[0]; \
    const uint32_t ne11 = src1->ne[1]; \
    const uint32_t ne12 = src1->ne[2]; \
    const uint32_t ne13 = src1->ne[3]; \
                                       \
    const uint32_t ne0 = dst->ne[0];   \
    const uint32_t ne1 = dst->ne[1];   \
    const uint32_t ne2 = dst->ne[2];   \
    const uint32_t ne3 = dst->ne[3];   \
                                       \
    const uint32_t nb00 = src0->nb[0]; \
    const uint32_t nb01 = src0->nb[1]; \
    const uint32_t nb02 = src0->nb[2]; \
    const uint32_t nb03 = src0->nb[3]; \
                                       \
    const uint32_t nb10 = src1->nb[0]; \
    const uint32_t nb11 = src1->nb[1]; \
    const uint32_t nb12 = src1->nb[2]; \
    const uint32_t nb13 = src1->nb[3]; \
                                       \
    const uint32_t nb0 = dst->nb[0];   \
    const uint32_t nb1 = dst->nb[1];   \
    const uint32_t nb2 = dst->nb[2];   \
    const uint32_t nb3 = dst->nb[3];   \
                                       \
    const uint32_t src0_nrows_per_thread = bctx->nrows_per_thread;

static void binary_job_f32_per_thread(struct htp_binary_context * bctx,
                                      uint32_t                 nth,
                                      uint32_t                 ith,
                                      dma_queue *              dma_queue) {
    struct htp_ops_context * octx = bctx->octx;
    htp_binary_preamble;

    const size_t src0_row_size = nb01;
    const size_t src1_row_size = nb11;
    const size_t dst_row_size  = nb1;

    const size_t src0_row_size_aligned = hex_round_up(src0_row_size, VLEN);
    const size_t src1_row_size_aligned = hex_round_up(src1_row_size, VLEN);
    const size_t dst_row_size_aligned  = hex_round_up(dst_row_size, VLEN);

    const uint32_t src0_nrows = ne01 * ne02 * ne03;
    const uint32_t src0_start_row = src0_nrows_per_thread * ith;
    const uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, src0_nrows);

    if (src0_start_row >= src0_end_row) return;

    if (nb00 != sizeof(float)) {
        FARF(ERROR, "binary-f32: src0 inner dim not contiguous");
        return;
    }

    uint8_t * src0_spad_base = octx->src0_spad.data + (ith * octx->src0_spad.size_per_thread);
    uint8_t * src1_spad_base = octx->src1_spad.data + (ith * octx->src1_spad.size_per_thread);
    uint8_t * dst_spad_base  = octx->dst_spad.data + (ith * octx->dst_spad.size_per_thread);

    size_t src0_half = octx->src0_spad.size_per_thread / 2;
    size_t src1_half = octx->src1_spad.size_per_thread / 2;
    size_t dst_half  = octx->dst_spad.size_per_thread / 2;

    const int BLOCK = src0_half / src0_row_size_aligned;
    if (BLOCK == 0) {
        FARF(ERROR, "binary-f32: VTCM too small");
        return;
    }

    const uint32_t ne02_ne01 = ne02 * ne01;
    const bool broadcast_simple = (ne11 == ne01 && ne12 == ne02 && ne13 == ne03);
    hvx_elemwise_f32_func func_HVX = func_table_HVX_opt[bctx->op];

    for (uint32_t ir = src0_start_row, spad_idx = 0; ir < src0_end_row && spad_idx < 2; ir += BLOCK, spad_idx++) {
        const uint32_t block_size = MIN(BLOCK, src0_end_row - ir);

        // Fence (pass dst base for this block)
        dma_queue_push_vtcm_to_ddr(dma_queue,
            dma_make_ptr(dst->data, dst_spad_base + (spad_idx * dst_half)),
            dst_row_size, dst_row_size_aligned, 0);

        for (uint32_t r = 0; r < block_size; r++) {
            uint32_t curr_ir = ir + r;
            const uint32_t i03 = fastdiv(curr_ir, &bctx->src0_div21);
            const uint32_t i02 = fastdiv(curr_ir - i03 * ne02_ne01, &bctx->src0_div1);
            const uint32_t i01 = (curr_ir - i03 * ne02_ne01 - i02 * ne01);

            // src0
            const uint8_t * s0_ptr = (const uint8_t *)src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01;
            uint8_t * s0_spad = src0_spad_base + (spad_idx * src0_half) + r * src0_row_size_aligned;
            dma_queue_push_ddr_to_vtcm(dma_queue, dma_make_ptr(s0_spad, s0_ptr), src0_row_size_aligned, src0_row_size, 1);

            // src1
            const uint8_t * s1_ptr;
            if (broadcast_simple) {
                s1_ptr = (const uint8_t *)src1->data + i03 * nb13 + i02 * nb12 + i01 * nb11;
            } else {
                const uint32_t i13 = fastmodulo(i03, ne13, &bctx->src1_div3);
                const uint32_t i12 = fastmodulo(i02, ne12, &bctx->src1_div2);
                const uint32_t i11 = fastmodulo(i01, ne11, &bctx->src1_div1);
                s1_ptr = (const uint8_t *)src1->data + i13 * nb13 + i12 * nb12 + i11 * nb11;
            }
            uint8_t * s1_spad = src1_spad_base + (spad_idx * src1_half) + r * src1_row_size_aligned;

            if (ne10 == 1) {
                // scalar broadcast
                dma_queue_push_ddr_to_vtcm(dma_queue, dma_make_ptr(s1_spad, s1_ptr), 128, 4, 1);
            } else {
                dma_queue_push_ddr_to_vtcm(dma_queue, dma_make_ptr(s1_spad, s1_ptr), src1_row_size_aligned, src1_row_size, 1);
            }
        }
    }

    for (uint32_t ir = src0_start_row; ir < src0_end_row; ir += BLOCK) {
        const uint32_t block_size = MIN(BLOCK, src0_end_row - ir);

        float * dst_block_base = (float *) dma_queue_pop(dma_queue).src; // Pop fence

        for (uint32_t r = 0; r < block_size; r++) {
            dma_ptr s0_dptr = dma_queue_pop(dma_queue);
            dma_ptr s1_dptr = dma_queue_pop(dma_queue);

            uint8_t * s0_vec = (uint8_t *)s0_dptr.dst;
            uint8_t * s1_vec = (uint8_t *)s1_dptr.dst;
            uint8_t * d_vec  = (uint8_t *)dst_block_base + r * dst_row_size_aligned;

            if (ne10 == 1) {
                hvx_splat_f32_a(d_vec, *(float *)s1_vec, ne00);
                func_HVX(d_vec, s0_vec, d_vec, ne00);
            } else {
                func_HVX(d_vec, s0_vec, s1_vec, ne00);
            }
        }

        // Push writeback
        for (uint32_t r = 0; r < block_size; r++) {
            uint32_t curr_ir = ir + r;
            const uint32_t i03 = fastdiv(curr_ir, &bctx->src0_div21);
            const uint32_t i02 = fastdiv(curr_ir - i03 * ne02_ne01, &bctx->src0_div1);
            const uint32_t i01 = (curr_ir - i03 * ne02_ne01 - i02 * ne01);

            uint8_t * d_ptr = (uint8_t *)dst->data + i03 * nb3 + i02 * nb2 + i01 * nb1;
            uint8_t * d_spad = (uint8_t *)dst_block_base + r * dst_row_size_aligned;

            dma_queue_push_vtcm_to_ddr(dma_queue, dma_make_ptr(d_ptr, d_spad), dst_row_size, dst_row_size_aligned, 1);
        }

        uint32_t pref_ir = ir + 2 * BLOCK;
        if (pref_ir < src0_end_row) {
            uint32_t pref_block_size = MIN(BLOCK, src0_end_row - pref_ir);
            // We need correct spad base. (ir / BLOCK) toggles between even and odd
            // We need to calculate spad_idx based on pref_ir relative to start
            uint32_t rel_block_idx = (pref_ir - src0_start_row) / BLOCK;
            uint32_t spad_idx = rel_block_idx % 2;

            uint8_t * p_src0_base = octx->src0_spad.data + (ith * octx->src0_spad.size_per_thread) + (spad_idx * src0_half);
            uint8_t * p_src1_base = octx->src1_spad.data + (ith * octx->src1_spad.size_per_thread) + (spad_idx * src1_half);
            uint8_t * p_dst_base  = octx->dst_spad.data + (ith * octx->dst_spad.size_per_thread) + (spad_idx * dst_half);

            // Fence
             dma_queue_push_vtcm_to_ddr(dma_queue, dma_make_ptr(dst->data, p_dst_base), dst_row_size, dst_row_size_aligned, 0);

             for (uint32_t r = 0; r < pref_block_size; r++) {
                 uint32_t curr_ir = pref_ir + r;
                 const uint32_t i03 = fastdiv(curr_ir, &bctx->src0_div21);
                 const uint32_t i02 = fastdiv(curr_ir - i03 * ne02_ne01, &bctx->src0_div1);
                 const uint32_t i01 = (curr_ir - i03 * ne02_ne01 - i02 * ne01);

                 const uint8_t * s0_ptr = (const uint8_t *)src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01;
                 uint8_t * s0_spad = p_src0_base + r * src0_row_size_aligned;
                 dma_queue_push_ddr_to_vtcm(dma_queue, dma_make_ptr(s0_spad, s0_ptr), src0_row_size_aligned, src0_row_size, 1);

                 const uint8_t * s1_ptr;
                 if (broadcast_simple) {
                     s1_ptr = (const uint8_t *)src1->data + i03 * nb13 + i02 * nb12 + i01 * nb11;
                 } else {
                     const uint32_t i13 = fastmodulo(i03, ne13, &bctx->src1_div3);
                     const uint32_t i12 = fastmodulo(i02, ne12, &bctx->src1_div2);
                     const uint32_t i11 = fastmodulo(i01, ne11, &bctx->src1_div1);
                     s1_ptr = (const uint8_t *)src1->data + i13 * nb13 + i12 * nb12 + i11 * nb11;
                 }
                 uint8_t * s1_spad = p_src1_base + r * src1_row_size_aligned;

                 if (ne10 == 1) {
                     dma_queue_push_ddr_to_vtcm(dma_queue, dma_make_ptr(s1_spad, s1_ptr), 128, 4, 1);
                 } else {
                     dma_queue_push_ddr_to_vtcm(dma_queue, dma_make_ptr(s1_spad, s1_ptr), src1_row_size_aligned, src1_row_size, 1);
                 }
             }
        }
    }
    dma_queue_flush(dma_queue);
}

static void binary_job_dispatcher_f32_dma(unsigned int n, unsigned int i, void * data) {
    struct htp_binary_context * bctx = (struct htp_binary_context *) data;
    binary_job_f32_per_thread(bctx, n, i, bctx->octx->ctx->dma[i]);
}

static void binary_add_id_job_f32_per_thread(struct htp_ops_context * octx,
                                             uint8_t *                spad_data,
                                             uint32_t                 nth,
                                             uint32_t                 ith,
                                             hvx_elemwise_f32_func    func_HVX) {
#define htp_binary_preamble_octx            \
    const struct htp_tensor * src0 = &octx->src0; \
    const struct htp_tensor * src1 = &octx->src1; \
    const struct htp_tensor * src2 = &octx->src2; \
    struct htp_tensor *       dst  = &octx->dst;  \
                                       \
    const uint32_t ne00 = src0->ne[0]; \
    const uint32_t ne01 = src0->ne[1]; \
    const uint32_t ne02 = src0->ne[2]; \
    const uint32_t ne03 = src0->ne[3]; \
                                       \
    const uint32_t ne10 = src1->ne[0]; \
    const uint32_t ne11 = src1->ne[1]; \
    const uint32_t ne12 = src1->ne[2]; \
    const uint32_t ne13 = src1->ne[3]; \
                                       \
    const uint32_t ne0 = dst->ne[0];   \
    const uint32_t ne1 = dst->ne[1];   \
    const uint32_t ne2 = dst->ne[2];   \
    const uint32_t ne3 = dst->ne[3];   \
                                       \
    const uint32_t nb00 = src0->nb[0]; \
    const uint32_t nb01 = src0->nb[1]; \
    const uint32_t nb02 = src0->nb[2]; \
    const uint32_t nb03 = src0->nb[3]; \
                                       \
    const uint32_t nb10 = src1->nb[0]; \
    const uint32_t nb11 = src1->nb[1]; \
    const uint32_t nb12 = src1->nb[2]; \
    const uint32_t nb13 = src1->nb[3]; \
                                       \
    const uint32_t nb0 = dst->nb[0];   \
    const uint32_t nb1 = dst->nb[1];   \
    const uint32_t nb2 = dst->nb[2];   \
    const uint32_t nb3 = dst->nb[3];   \
                                       \
    const uint32_t src0_nrows_per_thread = octx->src0_nrows_per_thread;

    htp_binary_preamble_octx;

    const size_t src0_row_size = nb01;
    const size_t src1_row_size = nb11;
    const size_t dst_row_size  = nb1;

    const uint32_t src0_nrows = ne01 * ne02 * ne03;  // src0 rows

    const uint32_t src0_start_row = src0_nrows_per_thread * ith;
    const uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, src0_nrows);

    // no work for this thread
    if (src0_start_row >= src0_end_row) {
        return;
    }

    uint64_t t1, t2;
    t1 = HAP_perf_get_qtimer_count();

    const uint8_t * restrict data_src0 = (const uint8_t *) src0->data;
    const uint8_t * restrict data_src1 = (const uint8_t *) src1->data;
    uint8_t * restrict data_dst        = (uint8_t *) dst->data;

    const uint32_t ne02_ne01  = ne02 * ne01;
    for (uint32_t ir = src0_start_row; ir < src0_end_row; ir++) {
        // src0 indices
        const uint32_t i03 = fastdiv(ir, &octx->src0_div21);
        const uint32_t i02 = fastdiv(ir - i03 * ne02_ne01, &octx->src0_div1);
        const uint32_t i01 = (ir - i03 * ne02_ne01 - i02 * ne01);

        // src1 indices
        const int i11 = *(int32_t *) ((char *) src2->data + i01 * src2->nb[0] + i02 * src2->nb[1]);
        assert(i11 >= 0 && i11 < ne11);

        float * restrict dst_ptr        = (float *) (data_dst + i03 * nb3 + i02 * nb2 + i01 * nb1);
        const float * restrict src0_ptr = (const float *) (data_src0 + i03 * nb03 + i02 * nb02 + i01 * nb01);
        const float * restrict src1_ptr = (const float *) (data_src1 + 0 + 0 + i11 * nb11);

        if (ir + 1 < src0_end_row) {
            hex_l2fetch(src0_ptr + ne00, src0_row_size, src0_row_size, 1);
            if (src1_row_size == src0_row_size) {
                hex_l2fetch(src1_ptr + ne10, src1_row_size, src1_row_size, 1);
            }
        }

        const uint32_t nr0 = ne00 / ne10;
        if (nr0 > 1) {
            for (uint32_t r = 0; r < nr0; r++) {
                memcpy(spad_data + r * nb10, (const uint8_t *) src1_ptr, nb10);
            }
            func_HVX((uint8_t *) dst_ptr, (const uint8_t *) src0_ptr, (const uint8_t *) spad_data, ne00);
        } else {
            func_HVX((uint8_t *) dst_ptr, (const uint8_t *) src0_ptr, (const uint8_t *) src1_ptr, ne00);
        }
    }

    t2 = HAP_perf_get_qtimer_count();

    FARF(HIGH, "add-id-f32 %d/%d: %ux%ux%ux%u (%u:%u) x %ux%ux%ux%u (%ux%ux%ux%u) -> %ux%ux%ux%u usec %u\n", ith, nth,
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src0_start_row, src0_end_row, src1->ne[0], src1->ne[1],
         src1->ne[2], src1->ne[3], src2->ne[0], src2->ne[1], src2->ne[2], src2->ne[3], dst->ne[0], dst->ne[1],
         dst->ne[2], dst->ne[3], (unsigned) HAP_perf_qtimer_count_to_us(t2 - t1));
}

static void binary_job_dispatcher_f32(unsigned int n, unsigned int i, void * data) {
    struct htp_ops_context * octx = (struct htp_ops_context *) data;
    if (octx->op == HTP_OP_ADD_ID) {
         binary_add_id_job_f32_per_thread(octx, octx->src0_spad.data, n, i, hvx_add_f32);
    }
}

static int execute_op_binary_f32(struct htp_ops_context * octx) {
    int err = HTP_STATUS_OK;

    const struct htp_tensor * src0 = &octx->src0;
    const struct htp_tensor * src1 = &octx->src1;
    struct htp_tensor *       dst  = &octx->dst;

    const char *      op_type = NULL;
    bool is_add_id = false;

    switch (octx->op) {
        case HTP_OP_MUL:
            op_type        = "mul-f32";
            break;

        case HTP_OP_ADD:
            op_type        = "add-f32";
            break;

        case HTP_OP_SUB:
            op_type        = "sub-f32";
            break;

        case HTP_OP_DIV:
            op_type        = "div-f32";
            break;

        case HTP_OP_ADD_ID:
            op_type        = "add-id-f32";
            is_add_id      = true;
            break;

        default:
            FARF(ERROR, "Unsupported binary-Op %u\n", octx->op);
            return HTP_STATUS_NO_SUPPORT;
    }

    const int      n_threads  = octx->n_threads;
    const uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];

    const size_t src0_row_size = src0->nb[1];
    const size_t src1_row_size = src1->nb[1];
    const size_t dst_row_size  = dst->nb[1];

    const size_t src0_row_size_aligned = hex_round_up(src0_row_size, VLEN);
    const size_t src1_row_size_aligned = hex_round_up(src1_row_size, VLEN);
    const size_t dst_row_size_aligned  = hex_round_up(dst_row_size, VLEN);

    if (is_add_id) {
        // Old logic for ADD_ID
        // VTCM scratchpads for all tensors
        octx->dst_spad.size  = hex_round_up(dst_row_size, 128) * n_threads;
        octx->src0_spad.size = hex_round_up(src0_row_size, 128) * n_threads;
        octx->src1_spad.size = hex_round_up(src1_row_size, 128) * n_threads;

        size_t spad_size = octx->src0_spad.size + octx->src1_spad.size + octx->dst_spad.size;

        if (octx->ctx->vtcm_size < spad_size) {
            return HTP_STATUS_VTCM_TOO_SMALL;
        }

        octx->src0_spad.data = octx->ctx->vtcm_base;
        octx->src1_spad.data = octx->src0_spad.data + octx->src0_spad.size;
        octx->dst_spad.data  = octx->src1_spad.data + octx->src1_spad.size;

        if (!(octx->flags & HTP_OPFLAGS_SKIP_COMPUTE)) {
            uint32_t n_jobs = MIN(n_threads, src0_nrows);
            octx->src0_nrows_per_thread = (src0_nrows + n_jobs - 1) / n_jobs;

            octx->src0_div21 = init_fastdiv_values(src0->ne[2] * src0->ne[1]);
            octx->src0_div3  = init_fastdiv_values(src0->ne[3]);
            octx->src0_div2  = init_fastdiv_values(src0->ne[2]);
            octx->src0_div1  = init_fastdiv_values(src0->ne[1]);

            worker_pool_run_func(octx->ctx->worker_pool, binary_job_dispatcher_f32, octx, n_jobs);
        }
    } else {
        // New DMA logic
        size_t vtcm_per_thread = octx->ctx->vtcm_size / n_threads;

        // Align to 128
        vtcm_per_thread = (vtcm_per_thread / 128) * 128;

        // Partition vtcm_per_thread into 3 parts (src0, src1, dst)
        size_t part_size = (vtcm_per_thread / 3 / 128) * 128;

        octx->src0_spad.size_per_thread = part_size;
        octx->src1_spad.size_per_thread = part_size;
        octx->dst_spad.size_per_thread  = part_size;

        octx->src0_spad.data = octx->ctx->vtcm_base;
        octx->src1_spad.data = octx->src0_spad.data + n_threads * part_size;
        octx->dst_spad.data  = octx->src1_spad.data + n_threads * part_size;

        octx->src0_spad.size = n_threads * part_size;

        if (part_size < 2 * src0_row_size_aligned) {
             FARF(ERROR, "binary-f32: VTCM too small for DMA path");
             return HTP_STATUS_VTCM_TOO_SMALL;
        }

        if (!(octx->flags & HTP_OPFLAGS_SKIP_COMPUTE)) {
            uint32_t n_jobs = MIN(n_threads, src0_nrows);

            struct htp_binary_context bctx;
            bctx.octx = octx;
            bctx.op = octx->op;
            bctx.nrows_per_thread = (src0_nrows + n_jobs - 1) / n_jobs;

            bctx.src0_div21 = init_fastdiv_values(src0->ne[2] * src0->ne[1]);
            bctx.src0_div1  = init_fastdiv_values(src0->ne[1]);

            bctx.src1_div3  = init_fastdiv_values(src1->ne[3]);
            bctx.src1_div2  = init_fastdiv_values(src1->ne[2]);
            bctx.src1_div1  = init_fastdiv_values(src1->ne[1]);

            worker_pool_run_func(octx->ctx->worker_pool, binary_job_dispatcher_f32_dma, &bctx, n_jobs);
        }
    }

    return err;
}

int op_binary(struct htp_ops_context * octx) {
    int err = HTP_STATUS_OK;

    switch (octx->src0.type) {
        case HTP_TYPE_F32:
            err = execute_op_binary_f32(octx);
            break;

        default:
            err = HTP_STATUS_NO_SUPPORT;
            break;
    }

    return err;
}
