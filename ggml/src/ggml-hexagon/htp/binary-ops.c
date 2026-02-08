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

// Context for binary operations
struct htp_binary_context {
    struct htp_ops_context * octx;
    // Fastdivs for decomposing flattened row index into dimensions
    struct fastdiv_values dim1_div;    // Divisor for ne01
    struct fastdiv_values dim2_div;    // Divisor for ne02
    struct fastdiv_values dim12_div;   // Divisor for ne01 * ne02

    uint32_t nrows_per_thread;
    bool split_at_ne01;
    bool split_at_ne02;
};

#define htp_binary_preamble            \
    const struct htp_tensor * src0 = &octx->src0; \
    const struct htp_tensor * src1 = &octx->src1; \
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
    const uint32_t nb01 = src0->nb[1]; \
    const uint32_t nb02 = src0->nb[2]; \
    const uint32_t nb03 = src0->nb[3]; \
                                       \
    const uint32_t nb11 = src1->nb[1]; \
    const uint32_t nb12 = src1->nb[2]; \
    const uint32_t nb13 = src1->nb[3]; \
                                       \
    const uint32_t nb1 = dst->nb[1];   \
    const uint32_t nb2 = dst->nb[2];   \
    const uint32_t nb3 = dst->nb[3];

static uint32_t calc_block_size(struct htp_binary_context * bctx, uint32_t ir, uint32_t end_row,
                                uint32_t ne01, uint32_t ne02,
                                int BLOCK_MAX) {
    uint32_t i03, i02, i01, rem;
    i03 = fastdiv(ir, &bctx->dim12_div);
    rem = ir - i03 * (ne02 * ne01);
    i02 = fastdiv(rem, &bctx->dim1_div);
    i01 = rem - i02 * ne01;

    uint32_t rows_left = end_row - ir;
    uint32_t block_limit = rows_left;

    if (bctx->split_at_ne01) {
        block_limit = MIN(block_limit, ne01 - i01);
    }
    if (bctx->split_at_ne02) {
         uint32_t rows_in_plane = (ne02 * ne01) - rem;
         block_limit = MIN(block_limit, rows_in_plane);
    }

    return MIN(BLOCK_MAX, block_limit);
}

// Scalar src1 implementation (ne10 == 1)
static void binary_job_f32_scalar_per_thread(struct htp_binary_context * bctx,
                                             uint32_t                 nth,
                                             uint32_t                 ith) {
    struct htp_ops_context * octx = bctx->octx;
    htp_binary_preamble;

    const size_t src0_row_size = nb01;
    const size_t dst_row_size  = nb1;

    const uint32_t total_rows = ne01 * ne02 * ne03;
    const uint32_t start_row = bctx->nrows_per_thread * ith;
    const uint32_t end_row   = MIN(start_row + bctx->nrows_per_thread, total_rows);

    if (start_row >= end_row) {
        return;
    }

    uint64_t t1, t2;
    t1 = HAP_perf_get_qtimer_count();

    const size_t src0_row_size_aligned = hex_round_up(src0_row_size, VLEN);
    const size_t dst_row_size_aligned  = hex_round_up(dst_row_size, VLEN);

    uint8_t * src0_spad_base = octx->src0_spad.data + (ith * octx->src0_spad.size_per_thread);
    uint8_t * dst_spad_base  = octx->dst_spad.data  + (ith * octx->dst_spad.size_per_thread);

    size_t src0_spad_half = octx->src0_spad.size_per_thread / 2;
    size_t dst_spad_half  = octx->dst_spad.size_per_thread  / 2;

    const int BLOCK_MAX = src0_spad_half / src0_row_size_aligned;
    if (BLOCK_MAX == 0) {
        FARF(ERROR, "binary-scalar-f32: VTCM too small\n");
        return;
    }

    dma_queue * q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = start_row;
    int spad_idx = 0;

    // Preamble
    for (int k = 0; k < 2 && ir_prefetch < end_row; k++) {
        uint32_t current_block_size = calc_block_size(bctx, ir_prefetch, end_row,
                                                      ne01, ne02, BLOCK_MAX);

        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir_prefetch, &bctx->dim12_div);
        rem = ir_prefetch - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->dim1_div);
        i01 = rem - i02 * ne01;

        uint8_t * src0_curr = (uint8_t *)src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01;
        uint8_t * dst_curr  = (uint8_t *)dst->data  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_base + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_base  + spad_idx * dst_spad_half;

        // Dummy push for sync (Output slot)
        dma_queue_push_vtcm_to_ddr(q, dma_make_ptr(dst_curr, d_spad), nb1, dst_row_size_aligned, 0);

        // Push Input (src0 only)
        dma_queue_push(q, dma_make_ptr(s0_spad, src0_curr), src0_row_size_aligned, nb01, ne00 * sizeof(float), current_block_size);

        ir_prefetch += current_block_size;
        spad_idx ^= 1;
    }

    // Main loop
    spad_idx = 0;
    for (uint32_t ir = start_row; ir < end_row; ) {
        uint32_t current_block_size = calc_block_size(bctx, ir, end_row,
                                                      ne01, ne02, BLOCK_MAX);

        // Pop Output Status
        dma_queue_pop(q);
        // Pop Input (src0)
        void * src0_ptr = dma_queue_pop(q).dst;

        uint8_t * d_spad = dst_spad_base + spad_idx * dst_spad_half;

        // Calculate src1 base address for this block
        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir, &bctx->dim12_div);
        rem = ir - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->dim1_div);
        i01 = rem - i02 * ne01;

        uint32_t i13 = (ne13 == 1) ? 0 : i03;
        uint32_t i12 = (ne12 == 1) ? 0 : i02;
        uint32_t i11 = (ne11 == 1) ? 0 : i01;

        uint8_t * src1_ptr = (uint8_t *)src1->data + i13 * nb13 + i12 * nb12 + i11 * nb11;
        uint32_t s1_stride = (ne11 == 1) ? 0 : nb11;

        // Compute
        for (uint32_t r = 0; r < current_block_size; r++) {
            uint8_t * r_src0 = (uint8_t *)src0_ptr + r * src0_row_size_aligned;
            uint8_t * r_dst  = d_spad + r * dst_row_size_aligned;

            float val = *(float *)src1_ptr;
            src1_ptr += s1_stride;

            switch (octx->op) {
                case HTP_OP_ADD: hvx_add_scalar_f32_aa(r_dst, r_src0, val, ne00); break;
                case HTP_OP_SUB: hvx_sub_scalar_f32_aa(r_dst, r_src0, val, ne00); break;
                case HTP_OP_MUL: hvx_mul_scalar_f32_aa(r_dst, r_src0, val, ne00); break;
                case HTP_OP_DIV: hvx_mul_scalar_f32_aa(r_dst, r_src0, 1.0f / val, ne00); break;
                default: break;
            }
        }

        // Push Output
        uint8_t * dst_curr = (uint8_t *)dst->data + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_queue_push(q, dma_make_ptr(dst_curr, d_spad), nb1, dst_row_size_aligned, ne00 * sizeof(float), current_block_size);

        // Prefetch
        if (ir_prefetch < end_row) {
             uint32_t next_block_size = calc_block_size(bctx, ir_prefetch, end_row,
                                                        ne01, ne02, BLOCK_MAX);

             uint32_t p03, p02, p01, prem;
             p03 = fastdiv(ir_prefetch, &bctx->dim12_div);
             prem = ir_prefetch - p03 * (ne02 * ne01);
             p02 = fastdiv(prem, &bctx->dim1_div);
             p01 = prem - p02 * ne01;

             uint8_t * s0_next = (uint8_t *)src0->data + p03 * nb03 + p02 * nb02 + p01 * nb01;
             uint8_t * s0_spad = src0_spad_base + spad_idx * src0_spad_half;

             dma_queue_push(q, dma_make_ptr(s0_spad, s0_next), src0_row_size_aligned, nb01, ne00 * sizeof(float), next_block_size);
             ir_prefetch += next_block_size;
        }

        ir += current_block_size;
        spad_idx ^= 1;
    }

    dma_queue_flush(q);

    t2 = HAP_perf_get_qtimer_count();
    FARF(HIGH, "binary-scalar-f32 %d/%d: %ux%ux%ux%u (%u:%u) usec %u\n", ith, nth,
         ne00, ne01, ne02, ne03, start_row, end_row,
         (unsigned) HAP_perf_qtimer_count_to_us(t2 - t1));
}

// Vector src1 implementation (ne10 == ne00)
static void binary_job_f32_vector_per_thread(struct htp_binary_context * bctx,
                                             uint32_t                 nth,
                                             uint32_t                 ith) {
    struct htp_ops_context * octx = bctx->octx;
    htp_binary_preamble;

    const size_t src0_row_size = nb01;
    const size_t src1_row_size = nb11;
    const size_t dst_row_size  = nb1;

    const uint32_t total_rows = ne01 * ne02 * ne03;
    const uint32_t start_row = bctx->nrows_per_thread * ith;
    const uint32_t end_row   = MIN(start_row + bctx->nrows_per_thread, total_rows);

    if (start_row >= end_row) {
        return;
    }

    uint64_t t1, t2;
    t1 = HAP_perf_get_qtimer_count();

    const size_t src0_row_size_aligned = hex_round_up(src0_row_size, VLEN);
    const size_t src1_row_size_aligned = hex_round_up(src1_row_size, VLEN);
    const size_t dst_row_size_aligned  = hex_round_up(dst_row_size, VLEN);

    uint8_t * src0_spad_base = octx->src0_spad.data + (ith * octx->src0_spad.size_per_thread);
    uint8_t * src1_spad_base = octx->src1_spad.data + (ith * octx->src1_spad.size_per_thread);
    uint8_t * dst_spad_base  = octx->dst_spad.data  + (ith * octx->dst_spad.size_per_thread);

    size_t src0_spad_half = octx->src0_spad.size_per_thread / 2;
    size_t src1_spad_half = octx->src1_spad.size_per_thread / 2;
    size_t dst_spad_half  = octx->dst_spad.size_per_thread  / 2;

    const int BLOCK_MAX = src0_spad_half / src0_row_size_aligned;
    if (BLOCK_MAX == 0) {
        FARF(ERROR, "binary-vector-f32: VTCM too small\n");
        return;
    }

    dma_queue * q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = start_row;
    int spad_idx = 0;

    // Preamble
    for (int k = 0; k < 2 && ir_prefetch < end_row; k++) {
        uint32_t current_block_size = calc_block_size(bctx, ir_prefetch, end_row,
                                                      ne01, ne02, BLOCK_MAX);

        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir_prefetch, &bctx->dim12_div);
        rem = ir_prefetch - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->dim1_div);
        i01 = rem - i02 * ne01;

        uint32_t i13 = (ne13 == 1) ? 0 : i03;
        uint32_t i12 = (ne12 == 1) ? 0 : i02;
        uint32_t i11 = (ne11 == 1) ? 0 : i01;

        uint8_t * src0_curr = (uint8_t *)src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01;
        uint8_t * src1_base = (uint8_t *)src1->data + i13 * nb13 + i12 * nb12 + i11 * nb11;
        uint8_t * dst_curr  = (uint8_t *)dst->data  + i03 * nb3  + i02 * nb2  + i01 * nb1;
        uint32_t src1_dma_stride = (ne11 == 1) ? 0 : nb11;

        uint8_t * s0_spad = src0_spad_base + spad_idx * src0_spad_half;
        uint8_t * s1_spad = src1_spad_base + spad_idx * src1_spad_half;
        uint8_t * d_spad  = dst_spad_base  + spad_idx * dst_spad_half;

        // Dummy push for sync
        dma_queue_push_vtcm_to_ddr(q, dma_make_ptr(dst_curr, d_spad), nb1, dst_row_size_aligned, 0);

        // Push Inputs
        dma_queue_push(q, dma_make_ptr(s0_spad, src0_curr), src0_row_size_aligned, nb01, ne00 * sizeof(float), current_block_size);
        dma_queue_push(q, dma_make_ptr(s1_spad, src1_base), src1_row_size_aligned, src1_dma_stride, ne00 * sizeof(float), current_block_size);

        ir_prefetch += current_block_size;
        spad_idx ^= 1;
    }

    // Main loop
    spad_idx = 0;
    for (uint32_t ir = start_row; ir < end_row; ) {
        uint32_t current_block_size = calc_block_size(bctx, ir, end_row,
                                                      ne01, ne02, BLOCK_MAX);

        // Pop Output Status
        dma_queue_pop(q);
        // Pop Inputs
        void * src0_ptr = dma_queue_pop(q).dst;
        void * src1_ptr = dma_queue_pop(q).dst;

        uint8_t * d_spad = dst_spad_base + spad_idx * dst_spad_half;

        // Compute
        for (uint32_t r = 0; r < current_block_size; r++) {
            uint8_t * r_src0 = (uint8_t *)src0_ptr + r * src0_row_size_aligned;
            uint8_t * r_src1 = (uint8_t *)src1_ptr + r * src1_row_size_aligned;
            uint8_t * r_dst  = d_spad + r * dst_row_size_aligned;

            switch (octx->op) {
                case HTP_OP_ADD: hvx_add_f32_aa(r_dst, r_src0, r_src1, ne00); break;
                case HTP_OP_SUB: hvx_sub_f32_aa(r_dst, r_src0, r_src1, ne00); break;
                case HTP_OP_MUL: hvx_mul_f32_aa(r_dst, r_src0, r_src1, ne00); break;
                case HTP_OP_DIV: hvx_div_f32_aa(r_dst, r_src0, r_src1, ne00); break;
                default: break;
            }
        }

        // Push Output
        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir, &bctx->dim12_div);
        rem = ir - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->dim1_div);
        i01 = rem - i02 * ne01;
        uint8_t * dst_curr = (uint8_t *)dst->data + i03 * nb3 + i02 * nb2 + i01 * nb1;

        dma_queue_push(q, dma_make_ptr(dst_curr, d_spad), nb1, dst_row_size_aligned, ne00 * sizeof(float), current_block_size);

        // Prefetch next block
        if (ir_prefetch < end_row) {
             uint32_t next_block_size = calc_block_size(bctx, ir_prefetch, end_row,
                                                        ne01, ne02, BLOCK_MAX);

             uint32_t p03, p02, p01, prem;
             p03 = fastdiv(ir_prefetch, &bctx->dim12_div);
             prem = ir_prefetch - p03 * (ne02 * ne01);
             p02 = fastdiv(prem, &bctx->dim1_div);
             p01 = prem - p02 * ne01;

             uint32_t p13 = (ne13 == 1) ? 0 : p03;
             uint32_t p12 = (ne12 == 1) ? 0 : p02;
             uint32_t p11 = (ne11 == 1) ? 0 : p01;

             uint8_t * s0_next = (uint8_t *)src0->data + p03 * nb03 + p02 * nb02 + p01 * nb01;
             uint8_t * s1_next = (uint8_t *)src1->data + p13 * nb13 + p12 * nb12 + p11 * nb11;
             uint32_t s1_stride = (ne11 == 1) ? 0 : nb11;

             uint8_t * s0_spad = src0_spad_base + spad_idx * src0_spad_half;
             uint8_t * s1_spad = src1_spad_base + spad_idx * src1_spad_half;

             dma_queue_push(q, dma_make_ptr(s0_spad, s0_next), src0_row_size_aligned, nb01, ne00 * sizeof(float), next_block_size);
             dma_queue_push(q, dma_make_ptr(s1_spad, s1_next), src1_row_size_aligned, s1_stride, ne00 * sizeof(float), next_block_size);

             ir_prefetch += next_block_size;
        }

        ir += current_block_size;
        spad_idx ^= 1;
    }

    dma_queue_flush(q);

    t2 = HAP_perf_get_qtimer_count();
    FARF(HIGH, "binary-vector-f32 %d/%d: %ux%ux%ux%u (%u:%u) usec %u\n", ith, nth,
         ne00, ne01, ne02, ne03, start_row, end_row,
         (unsigned) HAP_perf_qtimer_count_to_us(t2 - t1));
}

static void binary_job_f32(unsigned int n, unsigned int i, void * data) {
    struct htp_binary_context * bctx = (struct htp_binary_context *) data;
    if (bctx->octx->src1.ne[0] == 1) {
        binary_job_f32_scalar_per_thread(bctx, n, i);
    } else {
        binary_job_f32_vector_per_thread(bctx, n, i);
    }
}

static int execute_op_binary_f32(struct htp_ops_context * octx) {
    int err = HTP_STATUS_OK;
    const struct htp_tensor * src0 = &octx->src0;
    const struct htp_tensor * src1 = &octx->src1;
    struct htp_tensor *       dst  = &octx->dst;

    const uint32_t n_threads  = octx->n_threads;
    const uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];

    const size_t src0_row_size = src0->nb[1];
    const size_t src1_row_size = src1->nb[1];
    const size_t dst_row_size  = dst->nb[1];

    // Align to VLEN
    const size_t src0_row_size_aligned = hex_round_up(src0_row_size, VLEN);
    const size_t src1_row_size_aligned = hex_round_up(src1_row_size, VLEN);
    const size_t dst_row_size_aligned  = hex_round_up(dst_row_size, VLEN);

    bool is_scalar = (src1->ne[0] == 1);

    // Double buffering requires 2x rows per thread
    // If scalar, we don't need src1_spad
    size_t spad_row_total;
    if (is_scalar) {
        spad_row_total = 2 * (src0_row_size_aligned + dst_row_size_aligned);
    } else {
        spad_row_total = 2 * (src0_row_size_aligned + src1_row_size_aligned + dst_row_size_aligned);
    }

    size_t rows_per_buffer = octx->ctx->vtcm_size / (n_threads * spad_row_total);

    if (rows_per_buffer < 1) {
         FARF(ERROR, "binary-f32: VTCM too small\n");
         return HTP_STATUS_VTCM_TOO_SMALL;
    }

    // We allocate total per thread
    octx->src0_spad.size_per_thread = rows_per_buffer * 2 * src0_row_size_aligned;
    octx->src1_spad.size_per_thread = is_scalar ? 0 : (rows_per_buffer * 2 * src1_row_size_aligned);
    octx->dst_spad.size_per_thread  = rows_per_buffer * 2 * dst_row_size_aligned;

    octx->src0_spad.size = n_threads * octx->src0_spad.size_per_thread;
    octx->src1_spad.size = n_threads * octx->src1_spad.size_per_thread;
    octx->dst_spad.size  = n_threads * octx->dst_spad.size_per_thread;

    if (octx->ctx->vtcm_size < (octx->src0_spad.size + octx->src1_spad.size + octx->dst_spad.size)) {
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    octx->src0_spad.data = octx->ctx->vtcm_base;
    // If scalar, src1_spad.data might not be valid, but we set it anyway correctly offset
    octx->src1_spad.data = octx->src0_spad.data + octx->src0_spad.size;
    octx->dst_spad.data  = octx->src1_spad.data + octx->src1_spad.size;

    if (!(octx->flags & HTP_OPFLAGS_SKIP_COMPUTE)) {
        uint32_t n_jobs = MIN(n_threads, src0_nrows);

        struct htp_binary_context bctx;
        bctx.octx = octx;
        bctx.nrows_per_thread = (src0_nrows + n_jobs - 1) / n_jobs;

        bctx.dim1_div = init_fastdiv_values(src0->ne[1]);
        bctx.dim2_div = init_fastdiv_values(src0->ne[2]);
        bctx.dim12_div = init_fastdiv_values(src0->ne[1] * src0->ne[2]);

        // Contiguity checks for optimization (merging planes)
        bool src0_contig_dim1 = (src0->nb[2] == src0->ne[1] * src0->nb[1]);
        bool dst_contig_dim1  = (dst->nb[2] == src0->ne[1] * dst->nb[1]);

        bool src0_contig_dim2 = (src0->nb[3] == src0->ne[2] * src0->nb[2]);
        bool dst_contig_dim2  = (dst->nb[3] == src0->ne[2] * dst->nb[2]);

        // Splitting logic: split at boundary if not contiguous or if src1 broadcast logic requires it
        bctx.split_at_ne01 = (src0->ne[2] > 1) &&
                             ((src1->ne[1] > 1) || (src1->ne[2] > 1) || !src0_contig_dim1 || !dst_contig_dim1);

        bctx.split_at_ne02 = (src0->ne[3] > 1) &&
                             ((src1->ne[2] > 1) || (src1->ne[3] > 1) || !src0_contig_dim2 || !dst_contig_dim2);

        worker_pool_run_func(octx->ctx->worker_pool, binary_job_f32, &bctx, n_jobs);
    }
    return err;
}

int op_binary(struct htp_ops_context * octx) {
    if (octx->src0.type == HTP_TYPE_F32) {
        return execute_op_binary_f32(octx);
    }
    return HTP_STATUS_NO_SUPPORT;
}
