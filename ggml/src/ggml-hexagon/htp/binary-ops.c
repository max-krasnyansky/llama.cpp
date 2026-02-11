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

struct htp_binary_context {
    struct htp_ops_context * octx;
    struct fastdiv_values dim1_div;
    struct fastdiv_values dim2_div;
    struct fastdiv_values dim12_div;

    // Divisors for src1 complex broadcasting
    struct fastdiv_values src1_dim1_div; // ne11
    struct fastdiv_values src1_dim2_div; // ne12
    struct fastdiv_values src1_dim3_div; // ne13

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

// Macro for scalar op switch
#define COMPUTE_SCALAR_OP(DST, SRC, VAL, N) \
    switch (octx->op) { \
        case HTP_OP_ADD: hvx_add_scalar_f32_aa(DST, SRC, VAL, N); break; \
        case HTP_OP_SUB: hvx_sub_scalar_f32_aa(DST, SRC, VAL, N); break; \
        case HTP_OP_MUL: hvx_mul_scalar_f32_aa(DST, SRC, VAL, N); break; \
        case HTP_OP_DIV: hvx_mul_scalar_f32_aa(DST, SRC, 1.0f / (VAL), N); break; \
        default: break; \
    }

// Macro for vector op switch (All Aligned)
#define COMPUTE_VECTOR_OP_AAA(DST, SRC0, SRC1, N) \
    switch (octx->op) { \
        case HTP_OP_ADD: hvx_add_f32_aaa(DST, SRC0, SRC1, N); break; \
        case HTP_OP_SUB: hvx_sub_f32_aaa(DST, SRC0, SRC1, N); break; \
        case HTP_OP_MUL: hvx_mul_f32_aaa(DST, SRC0, SRC1, N); break; \
        case HTP_OP_DIV: hvx_div_f32_aaa(DST, SRC0, SRC1, N); break; \
        default: break; \
    }

// Macro for vector op switch (Dst Aligned, Src0 Aligned, Src1 Unaligned)
#define COMPUTE_VECTOR_OP_AAU(DST, SRC0, SRC1, N) \
    switch (octx->op) { \
        case HTP_OP_ADD: hvx_add_f32_aau(DST, SRC0, SRC1, N); break; \
        case HTP_OP_SUB: hvx_sub_f32_aau(DST, SRC0, SRC1, N); break; \
        case HTP_OP_MUL: hvx_mul_f32_aau(DST, SRC0, SRC1, N); break; \
        case HTP_OP_DIV: hvx_div_f32_aau(DST, SRC0, SRC1, N); break; \
        default: break; \
    }

// 1. Scalar src1 (ne10 == 1)
static void binary_job_scalar(struct htp_binary_context * bctx, uint32_t nth, uint32_t ith) {
    struct htp_ops_context * octx = bctx->octx;
    htp_binary_preamble;

    const uint32_t total_rows = ne01 * ne02 * ne03;
    const uint32_t start_row = bctx->nrows_per_thread * ith;
    const uint32_t end_row   = MIN(start_row + bctx->nrows_per_thread, total_rows);
    if (start_row >= end_row) return;

    const size_t src0_row_size_aligned = hex_round_up(nb01, VLEN);
    const size_t dst_row_size_aligned  = hex_round_up(nb1, VLEN);

    uint8_t * src0_spad_base = octx->src0_spad.data + (ith * octx->src0_spad.size_per_thread);
    uint8_t * dst_spad_base  = octx->dst_spad.data  + (ith * octx->dst_spad.size_per_thread);
    size_t src0_spad_half = octx->src0_spad.size_per_thread / 2;
    size_t dst_spad_half  = octx->dst_spad.size_per_thread  / 2;
    const int BLOCK_MAX = src0_spad_half / src0_row_size_aligned;

    dma_queue * q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = start_row;
    int spad_idx = 0;

    // Preamble
    for (int k = 0; k < 2 && ir_prefetch < end_row; k++) {
        uint32_t current_block_size = calc_block_size(bctx, ir_prefetch, end_row, ne01, ne02, BLOCK_MAX);
        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir_prefetch, &bctx->dim12_div);
        rem = ir_prefetch - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->dim1_div);
        i01 = rem - i02 * ne01;

        uint8_t * src0_curr = (uint8_t *)src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01;
        uint8_t * dst_curr  = (uint8_t *)dst->data  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_base + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_base  + spad_idx * dst_spad_half;

        dma_queue_push_vtcm_to_ddr(q, dma_make_ptr(dst_curr, d_spad), nb1, dst_row_size_aligned, 0);
        dma_queue_push(q, dma_make_ptr(s0_spad, src0_curr), src0_row_size_aligned, nb01, ne00 * sizeof(float), current_block_size);
        ir_prefetch += current_block_size;
        spad_idx ^= 1;
    }

    // Main loop
    for (uint32_t ir = start_row; ir < end_row; ) {
        uint32_t current_block_size = calc_block_size(bctx, ir, end_row, ne01, ne02, BLOCK_MAX);

        uint8_t * d_spad = (uint8_t *) dma_queue_pop(q).src;
        uint8_t * s0_spad = (uint8_t *) dma_queue_pop(q).dst;

        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir, &bctx->dim12_div);
        rem = ir - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->dim1_div);
        i01 = rem - i02 * ne01;

        // src1 indices (broadcast/repeat)
        uint32_t i13 = fastmodulo(i03, ne13, &bctx->src1_dim3_div);
        uint32_t i12 = fastmodulo(i02, ne12, &bctx->src1_dim2_div);
        uint32_t i11 = fastmodulo(i01, ne11, &bctx->src1_dim1_div);

        uint8_t * src1_ptr = (uint8_t *)src1->data + i13 * nb13 + i12 * nb12 + i11 * nb11;
        uint32_t s1_stride = (ne11 == 1) ? 0 : nb11;

        for (uint32_t r = 0; r < current_block_size; r++) {
            uint8_t * r_src0 = s0_spad + r * src0_row_size_aligned;
            uint8_t * r_dst  = d_spad + r * dst_row_size_aligned;
            float val = *(float *)src1_ptr;
            src1_ptr += s1_stride;
            COMPUTE_SCALAR_OP(r_dst, r_src0, val, ne00);
        }

        uint8_t * dst_curr = (uint8_t *)dst->data + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_queue_push(q, dma_make_ptr(dst_curr, d_spad), nb1, dst_row_size_aligned, ne00 * sizeof(float), current_block_size);

        if (ir_prefetch < end_row) {
             uint32_t next_block_size = calc_block_size(bctx, ir_prefetch, end_row, ne01, ne02, BLOCK_MAX);
             uint32_t p03, p02, p01, prem;
             p03 = fastdiv(ir_prefetch, &bctx->dim12_div);
             prem = ir_prefetch - p03 * (ne02 * ne01);
             p02 = fastdiv(prem, &bctx->dim1_div);
             p01 = prem - p02 * ne01;
             uint8_t * s0_next = (uint8_t *)src0->data + p03 * nb03 + p02 * nb02 + p01 * nb01;
             dma_queue_push(q, dma_make_ptr(s0_spad, s0_next), src0_row_size_aligned, nb01, ne00 * sizeof(float), next_block_size);
             ir_prefetch += next_block_size;
        }
        ir += current_block_size;
    }
    dma_queue_flush(q);
}

// 2. Vector Same Shape (ne1x == ne0x)
static void binary_job_vector_same_shape(struct htp_binary_context * bctx, uint32_t nth, uint32_t ith) {
    struct htp_ops_context * octx = bctx->octx;
    htp_binary_preamble;

    const uint32_t total_rows = ne01 * ne02 * ne03;
    const uint32_t start_row = bctx->nrows_per_thread * ith;
    const uint32_t end_row   = MIN(start_row + bctx->nrows_per_thread, total_rows);
    if (start_row >= end_row) return;

    const size_t src0_row_size_aligned = hex_round_up(nb01, VLEN);
    const size_t src1_row_size_aligned = hex_round_up(nb11, VLEN);
    const size_t dst_row_size_aligned  = hex_round_up(nb1, VLEN);

    uint8_t * src0_spad_base = octx->src0_spad.data + (ith * octx->src0_spad.size_per_thread);
    uint8_t * src1_spad_base = octx->src1_spad.data + (ith * octx->src1_spad.size_per_thread);
    uint8_t * dst_spad_base  = octx->dst_spad.data  + (ith * octx->dst_spad.size_per_thread);

    size_t src0_spad_half = octx->src0_spad.size_per_thread / 2;
    size_t src1_spad_half = octx->src1_spad.size_per_thread / 2;
    size_t dst_spad_half  = octx->dst_spad.size_per_thread  / 2;
    const int BLOCK_MAX = src0_spad_half / src0_row_size_aligned;

    dma_queue * q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = start_row;
    int spad_idx = 0;

    for (int k = 0; k < 2 && ir_prefetch < end_row; k++) {
        uint32_t current_block_size = calc_block_size(bctx, ir_prefetch, end_row, ne01, ne02, BLOCK_MAX);
        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir_prefetch, &bctx->dim12_div);
        rem = ir_prefetch - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->dim1_div);
        i01 = rem - i02 * ne01;

        uint8_t * src0_curr = (uint8_t *)src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01;
        uint8_t * src1_base = (uint8_t *)src1->data + i03 * nb13 + i02 * nb12 + i01 * nb11;
        uint8_t * dst_curr  = (uint8_t *)dst->data  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_base + spad_idx * src0_spad_half;
        uint8_t * s1_spad = src1_spad_base + spad_idx * src1_spad_half;
        uint8_t * d_spad  = dst_spad_base  + spad_idx * dst_spad_half;

        dma_queue_push_vtcm_to_ddr(q, dma_make_ptr(dst_curr, d_spad), nb1, dst_row_size_aligned, 0);
        dma_queue_push(q, dma_make_ptr(s0_spad, src0_curr), src0_row_size_aligned, nb01, ne00 * sizeof(float), current_block_size);
        dma_queue_push(q, dma_make_ptr(s1_spad, src1_base), src1_row_size_aligned, nb11, ne00 * sizeof(float), current_block_size);
        ir_prefetch += current_block_size;
        spad_idx ^= 1;
    }

    for (uint32_t ir = start_row; ir < end_row; ) {
        uint32_t current_block_size = calc_block_size(bctx, ir, end_row, ne01, ne02, BLOCK_MAX);
        uint8_t * d_spad = (uint8_t *) dma_queue_pop(q).src;
        uint8_t * s0_spad = (uint8_t *) dma_queue_pop(q).dst;
        uint8_t * s1_spad = (uint8_t *) dma_queue_pop(q).dst;

        for (uint32_t r = 0; r < current_block_size; r++) {
            uint8_t * r_src0 = s0_spad + r * src0_row_size_aligned;
            uint8_t * r_src1 = s1_spad + r * src1_row_size_aligned;
            uint8_t * r_dst  = d_spad + r * dst_row_size_aligned;
            COMPUTE_VECTOR_OP_AAA(r_dst, r_src0, r_src1, ne00);
        }

        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir, &bctx->dim12_div);
        rem = ir - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->dim1_div);
        i01 = rem - i02 * ne01;
        uint8_t * dst_curr = (uint8_t *)dst->data + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_queue_push(q, dma_make_ptr(dst_curr, d_spad), nb1, dst_row_size_aligned, ne00 * sizeof(float), current_block_size);

        if (ir_prefetch < end_row) {
             uint32_t next_block_size = calc_block_size(bctx, ir_prefetch, end_row, ne01, ne02, BLOCK_MAX);
             uint32_t p03, p02, p01, prem;
             p03 = fastdiv(ir_prefetch, &bctx->dim12_div);
             prem = ir_prefetch - p03 * (ne02 * ne01);
             p02 = fastdiv(prem, &bctx->dim1_div);
             p01 = prem - p02 * ne01;
             uint8_t * s0_next = (uint8_t *)src0->data + p03 * nb03 + p02 * nb02 + p01 * nb01;
             uint8_t * s1_next = (uint8_t *)src1->data + p03 * nb13 + p02 * nb12 + p01 * nb11;
             dma_queue_push(q, dma_make_ptr(s0_spad, s0_next), src0_row_size_aligned, nb01, ne00 * sizeof(float), next_block_size);
             dma_queue_push(q, dma_make_ptr(s1_spad, s1_next), src1_row_size_aligned, nb11, ne00 * sizeof(float), next_block_size);
             ir_prefetch += next_block_size;
        }
        ir += current_block_size;
    }
    dma_queue_flush(q);
}

// 3. Row Broadcast (ne11 == 1, ne12 == 1, single row src1)
static void binary_job_vector_row_broadcast(struct htp_binary_context * bctx, uint32_t nth, uint32_t ith) {
    struct htp_ops_context * octx = bctx->octx;
    htp_binary_preamble;

    const uint32_t total_rows = ne01 * ne02 * ne03;
    const uint32_t start_row = bctx->nrows_per_thread * ith;
    const uint32_t end_row   = MIN(start_row + bctx->nrows_per_thread, total_rows);
    if (start_row >= end_row) return;

    const size_t src0_row_size_aligned = hex_round_up(nb01, VLEN);
    const size_t src1_row_size_aligned = hex_round_up(nb11, VLEN); // ne11=1, so nb11 irrelevant, but size is row size
    const size_t dst_row_size_aligned  = hex_round_up(nb1, VLEN);

    uint8_t * src0_spad_base = octx->src0_spad.data + (ith * octx->src0_spad.size_per_thread);
    // src1 spad allocated for just 1 row
    uint8_t * src1_spad = octx->src1_spad.data + (ith * octx->src1_spad.size_per_thread);
    uint8_t * dst_spad_base  = octx->dst_spad.data  + (ith * octx->dst_spad.size_per_thread);

    size_t src0_spad_half = octx->src0_spad.size_per_thread / 2;
    size_t dst_spad_half  = octx->dst_spad.size_per_thread  / 2;
    const int BLOCK_MAX = src0_spad_half / src0_row_size_aligned;

    dma_queue * q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = start_row;
    int spad_idx = 0;

    // Fetch src1 once
    dma_queue_push(q, dma_make_ptr(src1_spad, src1->data), src1_row_size_aligned, 0, ne00 * sizeof(float), 1);
    // Wait for it (to ensure valid before loop use)
    void * s1_ptr = dma_queue_pop(q).dst;

    for (int k = 0; k < 2 && ir_prefetch < end_row; k++) {
        uint32_t current_block_size = calc_block_size(bctx, ir_prefetch, end_row, ne01, ne02, BLOCK_MAX);
        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir_prefetch, &bctx->dim12_div);
        rem = ir_prefetch - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->dim1_div);
        i01 = rem - i02 * ne01;

        uint8_t * src0_curr = (uint8_t *)src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01;
        uint8_t * dst_curr  = (uint8_t *)dst->data  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_base + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_base  + spad_idx * dst_spad_half;

        dma_queue_push_vtcm_to_ddr(q, dma_make_ptr(dst_curr, d_spad), nb1, dst_row_size_aligned, 0);
        dma_queue_push(q, dma_make_ptr(s0_spad, src0_curr), src0_row_size_aligned, nb01, ne00 * sizeof(float), current_block_size);
        ir_prefetch += current_block_size;
        spad_idx ^= 1;
    }

    for (uint32_t ir = start_row; ir < end_row; ) {
        uint32_t current_block_size = calc_block_size(bctx, ir, end_row, ne01, ne02, BLOCK_MAX);
        uint8_t * d_spad = (uint8_t *) dma_queue_pop(q).src;
        uint8_t * s0_spad = (uint8_t *) dma_queue_pop(q).dst;

        for (uint32_t r = 0; r < current_block_size; r++) {
            uint8_t * r_src0 = s0_spad + r * src0_row_size_aligned;
            uint8_t * r_src1 = (uint8_t *)s1_ptr; // Constant
            uint8_t * r_dst  = d_spad + r * dst_row_size_aligned;
            COMPUTE_VECTOR_OP_AAA(r_dst, r_src0, r_src1, ne00);
        }

        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir, &bctx->dim12_div);
        rem = ir - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->dim1_div);
        i01 = rem - i02 * ne01;
        uint8_t * dst_curr = (uint8_t *)dst->data + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_queue_push(q, dma_make_ptr(dst_curr, d_spad), nb1, dst_row_size_aligned, ne00 * sizeof(float), current_block_size);

        if (ir_prefetch < end_row) {
             uint32_t next_block_size = calc_block_size(bctx, ir_prefetch, end_row, ne01, ne02, BLOCK_MAX);
             uint32_t p03, p02, p01, prem;
             p03 = fastdiv(ir_prefetch, &bctx->dim12_div);
             prem = ir_prefetch - p03 * (ne02 * ne01);
             p02 = fastdiv(prem, &bctx->dim1_div);
             p01 = prem - p02 * ne01;
             uint8_t * s0_next = (uint8_t *)src0->data + p03 * nb03 + p02 * nb02 + p01 * nb01;
             dma_queue_push(q, dma_make_ptr(s0_spad, s0_next), src0_row_size_aligned, nb01, ne00 * sizeof(float), next_block_size);
             ir_prefetch += next_block_size;
        }
        ir += current_block_size;
    }
    dma_queue_flush(q);
}

// 4. Vector Complex (ne10 == ne00, complex broadcast)
static void binary_job_vector_complex(struct htp_binary_context * bctx, uint32_t nth, uint32_t ith) {
    struct htp_ops_context * octx = bctx->octx;
    htp_binary_preamble;

    const uint32_t total_rows = ne01 * ne02 * ne03;
    const uint32_t start_row = bctx->nrows_per_thread * ith;
    const uint32_t end_row   = MIN(start_row + bctx->nrows_per_thread, total_rows);
    if (start_row >= end_row) return;

    const size_t src0_row_size_aligned = hex_round_up(nb01, VLEN);
    const size_t dst_row_size_aligned  = hex_round_up(nb1, VLEN);

    uint8_t * src0_spad_base = octx->src0_spad.data + (ith * octx->src0_spad.size_per_thread);
    uint8_t * dst_spad_base  = octx->dst_spad.data  + (ith * octx->dst_spad.size_per_thread);
    size_t src0_spad_half = octx->src0_spad.size_per_thread / 2;
    size_t dst_spad_half  = octx->dst_spad.size_per_thread  / 2;
    const int BLOCK_MAX = src0_spad_half / src0_row_size_aligned;

    dma_queue * q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = start_row;
    int spad_idx = 0;

    for (int k = 0; k < 2 && ir_prefetch < end_row; k++) {
        uint32_t current_block_size = calc_block_size(bctx, ir_prefetch, end_row, ne01, ne02, BLOCK_MAX);
        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir_prefetch, &bctx->dim12_div);
        rem = ir_prefetch - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->dim1_div);
        i01 = rem - i02 * ne01;

        uint8_t * src0_curr = (uint8_t *)src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01;
        uint8_t * dst_curr  = (uint8_t *)dst->data  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_base + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_base  + spad_idx * dst_spad_half;

        dma_queue_push_vtcm_to_ddr(q, dma_make_ptr(dst_curr, d_spad), nb1, dst_row_size_aligned, 0);
        dma_queue_push(q, dma_make_ptr(s0_spad, src0_curr), src0_row_size_aligned, nb01, ne00 * sizeof(float), current_block_size);
        ir_prefetch += current_block_size;
        spad_idx ^= 1;
    }

    for (uint32_t ir = start_row; ir < end_row; ) {
        uint32_t current_block_size = calc_block_size(bctx, ir, end_row, ne01, ne02, BLOCK_MAX);
        uint8_t * d_spad = (uint8_t *) dma_queue_pop(q).src;
        uint8_t * s0_spad = (uint8_t *) dma_queue_pop(q).dst;

        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir, &bctx->dim12_div);
        rem = ir - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->dim1_div);
        i01 = rem - i02 * ne01;

        for (uint32_t r = 0; r < current_block_size; r++) {
            uint32_t r_i01 = i01 + r; // Assuming calc_block_size prevents wrap
            uint32_t i13 = fastmodulo(i03, ne13, &bctx->src1_dim3_div);
            uint32_t i12 = fastmodulo(i02, ne12, &bctx->src1_dim2_div);
            uint32_t i11 = fastmodulo(r_i01, ne11, &bctx->src1_dim1_div);

            uint8_t * r_src0 = s0_spad + r * src0_row_size_aligned;
            uint8_t * r_src1 = (uint8_t *)src1->data + i13 * nb13 + i12 * nb12 + i11 * nb11;
            uint8_t * r_dst  = d_spad + r * dst_row_size_aligned;

            // Read src1 from DDR (unaligned)
            COMPUTE_VECTOR_OP_AAU(r_dst, r_src0, r_src1, ne00);
        }

        uint8_t * dst_curr = (uint8_t *)dst->data + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_queue_push(q, dma_make_ptr(dst_curr, d_spad), nb1, dst_row_size_aligned, ne00 * sizeof(float), current_block_size);

        if (ir_prefetch < end_row) {
             uint32_t next_block_size = calc_block_size(bctx, ir_prefetch, end_row, ne01, ne02, BLOCK_MAX);
             uint32_t p03, p02, p01, prem;
             p03 = fastdiv(ir_prefetch, &bctx->dim12_div);
             prem = ir_prefetch - p03 * (ne02 * ne01);
             p02 = fastdiv(prem, &bctx->dim1_div);
             p01 = prem - p02 * ne01;
             uint8_t * s0_next = (uint8_t *)src0->data + p03 * nb03 + p02 * nb02 + p01 * nb01;
             dma_queue_push(q, dma_make_ptr(s0_spad, s0_next), src0_row_size_aligned, nb01, ne00 * sizeof(float), next_block_size);
             ir_prefetch += next_block_size;
        }
        ir += current_block_size;
    }
    dma_queue_flush(q);
}

// 5. Element Repeat (ne10 != ne00)
static void binary_job_element_repeat(struct htp_binary_context * bctx, uint32_t nth, uint32_t ith) {
    // Similar to complex, but inner loop repeats src1
    struct htp_ops_context * octx = bctx->octx;
    htp_binary_preamble;

    const uint32_t total_rows = ne01 * ne02 * ne03;
    const uint32_t start_row = bctx->nrows_per_thread * ith;
    const uint32_t end_row   = MIN(start_row + bctx->nrows_per_thread, total_rows);
    if (start_row >= end_row) return;

    const size_t src0_row_size_aligned = hex_round_up(nb01, VLEN);
    const size_t dst_row_size_aligned  = hex_round_up(nb1, VLEN);

    uint8_t * src0_spad_base = octx->src0_spad.data + (ith * octx->src0_spad.size_per_thread);
    uint8_t * dst_spad_base  = octx->dst_spad.data  + (ith * octx->dst_spad.size_per_thread);
    size_t src0_spad_half = octx->src0_spad.size_per_thread / 2;
    size_t dst_spad_half  = octx->dst_spad.size_per_thread  / 2;
    const int BLOCK_MAX = src0_spad_half / src0_row_size_aligned;

    dma_queue * q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = start_row;
    int spad_idx = 0;

    for (int k = 0; k < 2 && ir_prefetch < end_row; k++) {
        uint32_t current_block_size = calc_block_size(bctx, ir_prefetch, end_row, ne01, ne02, BLOCK_MAX);
        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir_prefetch, &bctx->dim12_div);
        rem = ir_prefetch - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->dim1_div);
        i01 = rem - i02 * ne01;

        uint8_t * src0_curr = (uint8_t *)src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01;
        uint8_t * dst_curr  = (uint8_t *)dst->data  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_base + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_base  + spad_idx * dst_spad_half;

        dma_queue_push_vtcm_to_ddr(q, dma_make_ptr(dst_curr, d_spad), nb1, dst_row_size_aligned, 0);
        dma_queue_push(q, dma_make_ptr(s0_spad, src0_curr), src0_row_size_aligned, nb01, ne00 * sizeof(float), current_block_size);
        ir_prefetch += current_block_size;
        spad_idx ^= 1;
    }

    for (uint32_t ir = start_row; ir < end_row; ) {
        uint32_t current_block_size = calc_block_size(bctx, ir, end_row, ne01, ne02, BLOCK_MAX);
        uint8_t * d_spad = (uint8_t *) dma_queue_pop(q).src;
        uint8_t * s0_spad = (uint8_t *) dma_queue_pop(q).dst;

        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir, &bctx->dim12_div);
        rem = ir - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->dim1_div);
        i01 = rem - i02 * ne01;

        for (uint32_t r = 0; r < current_block_size; r++) {
            uint32_t r_i01 = i01 + r;
            uint32_t i13 = fastmodulo(i03, ne13, &bctx->src1_dim3_div);
            uint32_t i12 = fastmodulo(i02, ne12, &bctx->src1_dim2_div);
            uint32_t i11 = fastmodulo(r_i01, ne11, &bctx->src1_dim1_div);

            uint8_t * r_src0 = s0_spad + r * src0_row_size_aligned;
            uint8_t * r_src1_row = (uint8_t *)src1->data + i13 * nb13 + i12 * nb12 + i11 * nb11;
            uint8_t * r_dst  = d_spad + r * dst_row_size_aligned;

            // Repeat src1 row
            for (uint32_t c = 0; c < ne00; c += ne10) {
                // Determine length: ne10 or remaining
                uint32_t len = MIN(ne10, ne00 - c);
                // r_dst and r_src0 are aligned to 128 at start.
                // c * 4 offset might make them unaligned if c is not multiple of 32.
                // ne10 (src1 width) might not be multiple of 32.
                // So subsequent iterations might have unaligned dst/src0.
                // We should use generic hvx_op here or AAU/UUU variants if we knew alignments.
                // Simplest is to call generic dispatcher hvx_op(dst, src0, src1, len).
                switch (octx->op) { \
                    case HTP_OP_ADD: hvx_add_f32(r_dst + c * sizeof(float), r_src0 + c * sizeof(float), r_src1_row, len); break; \
                    case HTP_OP_SUB: hvx_sub_f32(r_dst + c * sizeof(float), r_src0 + c * sizeof(float), r_src1_row, len); break; \
                    case HTP_OP_MUL: hvx_mul_f32(r_dst + c * sizeof(float), r_src0 + c * sizeof(float), r_src1_row, len); break; \
                    case HTP_OP_DIV: hvx_div_f32(r_dst + c * sizeof(float), r_src0 + c * sizeof(float), r_src1_row, len); break; \
                    default: break; \
                }
            }
        }

        uint8_t * dst_curr = (uint8_t *)dst->data + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_queue_push(q, dma_make_ptr(dst_curr, d_spad), nb1, dst_row_size_aligned, ne00 * sizeof(float), current_block_size);

        if (ir_prefetch < end_row) {
             uint32_t next_block_size = calc_block_size(bctx, ir_prefetch, end_row, ne01, ne02, BLOCK_MAX);
             uint32_t p03, p02, p01, prem;
             p03 = fastdiv(ir_prefetch, &bctx->dim12_div);
             prem = ir_prefetch - p03 * (ne02 * ne01);
             p02 = fastdiv(prem, &bctx->dim1_div);
             p01 = prem - p02 * ne01;
             uint8_t * s0_next = (uint8_t *)src0->data + p03 * nb03 + p02 * nb02 + p01 * nb01;
             dma_queue_push(q, dma_make_ptr(s0_spad, s0_next), src0_row_size_aligned, nb01, ne00 * sizeof(float), next_block_size);
             ir_prefetch += next_block_size;
        }
        ir += current_block_size;
    }
    dma_queue_flush(q);
}

static void binary_job_f32(unsigned int n, unsigned int i, void * data) {
    struct htp_binary_context * bctx = (struct htp_binary_context *) data;
    const struct htp_tensor * src0 = &bctx->octx->src0;
    const struct htp_tensor * src1 = &bctx->octx->src1;

    if (src1->ne[0] == 1) {
        binary_job_scalar(bctx, n, i);
    } else if (src1->ne[0] == src0->ne[0] &&
               (src1->ne[1] == src0->ne[1] || src1->ne[1] == 1) &&
               (src1->ne[2] == src0->ne[2] || src1->ne[2] == 1) &&
               (src1->ne[3] == src0->ne[3] || src1->ne[3] == 1)) {
        if (src1->ne[1] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1) {
            binary_job_vector_row_broadcast(bctx, n, i);
        } else {
            binary_job_vector_same_shape(bctx, n, i);
        }
    } else if (src1->ne[0] == src0->ne[0]) {
        binary_job_vector_complex(bctx, n, i);
    } else {
        binary_job_element_repeat(bctx, n, i);
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

    const size_t src0_row_size_aligned = hex_round_up(src0_row_size, VLEN);
    const size_t src1_row_size_aligned = hex_round_up(src1_row_size, VLEN);
    const size_t dst_row_size_aligned  = hex_round_up(dst_row_size, VLEN);

    bool is_scalar = (src1->ne[0] == 1);

    // Determine which kernel we will use to alloc memory
    bool use_vector_same = !is_scalar && src1->ne[0] == src0->ne[0] &&
               (src1->ne[1] == src0->ne[1] || src1->ne[1] == 1) &&
               (src1->ne[2] == src0->ne[2] || src1->ne[2] == 1) &&
               (src1->ne[3] == src0->ne[3] || src1->ne[3] == 1);

    bool is_row_bcast = use_vector_same && (src1->ne[1] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1);

    size_t spad_row_total;
    if (is_scalar) {
        spad_row_total = 2 * (src0_row_size_aligned + dst_row_size_aligned);
    } else if (is_row_bcast) {
        spad_row_total = 2 * (src0_row_size_aligned + dst_row_size_aligned); // src1 allocated separately
    } else if (use_vector_same) {
        spad_row_total = 2 * (src0_row_size_aligned + src1_row_size_aligned + dst_row_size_aligned);
    } else {
        // complex/repeat: we don't DMA src1 (or minimal)
        spad_row_total = 2 * (src0_row_size_aligned + dst_row_size_aligned);
    }

    size_t rows_per_buffer = octx->ctx->vtcm_size / (n_threads * spad_row_total);
    // Adjust for static src1 in row_bcast case
    if (is_row_bcast) {
        // Reduce slightly to fit static row
        size_t needed_static = src1_row_size_aligned;
        if (octx->ctx->vtcm_size < needed_static * n_threads) return HTP_STATUS_VTCM_TOO_SMALL;
        size_t avail = octx->ctx->vtcm_size - needed_static * n_threads;
        rows_per_buffer = avail / (n_threads * spad_row_total);
    }

    if (rows_per_buffer < 1) {
         FARF(ERROR, "binary-f32: VTCM too small\n");
         return HTP_STATUS_VTCM_TOO_SMALL;
    }

    octx->src0_spad.size_per_thread = rows_per_buffer * 2 * src0_row_size_aligned;
    octx->dst_spad.size_per_thread  = rows_per_buffer * 2 * dst_row_size_aligned;

    if (is_scalar || !use_vector_same) {
        octx->src1_spad.size_per_thread = 0;
    } else if (is_row_bcast) {
        octx->src1_spad.size_per_thread = src1_row_size_aligned; // 1 row
    } else {
        octx->src1_spad.size_per_thread = rows_per_buffer * 2 * src1_row_size_aligned;
    }

    octx->src0_spad.size = n_threads * octx->src0_spad.size_per_thread;
    octx->src1_spad.size = n_threads * octx->src1_spad.size_per_thread;
    octx->dst_spad.size  = n_threads * octx->dst_spad.size_per_thread;

    if (octx->ctx->vtcm_size < (octx->src0_spad.size + octx->src1_spad.size + octx->dst_spad.size)) {
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    octx->src0_spad.data = octx->ctx->vtcm_base;
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

        bctx.src1_dim1_div = init_fastdiv_values(src1->ne[1]);
        bctx.src1_dim2_div = init_fastdiv_values(src1->ne[2]);
        bctx.src1_dim3_div = init_fastdiv_values(src1->ne[3]);

        bool src0_contig_dim1 = (src0->nb[2] == src0->ne[1] * src0->nb[1]);
        bool dst_contig_dim1  = (dst->nb[2] == src0->ne[1] * dst->nb[1]);

        bool src0_contig_dim2 = (src0->nb[3] == src0->ne[2] * src0->nb[2]);
        bool dst_contig_dim2  = (dst->nb[3] == src0->ne[2] * dst->nb[2]);

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
