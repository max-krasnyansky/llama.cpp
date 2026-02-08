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

static void binary_job_f32_per_thread(struct htp_binary_context * bctx,
                                      uint32_t                 nth,
                                      uint32_t                 ith) {
    struct htp_ops_context * octx = bctx->octx;
    htp_binary_preamble;

    const size_t src0_row_size = nb01;
    const size_t src1_row_size = nb11;
    const size_t dst_row_size  = nb1;

    // We process rows based on src0 dimensions
    const uint32_t total_rows = ne01 * ne02 * ne03;

    const uint32_t start_row = bctx->nrows_per_thread * ith;
    const uint32_t end_row   = MIN(start_row + bctx->nrows_per_thread, total_rows);

    if (start_row >= end_row) {
        return;
    }

    uint64_t t1, t2;
    t1 = HAP_perf_get_qtimer_count();

    // Align row sizes to VLEN (128 bytes)
    const size_t src0_row_size_aligned = hex_round_up(src0_row_size, VLEN);
    // src1 row size depends on whether it's broadcasted (ne10 == 1) or full (ne10 == ne00)
    // If ne10 == 1, we only fetch sizeof(float) (or VLEN aligned)
    const size_t src1_real_row_size = (ne10 == 1) ? sizeof(float) : src1_row_size;
    const size_t src1_row_size_aligned = hex_round_up(src1_real_row_size, VLEN);
    const size_t dst_row_size_aligned  = hex_round_up(dst_row_size, VLEN);

    // Get VTCM scratchpads
    uint8_t * src0_spad_base = octx->src0_spad.data + (ith * octx->src0_spad.size_per_thread);
    uint8_t * src1_spad_base = octx->src1_spad.data + (ith * octx->src1_spad.size_per_thread);
    uint8_t * dst_spad_base  = octx->dst_spad.data  + (ith * octx->dst_spad.size_per_thread);

    // Double buffering (ping-pong)
    size_t src0_spad_half = octx->src0_spad.size_per_thread / 2;
    size_t src1_spad_half = octx->src1_spad.size_per_thread / 2;
    size_t dst_spad_half  = octx->dst_spad.size_per_thread  / 2;

    // Block size in rows
    const int BLOCK_MAX = src0_spad_half / src0_row_size_aligned;
    if (BLOCK_MAX == 0) {
        FARF(ERROR, "binary-f32: VTCM too small for even 1 row per thread\n");
        return;
    }

    dma_queue * q = octx->ctx->dma[ith];

    // Main loop over rows
    // We iterate 'ir' but we need to respect broadcasting boundaries for src1.
    // src1 might reset or jump at ne01 or ne02*ne01 boundaries if dims don't match.
    // To simplify, we recompute src1 address for every block or sub-block.

    // We process in chunks of BLOCK_MAX.
    // However, if src1 is not contiguous across rows (e.g. wrap around), we must handle it.
    // If ne11 == 1, src1 stride is 0 (repeats). Contiguous in DMA terms (stride 0).
    // If ne11 == ne01, src1 stride is nb11. Contiguous.
    // But if we cross ne01 boundary (i.e. i01 wraps), src1 might need to jump if ne12=1.
    // So we should clamp block size to not cross ne01 boundaries if broadcasting logic requires it.

    // Safer splitting logic:
    // We must split at ne01 boundary if we move to next row (ne02 > 1) AND:
    // 1. src1 varies along dim1 (ne11 > 1) -> we need to check if nb12 is contiguous continuation. Safest is to split.
    // 2. src1 varies along dim2 (ne12 > 1) -> we need to jump by nb12. Stride nb11 or 0 won't handle it.
    // If src1 is constant in both (ne11=1, ne12=1), stride 0 works across boundary.
    bool split_at_ne01 = (ne02 > 1) && ((ne11 > 1) || (ne12 > 1));
    bool split_at_ne02 = (ne03 > 1) && ((ne12 > 1) || (ne13 > 1));

    for (uint32_t ir = start_row; ir < end_row; ) {
        // Decompose ir to find i01, i02, i03
        // ir = i03 * (ne02*ne01) + i02 * ne01 + i01
        uint32_t i03, i02, i01;
        uint32_t rem;

        i03 = fastdiv(ir, &bctx->dim12_div);
        rem = ir - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->dim1_div);
        i01 = rem - i02 * ne01;

        // Determine max rows we can process contiguously regarding src1 broadcasting
        uint32_t rows_left = end_row - ir;
        uint32_t block_limit = rows_left;

        if (split_at_ne01) {
            block_limit = MIN(block_limit, ne01 - i01);
        }
        if (split_at_ne02) {
             // If we are splitting at ne02 (i.e. ne01*ne02 boundary), we need to check distance to next ne02 boundary
             // distance = (ne02 - i02) * ne01 - i01?
             // Actually, simplest is just clamp to end of current "plane"
             // But if split_at_ne01 is set, we already clamp to ne01.
             // If ne12 == ne02, we don't split at ne01. But if ne13 == 1, we split at ne02.
             // If ne12 == ne02, we continue across ne01 boundary.
             // But if ne13 == 1, we must stop when i02 wraps (i.e. at ne02*ne01).
             uint32_t rows_in_plane = (ne02 * ne01) - rem;
             block_limit = MIN(block_limit, rows_in_plane);
        }

        uint32_t current_block_size = MIN(BLOCK_MAX, block_limit);

        // Map indices to src1 indices
        // i13 = i03 % ne13; i12 = i02 % ne12; i11 = i01 % ne11;
        // Since ne1x is 1 or ne0x:
        uint32_t i13 = (ne13 == 1) ? 0 : i03;
        uint32_t i12 = (ne12 == 1) ? 0 : i02;
        uint32_t i11 = (ne11 == 1) ? 0 : i01;

        // src1 address
        uint8_t * src1_base = (uint8_t *)src1->data + i13 * nb13 + i12 * nb12 + i11 * nb11;

        // src1 stride for DMA
        // If ne11 == 1, stride is 0. If ne11 == ne01, stride is nb11.
        uint32_t src1_dma_stride = (ne11 == 1) ? 0 : nb11;

        // Issue DMA for this block (sub-divided into BLOCK_MAX chunks effectively, but here we handled BLOCK_MAX loop inside?)
        // Wait, loop structure in act-ops iterates ir += BLOCK.
        // Here I have a variable block size due to broadcasting.
        // I should probably loop current_block_size in chunks of BLOCK_MAX?
        // No, current_block_size IS <= BLOCK_MAX.
        // So I can just process 'current_block_size' rows.

        // We use ping-pong buffering. But since 'current_block_size' might be small (e.g. at boundaries),
        // we just do one transfer per iteration of outer loop?
        // Or we should try to pipeline?
        // Pipelining variable sized blocks is tricky.
        // Let's stick to single-buffer or simple ping-pong if blocks are uniform.
        // Given complexity, let's just do sequential DMA fetch -> compute -> store for correctness first.
        // Optimization: if BLOCK_MAX is large enough, we usually process many rows.

        // Let's use 2-stage pipeline manually.
        // Actually, act-ops uses a loop `for (ir...; ir < end; ir += BLOCK)` and prefetches next.
        // Here my `block_limit` might change.
        // But if `ne11 == ne01`, block_limit is large.
        // If `ne11 == 1`, block_limit is large.
        // The only case it's small is if `ne11` matches but `ne12` doesn't (reset every ne01).
        // Then `current_block_size` <= ne01.

        // I will implement a loop over `current_block_size` processing.

        // Src0 address
        uint8_t * src0_curr = (uint8_t *)src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01;
        uint8_t * dst_curr  = (uint8_t *)dst->data  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        // Use spad_idx = 0 for simplicity or toggle?
        // Let's just use 0 (half buffer) and process synchronously for now to ensure correctness with complex striding.
        // To optimize, one would prefetch 'next' block indices.

        // DMA In
        // Use dma_queue_push directly to specify width vs stride correctly.
        // src0
        dma_queue_push(q, dma_make_ptr(src0_spad_base, src0_curr),
                       src0_row_size_aligned, // dst_stride (VTCM)
                       nb01,                  // src_stride (DDR)
                       ne00 * sizeof(float),  // width
                       current_block_size);

        // src1
        uint32_t src1_width = (ne10 == 1) ? sizeof(float) : (ne00 * sizeof(float));
        dma_queue_push(q, dma_make_ptr(src1_spad_base, src1_base),
                       src1_row_size_aligned, // dst_stride (VTCM)
                       src1_dma_stride,       // src_stride (DDR)
                       src1_width,            // width
                       current_block_size);

        // Wait for DMA
        // dma_queue_pop returns pointers.
        // Since we pushed 2, we pop 2.
        void * src0_ptr_vtcm = dma_queue_pop(q).dst;
        void * src1_ptr_vtcm = dma_queue_pop(q).dst;

        // Compute
        for (uint32_t r = 0; r < current_block_size; r++) {
            uint8_t * r_src0 = (uint8_t *)src0_ptr_vtcm + r * src0_row_size_aligned;
            uint8_t * r_src1 = (uint8_t *)src1_ptr_vtcm + r * src1_row_size_aligned;
            uint8_t * r_dst  = dst_spad_base + r * dst_row_size_aligned; // Use same buffer for dst

            // Check if src1 is scalar (ne10 == 1)
            // Note: ne10 is the dimension 0 size.
            if (ne10 == 1) {
                // Scalar op
                float val = *(float *)r_src1; // Fetch scalar
                switch (octx->op) {
                    case HTP_OP_ADD:
                        hvx_add_scalar_f32_aa(r_dst, r_src0, val, ne00);
                        break;
                    case HTP_OP_SUB:
                        hvx_sub_scalar_f32_aa(r_dst, r_src0, val, ne00);
                        break;
                    case HTP_OP_MUL:
                        hvx_mul_scalar_f32_aa(r_dst, r_src0, val, ne00);
                        break;
                    case HTP_OP_DIV:
                        hvx_mul_scalar_f32_aa(r_dst, r_src0, 1.0f / val, ne00);
                        break;
                    default:
                        break;
                }
            } else {
                // Vector op
                switch (octx->op) {
                    case HTP_OP_ADD:
                        hvx_add_f32_aa(r_dst, r_src0, r_src1, ne00);
                        break;
                    case HTP_OP_SUB:
                        hvx_sub_f32_aa(r_dst, r_src0, r_src1, ne00);
                        break;
                    case HTP_OP_MUL:
                        hvx_mul_f32_aa(r_dst, r_src0, r_src1, ne00);
                        break;
                    case HTP_OP_DIV:
                        hvx_div_f32_aa(r_dst, r_src0, r_src1, ne00);
                        break;
                    default:
                        break;
                }
            }
        }

        // DMA Out
        dma_queue_push(q, dma_make_ptr(dst_curr, dst_spad_base),
                       nb1,                   // dst_stride (DDR)
                       dst_row_size_aligned,  // src_stride (VTCM)
                       ne00 * sizeof(float),  // width
                       current_block_size);

        // Force flush for synchronous execution
        dma_queue_flush(q);

        ir += current_block_size;
    }

    t2 = HAP_perf_get_qtimer_count();
    FARF(HIGH, "binary-f32 %d/%d: %ux%ux%ux%u (%u:%u) usec %u\n", ith, nth,
         ne00, ne01, ne02, ne03, start_row, end_row,
         (unsigned) HAP_perf_qtimer_count_to_us(t2 - t1));
}

static void binary_job_f32(unsigned int n, unsigned int i, void * data) {
    struct htp_binary_context * bctx = (struct htp_binary_context *) data;
    binary_job_f32_per_thread(bctx, n, i);
}

static int execute_op_binary_f32(struct htp_ops_context * octx) {
    int err = HTP_STATUS_OK;
    const struct htp_tensor * src0 = &octx->src0;
    const struct htp_tensor * src1 = &octx->src1;
    struct htp_tensor *       dst  = &octx->dst;

    const uint32_t n_threads  = octx->n_threads;
    const uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];

    const size_t src0_row_size = src0->nb[1];
    // For spad calculation, we use actual row size of src1 if it's vector, else scalar size
    const size_t src1_real_row_size = (src1->ne[0] == 1) ? sizeof(float) : src1->nb[1];
    const size_t dst_row_size  = dst->nb[1];

    // Align to VLEN
    const size_t src0_row_size_aligned = hex_round_up(src0_row_size, VLEN);
    const size_t src1_row_size_aligned = hex_round_up(src1_real_row_size, VLEN);
    const size_t dst_row_size_aligned  = hex_round_up(dst_row_size, VLEN);

    // Calc spad size per thread
    // We allocate 2 buffers per thread for ping-pong (although we use sync implementation for now, allocation supports it)
    // Actually, sync implementation uses 1 buffer set. But let's keep allocation robust.
    size_t src0_spad_per_thread = src0_row_size_aligned * 4; // Arbitrary small number of rows buffering? No, we need enough for BLOCK.

    // Let's allocate based on total VTCM
    size_t spad_row_total = src0_row_size_aligned + src1_row_size_aligned + dst_row_size_aligned;
    size_t available_rows = octx->ctx->vtcm_size / (n_threads * spad_row_total);

    // Ensure at least 1 row (actually 2 for ping pong would be better, but we used sync)
    if (available_rows < 1) {
         FARF(ERROR, "binary-f32: VTCM too small\n");
         return HTP_STATUS_VTCM_TOO_SMALL;
    }

    size_t rows_per_buffer = available_rows; // Use all available space

    octx->src0_spad.size_per_thread = rows_per_buffer * src0_row_size_aligned;
    octx->src1_spad.size_per_thread = rows_per_buffer * src1_row_size_aligned;
    octx->dst_spad.size_per_thread  = rows_per_buffer * dst_row_size_aligned;

    octx->src0_spad.size = n_threads * octx->src0_spad.size_per_thread;
    octx->src1_spad.size = n_threads * octx->src1_spad.size_per_thread;
    octx->dst_spad.size  = n_threads * octx->dst_spad.size_per_thread;

    if (octx->ctx->vtcm_size < (octx->src0_spad.size + octx->src1_spad.size + octx->dst_spad.size)) {
        // Fallback or error?
        // Should not happen if calculation correct
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

        // Init fastdivs
        bctx.dim1_div = init_fastdiv_values(src0->ne[1]);
        bctx.dim2_div = init_fastdiv_values(src0->ne[2]);
        bctx.dim12_div = init_fastdiv_values(src0->ne[1] * src0->ne[2]);

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
