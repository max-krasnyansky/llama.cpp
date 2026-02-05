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

typedef void (*hvx_elemwise_f32_func)(uint8_t * data_dst, const uint8_t * src0, const uint8_t * src1, const uint32_t num_elems);

static hvx_elemwise_f32_func func_table_HVX[]     = { hvx_mul_f32, hvx_add_f32, hvx_sub_f32 };
static hvx_elemwise_f32_func func_table_HVX_opt[] = { hvx_mul_f32_aa, hvx_add_f32_aa, hvx_sub_f32_aa };

struct htp_binary_context {
    struct htp_ops_context * octx;

    uint32_t src0_nrows_per_thread;

    struct fastdiv_values src0_div1;
    struct fastdiv_values src0_div2;
    struct fastdiv_values src0_div3;
    struct fastdiv_values src0_div21;

    struct fastdiv_values src1_div1;
    struct fastdiv_values src1_div2;
    struct fastdiv_values src1_div3;
    struct fastdiv_values src1_div21;
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
    const uint32_t src0_nrows_per_thread = octx->src0_nrows_per_thread;

#define htp_binary_context_preamble            \
    struct htp_ops_context * octx = bctx->octx; \
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
    const uint32_t src0_nrows_per_thread = bctx->src0_nrows_per_thread;

static void binary_job_f32_per_thread(struct htp_binary_context * bctx,
                                      uint32_t                    nth,
                                      uint32_t                    ith,
                                      enum htp_op                 op,
                                      dma_queue *                 dma_queue) {
    htp_binary_context_preamble;

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

    const size_t src0_row_size_aligned = hex_round_up(src0_row_size, VLEN);
    const size_t src1_row_size_aligned = hex_round_up(src1_row_size, VLEN);
    const size_t dst_row_size_aligned  = hex_round_up(dst_row_size, VLEN);

    uint8_t * src0_spad_base = octx->src0_spad.data + ith * octx->src0_spad.size_per_thread;
    uint8_t * src1_spad_base = octx->src1_spad.data + ith * octx->src1_spad.size_per_thread;
    uint8_t * dst_spad_base  = octx->dst_spad.data  + ith * octx->dst_spad.size_per_thread;

    size_t src0_spad_half = octx->src0_spad.size_per_thread / 2;
    size_t src1_spad_half = octx->src1_spad.size_per_thread / 2;
    size_t dst_spad_half  = octx->dst_spad.size_per_thread / 2;

    int BLOCK = src0_spad_half / src0_row_size_aligned;
    if (BLOCK == 0) return;

    const uint8_t * data_src0 = (const uint8_t *) src0->data;
    const uint8_t * data_src1 = (const uint8_t *) src1->data;
    uint8_t *       data_dst  = (uint8_t *) dst->data;

    hvx_elemwise_f32_func func_HVX = func_table_HVX_opt[op];

    uint32_t next_ir = src0_start_row;
    int spad_idx = 0;

    const uint32_t ne02_ne01 = ne02 * ne01;

    // Prime 2 blocks
    for (int k = 0; k < 2 && next_ir < src0_end_row; k++) {
        uint32_t block_size = MIN(BLOCK, src0_end_row - next_ir);

        uint32_t i03 = fastdiv(next_ir, &bctx->src0_div21);
        uint32_t i02 = fastdiv(next_ir - i03 * ne02_ne01, &bctx->src0_div1);
        uint32_t i01 = (next_ir - i03 * ne02_ne01 - i02 * ne01);
        block_size = MIN(block_size, ne01 - i01);

        uint32_t i13 = fastmodulo(i03, ne13, &bctx->src1_div3);
        uint32_t i12 = fastmodulo(i02, ne12, &bctx->src1_div2);
        uint32_t i11 = fastmodulo(i01, ne11, &bctx->src1_div1);

        const uint8_t * s1_addr = data_src1 + i13 * nb13 + i12 * nb12 + i11 * src1_row_size;
        size_t s1_stride = nb11;
        if (ne11 == 1 && ne01 > 1) {
            s1_stride = 0;
        } else if (ne11 != ne01) {
            block_size = 1;
            s1_stride = 0;
        }

        const uint8_t * s0_addr = data_src0 + next_ir * src0_row_size;

        dma_queue_push_vtcm_to_ddr(dma_queue,
             dma_make_ptr(data_dst, dst_spad_base + spad_idx * dst_spad_half),
             dst_row_size, dst_row_size_aligned, 0);

        dma_queue_push_ddr_to_vtcm(dma_queue,
             dma_make_ptr(src0_spad_base + spad_idx * src0_spad_half, s0_addr),
             src0_row_size_aligned, src0_row_size, block_size);

        size_t s1_width = ne10 * sizeof(float);
        dma_queue_push(dma_queue,
             dma_make_ptr(src1_spad_base + spad_idx * src1_spad_half, s1_addr),
             src1_row_size_aligned, s1_stride, s1_width, block_size);

        next_ir += block_size;
        spad_idx = (spad_idx + 1) % 2;
    }

    spad_idx = 0;
    for (uint32_t proc_ir = src0_start_row; proc_ir < src0_end_row; ) {
        uint32_t block_size = MIN(BLOCK, src0_end_row - proc_ir);

        uint32_t i03 = fastdiv(proc_ir, &bctx->src0_div21);
        uint32_t i02 = fastdiv(proc_ir - i03 * ne02_ne01, &bctx->src0_div1);
        uint32_t i01 = (proc_ir - i03 * ne02_ne01 - i02 * ne01);
        block_size = MIN(block_size, ne01 - i01);

        uint32_t i13 = fastmodulo(i03, ne13, &bctx->src1_div3);
        uint32_t i12 = fastmodulo(i02, ne12, &bctx->src1_div2);
        uint32_t i11 = fastmodulo(i01, ne11, &bctx->src1_div1);
        if (ne11 != ne01 && !(ne11 == 1 && ne01 > 1)) block_size = 1;

        dma_ptr dp_dst = dma_queue_pop(dma_queue);
        dma_ptr dp_s0  = dma_queue_pop(dma_queue);
        dma_ptr dp_s1  = dma_queue_pop(dma_queue);

        float * s0_buf = (float*)dp_s0.dst;
        float * s1_buf = (float*)dp_s1.dst;
        float * dst_buf = (float*)dp_dst.src;

        uint32_t nr0 = ne00 / ne10;
        for (uint32_t b = 0; b < block_size; b++) {
             float * p_s0 = s0_buf + b * (src0_row_size_aligned/4);
             float * p_s1 = s1_buf + b * (src1_row_size_aligned/4);
             float * p_d  = dst_buf + b * (dst_row_size_aligned/4);

             if (nr0 > 1) {
                 float val = *p_s1;
                 hvx_splat_f32_a((uint8_t*)p_s1, val, nr0);
             }
             func_HVX((uint8_t*)p_d, (const uint8_t*)p_s0, (const uint8_t*)p_s1, ne00);
        }

        dma_queue_push_vtcm_to_ddr(dma_queue,
             dma_make_ptr(data_dst + proc_ir * dst_row_size, dst_buf),
             dst_row_size, dst_row_size_aligned, block_size);

        if (next_ir < src0_end_row) {
            uint32_t pb_size = MIN(BLOCK, src0_end_row - next_ir);

            uint32_t ni03 = fastdiv(next_ir, &bctx->src0_div21);
            uint32_t ni02 = fastdiv(next_ir - ni03 * ne02_ne01, &bctx->src0_div1);
            uint32_t ni01 = (next_ir - ni03 * ne02_ne01 - ni02 * ne01);
            pb_size = MIN(pb_size, ne01 - ni01);

            uint32_t ni13 = fastmodulo(ni03, ne13, &bctx->src1_div3);
            uint32_t ni12 = fastmodulo(ni02, ne12, &bctx->src1_div2);
            uint32_t ni11 = fastmodulo(ni01, ne11, &bctx->src1_div1);

            const uint8_t * ns1_addr = data_src1 + ni13 * nb13 + ni12 * nb12 + ni11 * src1_row_size;
            size_t ns1_stride = nb11;
            if (ne11 == 1 && ne01 > 1) {
                ns1_stride = 0;
            } else if (ne11 != ne01) {
                pb_size = 1;
                ns1_stride = 0;
            }

            const uint8_t * ns0_addr = data_src0 + next_ir * src0_row_size;

            dma_queue_push_vtcm_to_ddr(dma_queue,
                 dma_make_ptr(data_dst, dst_spad_base + spad_idx * dst_spad_half),
                 dst_row_size, dst_row_size_aligned, 0);

            dma_queue_push_ddr_to_vtcm(dma_queue,
                 dma_make_ptr(src0_spad_base + spad_idx * src0_spad_half, ns0_addr),
                 src0_row_size_aligned, src0_row_size, pb_size);

            size_t ns1_width = ne10 * sizeof(float);
            dma_queue_push(dma_queue,
                 dma_make_ptr(src1_spad_base + spad_idx * src1_spad_half, ns1_addr),
                 src1_row_size_aligned, ns1_stride, ns1_width, pb_size);

            next_ir += pb_size;
            spad_idx = (spad_idx + 1) % 2;
        }

        proc_ir += block_size;
    }

    dma_queue_flush(dma_queue);

    t2 = HAP_perf_get_qtimer_count();

    FARF(HIGH, "binary-f32 %d/%d: %ux%ux%ux%u (%u:%u) x %ux%ux%ux%u -> %ux%ux%ux%u usec %u\n", ith, nth,
         ne00, ne01, ne02, ne03, src0_start_row, src0_end_row, ne10, ne11, ne12, ne13, ne0, ne1, ne2, ne3,
         (unsigned) HAP_perf_qtimer_count_to_us(t2 - t1));
}

static void binary_add_id_job_f32_per_thread(struct htp_binary_context * bctx,
                                             uint32_t                 nth,
                                             uint32_t                 ith,
                                             hvx_elemwise_f32_func    func_HVX) {
    htp_binary_context_preamble;

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

    uint8_t * spad_data = octx->src0_spad.data + ith * octx->src0_spad.size_per_thread;

    uint64_t t1, t2;
    t1 = HAP_perf_get_qtimer_count();

    const uint8_t * restrict data_src0 = (const uint8_t *) src0->data;
    const uint8_t * restrict data_src1 = (const uint8_t *) src1->data;
    uint8_t * restrict data_dst        = (uint8_t *) dst->data;

    const uint32_t ne02_ne01  = ne02 * ne01;
    for (uint32_t ir = src0_start_row; ir < src0_end_row; ir++) {
        // src0 indices
        const uint32_t i03 = fastdiv(ir, &bctx->src0_div21);
        const uint32_t i02 = fastdiv(ir - i03 * ne02_ne01, &bctx->src0_div1);
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
    struct htp_binary_context * bctx = (struct htp_binary_context *) data;
    struct htp_ops_context *    octx = bctx->octx;

    switch (octx->op) {
        case HTP_OP_MUL:
        case HTP_OP_ADD:
        case HTP_OP_SUB:
            binary_job_f32_per_thread(bctx, n, i, octx->op, octx->ctx->dma[i]);
            break;

        case HTP_OP_ADD_ID:
            binary_add_id_job_f32_per_thread(bctx, n, i, hvx_add_f32);
            break;

        default:
            FARF(ERROR, "Unknown Binary Op %u", octx->op);
            break;
    }
}

static int execute_op_binary_f32(struct htp_ops_context * octx) {
    int err = HTP_STATUS_OK;

    const struct htp_tensor * src0 = &octx->src0;
    const struct htp_tensor * src1 = &octx->src1;
    struct htp_tensor *       dst  = &octx->dst;

    worker_callback_t binary_op_func;
    const char *      op_type = NULL;

    switch (octx->op) {
        case HTP_OP_MUL:
            binary_op_func = binary_job_dispatcher_f32;
            op_type        = "mul-f32";
            break;

        case HTP_OP_ADD:
            binary_op_func = binary_job_dispatcher_f32;
            op_type        = "add-f32";
            break;

        case HTP_OP_SUB:
            binary_op_func = binary_job_dispatcher_f32;
            op_type        = "sub-f32";
            break;

        case HTP_OP_ADD_ID:
            binary_op_func = binary_job_dispatcher_f32;
            op_type        = "add-id-f32";
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

    // If inner broadcast, we might need larger src1 spad to splat
    size_t src1_spad_req = src1_row_size_aligned;
    if (src0->ne[0] / src1->ne[0] > 1) {
        src1_spad_req = MAX(src1_spad_req, src0_row_size_aligned);
    }

    // Determine row allocation in VTCM
    size_t spad_per_row_aligned = src0_row_size_aligned + src1_spad_req + dst_row_size_aligned;
    size_t total_rows_vtcm = octx->ctx->vtcm_size / spad_per_row_aligned;
    size_t rows_per_thread = total_rows_vtcm / n_threads;

    // We want at least 2 rows per thread for double buffering
    if (rows_per_thread < 2) {
        // Fallback: use minimum but check total size later
        rows_per_thread = 2;
    }
    // Make even for simpler ping-pong
    rows_per_thread &= ~1;
    if (rows_per_thread < 2) rows_per_thread = 2;

    octx->src0_spad.size_per_thread = rows_per_thread * src0_row_size_aligned;
    octx->src1_spad.size_per_thread = rows_per_thread * src1_spad_req;
    octx->dst_spad.size_per_thread  = rows_per_thread * dst_row_size_aligned;

    octx->src0_spad.size = octx->src0_spad.size_per_thread * n_threads;
    octx->src1_spad.size = octx->src1_spad.size_per_thread * n_threads;
    octx->dst_spad.size  = octx->dst_spad.size_per_thread * n_threads;

    size_t spad_size = octx->src0_spad.size + octx->src1_spad.size + octx->dst_spad.size;

    FARF(HIGH,
         "%s: (%ux%ux%ux%u) * (%ux%ux%ux%u) -> (%ux%ux%ux%u) : src0-spad-size %u src1-spad-size %u dst-spad-size %u\n",
         op_type, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src1->ne[0], src1->ne[1], src1->ne[2],
         src1->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], octx->src0_spad.size, octx->src1_spad.size,
         octx->dst_spad.size);

    // Make sure the reserved vtcm size is sufficient
    if (octx->ctx->vtcm_size < spad_size) {
        FARF(ERROR, "binary-%s : current VTCM reservation %zu is too small, needed %zu\n", op_type,
             octx->ctx->vtcm_size, spad_size);
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    octx->src0_spad.data = octx->ctx->vtcm_base;
    octx->src1_spad.data = octx->src0_spad.data + octx->src0_spad.size;
    octx->dst_spad.data  = octx->src1_spad.data + octx->src1_spad.size;

    if (!(octx->flags & HTP_OPFLAGS_SKIP_COMPUTE)) {
        uint32_t n_jobs = MIN(n_threads, src0_nrows);

        struct htp_binary_context bctx;
        bctx.octx = octx;
        bctx.src0_nrows_per_thread = (src0_nrows + n_jobs - 1) / n_jobs;

        bctx.src0_div21 = init_fastdiv_values(src0->ne[2] * src0->ne[1]);
        bctx.src0_div3  = init_fastdiv_values(src0->ne[3]);
        bctx.src0_div2  = init_fastdiv_values(src0->ne[2]);
        bctx.src0_div1  = init_fastdiv_values(src0->ne[1]);

        bctx.src1_div21 = init_fastdiv_values(src1->ne[2] * src1->ne[1]);
        bctx.src1_div3  = init_fastdiv_values(src1->ne[3]);
        bctx.src1_div2  = init_fastdiv_values(src1->ne[2]);
        bctx.src1_div1  = init_fastdiv_values(src1->ne[1]);

        worker_pool_run_func(octx->ctx->worker_pool, binary_op_func, &bctx, n_jobs);
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
