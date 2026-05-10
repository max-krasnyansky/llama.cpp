#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <HAP_perf.h>

#include <math.h>
#include <string.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "hex-dma.h"
#include "hvx-utils.h"
#include "vtcm-utils.h"

struct htp_concat_context {
    struct htp_ops_context * octx;

    int32_t dim;
    uint32_t type_size;
    uint32_t nrows_per_thread;
    uint32_t total_rows; // ne1 * ne2 * ne3

    uint32_t src0_dim_ne;
    uint32_t src1_dim_ne;

    uint32_t elements_per_chunk; // Used for double buffering block size inside a single row
};

static void concat_job_per_thread(unsigned int nth, unsigned int ith, void * data) {
    const struct htp_concat_context * cctx = (const struct htp_concat_context *) data;
    struct htp_ops_context * octx = cctx->octx;

    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * src1 = octx->src[1];
    const struct htp_tensor * dst  = octx->dst;

    const uint32_t ne00 = src0->ne[0];
    const uint32_t ne01 = src0->ne[1];
    const uint32_t ne02 = src0->ne[2];
    const uint32_t ne03 = src0->ne[3];

    const uint32_t nb00 = src0->nb[0];
    const uint32_t nb01 = src0->nb[1];
    const uint32_t nb02 = src0->nb[2];
    const uint32_t nb03 = src0->nb[3];

    const uint32_t nb10 = src1->nb[0];
    const uint32_t nb11 = src1->nb[1];
    const uint32_t nb12 = src1->nb[2];
    const uint32_t nb13 = src1->nb[3];

    const uint32_t ne0 = dst->ne[0];
    const uint32_t ne1 = dst->ne[1];
    const uint32_t ne2 = dst->ne[2];
    const uint32_t ne3 = dst->ne[3];

    const uint32_t nb0 = dst->nb[0];
    const uint32_t nb1 = dst->nb[1];
    const uint32_t nb2 = dst->nb[2];
    const uint32_t nb3 = dst->nb[3];

    const uint32_t row_start = cctx->nrows_per_thread * ith;
    const uint32_t row_end   = MIN(row_start + cctx->nrows_per_thread, cctx->total_rows);

    const int32_t dim = cctx->dim;

    uint64_t o[4] = {0, 0, 0, 0};
    o[dim] = cctx->src0_dim_ne;

    dma_queue * dma = octx->ctx->dma[ith];

    uint32_t block_size_bytes = octx->src0_spad.size_per_thread / 2; // For ping-pong buffer
    uint8_t * vtcm_ping = octx->src0_spad.data + ith * octx->src0_spad.size_per_thread;
    uint8_t * vtcm_pong = vtcm_ping + block_size_bytes;

    uint8_t * vtcm_buf[2] = {vtcm_ping, vtcm_pong};

    for (uint32_t flat_idx = row_start; flat_idx < row_end; flat_idx++) {
        uint32_t i1 = flat_idx % ne1;
        uint32_t i2 = (flat_idx / ne1) % ne2;
        uint32_t i3 = flat_idx / (ne1 * ne2);

        uint32_t blocks = (ne0 + cctx->elements_per_chunk - 1) / cctx->elements_per_chunk;

        for (uint32_t b = 0; b < blocks; b++) {
            uint32_t elem_offset = b * cctx->elements_per_chunk;
            uint32_t num_elems = MIN(cctx->elements_per_chunk, ne0 - elem_offset);

            uint8_t * cur_vtcm = vtcm_buf[b % 2];

            if (b > 0) {
                // Wait for the previous block's write-back to finish
                dma_queue_pop(dma);
            }

            if (dim == 0) {
                uint32_t s0_elems = 0;
                uint32_t s1_elems = 0;

                if (elem_offset < ne00) {
                    s0_elems = MIN(num_elems, ne00 - elem_offset);
                }
                if (elem_offset + num_elems > ne00) {
                    uint32_t start_s1 = MAX(elem_offset, ne00) - ne00;
                    s1_elems = num_elems - s0_elems;

                    const uint8_t * src1_ptr = (const uint8_t *)src1->data + start_s1*nb10 + (i1 - o[1])*nb11 + (i2 - o[2])*nb12 + (i3 - o[3])*nb13;
                    dma_queue_push_single_2d(dma, dma_make_ptr(cur_vtcm + s0_elems * cctx->type_size, src1_ptr), cctx->type_size, nb10, cctx->type_size, s1_elems);
                }

                if (s0_elems > 0) {
                    const uint8_t * src0_ptr = (const uint8_t *)src0->data + (elem_offset)*nb00 + (i1)*nb01 + (i2)*nb02 + (i3)*nb03;
                    dma_queue_push_single_2d(dma, dma_make_ptr(cur_vtcm, src0_ptr), cctx->type_size, nb00, cctx->type_size, s0_elems);
                }

                if (s0_elems > 0) dma_queue_pop(dma);
                if (s1_elems > 0) dma_queue_pop(dma);

            } else {
                const uint8_t * src_ptr;
                uint32_t src_nb0;

                if (i1 < ne01 && i2 < ne02 && i3 < ne03) {
                    src_ptr = (const uint8_t *)src0->data + (elem_offset)*nb00 + (i1)*nb01 + (i2)*nb02 + (i3)*nb03;
                    src_nb0 = nb00;
                } else {
                    src_ptr = (const uint8_t *)src1->data + (elem_offset)*nb10 + (i1 - o[1])*nb11 + (i2 - o[2])*nb12 + (i3 - o[3])*nb13;
                    src_nb0 = nb10;
                }

                dma_queue_push_single_2d(dma, dma_make_ptr(cur_vtcm, src_ptr), cctx->type_size, src_nb0, cctx->type_size, num_elems);
                dma_queue_pop(dma);
            }

            uint8_t * dst_ptr = (uint8_t *)dst->data + (elem_offset)*nb0 + i1*nb1 + i2*nb2 + i3*nb3;
            dma_queue_push_single_2d(dma, dma_make_ptr(dst_ptr, cur_vtcm), nb0, cctx->type_size, cctx->type_size, num_elems);
        }

        if (blocks > 0) {
            dma_queue_pop(dma); // pop the final block in the row
        }
    }
}

int op_concat(struct htp_ops_context * octx) {
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * src1 = octx->src[1];
    const struct htp_tensor * dst  = octx->dst;

    if (src0->type != dst->type || src1->type != dst->type) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    uint32_t type_size;
    switch (src0->type) {
        case HTP_TYPE_F32: type_size = 4; break;
        case HTP_TYPE_F16: type_size = 2; break;
        default:
            return HTP_STATUS_NO_SUPPORT;
    }

    if (octx->flags & HTP_OPFLAGS_SKIP_COMPUTE) {
        return HTP_STATUS_OK;
    }

    const int32_t dim = octx->op_params[0];
    if (dim < 0 || dim >= 4) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    uint32_t total_rows = dst->ne[1] * dst->ne[2] * dst->ne[3];
    uint32_t n_threads = MIN(octx->n_threads, total_rows);

    uint32_t max_block_size = 4096; // Some reasonable block size for double buffering inside VTCM
    uint32_t elements_per_chunk = max_block_size / type_size;
    uint32_t row_size_aligned = hex_round_up(max_block_size * 2, 128); // Ping-pong

    octx->src0_spad.size_per_thread = row_size_aligned;
    octx->src0_spad.size = n_threads * row_size_aligned;

    if (vtcm_alloc_spad(&octx->src0_spad, octx->ctx) != HTP_STATUS_OK) {
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    struct htp_concat_context cctx;
    cctx.octx = octx;
    cctx.dim = dim;
    cctx.type_size = type_size;
    cctx.nrows_per_thread = (total_rows + n_threads - 1) / n_threads;
    cctx.total_rows = total_rows;
    cctx.src0_dim_ne = src0->ne[dim];
    cctx.src1_dim_ne = src1->ne[dim];
    cctx.elements_per_chunk = elements_per_chunk;

    worker_pool_run_func(octx->ctx->worker_pool, concat_job_per_thread, &cctx, n_threads);

    return HTP_STATUS_OK;
}
