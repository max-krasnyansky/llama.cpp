#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <HAP_perf.h>

#include <math.h>
#include <string.h>

#include "hex-dma.h"
#include "hex-fastdiv.h"
#include "hvx-utils.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-ops.h"

struct htp_concat_context {
    struct htp_ops_context * octx;

    int32_t dim;

    size_t type_size;

    uint32_t nr;
    uint32_t dr; // rows per thread

    struct fastdiv_values div_ne1;
    struct fastdiv_values div_ne2;

    bool fast_src0;
};

#define concat_preamble                              \
    const struct htp_tensor *src0 = octx->src[0];    \
    const struct htp_tensor *src1 = octx->src[1];    \
    const struct htp_tensor *dst  = octx->dst;       \
                                                     \
    const uint32_t ne00 = src0->ne[0];               \
    const uint32_t ne01 = src0->ne[1];               \
    const uint32_t ne02 = src0->ne[2];               \
    const uint32_t ne03 = src0->ne[3];               \
                                                     \
    const uint32_t nb00 = src0->nb[0];               \
    const uint32_t nb01 = src0->nb[1];               \
    const uint32_t nb02 = src0->nb[2];               \
    const uint32_t nb03 = src0->nb[3];               \
                                                     \
    const uint32_t ne10 = src1->ne[0];               \
    const uint32_t ne11 = src1->ne[1];               \
    const uint32_t ne12 = src1->ne[2];               \
    const uint32_t ne13 = src1->ne[3];               \
                                                     \
    const uint32_t nb10 = src1->nb[0];               \
    const uint32_t nb11 = src1->nb[1];               \
    const uint32_t nb12 = src1->nb[2];               \
    const uint32_t nb13 = src1->nb[3];               \
                                                     \
    const uint32_t ne0 = dst->ne[0];                 \
    const uint32_t ne1 = dst->ne[1];                 \
    const uint32_t ne2 = dst->ne[2];                 \
    const uint32_t ne3 = dst->ne[3];                 \
                                                     \
    const uint32_t nb0 = dst->nb[0];                 \
    const uint32_t nb1 = dst->nb[1];                 \
    const uint32_t nb2 = dst->nb[2];                 \
    const uint32_t nb3 = dst->nb[3];


static void concat_thread_dim0_fast_src0_vgather_f32(unsigned int nth, unsigned int ith, void *data) {
    struct htp_concat_context * cctx = (struct htp_concat_context *)data;
    struct htp_ops_context * octx = cctx->octx;

    concat_preamble;

    const size_t type_size = 4;
    const uint32_t chunk_i1 = 128 / type_size;

    uint64_t qt = HAP_perf_get_qtimer_count();

    const uint32_t ir0 = cctx->dr * ith;
    const uint32_t ir1 = MIN(ir0 + cctx->dr, cctx->nr); // iterating over ne2 * ne3

    dma_queue * dma_queue = octx->ctx->dma[ith];
    uint8_t * src1_vtcm = octx->src1_spad.data + (ith * octx->src1_spad.size_per_thread);
    uint8_t * dst_vtcm  = octx->dst_spad.data  + (ith * octx->dst_spad.size_per_thread);

    int32_t offsets[32] __attribute__((aligned(128)));
    for (int k = 0; k < 32; k++) {
        offsets[k] = k * 128; // row size in VTCM is exactly chunk_i1 * type_size = 128 bytes
    }
    HVX_Vector v_off = hvx_vmem(offsets);

    for (uint32_t r = ir0; r < ir1; ++r) {
        const uint32_t i2 = fastmodulo(r, ne2, &cctx->div_ne2);
        const uint32_t i3 = fastdiv(r, &cctx->div_ne2);

        for (uint32_t i1_blk = 0; i1_blk < ne11; i1_blk += chunk_i1) {
            uint32_t cur_ne11 = MIN(chunk_i1, ne11 - i1_blk);

            // DMA src1 to VTCM: ne10 rows, row_size=cur_ne11 * 4, stride=nb10
            const char * x1 = (const char *)src1->data + i1_blk*nb11 + i2*nb12 + i3*nb13;
            dma_queue_push(dma_queue, dma_make_ptr(src1_vtcm, x1), 128, nb10, cur_ne11 * type_size, ne10);
            dma_queue_pop(dma_queue);

            for (uint32_t i0_blk = 0; i0_blk < ne10; i0_blk += chunk_i1) {
                uint32_t cur_ne10 = MIN(chunk_i1, ne10 - i0_blk);
                for (uint32_t c = 0; c < cur_ne11; ++c) {
                    uint8_t * src_ptr = src1_vtcm + i0_blk * 128 + c * type_size;
                    uint8_t * dst_ptr = dst_vtcm  + c * (ne10 * type_size) + i0_blk * type_size;
                    HVX_Vector v = Q6_V_vzero();
                    Q6_vgather_ARMVw(&v, (Word32)src_ptr, 128, v_off);
                    // Write 32 elements. If cur_ne10 < 32, padding protects next rows.
                    hvx_vec_store_u(dst_ptr, type_size * cur_ne10, v);
                }
            }

            // DMA transposed VTCM back to DDR dst
            char * y1 = (char *)dst->data + i1_blk*nb1 + i2*nb2 + i3*nb3 + ne00 * nb0;
            dma_queue_push(dma_queue, dma_make_ptr(y1, dst_vtcm), nb1, ne10 * type_size, ne10 * type_size, cur_ne11);
            dma_queue_pop(dma_queue);
        }
    }

    qt = HAP_perf_qtimer_count_to_us(HAP_perf_get_qtimer_count() - qt);
    FARF(HIGH, "concat-fast-src0-vgather-f32 %d/%d: usec %u\n", ith, nth, (unsigned) qt);
}

static void concat_thread_dim0_fast_src0_vgather_f16(unsigned int nth, unsigned int ith, void *data) {
    struct htp_concat_context * cctx = (struct htp_concat_context *)data;
    struct htp_ops_context * octx = cctx->octx;

    concat_preamble;

    const size_t type_size = 2;
    const uint32_t chunk_i1 = 128 / type_size; // 64

    uint64_t qt = HAP_perf_get_qtimer_count();

    const uint32_t ir0 = cctx->dr * ith;
    const uint32_t ir1 = MIN(ir0 + cctx->dr, cctx->nr); // iterating over ne2 * ne3

    dma_queue * dma_queue = octx->ctx->dma[ith];
    uint8_t * src1_vtcm = octx->src1_spad.data + (ith * octx->src1_spad.size_per_thread);
    uint8_t * dst_vtcm  = octx->dst_spad.data  + (ith * octx->dst_spad.size_per_thread);

    int16_t offsets[64] __attribute__((aligned(128)));
    for (int k = 0; k < 64; k++) {
        offsets[k] = k * 128; // row size in VTCM is exactly 128 bytes
    }
    HVX_Vector v_off = hvx_vmem(offsets);

    for (uint32_t r = ir0; r < ir1; ++r) {
        const uint32_t i2 = fastmodulo(r, ne2, &cctx->div_ne2);
        const uint32_t i3 = fastdiv(r, &cctx->div_ne2);

        for (uint32_t i1_blk = 0; i1_blk < ne11; i1_blk += chunk_i1) {
            uint32_t cur_ne11 = MIN(chunk_i1, ne11 - i1_blk);

            // DMA src1 to VTCM
            const char * x1 = (const char *)src1->data + i1_blk*nb11 + i2*nb12 + i3*nb13;
            dma_queue_push(dma_queue, dma_make_ptr(src1_vtcm, x1), 128, nb10, cur_ne11 * type_size, ne10);
            dma_queue_pop(dma_queue);

            for (uint32_t i0_blk = 0; i0_blk < ne10; i0_blk += chunk_i1) {
                uint32_t cur_ne10 = MIN(chunk_i1, ne10 - i0_blk);
                for (uint32_t c = 0; c < cur_ne11; ++c) {
                    uint8_t * src_ptr = src1_vtcm + i0_blk * 128 + c * type_size;
                    uint8_t * dst_ptr = dst_vtcm  + c * (ne10 * type_size) + i0_blk * type_size;
                    HVX_Vector v = Q6_V_vzero();
                    Q6_vgather_ARMVh(&v, (Word32)src_ptr, 128, v_off);
                    // Write 64 elements. If cur_ne10 < 64, padding protects next rows.
                    hvx_vec_store_u(dst_ptr, type_size * cur_ne10, v);
                }
            }

            // DMA transposed VTCM back to DDR dst
            char * y1 = (char *)dst->data + i1_blk*nb1 + i2*nb2 + i3*nb3 + ne00 * nb0;
            dma_queue_push(dma_queue, dma_make_ptr(y1, dst_vtcm), nb1, ne10 * type_size, ne10 * type_size, cur_ne11);
            dma_queue_pop(dma_queue);
        }
    }

    qt = HAP_perf_qtimer_count_to_us(HAP_perf_get_qtimer_count() - qt);
    FARF(HIGH, "concat-fast-src0-vgather-f16 %d/%d: usec %u\n", ith, nth, (unsigned) qt);
}

static void concat_thread_dim0_fast_src0(unsigned int nth, unsigned int ith, void *data) {
    struct htp_concat_context * cctx = (struct htp_concat_context *)data;
    struct htp_ops_context * octx = cctx->octx;

    concat_preamble;

    const size_t type_size = cctx->type_size;

    uint64_t qt = HAP_perf_get_qtimer_count();

    const uint32_t ir0 = cctx->dr * ith;
    const uint32_t ir1 = MIN(ir0 + cctx->dr, cctx->nr);

    for (uint32_t r = ir0; r < ir1; ++r) {
        const uint32_t i1 = fastmodulo(r, ne1, &cctx->div_ne1);
        uint32_t rem = fastdiv(r, &cctx->div_ne1);
        const uint32_t i2 = fastmodulo(rem, ne2, &cctx->div_ne2);
        const uint32_t i3 = fastdiv(rem, &cctx->div_ne2);

        char * y1 = (char *)dst->data + i1*nb1 + i2*nb2 + i3*nb3 + ne00 * nb0;

        if (nb10 == type_size && nb0 == type_size) {
            const char * x1 = (const char *)src1->data + i1*nb11 + i2*nb12 + i3*nb13;
            hvx_copy_uu((uint8_t*)y1, (const uint8_t*)x1, src1->ne[0], type_size);
        } else {
            for (uint32_t i0 = 0; i0 < src1->ne[0]; i0++) {
                const char * x = (const char *)src1->data + i0*nb10 + i1*nb11 + i2*nb12 + i3*nb13;
                char * y = y1 + i0*nb0;
                memcpy(y, x, type_size);
            }
        }
    }

    qt = HAP_perf_qtimer_count_to_us(HAP_perf_get_qtimer_count() - qt);
    FARF(HIGH, "concat-fast-src0 %d/%d: dim 0 x %ux%ux%ux%u / %ux%ux%ux%u -> %ux%ux%ux%u usec %u\n", ith, nth,
         ne00, ne01, ne02, ne03, src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3], ne0, ne1, ne2, ne3, (unsigned) qt);
}


static void concat_thread(unsigned int nth, unsigned int ith, void *data) {
    struct htp_concat_context * cctx = (struct htp_concat_context *)data;
    struct htp_ops_context * octx = cctx->octx;

    concat_preamble;

    const int32_t dim = cctx->dim;
    const size_t type_size = cctx->type_size;

    uint64_t qt = HAP_perf_get_qtimer_count();

    const uint32_t ir0 = cctx->dr * ith;
    const uint32_t ir1 = MIN(ir0 + cctx->dr, cctx->nr);

    uint64_t o[4] = {0, 0, 0, 0};
    o[dim] = src0->ne[dim];

    for (uint32_t r = ir0; r < ir1; ++r) {
        const uint32_t i1 = fastmodulo(r, ne1, &cctx->div_ne1);
        uint32_t rem = fastdiv(r, &cctx->div_ne1);
        const uint32_t i2 = fastmodulo(rem, ne2, &cctx->div_ne2);
        const uint32_t i3 = fastdiv(rem, &cctx->div_ne2);

        if (dim == 0) {
            char * y0 = (char *)dst->data + i1*nb1 + i2*nb2 + i3*nb3;
            char * y1 = y0 + ne00 * nb0;

            if (nb00 == type_size && nb0 == type_size) {
                const char * x0 = (const char *)src0->data + i1*nb01 + i2*nb02 + i3*nb03;
                hvx_copy_uu((uint8_t*)y0, (const uint8_t*)x0, ne00, type_size);
            } else {
                for (uint32_t i0 = 0; i0 < ne00; i0++) {
                    const char * x = (const char *)src0->data + i0*nb00 + i1*nb01 + i2*nb02 + i3*nb03;
                    char * y = y0 + i0*nb0;
                    memcpy(y, x, type_size);
                }
            }

            if (nb10 == type_size && nb0 == type_size) {
                const char * x1 = (const char *)src1->data + i1*nb11 + i2*nb12 + i3*nb13;
                hvx_copy_uu((uint8_t*)y1, (const uint8_t*)x1, src1->ne[0], type_size);
            } else {
                for (uint32_t i0 = 0; i0 < src1->ne[0]; i0++) {
                    const char * x = (const char *)src1->data + i0*nb10 + i1*nb11 + i2*nb12 + i3*nb13;
                    char * y = y1 + i0*nb0;
                    memcpy(y, x, type_size);
                }
            }
        } else {
            // dim != 0
            const char * x_base;
            uint32_t ne0_val;
            uint32_t nb0_val;

            if (i1 < ne01 && i2 < ne02 && i3 < ne03) {
                x_base = (const char *)src0->data + i1*nb01 + i2*nb02 + i3*nb03;
                ne0_val = ne00;
                nb0_val = nb00;
            } else {
                x_base = (const char *)src1->data + (i1 - o[1])*nb11 + (i2 - o[2])*nb12 + (i3 - o[3])*nb13;
                ne0_val = src1->ne[0];
                nb0_val = nb10;
            }
            char * y = (char *)dst->data + i1*nb1 + i2*nb2 + i3*nb3;

            if (nb0_val == type_size && nb0 == type_size) {
                hvx_copy_uu((uint8_t*)y, (const uint8_t*)x_base, ne0_val, type_size);
            } else {
                for (uint32_t i0 = 0; i0 < ne0_val; i0++) {
                    const char * x = x_base + i0*nb0_val;
                    char * y_elem = y + i0*nb0;
                    memcpy(y_elem, x, type_size);
                }
            }
        }
    }

    qt = HAP_perf_qtimer_count_to_us(HAP_perf_get_qtimer_count() - qt);
    FARF(HIGH, "concat %d/%d: dim %d x %ux%ux%ux%u / %ux%ux%ux%u -> %ux%ux%ux%u usec %u\n", ith, nth,
         dim, ne00, ne01, ne02, ne03, src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3], ne0, ne1, ne2, ne3, (unsigned) qt);
}

int op_concat(struct htp_ops_context * octx) {
    concat_preamble;

    const int32_t dim = octx->op_params[0];

    if (dim < 0 || dim >= 4) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    if (src0->type != src1->type || src0->type != dst->type) {
        return HTP_STATUS_NO_SUPPORT;
    }

    size_t type_size = 0;
    switch(src0->type) {
        case HTP_TYPE_F32: type_size = 4; break;
        case HTP_TYPE_F16: type_size = 2; break;
        case HTP_TYPE_I32: type_size = 4; break;
        default:
            return HTP_STATUS_NO_SUPPORT;
    }

    if (octx->flags & HTP_OPFLAGS_SKIP_COMPUTE) {
        return HTP_STATUS_OK;
    }

    const uint32_t total_rows = ne3 * ne2 * ne1;
    const uint32_t n_threads = MIN(total_rows, octx->n_threads);

    const bool fast_src0 = (dim == 0 && nb00 == type_size && nb0 == type_size && src0->ne[1] == dst->ne[1] && src0->ne[2] == dst->ne[2] && src0->ne[3] == dst->ne[3]);

    if (fast_src0) {
        // Fast path for src0: single DMA push for the entire tensor across rows
        dma_queue * dma_queue = octx->ctx->dma[0];

        // Calculate the contiguous chunk size per row
        size_t row_size = ne00 * type_size;

        // Use the DMA engine to copy all rows at once, using the correct src and dst strides
        dma_queue_push(dma_queue, dma_make_ptr((void*)dst->data, src0->data), nb1, nb01, row_size, total_rows);
        dma_queue_pop(dma_queue);
    }

    bool vgather_fallback = false;
    if (fast_src0 && (nb10 != type_size) && (nb11 == type_size)) {
        if (src0->type == HTP_TYPE_F32 || src0->type == HTP_TYPE_F16) {
            vgather_fallback = true;
        }
    }

    if (vgather_fallback) {
        // chunk_i1 is the number of elements along i1 we process in one VTCM block (128 bytes)
        const uint32_t chunk_i1 = 128 / type_size;

        octx->src1_spad.size_per_thread = hex_round_up(src1->ne[0] * 128, 128); // ne10 * chunk_i1 * type_size
        octx->dst_spad.size_per_thread  = hex_round_up(chunk_i1 * src1->ne[0] * type_size, 128) + 128; // padded by 128 bytes

        octx->src1_spad.size = octx->src1_spad.size_per_thread * n_threads;
        octx->dst_spad.size  = octx->dst_spad.size_per_thread * n_threads;

        size_t spad_size = octx->src1_spad.size + octx->dst_spad.size;

        if (octx->ctx->vtcm_size < spad_size) {
            FARF(ERROR, "concat: current VTCM reservation %zu is too small, needed %zu\n", octx->ctx->vtcm_size, spad_size);
            return HTP_STATUS_VTCM_TOO_SMALL;
        }

        octx->src1_spad.data = octx->ctx->vtcm_base;                        octx->src1_spad.src = NULL;
        octx->dst_spad.data  = octx->src1_spad.data + octx->src1_spad.size; octx->dst_spad.src = NULL;
    }

    struct htp_concat_context cctx;
    cctx.octx = octx;
    cctx.dim = dim;
    cctx.type_size = type_size;
    cctx.nr = total_rows;
    cctx.dr = (total_rows + n_threads - 1) / n_threads;
    cctx.div_ne1 = init_fastdiv_values(ne1);
    cctx.div_ne2 = init_fastdiv_values(ne2);
    cctx.fast_src0 = fast_src0;

    if (vgather_fallback) {
        // vgather worker groups outer loops into dr covering ne2*ne3, since it chunks i1 internally.
        const uint32_t total_outer = ne3 * ne2;
        cctx.nr = total_outer;
        cctx.dr = (total_outer + n_threads - 1) / n_threads;
        if (src0->type == HTP_TYPE_F32 || src0->type == HTP_TYPE_I32) {
            // F32 and I32 both use 4-byte vgather
            worker_pool_run_func(octx->ctx->worker_pool, concat_thread_dim0_fast_src0_vgather_f32, &cctx, n_threads);
        } else {
            worker_pool_run_func(octx->ctx->worker_pool, concat_thread_dim0_fast_src0_vgather_f16, &cctx, n_threads);
        }
    } else if (fast_src0) {
        worker_pool_run_func(octx->ctx->worker_pool, concat_thread_dim0_fast_src0, &cctx, n_threads);
    } else {
        worker_pool_run_func(octx->ctx->worker_pool, concat_thread, &cctx, n_threads);
    }

    return HTP_STATUS_OK;
}
