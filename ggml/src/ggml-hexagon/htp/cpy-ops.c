#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#ifdef HTP_DEBUG
#    define FARF_HIGH 1
#endif
#include <HAP_farf.h>
#include <HAP_mem.h>
#include <HAP_perf.h>
#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <math.h>
#include <string.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-msg.h"
#include "htp-ops.h"
#include "hvx-utils.h"
#include "ops-utils.h"

#define cpy_preamble \
    const uint32_t ne00 = octx->src0.ne[0]; \
    const uint32_t ne01 = octx->src0.ne[1]; \
    const uint32_t ne02 = octx->src0.ne[2]; \
    const uint32_t ne03 = octx->src0.ne[3]; \
                                            \
    const uint32_t nb01 = octx->src0.nb[1]; \
    const uint32_t nb02 = octx->src0.nb[2]; \
    const uint32_t nb03 = octx->src0.nb[3]; \
                                            \
    const uint32_t ne0 = octx->dst.ne[0];   \
    const uint32_t ne1 = octx->dst.ne[1];   \
    const uint32_t ne2 = octx->dst.ne[2];   \
    const uint32_t ne3 = octx->dst.ne[3];   \
                                            \
    const uint32_t nb1 = octx->dst.nb[1];   \
    const uint32_t nb2 = octx->dst.nb[2];   \
    const uint32_t nb3 = octx->dst.nb[3];   \
                                            \
    const uint32_t nr  = ne1;

static int cpy_thread_f32_f32(struct htp_ops_context * octx, const int nth, const int ith) {
    cpy_preamble;

    // parallelize by rows of dst
    const uint32_t dr  = octx->src0_nrows_per_thread;
    const uint32_t ir0 = dr * ith;
    const uint32_t ir1 = (ir0 + dr < nr) ? (ir0 + dr) : nr;

    // Check if we can do a simple copy (ne00 == ne0, contiguous row copy)
    // If not, we might need element-wise loop, but hvx_copy_fp32_uu handles arbitrary size n.
    // The main complexity is coordinate mapping for broadcasting.

    // If src0 dims are 1, we broadcast.
    // i_src = i_dst % src_dim. If src_dim == 1, i_src = 0.
    // fastmodulo handles (i_dst, src_dim, div).
    // If src_dim == dst_dim, i_src = i_dst (no modulo needed if they match).
    // But generalized: i_src = fastmodulo(i_dst, src_dim, div)

    // Optimization: if dim matches, avoid modulo.
    // If dim is 1, index is always 0.

    const bool broadcast_1 = (ne01 != ne1);
    const bool broadcast_2 = (ne02 != ne2);
    const bool broadcast_3 = (ne03 != ne3);

    // Assuming we iterate dst coordinates
    for (uint32_t i3 = 0; i3 < ne3; ++i3) {
        // Map i3 to src0 coord
        uint32_t i03 = i3;
        if (broadcast_3) {
            // if ne03 == 1, i03 = 0. else i03 = i3 % ne03.
            // But ggml broadcasting rules usually imply either equal or 1.
            // If they are not equal and not 1, it's invalid unless it's repeat, but CPY supports standard broadcasting.
            i03 = (ne03 == 1) ? 0 : fastmodulo(i3, ne03, &octx->cpy_div_ne03);
        }

        for (uint32_t i2 = 0; i2 < ne2; ++i2) {
            uint32_t i02 = i2;
            if (broadcast_2) {
                i02 = (ne02 == 1) ? 0 : fastmodulo(i2, ne02, &octx->cpy_div_ne02);
            }

            for (uint32_t i1 = ir0; i1 < ir1; ++i1) {
                uint32_t i01 = i1;
                if (broadcast_1) {
                    i01 = (ne01 == 1) ? 0 : fastmodulo(i1, ne01, &octx->cpy_div_ne01);
                }

                // If ne00 == ne0, we copy the whole row.
                // If ne00 == 1 and ne0 > 1, we broadcast the scalar to the row.
                // Other cases: technically possible but rare in simple CPY.
                // Let's support ne00 == ne0 and ne00 == 1.

                const uintptr_t src0_ptr = octx->src0.data + i01*nb01 + i02*nb02 + i03*nb03;
                const uintptr_t dst_ptr  = octx->dst.data  + i1*nb1   + i2*nb2   + i3*nb3;

                if (ne00 == ne0) {
                    hvx_copy_fp32_uu((uint8_t *)dst_ptr, (const uint8_t *)src0_ptr, ne0 * sizeof(float));
                } else if (ne00 == 1) {
                    hvx_bcast_fp32_u((uint8_t *)dst_ptr, *(const float *)src0_ptr, ne0);
                } else {
                    // General case (repeat/tile)? Not implemented for now, fallback or partial copy?
                    // GGML CPY usually expects broadcasting rules.
                    // If ne00 != 1 and ne00 != ne0, it might be a partial copy or repeat.
                    // For now, support only broadcast 1->N or copy N->N.
                    return HTP_STATUS_NO_SUPPORT;
                }
            }
        }
    }

    return HTP_STATUS_OK;
}

static void cpy_work_f32_f32(unsigned int n, unsigned int i, void *data) {
    cpy_thread_f32_f32((struct htp_ops_context *) data, n, i);
}

int op_cpy(struct htp_ops_context * octx) {
    cpy_preamble;

    if (octx->src0.type != HTP_TYPE_F32) {
        return HTP_STATUS_NO_SUPPORT;
    }

    if (octx->dst.type != HTP_TYPE_F32) {
        return HTP_STATUS_NO_SUPPORT;
    }

    if (octx->flags & HTP_OPFLAGS_SKIP_COMPUTE) {
        return HTP_STATUS_OK;
    }

    octx->cpy_div_ne01 = init_fastdiv_values(ne01);
    octx->cpy_div_ne02 = init_fastdiv_values(ne02);
    octx->cpy_div_ne03 = init_fastdiv_values(ne03);

    const uint32_t n_jobs = MIN(nr, octx->n_threads);
    octx->src0_nrows_per_thread = (nr + n_jobs - 1) / n_jobs;

    worker_pool_run_func(octx->ctx->worker_pool, cpy_work_f32_f32, octx, n_jobs);

    return HTP_STATUS_OK;
}
