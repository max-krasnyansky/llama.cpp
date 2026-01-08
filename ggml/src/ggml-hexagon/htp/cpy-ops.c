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
    struct htp_tensor *src0 = &octx->src0; \
    struct htp_tensor *dst  = &octx->dst;  \
                                       \
    const uint32_t ne00 = src0->ne[0]; \
    const uint32_t ne01 = src0->ne[1]; \
    const uint32_t ne02 = src0->ne[2]; \
    const uint32_t ne03 = src0->ne[3]; \
                                       \
    const uint32_t nb00 = src0->nb[0]; \
    const uint32_t nb01 = src0->nb[1]; \
    const uint32_t nb02 = src0->nb[2]; \
    const uint32_t nb03 = src0->nb[3]; \
                                       \
    const uint32_t  ne0 = dst->ne[0];  \
    const uint32_t  ne1 = dst->ne[1];  \
    const uint32_t  ne2 = dst->ne[2];  \
    const uint32_t  ne3 = dst->ne[3];  \
                                       \
    const uint32_t  nb0 = dst->nb[0];  \
    const uint32_t  nb1 = dst->nb[1];  \
    const uint32_t  nb2 = dst->nb[2];  \
    const uint32_t  nb3 = dst->nb[3];  \
                                       \
    const uint32_t   nr = ne01;

static int cpy_thread_f32_f32(struct htp_ops_context * octx, const int nth, const int ith) {
    cpy_preamble;

    // parallelize by src0 rows
    const uint32_t dr  = octx->src0_nrows_per_thread;
    const uint32_t ir0 = dr * ith;
    const uint32_t ir1 = (ir0 + dr) < nr ? (ir0 + dr) : nr;

    // row size equal
    if (ne00 == ne0 && (nb00 == sizeof(float)) && (nb0 == sizeof(float))) {
        // copy by rows
        for (uint32_t i03 = 0; i03 < ne03; i03++) {
            for (uint32_t i02 = 0; i02 < ne02; i02++) {
                for (uint32_t i01 = ir0; i01 < ir1; i01++) {
                    uint8_t* dst_ptr  = (uint8_t*) dst->data  + i01*nb1  + i02*nb2  + i03*nb3;
                    uint8_t* src0_ptr = (uint8_t*) src0->data + i01*nb01 + i02*nb02 + i03*nb03;
                    hvx_copy_fp32_uu(dst_ptr, src0_ptr, ne00);
                }
            }
        }
        return HTP_STATUS_OK;
    }

    // dst counters
    int64_t k10 = 0;
    int64_t i11 = 0;
    int64_t i12 = 0;
    int64_t i13 = 0;

    // number of blocks in a row
    const int64_t nk00 = ne00 / 1;
    const int64_t nk0  = ne0  / 1;

    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            k10 += nk00 * ir0;
            while (k10 >= nk0) {
                k10 -= nk0;
                if (++i11 == ne1) {
                    i11 = 0;
                    if (++i12 == ne2) {
                        i12 = 0;
                        if (++i13 == ne3) {
                            i13 = 0;
                        }
                    }
                }
            }
            for (int64_t i01 = ir0; i01 < ir1; i01++) {
                for (int64_t k00 = 0; k00 < nk00; k00++) {
                    const char * src0_ptr = ((char *) src0->data + k00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                          char * dst_ptr  = ((char *)  dst->data + k10*nb0  + i11*nb1  + i12*nb2  + i13*nb3);
                    memcpy(dst_ptr, src0_ptr, 4);

                    if (++k10 == nk0) {
                        k10 = 0;
                        if (++i11 == ne1) {
                            i11 = 0;
                            if (++i12 == ne2) {
                                i12 = 0;
                                if (++i13 == ne3) {
                                    i13 = 0;
                                }
                            }
                        }
                    }
                }
            }
            k10 += nk00 * (ne01 - ir1);
            while (k10 >= nk0) {
                k10 -= nk0;
                if (++i11 == ne1) {
                    i11 = 0;
                    if (++i12 == ne2) {
                        i12 = 0;
                        if (++i13 == ne3) {
                            i13 = 0;
                        }
                    }
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

    const bool reshape = (ne00 != ne0 || ne01 != ne1 || ne02 != ne2 || ne03 != ne3);
    if (reshape) {
        // Reshape mode needs different fastdivs
        octx->flags |= 0x80000000;
        octx->cpy_rshp_div_n0      = init_fastdiv_values(ne00);
        octx->cpy_rshp_div_n1n0    = init_fastdiv_values(ne01 * ne00);
        octx->cpy_rshp_div_n2n1n0  = init_fastdiv_values(ne02 * ne01 * ne00);
    } else {
        octx->cpy_div_ne01 = init_fastdiv_values(ne01);
        octx->cpy_div_ne02 = init_fastdiv_values(ne02);
        octx->cpy_div_ne03 = init_fastdiv_values(ne03);
    }

    const uint32_t n_jobs = MIN(nr, octx->n_threads);
    octx->src0_nrows_per_thread = (nr + n_jobs - 1) / n_jobs;

    worker_pool_run_func(octx->ctx->worker_pool, cpy_work_f32_f32, octx, n_jobs);

    return HTP_STATUS_OK;
}
