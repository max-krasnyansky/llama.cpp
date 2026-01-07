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

    const bool reshape = (octx->flags & 0x80000000) != 0;

    if (reshape) {
        // Reshape mode: linear copy by unraveling dst index to src coords
        // Check if src is contiguous
        const bool src_cont = (nb00 == sizeof(float)) &&
                              (nb01 == ne00 * sizeof(float)) &&
                              (nb02 == ne01 * ne00 * sizeof(float)) &&
                              (nb03 == ne02 * ne01 * ne00 * sizeof(float));

        for (uint32_t i3 = 0; i3 < ne3; ++i3) {
            for (uint32_t i2 = 0; i2 < ne2; ++i2) {
                for (uint32_t i1 = ir0; i1 < ir1; ++i1) {
                    const uintptr_t dst_row_ptr = octx->dst.data + i1*nb1 + i2*nb2 + i3*nb3;

                    // Linear index of the start of this row
                    uint32_t L_start = (i3 * ne2 * ne1 + i2 * ne1 + i1) * ne0;

                    if (src_cont) {
                        // Fast path: src is contiguous
                        const uintptr_t src0_ptr = octx->src0.data + L_start * sizeof(float);
                        hvx_copy_fp32_uu((uint8_t *)dst_row_ptr, (const uint8_t *)src0_ptr, ne0 * sizeof(float));
                    } else {
                        // Slow path: unravel linear index for each element
                        float * d = (float *)dst_row_ptr;
                        for (uint32_t i0 = 0; i0 < ne0; ++i0) {
                            uint32_t L = L_start + i0;

                            // Unravel L to (k3, k2, k1, k0)
                            uint32_t k3 = fastdiv(L, &octx->cpy_rshp_div_n2n1n0);
                            uint32_t rem2 = L - k3 * octx->cpy_rshp_div_n2n1n0.mp; // approximate fix, wait fastmodulo?
                            // Actually fastmodulo(L, divisor, &val) computes L % divisor.
                            // But here divisors are cumulative products.
                            // fastdiv returns quotient.
                            // rem2 = L % (ne02*ne01*ne00).
                            // But fastmodulo needs div struct for the divisor.
                            // We have struct for ne00, ne00*ne01, ne00*ne01*ne02.
                            // L / (ne00*ne01*ne02) = k3.
                            // We don't have struct for that divisor? Yes we added cpy_rshp_div_n2n1n0.
                            // Wait, L % (ne00*ne01*ne02) is not what we want.
                            // We want L - k3 * (ne00*ne01*ne02).
                            // But fastdiv uses precomputed multiplier.
                            // Let's just use fastmodulo if possible.

                            // Let's trust standard unraveling:
                            // k3 = L / (N2*N1*N0)
                            // r2 = L % (N2*N1*N0)
                            // k2 = r2 / (N1*N0)
                            // r1 = r2 % (N1*N0)
                            // k1 = r1 / N0
                            // k0 = r1 % N0

                            // We need div/mod for cumulative sizes.

                            // Recalculate using integer math if sizes are small? No, use fastdiv.
                            // But we need the 'stride' values for multiplication.
                            // struct fastdiv_values doesn't store the divisor 'd'.
                            // We can store them in context? Or recompute?
                            // Or just pass them.
                            // ne00, ne01, ne02 are available.

                            uint32_t N0 = ne00;
                            uint32_t N1N0 = ne00 * ne01;
                            uint32_t N2N1N0 = ne00 * ne01 * ne02;

                            // k3
                            k3 = fastdiv(L, &octx->cpy_rshp_div_n2n1n0);
                            uint32_t r2 = L - k3 * N2N1N0;

                            // k2
                            uint32_t k2 = fastdiv(r2, &octx->cpy_rshp_div_n1n0);
                            uint32_t r1 = r2 - k2 * N1N0;

                            // k1
                            uint32_t k1 = fastdiv(r1, &octx->cpy_rshp_div_n0);
                            uint32_t k0 = r1 - k1 * N0;

                            const uintptr_t src_ptr = octx->src0.data + k0*nb00 + k1*nb01 + k2*nb02 + k3*nb03;
                            d[i0] = *(const float *)src_ptr;
                        }
                    }
                }
            }
        }
        return HTP_STATUS_OK;
    }

    // Broadcast Mode
    const bool broadcast_1 = (ne01 != ne1);
    const bool broadcast_2 = (ne02 != ne2);
    const bool broadcast_3 = (ne03 != ne3);

    // Assuming we iterate dst coordinates
    for (uint32_t i3 = 0; i3 < ne3; ++i3) {
        // Map i3 to src0 coord
        uint32_t i03 = i3;
        if (broadcast_3) {
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

                const uintptr_t src0_ptr = octx->src0.data + i01*nb01 + i02*nb02 + i03*nb03;
                const uintptr_t dst_ptr  = octx->dst.data  + i1*nb1   + i2*nb2   + i3*nb3;

                if (ne00 == ne0) {
                    hvx_copy_fp32_uu((uint8_t *)dst_ptr, (const uint8_t *)src0_ptr, ne0 * sizeof(float));
                } else if (ne00 == 1) {
                    hvx_bcast_fp32_u((uint8_t *)dst_ptr, *(const float *)src0_ptr, ne0);
                } else {
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
