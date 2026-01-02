#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include <stdio.h>

#include "htp-ops.h"
#include "hvx-utils.h"

static int op_set_rows_worker(struct htp_ops_context * octx, const int ith, const int nth) {
    // src0 = values [nc, nr, ne2, ne3]
    // src1 = indices [nr, ne2, ne3, 1] - values are int32 or int64
    // dst = destination [ne0, ne1, ne2, ne3] - ne0 == nc

    const uint32_t ne00 = octx->src0.ne[0]; // row size
    const uint32_t ne01 = octx->src0.ne[1]; // num rows
    const uint32_t ne02 = octx->src0.ne[2];
    const uint32_t ne03 = octx->src0.ne[3];

    const uint32_t ne10 = octx->src1.ne[0]; // num rows
    const uint32_t ne11 = octx->src1.ne[1];
    const uint32_t ne12 = octx->src1.ne[2];

    const uint32_t nb00 = octx->src0.nb[0];
    const uint32_t nb01 = octx->src0.nb[1];
    const uint32_t nb02 = octx->src0.nb[2];
    const uint32_t nb03 = octx->src0.nb[3];

    const uint32_t nb10 = octx->src1.nb[0];
    const uint32_t nb11 = octx->src1.nb[1];
    const uint32_t nb12 = octx->src1.nb[2];

    const uint32_t nb1 = octx->dst.nb[1];
    const uint32_t nb2 = octx->dst.nb[2];
    const uint32_t nb3 = octx->dst.nb[3];

    const uint32_t dst_ne1 = octx->dst.ne[1];

    // parallelize by rows of src0
    const uint32_t nr = ne01;
    const uint32_t dr = (nr + nth - 1) / nth;
    const uint32_t ir0 = dr * ith;
    const uint32_t ir1 = (ir0 + dr < nr) ? (ir0 + dr) : nr;

    const bool is_i32 = (octx->src1.type == HTP_TYPE_I32);

    for (uint32_t i03 = 0; i03 < ne03; ++i03) {
        for (uint32_t i02 = 0; i02 < ne02; ++i02) {
            for (uint32_t i = ir0; i < ir1; ++i) {
                // map src0 row index to src1 index
                // logic from CPU implementation:
                // i12 = i03 % ne12
                // i11 = i02 % ne11
                // i10 = i

                const uint32_t i12 = fastmodulo(i03, ne12, &octx->set_rows_div_ne12);
                const uint32_t i11 = fastmodulo(i02, ne11, &octx->set_rows_div_ne11);
                const uint32_t i10 = i;

                const uintptr_t src1_addr = octx->src1.data + i10*nb10 + i11*nb11 + i12*nb12;

                int64_t i1;
                if (is_i32) {
                    i1 = *(int32_t *)src1_addr;
                } else {
                    i1 = *(int64_t *)src1_addr;
                }

                if (i1 < 0 || i1 >= dst_ne1) {
                    // ignore invalid indices
                    continue;
                }

                const uintptr_t src0_ptr = octx->src0.data + i*nb01 + i02*nb02 + i03*nb03;
                const uintptr_t dst_ptr  = octx->dst.data  + i1*nb1 + i02*nb2  + i03*nb3;

                // copy row
                hvx_copy_fp32_ua((uint8_t *)dst_ptr, (const uint8_t *)src0_ptr, ne00 * sizeof(float));
            }
        }
    }

    return HTP_STATUS_OK;
}

static void op_set_rows_worker_wrapper(void * data, int ith, int nth) {
    op_set_rows_worker((struct htp_ops_context *) data, ith, nth);
}

int op_set_rows(struct htp_ops_context * octx) {
    if (octx->src0.type != HTP_TYPE_F32 || octx->dst.type != HTP_TYPE_F32) {
        return HTP_STATUS_NO_SUPPORT;
    }

    if (octx->src1.type != HTP_TYPE_I32 && octx->src1.type != HTP_TYPE_I64) {
        return HTP_STATUS_NO_SUPPORT;
    }

    worker_pool_run(octx->ctx->worker_pool, op_set_rows_worker_wrapper, octx);

    return HTP_STATUS_OK;
}
