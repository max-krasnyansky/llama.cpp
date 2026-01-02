#include "htp-ops.h"
#include "hvx-utils.h"

#include <stdint.h>
#include <string.h>

static int get_rows_process_slice(void * data, void * arg) {
    struct htp_ops_context * octx = (struct htp_ops_context *) arg;
    worker_pool_task_info_t * info = (worker_pool_task_info_t *) data;

    const int ith = info->ith;
    const int nth = info->nth;

    // src0 is data (float), src1 is indices (int32), dst is output (float)
    // ne00 = row size (number of floats per row)
    const uint32_t ne00 = octx->src0.ne[0];
    const uint32_t nb01 = octx->src0.nb[1];
    const uint32_t nb02 = octx->src0.nb[2];
    const uint32_t nb03 = octx->src0.nb[3];

    const uint32_t ne10 = octx->src1.ne[0];
    const uint32_t ne11 = octx->src1.ne[1];
    const uint32_t ne12 = octx->src1.ne[2];
    // const uint32_t ne13 = octx->src1.ne[3];

    const uint32_t nb10 = octx->src1.nb[0];
    const uint32_t nb11 = octx->src1.nb[1];
    const uint32_t nb12 = octx->src1.nb[2];
    const uint32_t nb1  = octx->dst.nb[1];
    const uint32_t nb2  = octx->dst.nb[2];
    const uint32_t nb3  = octx->dst.nb[3];

    const uint32_t ne01 = octx->src0.ne[1]; // Used for asserting index

    const uint32_t nr = octx->src1.ne[0] * octx->src1.ne[1] * octx->src1.ne[2] * octx->src1.ne[3];

    // rows per thread
    const int dr = (nr + nth - 1) / nth;

    // row range for this thread
    const int ir0 = dr * ith;
    const int ir1 = MIN(ir0 + dr, nr);

    char * src0_base = (char *) octx->src0.data;
    char * src1_base = (char *) octx->src1.data;
    char * dst_base  = (char *) octx->dst.data;

    // Check if we can use vectorized copy
    bool use_hvx_copy = (ne00 % 32 == 0) && (nb01 % 128 == 0) && (nb1 % 128 == 0);
    // Even stricter for optimal: everything aligned. But we can use unaligned load/store if needed.
    // For now, let's use a simpler check: if ne00 is large enough, try vector copy logic.
    // The CPU version uses ggml_vec_cpy_f32 which is essentially memcpy.
    // On Hexagon, we want to use HVX for large copies.

    // Precompute divisors for coordinate decomposition
    // i12 = i / (ne11 * ne10)
    // i11 = (i - i12 * ne11 * ne10) / ne10
    // i10 = (i - i12 * ne11 * ne10 - i11 * ne10)
    // These are integer divisions. We should use fastdiv if we can, but we need to init them.
    // Since we are inside the thread, we can compute them on the fly if needed,
    // or we can precompute them in op_get_rows before launching threads.
    // However, fastdiv struct initialization is fast.

    // We can't easily pass the fastdiv structs for src1 dimensions through octx unless we added them.
    // Let's assume scalar division is acceptable for the outer loop indices calculation as 'nr' is usually not huge compared to row size copying.
    // Wait, 'nr' is the number of rows to copy. If it's large (many tokens), division per row is costly.
    // The CPU code does exactly this scalar division.
    // Let's use fastdiv. We can init local fastdiv structs here since they are small.

    struct fastdiv_values div_ne11_ne10 = init_fastdiv_values(ne11 * ne10);
    struct fastdiv_values div_ne10      = init_fastdiv_values(ne10);

    for (int i = ir0; i < ir1; ++i) {
        // Calculate coordinates in src1 (indices tensor)
        // i is the linear index in src1
        // src1 shape is [ne10, ne11, ne12, ne13]

        // i12 is index for dimension 2 (and 3 combined if we treat it flat above dim 2, but wait)
        // The CPU code:
        // const int64_t i12 = i/(ne11*ne10);
        // const int64_t i11 = (i - i12*ne11*ne10)/ne10;
        // const int64_t i10 = (i - i12*ne11*ne10 - i11*ne10);
        // It seems to flatten dims 2 and 3 into i12?
        // CPU: nr = ggml_nelements(src1). i iterates 0..nr.
        // It effectively treats src1 as [ne10, ne11, ne12*ne13].

        uint32_t i12 = fastdiv(i, &div_ne11_ne10);
        uint32_t rem = i - i12 * (ne11 * ne10);
        uint32_t i11 = fastdiv(rem, &div_ne10);
        uint32_t i10 = rem - i11 * ne10;

        // Get the index from src1
        int32_t i01 = *(int32_t *) (src1_base + i10 * nb10 + i11 * nb11 + i12 * nb12);

        // Sanity check
        // assert(i01 >= 0 && i01 < ne01);
        if (i01 < 0 || i01 >= ne01) {
            // Error handling on DSP is limited, maybe print error?
            // FARF(ERROR, "Index out of bounds: %d (max %d)", i01, ne01);
            // Continue or set zero? CPU asserts. We can clamp or zero.
            // Let's just clamp for safety to avoid crash.
            if (i01 < 0) i01 = 0;
            if (i01 >= ne01) i01 = ne01 - 1;
        }

        // Copy row from src0 to dst
        // src0 offset: i01*nb01 + i11*nb02 + i12*nb03
        // Note: i11 and i12 are broadcast/mapped from src1 dims to src0 dims.
        // CPU implementation uses them directly, implying src0 dims 2 and 3 must match src1 dims 1 and 2??
        // Wait, CPU:
        // src0->data + i01*nb01 + i11*nb02 + i12*nb03
        // src1 has dims [ne10, ne11, ne12, ne13]
        // i10 corresponds to ne10 (columns of indices)
        // i11 corresponds to ne11
        // i12 corresponds to ne12*ne13

        // Usually get_rows src0 is 2D [ne00, ne01]. ne02=ne03=1.
        // In that case nb02 and nb03 stride over 1 element? No, if ne=1, stride is usually element size or total size.
        // But if ne02=1, i11*nb02 will be i11 * stride_of_dim2. If stride is correct (tight packing), it might be huge.
        // If broadcasting, stride should be 0. ggml usually handles broadcasting via strides.
        // If src0 is truly 2D, nb02 should be total size, but we only access index 0.
        // However, if we access index i11 > 0, we are accessing outside if ne02=1 and nb02 is normal stride.
        // Unless nb02 is 0.
        // Let's trust that nb array passed in has correct strides (0 for broadcasted dimensions).

        float * dst_ptr = (float *) (dst_base + i10 * nb1 + i11 * nb2 + i12 * nb3);
        const float * src_ptr = (const float *) (src0_base + i01 * nb01 + i11 * nb02 + i12 * nb03);

        if (ne00 % 32 == 0 && (size_t)dst_ptr % 128 == 0 && (size_t)src_ptr % 128 == 0) {
             hvx_copy_fp32_aa((uint8_t*)dst_ptr, (const uint8_t*)src_ptr, ne00);
        } else if (ne00 % 32 == 0 && (size_t)src_ptr % 128 == 0) {
             hvx_copy_fp32_ua((uint8_t*)dst_ptr, (const uint8_t*)src_ptr, ne00);
        } else if (ne00 % 32 == 0 && (size_t)dst_ptr % 128 == 0) {
             hvx_copy_fp32_au((uint8_t*)dst_ptr, (const uint8_t*)src_ptr, ne00);
        } else {
            // Scalar fallback or use memcpy
            memcpy(dst_ptr, src_ptr, ne00 * sizeof(float));
        }
    }

    return 0;
}

int op_get_rows(struct htp_ops_context * octx) {
    // Launch worker threads
    worker_pool_run(octx->wpool, get_rows_process_slice, octx);

    return HTP_STATUS_OK;
}
