#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <HAP_farf.h>
#include <HAP_perf.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "ggml.h"

#include "hvx-utils.h"
#include "hex-dma.h"

#include "htp-ctx.h"
#include "htp-msg.h"
#include "htp-ops.h"

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

struct htp_argsort_context {
    struct htp_ops_context * octx;
    uint32_t                 nrows_per_thread;
    struct fastdiv_values    div_ne01;
    struct fastdiv_values    div_ne02_ne01;
};

// Scalar sort implementation since std::sort is not available.
// Sorts indices based on values.
static void quicksort_indices_asc(int32_t * indices, const float * data, int left, int right) {
    if (left >= right) return;

    int pivot_idx = indices[(left + right) / 2];
    float pivot = data[pivot_idx];
    int i = left;
    int j = right;

    while (i <= j) {
        while (data[indices[i]] < pivot) i++;
        while (data[indices[j]] > pivot) j--;
        if (i <= j) {
            int32_t tmp = indices[i];
            indices[i] = indices[j];
            indices[j] = tmp;
            i++;
            j--;
        }
    }

    if (left < j) quicksort_indices_asc(indices, data, left, j);
    if (i < right) quicksort_indices_asc(indices, data, i, right);
}

static void quicksort_indices_desc(int32_t * indices, const float * data, int left, int right) {
    if (left >= right) return;

    int pivot_idx = indices[(left + right) / 2];
    float pivot = data[pivot_idx];
    int i = left;
    int j = right;

    while (i <= j) {
        while (data[indices[i]] > pivot) i++;
        while (data[indices[j]] < pivot) j--;
        if (i <= j) {
            int32_t tmp = indices[i];
            indices[i] = indices[j];
            indices[j] = tmp;
            i++;
            j--;
        }
    }

    if (left < j) quicksort_indices_desc(indices, data, left, j);
    if (i < right) quicksort_indices_desc(indices, data, i, right);
}

static void htp_argsort_f32(unsigned int n, unsigned int i, void * data) {
    struct htp_argsort_context * actx = (struct htp_argsort_context *)data;
    struct htp_ops_context * octx = actx->octx;

    // Unpack context
    const struct htp_tensor * src0 = &octx->src0;
    const struct htp_tensor * dst = &octx->dst;

    // Scratchpad memory
    uint8_t * spad = octx->src0_spad.data + octx->src0_spad.size_per_thread * i;

    // Dimensions
    uint32_t ne00 = src0->ne[0];
    uint32_t ne01 = src0->ne[1];
    uint32_t ne02 = src0->ne[2];
    uint32_t ne03 = src0->ne[3];

    uint32_t nb01 = src0->nb[1];
    uint32_t nb02 = src0->nb[2];
    uint32_t nb03 = src0->nb[3];

    uint32_t nb1 = dst->nb[1];
    uint32_t nb2 = dst->nb[2];
    uint32_t nb3 = dst->nb[3];

    // Sort order
    enum ggml_sort_order order = (enum ggml_sort_order) octx->op_params[0];

    // Rows to process
    uint32_t total_rows = ne01 * ne02 * ne03;
    uint32_t rows_per_thread = actx->nrows_per_thread;
    uint32_t start_row = rows_per_thread * i;
    uint32_t end_row = MIN(start_row + rows_per_thread, total_rows);

    // Scratchpad layout:
    // We need space for one row of float data (values) and one row of int32 indices.
    // values: ne00 * sizeof(float)
    // indices: ne00 * sizeof(int32_t)
    // Padded to 128 bytes.

    size_t values_size = hex_round_up(ne00 * sizeof(float), 128);
    float * values_buf = (float *) spad;
    int32_t * indices_buf = (int32_t *) (spad + values_size);

    for (uint32_t r = start_row; r < end_row; r++) {
        // Calculate indices for 3D iteration flattened using fastdiv
        // uint32_t i03 = r / (ne02 * ne01);
        // uint32_t rem = r % (ne02 * ne01);
        // uint32_t i02 = rem / ne01;
        // uint32_t i01 = rem % ne01;

        uint32_t i03 = fastdiv(r, &actx->div_ne02_ne01);
        uint32_t rem = fastmodulo(r, ne02 * ne01, &actx->div_ne02_ne01);
        uint32_t i02 = fastdiv(rem, &actx->div_ne01);
        uint32_t i01 = rem - i02 * ne01;

        uint32_t src_offset = i03 * nb03 + i02 * nb02 + i01 * nb01;
        uint32_t dst_offset = i03 * nb3 + i02 * nb2 + i01 * nb1;

        uint8_t * src_ptr = (uint8_t *) src0->data + src_offset;
        uint8_t * dst_ptr = (uint8_t *) dst->data + dst_offset;

        // Prefetch and Copy row data to VTCM
        hex_l2fetch(src_ptr, ne00 * sizeof(float), ne00 * sizeof(float), 1);

        // Use vector copy if available/efficient, handles unaligned
        hvx_copy_f32_uu((uint8_t*)values_buf, src_ptr, ne00);

        // Initialize indices
        for (uint32_t j = 0; j < ne00; j++) {
            indices_buf[j] = j;
        }

        // Sort indices based on values
        if (order == GGML_SORT_ORDER_ASC) {
            quicksort_indices_asc(indices_buf, values_buf, 0, ne00 - 1);
        } else {
            quicksort_indices_desc(indices_buf, values_buf, 0, ne00 - 1);
        }

        // Copy indices back to DDR
        // Indices are 32-bit integers, effectively same as float for copy purposes size-wise
        hvx_copy_f32_uu(dst_ptr, (const uint8_t *) indices_buf, ne00);
    }
}

int op_argsort(struct htp_ops_context * octx) {
    // Check supported types
    if (octx->src0.type != HTP_TYPE_F32) {
        return HTP_STATUS_NO_SUPPORT;
    }

    // Allocate scratchpad
    // We need 1 row of float + 1 row of int32 per thread.
    uint32_t ne00 = octx->src0.ne[0];
    size_t values_size = hex_round_up(ne00 * sizeof(float), 128);
    size_t indices_size = hex_round_up(ne00 * sizeof(int32_t), 128);
    size_t spad_per_thread = values_size + indices_size;

    // Make sure we round up to 256 for alignment requirements
    spad_per_thread = hex_round_up(spad_per_thread, 256);

    size_t total_spad_size = spad_per_thread * octx->n_threads;

    if (octx->ctx->vtcm_size < total_spad_size) {
        FARF(ERROR, "argsort: VTCM size too small. Needed %zu, have %zu", total_spad_size, octx->ctx->vtcm_size);
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    octx->src0_spad.data = octx->ctx->vtcm_base;
    octx->src0_spad.size = total_spad_size;
    octx->src0_spad.size_per_thread = spad_per_thread;

    FARF(HIGH, "argsort: %ux%ux%ux%u -> %ux%ux%ux%u (0x%x, 0x%x)",
         octx->src0.ne[0], octx->src0.ne[1], octx->src0.ne[2], octx->src0.ne[3],
         octx->dst.ne[0], octx->dst.ne[1], octx->dst.ne[2], octx->dst.ne[3],
         octx->src0.data, octx->dst.data);

    uint32_t total_rows = octx->src0.ne[1] * octx->src0.ne[2] * octx->src0.ne[3];
    uint32_t n_jobs = MIN(total_rows, octx->n_threads);

    struct htp_argsort_context actx;
    actx.octx = octx;
    actx.nrows_per_thread = (total_rows + n_jobs - 1) / n_jobs;
    // Initialize fastdiv values
    actx.div_ne01 = init_fastdiv_values(octx->src0.ne[1]);
    actx.div_ne02_ne01 = init_fastdiv_values(octx->src0.ne[2] * octx->src0.ne[1]);

    // Run jobs
    worker_pool_run_func(octx->ctx->worker_pool, htp_argsort_f32, &actx, n_jobs);

    return HTP_STATUS_OK;
}
