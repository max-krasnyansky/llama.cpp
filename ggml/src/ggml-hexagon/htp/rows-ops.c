#include "htp-ops.h"
#include "hvx-utils.h"
#include "ops-utils.h"

// dst[i] = src0[src1[i]]
// src0: data
// src1: indices (int32)
// dst: data

static int op_get_rows_f32(struct htp_ops_context * octx) {
    const float * src0 = (const float *)octx->src0.data;
    const int32_t * indices = (const int32_t *)octx->src1.data;
    float * dst = (float *)octx->dst.data;

    // src0 has shape [ne00, ne01, ne02, ne03]
    // src1 has shape [ne10, ne11, ne12, ne13]
    // dst has shape [ne00, ne10, ne11, ne12] (typically)
    // Actually, GGML_OP_GET_ROWS: dst[i, j, k, l] = src0[src1[i, j, k, l], ...]
    // src0 is [row_size, n_rows, ...]
    // src1 is [n_indices, ...]
    // dst is [row_size, n_indices, ...]

    // We iterate over src1 elements (indices)
    // For each index idx = src1[i], we copy row src0[:, idx] to dst[:, i]

    size_t row_size = octx->src0.ne[0];
    size_t n_indices = octx->src1.ne[0] * octx->src1.ne[1] * octx->src1.ne[2] * octx->src1.ne[3]; // Total number of indices

    // Check if src0 is contiguous in dim0.
    // Assuming src0.nb[0] == sizeof(float)

    size_t stride_row = octx->src0.nb[1] / sizeof(float); // Stride between rows in elements

    for (size_t i = 0; i < n_indices; ++i) {
        int32_t idx = indices[i];
        const float * src_row = src0 + idx * stride_row;
        float * dst_row = dst + i * row_size;

        // Copy row using HVX
        size_t vec_size = 128 / sizeof(float);
        size_t j = 0;
        for (; j + vec_size <= row_size; j += vec_size) {
            HVX_Vector v = q6op_V_vldu_A((const HVX_Vector *)&src_row[j]);
            q6op_V_vstu_A((HVX_Vector *)&dst_row[j], v);
        }
        for (; j < row_size; ++j) {
            dst_row[j] = src_row[j];
        }
    }

    return HTP_STATUS_OK;
}

static int op_get_rows_f16(struct htp_ops_context * octx) {
    const _Float16 * src0 = (const _Float16 *)octx->src0.data;
    const int32_t * indices = (const int32_t *)octx->src1.data;
    _Float16 * dst = (_Float16 *)octx->dst.data;

    size_t row_size = octx->src0.ne[0];
    size_t n_indices = octx->src1.ne[0] * octx->src1.ne[1] * octx->src1.ne[2] * octx->src1.ne[3];
    size_t stride_row = octx->src0.nb[1] / sizeof(_Float16);

    for (size_t i = 0; i < n_indices; ++i) {
        int32_t idx = indices[i];
        const _Float16 * src_row = src0 + idx * stride_row;
        _Float16 * dst_row = dst + i * row_size;

        size_t vec_size = 128 / sizeof(_Float16);
        size_t j = 0;
        for (; j + vec_size <= row_size; j += vec_size) {
            HVX_Vector v = q6op_V_vldu_A((const HVX_Vector *)&src_row[j]);
            q6op_V_vstu_A((HVX_Vector *)&dst_row[j], v);
        }
        for (; j < row_size; ++j) {
            dst_row[j] = src_row[j];
        }
    }

    return HTP_STATUS_OK;
}

int op_get_rows(struct htp_ops_context * octx) {
    if (octx->src0.type == HTP_TYPE_F32) return op_get_rows_f32(octx);
    if (octx->src0.type == HTP_TYPE_F16) return op_get_rows_f16(octx);
    return HTP_STATUS_NO_SUPPORT;
}

// dst = src0
// dst[src1[i]] = src2[i]
// src0: original data (copied to dst)
// src1: indices
// src2: new data for rows
// dst: result

static int op_set_rows_f32(struct htp_ops_context * octx) {
    // 1. Copy src0 to dst
    const float * src0 = (const float *)octx->src0.data;
    float * dst = (float *)octx->dst.data;
    size_t total_size = octx->src0.ne[0] * octx->src0.ne[1] * octx->src0.ne[2] * octx->src0.ne[3];

    // HVX copy src0 -> dst
    size_t vec_size = 128 / sizeof(float);
    size_t k = 0;
    for (; k + vec_size <= total_size; k += vec_size) {
        HVX_Vector v = q6op_V_vldu_A((const HVX_Vector *)&src0[k]);
        q6op_V_vstu_A((HVX_Vector *)&dst[k], v);
    }
    for (; k < total_size; ++k) dst[k] = src0[k];

    // 2. Update rows
    const int32_t * indices = (const int32_t *)octx->src1.data;
    const float * src2 = (const float *)octx->src2.data; // The update values

    size_t row_size = octx->src0.ne[0]; // Elements per row
    size_t n_indices = octx->src1.ne[0] * octx->src1.ne[1] * octx->src1.ne[2] * octx->src1.ne[3];
    size_t stride_row = octx->dst.nb[1] / sizeof(float);

    for (size_t i = 0; i < n_indices; ++i) {
        int32_t idx = indices[i];
        float * dst_row = dst + idx * stride_row;
        const float * update_row = src2 + i * row_size; // src2 rows correspond to indices

        size_t j = 0;
        for (; j + vec_size <= row_size; j += vec_size) {
            HVX_Vector v = q6op_V_vldu_A((const HVX_Vector *)&update_row[j]);
            q6op_V_vstu_A((HVX_Vector *)&dst_row[j], v);
        }
        for (; j < row_size; ++j) {
            dst_row[j] = update_row[j];
        }
    }

    return HTP_STATUS_OK;
}

static int op_set_rows_f16(struct htp_ops_context * octx) {
    const _Float16 * src0 = (const _Float16 *)octx->src0.data;
    _Float16 * dst = (_Float16 *)octx->dst.data;
    size_t total_size = octx->src0.ne[0] * octx->src0.ne[1] * octx->src0.ne[2] * octx->src0.ne[3];

    size_t vec_size = 128 / sizeof(_Float16);
    size_t k = 0;
    for (; k + vec_size <= total_size; k += vec_size) {
        HVX_Vector v = q6op_V_vldu_A((const HVX_Vector *)&src0[k]);
        q6op_V_vstu_A((HVX_Vector *)&dst[k], v);
    }
    for (; k < total_size; ++k) dst[k] = src0[k];

    const int32_t * indices = (const int32_t *)octx->src1.data;
    const _Float16 * src2 = (const _Float16 *)octx->src2.data;

    size_t row_size = octx->src0.ne[0];
    size_t n_indices = octx->src1.ne[0] * octx->src1.ne[1] * octx->src1.ne[2] * octx->src1.ne[3];
    size_t stride_row = octx->dst.nb[1] / sizeof(_Float16);

    for (size_t i = 0; i < n_indices; ++i) {
        int32_t idx = indices[i];
        _Float16 * dst_row = dst + idx * stride_row;
        const _Float16 * update_row = src2 + i * row_size;

        size_t j = 0;
        for (; j + vec_size <= row_size; j += vec_size) {
            HVX_Vector v = q6op_V_vldu_A((const HVX_Vector *)&update_row[j]);
            q6op_V_vstu_A((HVX_Vector *)&dst_row[j], v);
        }
        for (; j < row_size; ++j) {
            dst_row[j] = update_row[j];
        }
    }

    return HTP_STATUS_OK;
}

int op_set_rows(struct htp_ops_context * octx) {
    if (octx->src0.type == HTP_TYPE_F32) return op_set_rows_f32(octx);
    if (octx->src0.type == HTP_TYPE_F16) return op_set_rows_f16(octx);
    return HTP_STATUS_NO_SUPPORT;
}
