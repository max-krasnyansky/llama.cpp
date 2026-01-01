#include "htp-ops.h"
#include "hvx-utils.h"
#include "ops-utils.h"

// dst = src0

static int op_cpy_f32(struct htp_ops_context * octx) {
    const float * src0 = (const float *)octx->src0.data;
    float * dst        = (float *)octx->dst.data;
    size_t size        = octx->src0.ne[0] * octx->src0.ne[1] * octx->src0.ne[2] * octx->src0.ne[3];

    // Check if contiguous
    // Basic copy
    size_t vec_size = 128 / sizeof(float);
    size_t i = 0;

    for (; i + vec_size <= size; i += vec_size) {
        HVX_Vector v = q6op_V_vldu_A((const HVX_Vector *)&src0[i]);
        q6op_V_vstu_A((HVX_Vector *)&dst[i], v);
    }

    for (; i < size; ++i) {
        dst[i] = src0[i];
    }

    return HTP_STATUS_OK;
}

static int op_cpy_f16(struct htp_ops_context * octx) {
    const _Float16 * src0 = (const _Float16 *)octx->src0.data;
    _Float16 * dst        = (_Float16 *)octx->dst.data;
    size_t size           = octx->src0.ne[0] * octx->src0.ne[1] * octx->src0.ne[2] * octx->src0.ne[3];

    size_t vec_size = 128 / sizeof(_Float16);
    size_t i = 0;

    for (; i + vec_size <= size; i += vec_size) {
        HVX_Vector v = q6op_V_vldu_A((const HVX_Vector *)&src0[i]);
        q6op_V_vstu_A((HVX_Vector *)&dst[i], v);
    }

    for (; i < size; ++i) {
        dst[i] = src0[i];
    }

    return HTP_STATUS_OK;
}

static int op_cpy_f32_to_f16(struct htp_ops_context * octx) {
    const float * src0 = (const float *)octx->src0.data;
    _Float16 * dst     = (_Float16 *)octx->dst.data;
    size_t size        = octx->src0.ne[0] * octx->src0.ne[1] * octx->src0.ne[2] * octx->src0.ne[3];

    // TODO: use HVX conversion
    for (size_t i = 0; i < size; ++i) {
        dst[i] = (_Float16)src0[i];
    }
    return HTP_STATUS_OK;
}

static int op_cpy_f16_to_f32(struct htp_ops_context * octx) {
    const _Float16 * src0 = (const _Float16 *)octx->src0.data;
    float * dst           = (float *)octx->dst.data;
    size_t size           = octx->src0.ne[0] * octx->src0.ne[1] * octx->src0.ne[2] * octx->src0.ne[3];

    for (size_t i = 0; i < size; ++i) {
        dst[i] = (float)src0[i];
    }
    return HTP_STATUS_OK;
}

int op_cpy(struct htp_ops_context * octx) {
    if (octx->src0.type == octx->dst.type) {
        if (octx->src0.type == HTP_TYPE_F32) return op_cpy_f32(octx);
        if (octx->src0.type == HTP_TYPE_F16) return op_cpy_f16(octx);
    } else {
        if (octx->src0.type == HTP_TYPE_F32 && octx->dst.type == HTP_TYPE_F16) return op_cpy_f32_to_f16(octx);
        if (octx->src0.type == HTP_TYPE_F16 && octx->dst.type == HTP_TYPE_F32) return op_cpy_f16_to_f32(octx);
    }
    return HTP_STATUS_NO_SUPPORT;
}
