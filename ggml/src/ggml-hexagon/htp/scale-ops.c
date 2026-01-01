#include "htp-ops.h"
#include "hvx-utils.h"
#include "ops-utils.h"

// dst = src0 * scale
// scale = src1 (tensor) or op_params

static int op_scale_f32(struct htp_ops_context * octx) {
    float scale = 1.0f;

    // Check if scale is provided in src1 (tensor)
    if (octx->src1.data != 0) {
        // src1 is a scalar tensor
        const float * s_ptr = (const float *)octx->src1.data;
        scale = *s_ptr;
    } else {
        // Retrieve scale from op_params (passed as int32_t, need to cast back to float)
        // GGML passes scale in op_params[0]
        int32_t scale_as_int = octx->op_params[0];
        memcpy(&scale, &scale_as_int, sizeof(float));
    }

    const float * src0 = (const float *)octx->src0.data;
    float * dst        = (float *)octx->dst.data;
    size_t size        = octx->src0.ne[0] * octx->src0.ne[1] * octx->src0.ne[2] * octx->src0.ne[3];

    // HVX loop
    HVX_Vector v_scale = Q6_V_vsplat_R(*(int*)&scale);

    size_t vec_size = 128 / sizeof(float); // 32 floats
    size_t i = 0;

    // Process 128 bytes at a time
    for (; i + vec_size <= size; i += vec_size) {
        HVX_Vector v_src = q6op_V_vldu_A((const HVX_Vector *)&src0[i]);
        HVX_Vector v_dst = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v_src, v_scale));
        q6op_V_vstu_A((HVX_Vector *)&dst[i], v_dst);
    }

    // Scalar leftover
    for (; i < size; ++i) {
        dst[i] = src0[i] * scale;
    }

    return HTP_STATUS_OK;
}

static int op_scale_f16(struct htp_ops_context * octx) {
    float scale_f = 1.0f;
    if (octx->src1.data != 0) {
        const float * s_ptr = (const float *)octx->src1.data; // src1 is usually F32 even if src0 is F16?
        // Check src1 type
        if (octx->src1.type == HTP_TYPE_F16) {
             const _Float16 * s_ptr_f16 = (const _Float16 *)octx->src1.data;
             scale_f = (float)*s_ptr_f16;
        } else {
             scale_f = *s_ptr;
        }
    } else {
        // Retrieve scale from op_params
        int32_t scale_as_int = octx->op_params[0];
        memcpy(&scale_f, &scale_as_int, sizeof(float));
    }

    const _Float16 * src0 = (const _Float16 *)octx->src0.data;
    _Float16 * dst        = (_Float16 *)octx->dst.data;
    size_t size           = octx->src0.ne[0] * octx->src0.ne[1] * octx->src0.ne[2] * octx->src0.ne[3];

    // HVX loop for F16
    // We can use Q6_Vhf_vmpy_VhfVhf if available, or convert to float.
    // Assuming V68+ supports HF mpy.

    // Construct scale vector
    _Float16 scale_h = (_Float16)scale_f;
    // splat 16-bit
    int scale_pair = (int)((unsigned short)scale_h | ((unsigned int)(unsigned short)scale_h << 16));
    HVX_Vector v_scale = Q6_V_vsplat_R(scale_pair);

    size_t vec_size = 128 / sizeof(_Float16); // 64 halves
    size_t i = 0;

    for (; i + vec_size <= size; i += vec_size) {
        HVX_Vector v_src = q6op_V_vldu_A((const HVX_Vector *)&src0[i]);
        // Use Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(v_src, v_scale))
        HVX_Vector v_dst = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(v_src, v_scale));
        q6op_V_vstu_A((HVX_Vector *)&dst[i], v_dst);
    }

    for (; i < size; ++i) {
        dst[i] = src0[i] * scale_h;
    }

    return HTP_STATUS_OK;
}

int op_scale(struct htp_ops_context * octx) {
    if (octx->src0.type == HTP_TYPE_F32) {
        return op_scale_f32(octx);
    } else if (octx->src0.type == HTP_TYPE_F16) {
        return op_scale_f16(octx);
    }
    return HTP_STATUS_NO_SUPPORT;
}
