import re

with open('ggml/src/ggml-hexagon/htp/matmul-ops.c', 'r') as f:
    content = f.read()

# Fix 1: `hvx_vec_reduce_sum_f32` returns `HVX_Vector`, so we need to extract the scalar.
s1 = """    *s0 = hvx_vec_reduce_sum_f32(r0_sum);"""
r1 = """    *s0 = *(float *)&hvx_vec_reduce_sum_f32(r0_sum);"""
content = content.replace(s1, r1)

s2 = """    s0[0] = hvx_vec_reduce_sum_f32(r0_sum);
    s0[1] = hvx_vec_reduce_sum_f32(r1_sum);"""
r2 = """    s0[0] = *(float *)&hvx_vec_reduce_sum_f32(r0_sum);
    s0[1] = *(float *)&hvx_vec_reduce_sum_f32(r1_sum);"""
content = content.replace(s2, r2)

s3 = """    s0[0] = hvx_vec_reduce_sum_f32(r0_c0_sum);
    s0[1] = hvx_vec_reduce_sum_f32(r1_c0_sum);
    s1[0] = hvx_vec_reduce_sum_f32(r0_c1_sum);
    s1[1] = hvx_vec_reduce_sum_f32(r1_c1_sum);"""
r3 = """    s0[0] = *(float *)&hvx_vec_reduce_sum_f32(r0_c0_sum);
    s0[1] = *(float *)&hvx_vec_reduce_sum_f32(r1_c0_sum);
    s1[0] = *(float *)&hvx_vec_reduce_sum_f32(r0_c1_sum);
    s1[1] = *(float *)&hvx_vec_reduce_sum_f32(r1_c1_sum);"""
content = content.replace(s3, r3)

# Fix 2: `GGML_FP16_TO_FP32` and `GGML_FP32_TO_FP16` implicitly declared. Use `(float)x` and `(__fp16)x` instead.
s4 = """    float d0 = GGML_FP16_TO_FP32(*(__fp16*)(y_d + 0));
    float d1 = GGML_FP16_TO_FP32(*(__fp16*)(y_d + 2));
    float d2 = GGML_FP16_TO_FP32(*(__fp16*)(y_d + 4));
    float d3 = GGML_FP16_TO_FP32(*(__fp16*)(y_d + 6));

    __fp16 hs0 = GGML_FP32_TO_FP16(d0 * s0);
    __fp16 hs1 = GGML_FP32_TO_FP16(d1 * s1);
    __fp16 hs2 = GGML_FP32_TO_FP16(d2 * s2);
    __fp16 hs3 = GGML_FP32_TO_FP16(d3 * s3);"""
r4 = """    float d0 = (float)*(__fp16*)(y_d + 0);
    float d1 = (float)*(__fp16*)(y_d + 2);
    float d2 = (float)*(__fp16*)(y_d + 4);
    float d3 = (float)*(__fp16*)(y_d + 6);

    __fp16 hs0 = (__fp16)(d0 * s0);
    __fp16 hs1 = (__fp16)(d1 * s1);
    __fp16 hs2 = (__fp16)(d2 * s2);
    __fp16 hs3 = (__fp16)(d3 * s3);"""
content = content.replace(s4, r4)

# Fix 3: `wtype` is undeclared in the sections where I added `if (wtype == HTP_TYPE_Q4_1)`.
# The variable is `src0->type` in those contexts!
# Let's replace `wtype` with `src0->type` in those two places.
s5 = """        if (wtype == HTP_TYPE_Q4_1) {
            quant_job_func = quantize_f32_q8_1x4x2;
            src1_row_size  = ne10 + ne10 / 32 * 2 + ne10 / 32 * 2; // q8_1x4x2 size
        } else {
            quant_job_func = quantize_f32_q8x4x2;
            src1_row_size  = q8x4x2_row_size(ne10);
        }"""
r5 = """        if (src0->type == HTP_TYPE_Q4_1) {
            quant_job_func = quantize_f32_q8_1x4x2;
            src1_row_size  = ne10 + ne10 / 32 * 2 + ne10 / 32 * 2; // q8_1x4x2 size
        } else {
            quant_job_func = quantize_f32_q8x4x2;
            src1_row_size  = q8x4x2_row_size(ne10);
        }"""
content = content.replace(s5, r5)

s6 = """    if (wtype == HTP_TYPE_Q4_1) {
        quant_job_func = quantize_f32_q8_1x4x2;
        src1_row_size  = ne10 + ne10 / 32 * 2 + ne10 / 32 * 2;
    } else {
        quant_job_func = quantize_f32_q8x4x2;
        src1_row_size  = q8x4x2_row_size(ne10);
    }"""
r6 = """    if (src0->type == HTP_TYPE_Q4_1) {
        quant_job_func = quantize_f32_q8_1x4x2;
        src1_row_size  = ne10 + ne10 / 32 * 2 + ne10 / 32 * 2;
    } else {
        quant_job_func = quantize_f32_q8x4x2;
        src1_row_size  = q8x4x2_row_size(ne10);
    }"""
content = content.replace(s6, r6)

# Fix 4: `src1_nrows` and `src1_row_size` missing from `struct htp_matmul_context`.
# Actually, the struct doesn't have `src1_nrows` and `src1_row_size`? Wait!
# In `quantize_f32_q8x4x2`:
# `uint32_t ne1 = src->ne[1];`
# Oh! `mmctx` doesn't have `src1_nrows`! `quantize_f32_q8x4x2` doesn't use `mmctx->src1_nrows`.
# It uses `src->ne[1]`. And `src1_row_size` is calculated.
s7 = """    uint32_t ne1  = mmctx->src1_nrows;

    uint32_t r_start = ith * nrows_per_thread;
    uint32_t r_end   = r_start + nrows_per_thread;
    if (r_end > ne1) r_end = ne1;
    if (r_start >= r_end) return;

    size_t src1_row_size = mmctx->src1_row_size;"""
r7 = """    uint32_t ne1  = src->ne[1];

    uint32_t r_start = ith * nrows_per_thread;
    uint32_t r_end   = r_start + nrows_per_thread;
    if (r_end > ne1) r_end = ne1;
    if (r_start >= r_end) return;

    size_t src1_row_size = ne0 + ne0 / 32 * 2 + ne0 / 32 * 2;"""
content = content.replace(s7, r7)

with open('ggml/src/ggml-hexagon/htp/matmul-ops.c', 'w') as f:
    f.write(content)
