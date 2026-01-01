#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#ifdef HTP_DEBUG
#    define FARF_HIGH 1
#endif

#include <HAP_farf.h>
#include <HAP_mem.h>
#include <HAP_perf.h>
#include <HAP_ps.h>
#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <math.h>
#include <qurt_thread.h>
#include <string.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-dma.h"
#include "htp-msg.h"
#include "htp-ops.h"
#include "hvx-utils.h"
#include "ops-utils.h"

struct htp_matmul_type {
    const char * type;
    void (*vec_dot)(const int n, float * restrict s, const void * restrict vx, const void * restrict vy);
    void (*vec_dot_rx2)(const int n,
                        float * restrict s,
                        const void * restrict vx,
                        uint32_t vx_row_size,
                        const void * restrict vy);
};

typedef struct {
    HVX_Vector v[2];
} HVX_Vector_x2;

typedef struct {
    HVX_Vector v[4];
} HVX_Vector_x4;

typedef struct {
    HVX_Vector v[8];
} HVX_Vector_x8;

// vdelta control to replicate first 4x fp32 values across lanes
static const uint8_t __attribute__((aligned(128))) repl_4x_fp32[128] = {
    0x00, 0x00, 0x00, 0x00, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04, 0x10, 0x10, 0x10,
    0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x20, 0x20,
    0x20, 0x20, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04, 0x10, 0x10, 0x10, 0x10, 0x04,
    0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04, 0x40, 0x40, 0x40, 0x40,
    0x44, 0x44, 0x44, 0x44, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04, 0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04,
    0x04, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04, 0x20, 0x20, 0x20, 0x20, 0x04, 0x04,
    0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04, 0x10, 0x10, 0x10, 0x10,
};

// vdelta control to replicate and interleave first 8x fp32 values across lanes
static const uint8_t __attribute__((aligned(128))) repl_interleave_8x_fp32[128] = {
    0x00, 0x00, 0x00, 0x00, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04, 0x00, 0x00, 0x00,
    0x00, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x20, 0x20,
    0x20, 0x20, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04, 0x20, 0x20, 0x20, 0x20, 0x04,
    0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04, 0x40, 0x40, 0x40, 0x40,
    0x44, 0x44, 0x44, 0x44, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04, 0x40, 0x40, 0x40, 0x40, 0x44, 0x44, 0x44,
    0x44, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04, 0x20, 0x20, 0x20, 0x20, 0x04, 0x04,
    0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04, 0x20, 0x20, 0x20, 0x20,
};

// vdelta control to replicate first fp32 value across all elements
static const uint8_t __attribute__((aligned(128))) repl_1x_fp32[128] = {
    0x00, 0x00, 0x00, 0x00, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04, 0x10, 0x10, 0x10,
    0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04, 0x20, 0x20, 0x20, 0x20, 0x04, 0x04,
    0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04, 0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08,
    0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04, 0x40, 0x40, 0x40, 0x40, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08,
    0x04, 0x04, 0x04, 0x04, 0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04,
    0x04, 0x20, 0x20, 0x20, 0x20, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04, 0x10, 0x10,
    0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
};

// vdelta control to replicate first fp16 value across all elements
static const uint8_t __attribute__((aligned(128))) repl_1x_fp16[128] = {
    0x00, 0x00, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02, 0x08, 0x08, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02, 0x10, 0x10, 0x02,
    0x02, 0x04, 0x04, 0x02, 0x02, 0x08, 0x08, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02, 0x20, 0x20, 0x02, 0x02, 0x04, 0x04,
    0x02, 0x02, 0x08, 0x08, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02, 0x10, 0x10, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02, 0x08,
    0x08, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02, 0x40, 0x40, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02, 0x08, 0x08, 0x02, 0x02,
    0x04, 0x04, 0x02, 0x02, 0x10, 0x10, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02, 0x08, 0x08, 0x02, 0x02, 0x04, 0x04, 0x02,
    0x02, 0x20, 0x20, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02, 0x08, 0x08, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02, 0x10, 0x10,
    0x02, 0x02, 0x04, 0x04, 0x02, 0x02, 0x08, 0x08, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02,
};

// vdelta control to replicate first fp16 value across all elements
static const uint8_t __attribute__((aligned(128))) repl_2x_fp16[128] = {
    0x00, 0x00, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02, 0x08, 0x08, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02,
    0x10, 0x10, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02, 0x08, 0x08, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02,
    0x20, 0x20, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02, 0x08, 0x08, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02,
    0x10, 0x10, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02, 0x08, 0x08, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02,
    0x00, 0x00, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02, 0x08, 0x08, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02,
    0x10, 0x10, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02, 0x08, 0x08, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02,
    0x20, 0x20, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02, 0x08, 0x08, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02,
    0x10, 0x10, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02, 0x08, 0x08, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02,
};

// vdelta control to expand first 32 e8m0 values into 32 uint32 elements
static const uint8_t __attribute__((aligned(128))) expand_x32_e8m0[128] = {
    0x00, 0x00, 0x00, 0x00, 0x01, 0x04, 0x00, 0x00, 0x02, 0x00, 0x08, 0x08, 0x01, 0x02, 0x00, 0x04, 0x04, 0x00, 0x00,
    0x00, 0x11, 0x10, 0x10, 0x10, 0x02, 0x00, 0x04, 0x00, 0x01, 0x02, 0x08, 0x08, 0x08, 0x08, 0x00, 0x00, 0x01, 0x04,
    0x00, 0x00, 0x22, 0x20, 0x20, 0x20, 0x21, 0x22, 0x20, 0x24, 0x04, 0x00, 0x00, 0x00, 0x09, 0x08, 0x00, 0x00, 0x02,
    0x00, 0x04, 0x00, 0x11, 0x12, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x01, 0x04, 0x00, 0x00, 0x02, 0x00, 0x08, 0x08,
    0x01, 0x02, 0x00, 0x04, 0x44, 0x40, 0x40, 0x40, 0x41, 0x40, 0x40, 0x40, 0x42, 0x40, 0x44, 0x40, 0x41, 0x42, 0x48,
    0x48, 0x08, 0x08, 0x00, 0x00, 0x01, 0x04, 0x00, 0x00, 0x12, 0x10, 0x10, 0x10, 0x01, 0x02, 0x00, 0x04, 0x04, 0x00,
    0x00, 0x00, 0x09, 0x08, 0x00, 0x00, 0x22, 0x20, 0x24, 0x20, 0x21, 0x22, 0x20, 0x20,
};

static const uint8_t __attribute__((aligned(VLEN))) kvalues_mxfp4_lut[] = {
    0,    0, 1,    0, 2,    0, 3, 0, 4, 0, 6, 0, 8, 0, 12, 0, 0, 0, 0xff, 0, 0xfe, 0, 0xfd, 0, 0xfc, 0,
    0xfa, 0, 0xf8, 0, 0xf4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0,    0, 0,    0, 0,    0,
};

// q4x4x2 and q8x4x2 are the flat q4/8_0 formats where all quants are stored first followed by all scales

static inline size_t q8x4x2_row_size(uint32_t ne) {
    // ensures perfect alignment of quants and full row
    const uint32_t qk = QK_Q8_0x4x2;
    const uint32_t nb = (ne + qk - 1) / qk;
    return htp_round_up(ne + nb * 8 * sizeof(__fp16), 128);
}

static inline HVX_Vector_x8 hvx_vec_load_q4x4x8(const uint8_t * restrict ptr) {
    const HVX_Vector * restrict vptr = (const HVX_Vector *) ptr;

    HVX_Vector v0_1 = vptr[0];  // first 256 elements (128 bytes)
    HVX_Vector v2_3 = vptr[1];  // ...
    HVX_Vector v4_5 = vptr[2];  // ...
    HVX_Vector v6_7 = vptr[3];  // ...

    const HVX_Vector mask_h4 = Q6_Vb_vsplat_R(0x0F);

    HVX_Vector v0 = Q6_V_vand_VV(v0_1, mask_h4);  // & 0x0F
    HVX_Vector v1 = Q6_Vub_vlsr_VubR(v0_1, 4);    // >> 4
    HVX_Vector v2 = Q6_V_vand_VV(v2_3, mask_h4);  // & 0x0F
    HVX_Vector v3 = Q6_Vub_vlsr_VubR(v2_3, 4);    // >> 4
    HVX_Vector v4 = Q6_V_vand_VV(v4_5, mask_h4);  // & 0x0F
    HVX_Vector v5 = Q6_Vub_vlsr_VubR(v4_5, 4);    // >> 4
    HVX_Vector v6 = Q6_V_vand_VV(v6_7, mask_h4);  // & 0x0F
    HVX_Vector v7 = Q6_Vub_vlsr_VubR(v6_7, 4);    // >> 4

    // Convert uint4 to int4 (i.e. x - 8)
    const HVX_Vector i8 = Q6_Vb_vsplat_R(8);
    v0                  = Q6_Vb_vsub_VbVb(v0, i8);
    v1                  = Q6_Vb_vsub_VbVb(v1, i8);
    v2                  = Q6_Vb_vsub_VbVb(v2, i8);
    v3                  = Q6_Vb_vsub_VbVb(v3, i8);
    v4                  = Q6_Vb_vsub_VbVb(v4, i8);
    v5                  = Q6_Vb_vsub_VbVb(v5, i8);
    v6                  = Q6_Vb_vsub_VbVb(v6, i8);
    v7                  = Q6_Vb_vsub_VbVb(v7, i8);

    HVX_Vector_x8 r = { v0, v1, v2, v3, v4, v5, v6, v7 };
    return r;
}

static inline HVX_Vector_x8 hvx_vec_load_mxfp4x4x8(const uint8_t * restrict ptr) {
    const HVX_Vector * restrict vptr = (const HVX_Vector *) ptr;

    HVX_Vector v0_1 = vptr[0];  // first 256 elements (128 bytes)
    HVX_Vector v2_3 = vptr[1];  // ...
    HVX_Vector v4_5 = vptr[2];  // ...
    HVX_Vector v6_7 = vptr[3];  // ...

    const HVX_Vector mask_h4 = Q6_Vb_vsplat_R(0x0F);

    HVX_Vector v0 = Q6_V_vand_VV(v0_1, mask_h4);  // & 0x0F
    HVX_Vector v1 = Q6_Vub_vlsr_VubR(v0_1, 4);    // >> 4
    HVX_Vector v2 = Q6_V_vand_VV(v2_3, mask_h4);  // & 0x0F
    HVX_Vector v3 = Q6_Vub_vlsr_VubR(v2_3, 4);    // >> 4
    HVX_Vector v4 = Q6_V_vand_VV(v4_5, mask_h4);  // & 0x0F
    HVX_Vector v5 = Q6_Vub_vlsr_VubR(v4_5, 4);    // >> 4
    HVX_Vector v6 = Q6_V_vand_VV(v6_7, mask_h4);  // & 0x0F
    HVX_Vector v7 = Q6_Vub_vlsr_VubR(v6_7, 4);    // >> 4

    HVX_Vector lut = *(const HVX_Vector *) kvalues_mxfp4_lut;
    v0             = Q6_Vb_vlut32_VbVbI(v0, lut, 0);
    v1             = Q6_Vb_vlut32_VbVbI(v1, lut, 0);
    v2             = Q6_Vb_vlut32_VbVbI(v2, lut, 0);
    v3             = Q6_Vb_vlut32_VbVbI(v3, lut, 0);
    v4             = Q6_Vb_vlut32_VbVbI(v4, lut, 0);
    v5             = Q6_Vb_vlut32_VbVbI(v5, lut, 0);
    v6             = Q6_Vb_vlut32_VbVbI(v6, lut, 0);
    v7             = Q6_Vb_vlut32_VbVbI(v7, lut, 0);

    HVX_Vector_x8 r = { v0, v1, v2, v3, v4, v5, v6, v7 };
    return r;
}

static inline HVX_Vector_x8 hvx_vec_load_q8x4x8(const uint8_t * restrict ptr) {
    const HVX_Vector * restrict vptr = (const HVX_Vector *) ptr;

    HVX_Vector v0 = vptr[0];  // first  128 vals
    HVX_Vector v1 = vptr[1];  // ...
    HVX_Vector v2 = vptr[2];  // ...
    HVX_Vector v3 = vptr[3];  // ...
    HVX_Vector v4 = vptr[4];  // ...
    HVX_Vector v5 = vptr[5];  // ...
    HVX_Vector v6 = vptr[6];  // ...
    HVX_Vector v7 = vptr[7];  // ...

    HVX_Vector_x8 r = { v0, v1, v2, v3, v4, v5, v6, v7 };
    return r;
}

static inline HVX_Vector_x4 hvx_vec_load_x4_f16(const uint8_t * restrict ptr) {
    const HVX_Vector * restrict vptr = (const HVX_Vector *) ptr;

    HVX_Vector v0 = vptr[0];  // first  64 vals
    HVX_Vector v1 = vptr[1];  // second 64 vals
    HVX_Vector v2 = vptr[2];  // third  64 vals
    HVX_Vector v3 = vptr[3];  // forth  64 vals

    HVX_Vector_x4 r = { v0, v1, v2, v3 };
    return r;
}

static inline HVX_Vector_x4 hvx_vec_load_x4_f32_as_f16(const uint8_t * restrict ptr) {
    const HVX_VectorPair * restrict vptr = (const HVX_VectorPair *) ptr;

    HVX_VectorPair v0 = vptr[0];  // first  64 vals
    HVX_VectorPair v1 = vptr[1];  // second 64 vals
    HVX_VectorPair v2 = vptr[2];  // third  64 vals
    HVX_VectorPair v3 = vptr[3];  // forth  64 vals

    HVX_Vector vq0_lo = Q6_Vqf32_vsub_VsfVsf(Q6_V_lo_W(v0), Q6_V_vzero());
    HVX_Vector vq0_hi = Q6_Vqf32_vsub_VsfVsf(Q6_V_hi_W(v0), Q6_V_vzero());
    HVX_Vector vq1_lo = Q6_Vqf32_vsub_VsfVsf(Q6_V_lo_W(v1), Q6_V_vzero());
    HVX_Vector vq1_hi = Q6_Vqf32_vsub_VsfVsf(Q6_V_hi_W(v1), Q6_V_vzero());
    HVX_Vector vq2_lo = Q6_Vqf32_vsub_VsfVsf(Q6_V_lo_W(v2), Q6_V_vzero());
    HVX_Vector vq2_hi = Q6_Vqf32_vsub_VsfVsf(Q6_V_hi_W(v2), Q6_V_vzero());
    HVX_Vector vq3_lo = Q6_Vqf32_vsub_VsfVsf(Q6_V_lo_W(v3), Q6_V_vzero());
    HVX_Vector vq3_hi = Q6_Vqf32_vsub_VsfVsf(Q6_V_hi_W(v3), Q6_V_vzero());

    HVX_Vector vh0 = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vq0_hi, vq0_lo));
    HVX_Vector vh1 = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vq1_hi, vq1_lo));
    HVX_Vector vh2 = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vq2_hi, vq2_lo));
    HVX_Vector vh3 = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vq3_hi, vq3_lo));

    // vcombine does a shuffle, use vdeal to undo

    HVX_Vector_x4 r = { Q6_Vh_vdeal_Vh(vh0), Q6_Vh_vdeal_Vh(vh1), Q6_Vh_vdeal_Vh(vh2), Q6_Vh_vdeal_Vh(vh3) };
    return r;
}

// Reduce multiply 1024 x 1024 int8 elements (32x q4/8 blocks in 8x HVX vectors).
// Accumulate each block into a single int32 value.
// Return a single HVX vector with 32x int32 accumulators.
// This version is parameterized to support less than 1024 elements.
// if() checks are optimized out at compile time -- make sure to pass N as a constexpr.

static inline HVX_Vector hvx_vec_rmpy_x8_n(HVX_Vector_x8 x, HVX_Vector_x8 y, unsigned int n) {
    HVX_Vector r0 = Q6_V_vsplat_R(0);
    HVX_Vector r1 = Q6_V_vsplat_R(0);
    HVX_Vector r2 = Q6_V_vsplat_R(0);
    HVX_Vector r3 = Q6_V_vsplat_R(0);
    HVX_Vector r4 = Q6_V_vsplat_R(0);
    HVX_Vector r5 = Q6_V_vsplat_R(0);
    HVX_Vector r6 = Q6_V_vsplat_R(0);
    HVX_Vector r7 = Q6_V_vsplat_R(0);

    HVX_VectorPair p3;
    HVX_VectorPair p2;
    HVX_VectorPair p1;
    HVX_VectorPair p0;

    if (n >=  128) { r0 = Q6_Vw_vrmpy_VbVb(x.v[0], y.v[0]); }
    if (n >=  256) { r1 = Q6_Vw_vrmpy_VbVb(x.v[1], y.v[1]); }
    if (n >=  384) { r2 = Q6_Vw_vrmpy_VbVb(x.v[2], y.v[2]); }
    if (n >=  512) { r3 = Q6_Vw_vrmpy_VbVb(x.v[3], y.v[3]); }
    if (n >=  640) { r4 = Q6_Vw_vrmpy_VbVb(x.v[4], y.v[4]); }
    if (n >=  768) { r5 = Q6_Vw_vrmpy_VbVb(x.v[5], y.v[5]); }
    if (n >=  896) { r6 = Q6_Vw_vrmpy_VbVb(x.v[6], y.v[6]); }
    if (n >= 1024) { r7 = Q6_Vw_vrmpy_VbVb(x.v[7], y.v[7]); }

    if (n >=  128) { p0 = Q6_W_vdeal_VVR(r1, r0, -4); }
    if (n >=  384) { p1 = Q6_W_vdeal_VVR(r3, r2, -4); }
    if (n >=  640) { p2 = Q6_W_vdeal_VVR(r5, r4, -4); }
    if (n >=  896) { p3 = Q6_W_vdeal_VVR(r7, r6, -4); }

    if (n >=  128) { r0 = Q6_Vw_vadd_VwVw(Q6_V_lo_W(p0), Q6_V_hi_W(p0)); }
    if (n >=  384) { r1 = Q6_Vw_vadd_VwVw(Q6_V_lo_W(p1), Q6_V_hi_W(p1)); }
    if (n >=  640) { r2 = Q6_Vw_vadd_VwVw(Q6_V_lo_W(p2), Q6_V_hi_W(p2)); }
    if (n >=  896) { r3 = Q6_Vw_vadd_VwVw(Q6_V_lo_W(p3), Q6_V_hi_W(p3)); }

    if (n >=  128) { p0 = Q6_W_vdeal_VVR(r1, r0, -4); }
    if (n >=  640) { p1 = Q6_W_vdeal_VVR(r3, r2, -4); }

    if (n >=  128) { r0 = Q6_Vw_vadd_VwVw(Q6_V_lo_W(p0), Q6_V_hi_W(p0)); }
    if (n >=  640) { r1 = Q6_Vw_vadd_VwVw(Q6_V_lo_W(p1), Q6_V_hi_W(p1)); }

    if (n >=  128) { p0 = Q6_W_vdeal_VVR(r1, r0, -4); }
    if (n >=  128) { r0 = Q6_Vw_vadd_VwVw(Q6_V_lo_W(p0), Q6_V_hi_W(p0)); }

    return r0;
}

static inline HVX_Vector hvx_vec_rmpy_x8_full(HVX_Vector_x8 x, HVX_Vector_x8 y) {
    return hvx_vec_rmpy_x8_n(x, y, 1024);
}

// Handle most common cases of tensors not multiple of 1024.
static inline HVX_Vector hvx_vec_rmpy_x8_nloe(HVX_Vector_x8 x, HVX_Vector_x8 y, unsigned int n) {
    if (n <= 256) { return hvx_vec_rmpy_x8_n(x, y, 256); };
    if (n <= 512) { return hvx_vec_rmpy_x8_n(x, y, 512); };
    if (n <= 768) { return hvx_vec_rmpy_x8_n(x, y, 768); };
    return hvx_vec_rmpy_x8_n(x, y, 1024);
}

static void vec_dot_q4x4x2_q8x4x2(const int n, float * restrict s, const void * restrict vx, const void * restrict vy) {
    assert(n % 32 == 0);  // min sub-block size
    assert((unsigned long) vx % 128 == 0);
    assert((unsigned long) vy % 128 == 0);

    const uint32_t qk = QK_Q4_0x4x2 * 4;

    const uint32_t x_dblk_size = 8 * 4 * 2;                                  // 32x __fp16
    const uint32_t x_qblk_size = qk / 2;                                     // int4
    const uint32_t x_qrow_size = n / 2;                                      // int4 (not padded)

    const uint32_t y_dblk_size = 8 * 4 * 2;                                  // 32x __fp16
    const uint32_t y_qblk_size = qk;                                         // int8
    const uint32_t y_qrow_size = n;                                          // int8 (not padded)

    const uint8_t * restrict r0_x_q = ((const uint8_t *) vx + 0);            // quants first
    const uint8_t * restrict r0_x_d = ((const uint8_t *) vx + x_qrow_size);  // then scales

    const uint8_t * restrict y_q = ((const uint8_t *) vy + 0);               // quants first
    const uint8_t * restrict y_d = ((const uint8_t *) vy + y_qrow_size);     // then scales

    // Row sum (qf32)
    HVX_Vector r0_sum = Q6_V_vsplat_R(0);

    // Multiply and accumulate into int32.
    // Compute combined scale (fp32).
    // Apply scale to acc and accumulate into the row sum (qf32).

    const uint32_t nb   = n / qk;  // num full blocks
    const uint32_t nloe = n % qk;  // num leftover elemements

    uint32_t i = 0;
    for (; i < nb; i++) {
        HVX_Vector_x8 vy_q = hvx_vec_load_q8x4x8(y_q + i * y_qblk_size);
        HVX_Vector_x8 r0_q = hvx_vec_load_q4x4x8(r0_x_q + i * x_qblk_size);

        HVX_Vector r0_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_full(r0_q, vy_q));

        HVX_Vector vy_d = Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (y_d + i * y_dblk_size));
        HVX_Vector r0_d = Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (r0_x_d + i * x_dblk_size));

        HVX_Vector r0_dd = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(r0_d, vy_d)));

        HVX_Vector r0_fa = Q6_Vqf32_vmpy_VsfVsf(r0_ia, r0_dd);

        r0_sum = Q6_Vqf32_vadd_Vqf32Vqf32(r0_sum, r0_fa);
    }

    // Process leftovers, we still load full 4x4x2 block but zero out unused scales/blocks
    if (nloe) {
        HVX_Vector_x8 vy_q = hvx_vec_load_q8x4x8(y_q + i * y_qblk_size);
        HVX_Vector_x8 r0_q = hvx_vec_load_q4x4x8(r0_x_q + i * x_qblk_size);

        HVX_Vector r0_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_nloe(r0_q, vy_q, nloe));

        HVX_Vector vy_d = Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (y_d + i * y_dblk_size));
        HVX_Vector r0_d = Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (r0_x_d + i * x_dblk_size));

        HVX_Vector r0_dd = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(r0_d, vy_d)));

        // Zero out unused scales
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe / 8);
        r0_dd                = Q6_V_vand_QV(bmask, r0_dd);

        HVX_Vector r0_fa = Q6_Vqf32_vmpy_VsfVsf(r0_ia, r0_dd);

        r0_sum = Q6_Vqf32_vadd_Vqf32Vqf32(r0_sum, r0_fa);
    }

    // Reduce and convert into fp32
    r0_sum = hvx_vec_fp32_reduce_sum(Q6_Vsf_equals_Vqf32(r0_sum));

    hvx_vec_store_u(&s[0], 4, r0_sum);
}

static void vec_dot_q4x4x2_q8x4x2_rx2(const int n,
                                      float * restrict s,
                                      const void * restrict vx,
                                      uint32_t vx_row_size,
                                      const void * restrict vy) {
    assert(n % 32 == 0);  // min sub-block size
    assert((unsigned long) vx % 128 == 0);
    assert((unsigned long) vy % 128 == 0);

    const uint32_t qk = QK_Q4_0x4x2 * 4;

    const uint32_t x_dblk_size = 8 * 4 * 2;                                                        // 32x __fp16
    const uint32_t x_qblk_size = qk / 2;                                                           // int4
    const uint32_t x_qrow_size = n / 2;                                                            // int4 (not padded)

    const uint32_t y_dblk_size = 8 * 4 * 2;                                                        // 32x __fp16
    const uint32_t y_qblk_size = qk;                                                               // int8
    const uint32_t y_qrow_size = n;                                                                // int8 (not padded)

    const uint8_t * restrict r0_x_q = ((const uint8_t *) (vx + (0 * vx_row_size)) + 0);            // quants first
    const uint8_t * restrict r0_x_d = ((const uint8_t *) (vx + (0 * vx_row_size)) + x_qrow_size);  // then scales

    const uint8_t * restrict r1_x_q = ((const uint8_t *) (vx + (1 * vx_row_size)) + 0);            // quants first
    const uint8_t * restrict r1_x_d = ((const uint8_t *) (vx + (1 * vx_row_size)) + x_qrow_size);  // then scales

    const uint8_t * restrict y_q = ((const uint8_t *) vy + 0);                                     // quants first
    const uint8_t * restrict y_d = ((const uint8_t *) vy + y_qrow_size);                           // then scales

    // Row sum (qf32)
    HVX_Vector r0_sum = Q6_V_vsplat_R(0);
    HVX_Vector r1_sum = Q6_V_vsplat_R(0);

    // Multiply and accumulate into int32.
    // Compute combined scale (fp32).
    // Apply scale to acc and accumulate into the row sum (qf32).

    const uint32_t nb   = n / qk;  // num full blocks
    const uint32_t nloe = n % qk;  // num leftover elemements

    uint32_t i = 0;
    for (; i < nb; i++) {
        HVX_Vector_x8 vy_q = hvx_vec_load_q8x4x8(y_q + i * y_qblk_size);
        HVX_Vector_x8 r0_q = hvx_vec_load_q4x4x8(r0_x_q + i * x_qblk_size);
        HVX_Vector_x8 r1_q = hvx_vec_load_q4x4x8(r1_x_q + i * x_qblk_size);

        HVX_Vector r0_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_full(r0_q, vy_q));
        HVX_Vector r1_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_full(r1_q, vy_q));

        HVX_Vector vy_d = Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (y_d + i * y_dblk_size));
        HVX_Vector r0_d = Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (r0_x_d + i * x_dblk_size));
        HVX_Vector r1_d = Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (r1_x_d + i * x_dblk_size));

        HVX_Vector r0_dd = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(r0_d, vy_d)));
        HVX_Vector r1_dd = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(r1_d, vy_d)));

        HVX_Vector r0_fa = Q6_Vqf32_vmpy_VsfVsf(r0_ia, r0_dd);
        HVX_Vector r1_fa = Q6_Vqf32_vmpy_VsfVsf(r1_ia, r1_dd);

        r0_sum = Q6_Vqf32_vadd_Vqf32Vqf32(r0_sum, r0_fa);
        r1_sum = Q6_Vqf32_vadd_Vqf32Vqf32(r1_sum, r1_fa);
    }

    // Process leftovers, we still load full 4x4x2 block but zero out unused scales/blocks
    if (nloe) {
        HVX_Vector_x8 vy_q = hvx_vec_load_q8x4x8(y_q + i * y_qblk_size);
        HVX_Vector_x8 r0_q = hvx_vec_load_q4x4x8(r0_x_q + i * x_qblk_size);
        HVX_Vector_x8 r1_q = hvx_vec_load_q4x4x8(r1_x_q + i * x_qblk_size);

        HVX_Vector r0_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_nloe(r0_q, vy_q, nloe));
        HVX_Vector r1_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_nloe(r1_q, vy_q, nloe));

        HVX_Vector vy_d = Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (y_d + i * y_dblk_size));
        HVX_Vector r0_d = Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (r0_x_d + i * x_dblk_size));
        HVX_Vector r1_d = Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (r1_x_d + i * x_dblk_size));

        HVX_Vector r0_dd = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(r0_d, vy_d)));
        HVX_Vector r1_dd = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(r1_d, vy_d)));

        // Zero out unused scales
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe / 8);
        r0_dd                = Q6_V_vand_QV(bmask, r0_dd);
        r1_dd                = Q6_V_vand_QV(bmask, r1_dd);

        HVX_Vector r0_fa = Q6_Vqf32_vmpy_VsfVsf(r0_ia, r0_dd);
        HVX_Vector r1_fa = Q6_Vqf32_vmpy_VsfVsf(r1_ia, r1_dd);

        r0_sum = Q6_Vqf32_vadd_Vqf32Vqf32(r0_sum, r0_fa);
        r1_sum = Q6_Vqf32_vadd_Vqf32Vqf32(r1_sum, r1_fa);
    }

    // Convert into fp32 and reduce
    r0_sum = hvx_vec_fp32_reduce_sum(Q6_Vsf_equals_Vqf32(r0_sum));
    r1_sum = hvx_vec_fp32_reduce_sum(Q6_Vsf_equals_Vqf32(r1_sum));
    HVX_VectorPair p0 = Q6_W_vshuff_VVR(r1_sum, r0_sum, 4);

    hvx_vec_store_u(&s[0], 8, Q6_V_lo_W(p0));
}

static void vec_dot_q8x4x2_q8x4x2(const int n, float * restrict s, const void * restrict vx, const void * restrict vy) {
    assert(n % 32 == 0);  // min sub-block size
    assert((unsigned long) vx % 128 == 0);
    assert((unsigned long) vy % 128 == 0);

    const uint32_t qk = QK_Q4_0x4x2 * 4;

    const uint32_t x_dblk_size = 8 * 4 * 2;                                  // 32x __fp16
    const uint32_t x_qblk_size = qk;                                         // int8
    const uint32_t x_qrow_size = n;                                          // int8 (not padded)

    const uint32_t y_dblk_size = 8 * 4 * 2;                                  // 32x __fp16
    const uint32_t y_qblk_size = qk;                                         // int8
    const uint32_t y_qrow_size = n;                                          // int8 (not padded)

    const uint8_t * restrict r0_x_q = ((const uint8_t *) vx + 0);            // quants first
    const uint8_t * restrict r0_x_d = ((const uint8_t *) vx + x_qrow_size);  // then scales

    const uint8_t * restrict y_q = ((const uint8_t *) vy + 0);               // quants first
    const uint8_t * restrict y_d = ((const uint8_t *) vy + y_qrow_size);     // then scales

    // Row sum (qf32)
    HVX_Vector r0_sum = Q6_V_vsplat_R(0);

    // Multiply and accumulate into int32.
    // Compute combined scale (fp32).
    // Apply scale to acc and accumulate into the row sum (qf32).

    const uint32_t nb   = n / qk;  // num full blocks
    int32_t        nloe = n % qk;  // num leftover elemements (must be signed)

    uint32_t i = 0;
    for (; i < nb; i++) {
        HVX_Vector_x8 vy_q = hvx_vec_load_q8x4x8(y_q + i * y_qblk_size);
        HVX_Vector_x8 r0_q = hvx_vec_load_q8x4x8(r0_x_q + i * x_qblk_size);

        HVX_Vector r0_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_full(r0_q, vy_q));

        HVX_Vector vy_d = Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (y_d + i * y_dblk_size));
        HVX_Vector r0_d = Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (r0_x_d + i * x_dblk_size));

        HVX_Vector r0_dd = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(r0_d, vy_d)));

        HVX_Vector r0_fa = Q6_Vqf32_vmpy_VsfVsf(r0_ia, r0_dd);

        r0_sum = Q6_Vqf32_vadd_Vqf32Vqf32(r0_sum, r0_fa);
    }

    // Process leftovers, we still load full 4x4x2 block but zero out unused scales/blocks
    if (nloe) {
        HVX_Vector_x8 vy_q = hvx_vec_load_q8x4x8(y_q + i * y_qblk_size);
        HVX_Vector_x8 r0_q = hvx_vec_load_q8x4x8(r0_x_q + i * x_qblk_size);

        HVX_Vector r0_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_nloe(r0_q, vy_q, nloe));

        HVX_Vector vy_d = Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (y_d + i * y_dblk_size));
        HVX_Vector r0_d = Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (r0_x_d + i * x_dblk_size));

        HVX_Vector r0_dd = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(r0_d, vy_d)));

        // Zero out unused scales
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe / 8);
        r0_dd                = Q6_V_vand_QV(bmask, r0_dd);

        HVX_Vector r0_fa = Q6_Vqf32_vmpy_VsfVsf(r0_ia, r0_dd);

        r0_sum = Q6_Vqf32_vadd_Vqf32Vqf32(r0_sum, r0_fa);
    }

    // Reduce and convert into fp32
    r0_sum = hvx_vec_fp32_reduce_sum(Q6_Vsf_equals_Vqf32(r0_sum));

    hvx_vec_store_u(&s[0], 4, r0_sum);
}

static void vec_dot_q8x4x2_q8x4x2_rx2(const int n,
                                      float * restrict s,
                                      const void * restrict vx,
                                      uint32_t vx_row_size,
                                      const void * restrict vy) {
    assert(n % 32 == 0);  // min sub-block size
    assert((unsigned long) vx % 128 == 0);
    assert((unsigned long) vy % 128 == 0);

    const uint32_t qk = QK_Q4_0x4x2 * 4;

    const uint32_t x_dblk_size = 8 * 4 * 2;                                                        // 32x __fp16
    const uint32_t x_qblk_size = qk;                                                               // int8
    const uint32_t x_qrow_size = n;                                                                // int8 (not padded)

    const uint32_t y_dblk_size = 8 * 4 * 2;                                                        // 32x __fp16
    const uint32_t y_qblk_size = qk;                                                               // int8
    const uint32_t y_qrow_size = n;                                                                // int8 (not padded)

    const uint8_t * restrict r0_x_q = ((const uint8_t *) (vx + (0 * vx_row_size)) + 0);            // quants first
    const uint8_t * restrict r0_x_d = ((const uint8_t *) (vx + (0 * vx_row_size)) + x_qrow_size);  // then scales

    const uint8_t * restrict r1_x_q = ((const uint8_t *) (vx + (1 * vx_row_size)) + 0);            // quants first
    const uint8_t * restrict r1_x_d = ((const uint8_t *) (vx + (1 * vx_row_size)) + x_qrow_size);  // then scales

    const uint8_t * restrict y_q = ((const uint8_t *) vy + 0);                                     // quants first
    const uint8_t * restrict y_d = ((const uint8_t *) vy + y_qrow_size);                           // then scales

    // Row sum (qf32)
    HVX_Vector r0_sum = Q6_V_vsplat_R(0);
    HVX_Vector r1_sum = Q6_V_vsplat_R(0);

    // Multiply and accumulate into int32.
    // Compute combined scale (fp32).
    // Apply scale to acc and accumulate into the row sum (qf32).

    const uint32_t nb   = n / qk;  // num full blocks
    int32_t        nloe = n % qk;  // num leftover elemements (must be signed)

    uint32_t i = 0;
    for (; i < nb; i++) {
        HVX_Vector_x8 vy_q = hvx_vec_load_q8x4x8(y_q + i * y_qblk_size);
        HVX_Vector_x8 r0_q = hvx_vec_load_q8x4x8(r0_x_q + i * x_qblk_size);
        HVX_Vector_x8 r1_q = hvx_vec_load_q8x4x8(r1_x_q + i * x_qblk_size);

        HVX_Vector r0_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_full(r0_q, vy_q));
        HVX_Vector r1_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_full(r1_q, vy_q));

        HVX_Vector vy_d = Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (y_d + i * y_dblk_size));
        HVX_Vector r0_d = Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (r0_x_d + i * x_dblk_size));
        HVX_Vector r1_d = Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (r1_x_d + i * x_dblk_size));

        HVX_Vector r0_dd = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(r0_d, vy_d)));
        HVX_Vector r1_dd = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(r1_d, vy_d)));

        HVX_Vector r0_fa = Q6_Vqf32_vmpy_VsfVsf(r0_ia, r0_dd);
        HVX_Vector r1_fa = Q6_Vqf32_vmpy_VsfVsf(r1_ia, r1_dd);

        r0_sum = Q6_Vqf32_vadd_Vqf32Vqf32(r0_sum, r0_fa);
        r1_sum = Q6_Vqf32_vadd_Vqf32Vqf32(r1_sum, r1_fa);
    }

    // Process leftovers, we still load full 4x4x2 block but zero out unused scales/blocks
    if (nloe) {
        HVX_Vector_x8 vy_q = hvx_vec_load_q8x4x8(y_q + i * y_qblk_size);
        HVX_Vector_x8 r0_q = hvx_vec_load_q8x4x8(r0_x_q + i * x_qblk_size);
        HVX_Vector_x8 r1_q = hvx_vec_load_q8x4x8(r1_x_q + i * x_qblk_size);

        HVX_Vector r0_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_nloe(r0_q, vy_q, nloe));
        HVX_Vector r1_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_nloe(r1_q, vy_q, nloe));

        HVX_Vector vy_d = Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (y_d + i * y_dblk_size));
        HVX_Vector r0_d = Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (r0_x_d + i * x_dblk_size));
        HVX_Vector r1_d = Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (r1_x_d + i * x_dblk_size));

        HVX_Vector r0_dd = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(r0_d, vy_d)));
        HVX_Vector r1_dd = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(r1_d, vy_d)));

        // Zero out unused scales
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe / 8);
        r0_dd                = Q6_V_vand_QV(bmask, r0_dd);
        r1_dd                = Q6_V_vand_QV(bmask, r1_dd);

        HVX_Vector r0_fa = Q6_Vqf32_vmpy_VsfVsf(r0_ia, r0_dd);
        HVX_Vector r1_fa = Q6_Vqf32_vmpy_VsfVsf(r1_ia, r1_dd);

        r0_sum = Q6_Vqf32_vadd_Vqf32Vqf32(r0_sum, r0_fa);
        r1_sum = Q6_Vqf32_vadd_Vqf32Vqf32(r1_sum, r1_fa);
    }

    // Convert into fp32 and reduce
    r0_sum = hvx_vec_fp32_reduce_sum(Q6_Vsf_equals_Vqf32(r0_sum));
    r1_sum = hvx_vec_fp32_reduce_sum(Q6_Vsf_equals_Vqf32(r1_sum));
    HVX_VectorPair p0 = Q6_W_vshuff_VVR(r1_sum, r0_sum, 4);

    hvx_vec_store_u(&s[0], 8, Q6_V_lo_W(p0));
}

static void vec_dot_mxfp4x4x2_q8x4x2(const int n,
                                     float * restrict s,
                                     const void * restrict vx,
                                     const void * restrict vy) {
    assert(n % 32 == 0);  // min sub-block size
    assert((unsigned long) vx % 128 == 0);
    assert((unsigned long) vy % 128 == 0);

    const uint32_t qk = QK_MXFP4x4x2 * 4;

    const uint32_t x_dblk_size = 8 * 4 * 1;                                  // 32x e8m0
    const uint32_t x_qblk_size = qk / 2;                                     // fp4
    const uint32_t x_qrow_size = n / 2;                                      // fp4 (not padded)

    const uint32_t y_dblk_size = 8 * 4 * 2;                                  // 32x __fp16
    const uint32_t y_qblk_size = qk;                                         // int8
    const uint32_t y_qrow_size = n;                                          // int8 (not padded)

    const uint8_t * restrict r0_x_q = ((const uint8_t *) vx + 0);            // quants first
    const uint8_t * restrict r0_x_d = ((const uint8_t *) vx + x_qrow_size);  // then scales

    const uint8_t * restrict y_q = ((const uint8_t *) vy + 0);               // quants first
    const uint8_t * restrict y_d = ((const uint8_t *) vy + y_qrow_size);     // then scales

    // Row sum (qf32)
    HVX_Vector r0_sum = Q6_V_vsplat_R(0);

    // Multiply and accumulate into int32.
    // Compute combined scale (fp32).
    // Apply scale to acc and accumulate into the row sum (qf32).

    const uint32_t nb   = n / qk;  // num full blocks
    int32_t        nloe = n % qk;  // num leftover elemements (must be signed)

    uint32_t i = 0;
    for (; i < nb; i++) {
        HVX_Vector_x8 vy_q = hvx_vec_load_q8x4x8(y_q + i * y_qblk_size);
        HVX_Vector_x8 r0_q = hvx_vec_load_mxfp4x4x8(r0_x_q + i * x_qblk_size);

        HVX_Vector r0_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_full(r0_q, vy_q));

        HVX_Vector vy_d = *(const HVX_UVector *) (y_d + i * y_dblk_size);
        HVX_Vector r0_d = *(const HVX_UVector *) (r0_x_d + i * x_dblk_size);

        // Convert vy_d from fp16 to fp32 while applying 0.5 scaling which is used for e8m0 halving
        HVX_Vector half = Q6_Vh_vsplat_R(0x3800);  // 0.5 in fp16
        vy_d            = Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(vy_d), half));
        vy_d            = Q6_Vsf_equals_Vqf32(vy_d);

        // Convert rX_d scales from e8m0 to fp32
        // Expand and zero-pad 32x uint8 e8m0 values to uint32s : 0 0 0 0, 0 0 0 1, 0 0 0 2, ...
        // Left shift with zero fill to create FP32
        // FIXME: might need to handle zero as a special case (see ggml-cpu code)
        HVX_Vector expand    = *(const HVX_Vector *) expand_x32_e8m0;
        HVX_Vector e8m0_mask = Q6_V_vsplat_R(0x000000ff);
        r0_d                 = Q6_V_vdelta_VV(r0_d, expand);
        r0_d                 = Q6_V_vand_VV(r0_d, e8m0_mask);
        r0_d                 = Q6_Vw_vasl_VwR(r0_d, 23);

        HVX_Vector r0_dd = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(r0_d, vy_d));

        HVX_Vector r0_fa = Q6_Vqf32_vmpy_VsfVsf(r0_ia, r0_dd);

        r0_sum = Q6_Vqf32_vadd_Vqf32Vqf32(r0_sum, r0_fa);
    }

    // Process leftovers
    if (nloe) {
        HVX_Vector_x8 vy_q = hvx_vec_load_q8x4x8(y_q + i * y_qblk_size);
        HVX_Vector_x8 r0_q = hvx_vec_load_mxfp4x4x8(r0_x_q + i * x_qblk_size);

        HVX_Vector r0_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_full(r0_q, vy_q));

        HVX_Vector vy_d = *(const HVX_UVector *) (y_d + i * y_dblk_size);
        HVX_Vector r0_d = *(const HVX_UVector *) (r0_x_d + i * x_dblk_size);

        // Convert vy_d from fp16 to fp32 while applying 0.5 scaling which is used for e8m0 halving
        HVX_Vector half = Q6_Vh_vsplat_R(0x3800);  // 0.5 in fp16
        vy_d            = Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(vy_d), half));
        vy_d            = Q6_Vsf_equals_Vqf32(vy_d);

        // Convert rX_d scales from e8m0 to fp32
        // Expand and zero-pad 32x uint8 e8m0 values to uint32s : 0 0 0 0, 0 0 0 1, 0 0 0 2, ...
        // Left shift with zero fill to create FP32
        // FIXME: might need to handle zero as a special case (see ggml-cpu code)
        HVX_Vector expand    = *(const HVX_Vector *) expand_x32_e8m0;
        HVX_Vector e8m0_mask = Q6_V_vsplat_R(0x000000ff);
        r0_d                 = Q6_V_vdelta_VV(r0_d, expand);
        r0_d                 = Q6_V_vand_VV(r0_d, e8m0_mask);
        r0_d                 = Q6_Vw_vasl_VwR(r0_d, 23);

        HVX_Vector r0_dd = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(r0_d, vy_d));

        // Zero-out unused scales
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe / 8);
        r0_dd                = Q6_V_vand_QV(bmask, r0_dd);

        HVX_Vector r0_fa = Q6_Vqf32_vmpy_VsfVsf(r0_ia, r0_dd);

        r0_sum = Q6_Vqf32_vadd_Vqf32Vqf32(r0_sum, r0_fa);
    }

    // Reduce and convert into fp32
    r0_sum = hvx_vec_fp32_reduce_sum(Q6_Vsf_equals_Vqf32(r0_sum));

    hvx_vec_store_u(&s[0], 4, r0_sum);
}

static void vec_dot_mxfp4x4x2_q8x4x2_rx2(const int n,
                                         float * restrict s,
                                         const void * restrict vx,
                                         uint32_t vx_row_size,
                                         const void * restrict vy) {
    assert(n % 32 == 0);  // min sub-block size
    assert((unsigned long) vx % 128 == 0);
    assert((unsigned long) vy % 128 == 0);

    const uint32_t qk = QK_MXFP4x4x2 * 4;

    const uint32_t x_dblk_size = 8 * 4 * 1;                                                        // 32x e8m0
    const uint32_t x_qblk_size = qk / 2;                                                           // fp4
    const uint32_t x_qrow_size = n / 2;                                                            // fp4 (not padded)

    const uint32_t y_dblk_size = 8 * 4 * 2;                                                        // 32x __fp16
    const uint32_t y_qblk_size = qk;                                                               // int8
    const uint32_t y_qrow_size = n;                                                                // int8 (not padded)

    const uint8_t * restrict r0_x_q = ((const uint8_t *) (vx + (0 * vx_row_size)) + 0);            // quants first
    const uint8_t * restrict r0_x_d = ((const uint8_t *) (vx + (0 * vx_row_size)) + x_qrow_size);  // then scales

    const uint8_t * restrict r1_x_q = ((const uint8_t *) (vx + (1 * vx_row_size)) + 0);            // quants first
    const uint8_t * restrict r1_x_d = ((const uint8_t *) (vx + (1 * vx_row_size)) + x_qrow_size);  // then scales

    const uint8_t * restrict y_q = ((const uint8_t *) vy + 0);                                     // quants first
    const uint8_t * restrict y_d = ((const uint8_t *) vy + y_qrow_size);                           // then scales

    // Row sum (qf32)
    HVX_Vector r0_sum = Q6_V_vsplat_R(0);
    HVX_Vector r1_sum = Q6_V_vsplat_R(0);

    // Multiply and accumulate into int32.
    // Compute combined scale (fp32).
    // Apply scale to acc and accumulate into the row sum (qf32).

    const uint32_t nb   = n / qk;  // num full blocks
    int32_t        nloe = n % qk;  // num leftover elemements (must be signed)

    uint32_t i = 0;
    for (; i < nb; i++) {
        HVX_Vector_x8 vy_q = hvx_vec_load_q8x4x8(y_q + i * y_qblk_size);
        HVX_Vector_x8 r0_q = hvx_vec_load_mxfp4x4x8(r0_x_q + i * x_qblk_size);
        HVX_Vector_x8 r1_q = hvx_vec_load_mxfp4x4x8(r1_x_q + i * x_qblk_size);

        HVX_Vector r0_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_full(r0_q, vy_q));
        HVX_Vector r1_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_full(r1_q, vy_q));

        HVX_Vector vy_d = *(const HVX_UVector *) (y_d + i * y_dblk_size);
        HVX_Vector r0_d = *(const HVX_UVector *) (r0_x_d + i * x_dblk_size);
        HVX_Vector r1_d = *(const HVX_UVector *) (r1_x_d + i * x_dblk_size);

        // Convert vy_d from fp16 to fp32 while applying 0.5 scaling which is used for e8m0 halving
        HVX_Vector half = Q6_Vh_vsplat_R(0x3800);  // 0.5 in fp16
        vy_d            = Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(vy_d), half));
        vy_d            = Q6_Vsf_equals_Vqf32(vy_d);

        // Convert rX_d scales from e8m0 to fp32
        // Expand and zero-pad 32x uint8 e8m0 values to uint32s : 0 0 0 0, 0 0 0 1, 0 0 0 2, ...
        // Left shift with zero fill to create FP32
        // FIXME: might need to handle zero as a special case (see ggml-cpu code)
        HVX_Vector expand    = *(const HVX_Vector *) expand_x32_e8m0;
        HVX_Vector e8m0_mask = Q6_V_vsplat_R(0x000000ff);
        r0_d                 = Q6_V_vdelta_VV(r0_d, expand);
        r0_d                 = Q6_V_vand_VV(r0_d, e8m0_mask);
        r0_d                 = Q6_Vw_vasl_VwR(r0_d, 23);
        r1_d                 = Q6_V_vdelta_VV(r1_d, expand);
        r1_d                 = Q6_V_vand_VV(r1_d, e8m0_mask);
        r1_d                 = Q6_Vw_vasl_VwR(r1_d, 23);

        HVX_Vector r0_dd = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(r0_d, vy_d));
        HVX_Vector r1_dd = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(r1_d, vy_d));

        HVX_Vector r0_fa = Q6_Vqf32_vmpy_VsfVsf(r0_ia, r0_dd);
        HVX_Vector r1_fa = Q6_Vqf32_vmpy_VsfVsf(r1_ia, r1_dd);

        r0_sum = Q6_Vqf32_vadd_Vqf32Vqf32(r0_sum, r0_fa);
        r1_sum = Q6_Vqf32_vadd_Vqf32Vqf32(r1_sum, r1_fa);
    }

    // Process leftovers
    if (nloe) {
        HVX_Vector_x8 vy_q = hvx_vec_load_q8x4x8(y_q + i * y_qblk_size);
        HVX_Vector_x8 r0_q = hvx_vec_load_mxfp4x4x8(r0_x_q + i * x_qblk_size);
        HVX_Vector_x8 r1_q = hvx_vec_load_mxfp4x4x8(r1_x_q + i * x_qblk_size);

        HVX_Vector r0_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_full(r0_q, vy_q));
        HVX_Vector r1_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_full(r1_q, vy_q));

        HVX_Vector vy_d = *(const HVX_UVector *) (y_d + i * y_dblk_size);
        HVX_Vector r0_d = *(const HVX_UVector *) (r0_x_d + i * x_dblk_size);
        HVX_Vector r1_d = *(const HVX_UVector *) (r1_x_d + i * x_dblk_size);

        // Convert vy_d from fp16 to fp32 while applying 0.5 scaling which is used for e8m0 halving
        HVX_Vector half = Q6_Vh_vsplat_R(0x3800);  // 0.5 in fp16
        vy_d            = Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(vy_d), half));
        vy_d            = Q6_Vsf_equals_Vqf32(vy_d);

        // Convert rX_d scales from e8m0 to fp32
        // Expand and zero-pad 32x uint8 e8m0 values to uint32s : 0 0 0 0, 0 0 0 1, 0 0 0 2, ...
        // Left shift with zero fill to create FP32
        // FIXME: might need to handle zero as a special case (see ggml-cpu code)
        HVX_Vector expand    = *(const HVX_Vector *) expand_x32_e8m0;
        HVX_Vector e8m0_mask = Q6_V_vsplat_R(0x000000ff);
        r0_d                 = Q6_V_vdelta_VV(r0_d, expand);
        r0_d                 = Q6_V_vand_VV(r0_d, e8m0_mask);
        r0_d                 = Q6_Vw_vasl_VwR(r0_d, 23);
        r1_d                 = Q6_V_vdelta_VV(r1_d, expand);
        r1_d                 = Q6_V_vand_VV(r1_d, e8m0_mask);
        r1_d                 = Q6_Vw_vasl_VwR(r1_d, 23);

        HVX_Vector r0_dd = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(r0_d, vy_d));
        HVX_Vector r1_dd = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(r1_d, vy_d));

        // Zero-out unused scales
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe / 8);
        r0_dd                = Q6_V_vand_QV(bmask, r0_dd);
        r1_dd                = Q6_V_vand_QV(bmask, r1_dd);

        HVX_Vector r0_fa = Q6_Vqf32_vmpy_VsfVsf(r0_ia, r0_dd);
        HVX_Vector r1_fa = Q6_Vqf32_vmpy_VsfVsf(r1_ia, r1_dd);

        r0_sum = Q6_Vqf32_vadd_Vqf32Vqf32(r0_sum, r0_fa);
        r1_sum = Q6_Vqf32_vadd_Vqf32Vqf32(r1_sum, r1_fa);
    }

    // Convert into fp32 and reduce
    r0_sum = hvx_vec_fp32_reduce_sum(Q6_Vsf_equals_Vqf32(r0_sum));
    r1_sum = hvx_vec_fp32_reduce_sum(Q6_Vsf_equals_Vqf32(r1_sum));
    HVX_VectorPair p0 = Q6_W_vshuff_VVR(r1_sum, r0_sum, 4);

    hvx_vec_store_u(&s[0], 8, Q6_V_lo_W(p0));
}

static void vec_dot_f16_f16(const int n, float * restrict s, const void * restrict x, const void * restrict y) {
    const HVX_UVector * restrict vx = (const HVX_UVector * restrict) x;
    const HVX_UVector * restrict vy = (const HVX_UVector * restrict) y;

    uint32_t nvec = n / 64;  // 128 bytes / 2 bytes = 64 elements
    uint32_t nloe = n % 64;

    HVX_Vector rsum = Q6_V_vsplat_R(0);

    for (uint32_t i = 0; i < nvec; ++i) {
        HVX_Vector xv = vx[i];
        HVX_Vector yv = vy[i];
        HVX_VectorPair p = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(xv), Q6_Vh_vshuff_Vh(yv));
        rsum = Q6_Vqf32_vadd_Vqf32Vqf32(rsum, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p), Q6_V_hi_W(p)));
    }

    if (nloe) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 2);
        HVX_Vector xv = Q6_V_vand_QV(bmask, vx[nvec]);
        HVX_Vector yv = vy[nvec];
        HVX_VectorPair p = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(xv), Q6_Vh_vshuff_Vh(yv));
        rsum = Q6_Vqf32_vadd_Vqf32Vqf32(rsum, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p), Q6_V_hi_W(p)));
    }

    rsum = Q6_Vsf_equals_Vqf32(hvx_vec_qf32_reduce_sum(rsum));
    hvx_vec_store_u(s, 4, rsum);
}

static void vec_dot_f16_f16_4x4(const int n,
                                float * restrict dst,
                                size_t dst_stride,
                                const void * restrict vx,
                                size_t vx_stride,
                                const void * restrict vy,
                                size_t vy_stride) {
    const HVX_UVector * restrict x0 = (const HVX_UVector * restrict) vx;
    const HVX_UVector * restrict x1 = (const HVX_UVector * restrict) ((const uint8_t *) vx + vx_stride);
    const HVX_UVector * restrict x2 = (const HVX_UVector * restrict) ((const uint8_t *) vx + 2 * vx_stride);
    const HVX_UVector * restrict x3 = (const HVX_UVector * restrict) ((const uint8_t *) vx + 3 * vx_stride);

    const HVX_UVector * restrict y0 = (const HVX_UVector * restrict) vy;
    const HVX_UVector * restrict y1 = (const HVX_UVector * restrict) ((const uint8_t *) vy + vy_stride);
    const HVX_UVector * restrict y2 = (const HVX_UVector * restrict) ((const uint8_t *) vy + 2 * vy_stride);
    const HVX_UVector * restrict y3 = (const HVX_UVector * restrict) ((const uint8_t *) vy + 3 * vy_stride);

    float * restrict d0 = dst;
    float * restrict d1 = (float *) ((uint8_t *) dst + dst_stride);
    float * restrict d2 = (float *) ((uint8_t *) dst + 2 * dst_stride);
    float * restrict d3 = (float *) ((uint8_t *) dst + 3 * dst_stride);

    HVX_Vector acc00 = Q6_V_vsplat_R(0);
    HVX_Vector acc01 = Q6_V_vsplat_R(0);
    HVX_Vector acc02 = Q6_V_vsplat_R(0);
    HVX_Vector acc03 = Q6_V_vsplat_R(0);

    HVX_Vector acc10 = Q6_V_vsplat_R(0);
    HVX_Vector acc11 = Q6_V_vsplat_R(0);
    HVX_Vector acc12 = Q6_V_vsplat_R(0);
    HVX_Vector acc13 = Q6_V_vsplat_R(0);

    HVX_Vector acc20 = Q6_V_vsplat_R(0);
    HVX_Vector acc21 = Q6_V_vsplat_R(0);
    HVX_Vector acc22 = Q6_V_vsplat_R(0);
    HVX_Vector acc23 = Q6_V_vsplat_R(0);

    HVX_Vector acc30 = Q6_V_vsplat_R(0);
    HVX_Vector acc31 = Q6_V_vsplat_R(0);
    HVX_Vector acc32 = Q6_V_vsplat_R(0);
    HVX_Vector acc33 = Q6_V_vsplat_R(0);

    uint32_t nvec = n / 64;  // 128 bytes / 2 bytes = 64 elements
    uint32_t nloe = n % 64;

    for (uint32_t i = 0; i < nvec; ++i) {
        HVX_Vector xv0 = x0[i];
        HVX_Vector xv1 = x1[i];
        HVX_Vector xv2 = x2[i];
        HVX_Vector xv3 = x3[i];

        HVX_Vector yv0 = y0[i];
        HVX_Vector yv1 = y1[i];
        HVX_Vector yv2 = y2[i];
        HVX_Vector yv3 = y3[i];

        // Process row 0
        HVX_VectorPair p00 = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(xv0), Q6_Vh_vshuff_Vh(yv0));
        HVX_VectorPair p01 = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(xv0), Q6_Vh_vshuff_Vh(yv1));
        HVX_VectorPair p02 = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(xv0), Q6_Vh_vshuff_Vh(yv2));
        HVX_VectorPair p03 = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(xv0), Q6_Vh_vshuff_Vh(yv3));

        acc00 = Q6_Vqf32_vadd_Vqf32Vqf32(acc00, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p00), Q6_V_hi_W(p00)));
        acc01 = Q6_Vqf32_vadd_Vqf32Vqf32(acc01, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p01), Q6_V_hi_W(p01)));
        acc02 = Q6_Vqf32_vadd_Vqf32Vqf32(acc02, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p02), Q6_V_hi_W(p02)));
        acc03 = Q6_Vqf32_vadd_Vqf32Vqf32(acc03, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p03), Q6_V_hi_W(p03)));

        // Process row 1
        HVX_VectorPair p10 = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(xv1), Q6_Vh_vshuff_Vh(yv0));
        HVX_VectorPair p11 = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(xv1), Q6_Vh_vshuff_Vh(yv1));
        HVX_VectorPair p12 = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(xv1), Q6_Vh_vshuff_Vh(yv2));
        HVX_VectorPair p13 = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(xv1), Q6_Vh_vshuff_Vh(yv3));

        acc10 = Q6_Vqf32_vadd_Vqf32Vqf32(acc10, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p10), Q6_V_hi_W(p10)));
        acc11 = Q6_Vqf32_vadd_Vqf32Vqf32(acc11, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p11), Q6_V_hi_W(p11)));
        acc12 = Q6_Vqf32_vadd_Vqf32Vqf32(acc12, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p12), Q6_V_hi_W(p12)));
        acc13 = Q6_Vqf32_vadd_Vqf32Vqf32(acc13, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p13), Q6_V_hi_W(p13)));

        // Process row 2
        HVX_VectorPair p20 = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(xv2), Q6_Vh_vshuff_Vh(yv0));
        HVX_VectorPair p21 = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(xv2), Q6_Vh_vshuff_Vh(yv1));
        HVX_VectorPair p22 = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(xv2), Q6_Vh_vshuff_Vh(yv2));
        HVX_VectorPair p23 = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(xv2), Q6_Vh_vshuff_Vh(yv3));

        acc20 = Q6_Vqf32_vadd_Vqf32Vqf32(acc20, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p20), Q6_V_hi_W(p20)));
        acc21 = Q6_Vqf32_vadd_Vqf32Vqf32(acc21, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p21), Q6_V_hi_W(p21)));
        acc22 = Q6_Vqf32_vadd_Vqf32Vqf32(acc22, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p22), Q6_V_hi_W(p22)));
        acc23 = Q6_Vqf32_vadd_Vqf32Vqf32(acc23, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p23), Q6_V_hi_W(p23)));

        // Process row 3
        HVX_VectorPair p30 = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(xv3), Q6_Vh_vshuff_Vh(yv0));
        HVX_VectorPair p31 = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(xv3), Q6_Vh_vshuff_Vh(yv1));
        HVX_VectorPair p32 = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(xv3), Q6_Vh_vshuff_Vh(yv2));
        HVX_VectorPair p33 = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(xv3), Q6_Vh_vshuff_Vh(yv3));

        acc30 = Q6_Vqf32_vadd_Vqf32Vqf32(acc30, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p30), Q6_V_hi_W(p30)));
        acc31 = Q6_Vqf32_vadd_Vqf32Vqf32(acc31, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p31), Q6_V_hi_W(p31)));
        acc32 = Q6_Vqf32_vadd_Vqf32Vqf32(acc32, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p32), Q6_V_hi_W(p32)));
        acc33 = Q6_Vqf32_vadd_Vqf32Vqf32(acc33, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p33), Q6_V_hi_W(p33)));
    }

    if (nloe) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 2);

        HVX_Vector xv0 = Q6_V_vand_QV(bmask, x0[nvec]);
        HVX_Vector xv1 = Q6_V_vand_QV(bmask, x1[nvec]);
        HVX_Vector xv2 = Q6_V_vand_QV(bmask, x2[nvec]);
        HVX_Vector xv3 = Q6_V_vand_QV(bmask, x3[nvec]);

        HVX_Vector yv0 = y0[nvec];
        HVX_Vector yv1 = y1[nvec];
        HVX_Vector yv2 = y2[nvec];
        HVX_Vector yv3 = y3[nvec];

        // Process row 0
        HVX_VectorPair p00 = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(xv0), Q6_Vh_vshuff_Vh(yv0));
        HVX_VectorPair p01 = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(xv0), Q6_Vh_vshuff_Vh(yv1));
        HVX_VectorPair p02 = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(xv0), Q6_Vh_vshuff_Vh(yv2));
        HVX_VectorPair p03 = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(xv0), Q6_Vh_vshuff_Vh(yv3));

        acc00 = Q6_Vqf32_vadd_Vqf32Vqf32(acc00, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p00), Q6_V_hi_W(p00)));
        acc01 = Q6_Vqf32_vadd_Vqf32Vqf32(acc01, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p01), Q6_V_hi_W(p01)));
        acc02 = Q6_Vqf32_vadd_Vqf32Vqf32(acc02, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p02), Q6_V_hi_W(p02)));
        acc03 = Q6_Vqf32_vadd_Vqf32Vqf32(acc03, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p03), Q6_V_hi_W(p03)));

        // Process row 1
        HVX_VectorPair p10 = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(xv1), Q6_Vh_vshuff_Vh(yv0));
        HVX_VectorPair p11 = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(xv1), Q6_Vh_vshuff_Vh(yv1));
        HVX_VectorPair p12 = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(xv1), Q6_Vh_vshuff_Vh(yv2));
        HVX_VectorPair p13 = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(xv1), Q6_Vh_vshuff_Vh(yv3));

        acc10 = Q6_Vqf32_vadd_Vqf32Vqf32(acc10, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p10), Q6_V_hi_W(p10)));
        acc11 = Q6_Vqf32_vadd_Vqf32Vqf32(acc11, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p11), Q6_V_hi_W(p11)));
        acc12 = Q6_Vqf32_vadd_Vqf32Vqf32(acc12, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p12), Q6_V_hi_W(p12)));
        acc13 = Q6_Vqf32_vadd_Vqf32Vqf32(acc13, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p13), Q6_V_hi_W(p13)));

        // Process row 2
        HVX_VectorPair p20 = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(xv2), Q6_Vh_vshuff_Vh(yv0));
        HVX_VectorPair p21 = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(xv2), Q6_Vh_vshuff_Vh(yv1));
        HVX_VectorPair p22 = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(xv2), Q6_Vh_vshuff_Vh(yv2));
        HVX_VectorPair p23 = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(xv2), Q6_Vh_vshuff_Vh(yv3));

        acc20 = Q6_Vqf32_vadd_Vqf32Vqf32(acc20, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p20), Q6_V_hi_W(p20)));
        acc21 = Q6_Vqf32_vadd_Vqf32Vqf32(acc21, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p21), Q6_V_hi_W(p21)));
        acc22 = Q6_Vqf32_vadd_Vqf32Vqf32(acc22, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p22), Q6_V_hi_W(p22)));
        acc23 = Q6_Vqf32_vadd_Vqf32Vqf32(acc23, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p23), Q6_V_hi_W(p23)));

        // Process row 3
        HVX_VectorPair p30 = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(xv3), Q6_Vh_vshuff_Vh(yv0));
        HVX_VectorPair p31 = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(xv3), Q6_Vh_vshuff_Vh(yv1));
        HVX_VectorPair p32 = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(xv3), Q6_Vh_vshuff_Vh(yv2));
        HVX_VectorPair p33 = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(xv3), Q6_Vh_vshuff_Vh(yv3));

        acc30 = Q6_Vqf32_vadd_Vqf32Vqf32(acc30, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p30), Q6_V_hi_W(p30)));
        acc31 = Q6_Vqf32_vadd_Vqf32Vqf32(acc31, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p31), Q6_V_hi_W(p31)));
        acc32 = Q6_Vqf32_vadd_Vqf32Vqf32(acc32, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p32), Q6_V_hi_W(p32)));
        acc33 = Q6_Vqf32_vadd_Vqf32Vqf32(acc33, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p33), Q6_V_hi_W(p33)));
    }

    acc00 = Q6_Vsf_equals_Vqf32(hvx_vec_qf32_reduce_sum(acc00));
    acc01 = Q6_Vsf_equals_Vqf32(hvx_vec_qf32_reduce_sum(acc01));
    acc02 = Q6_Vsf_equals_Vqf32(hvx_vec_qf32_reduce_sum(acc02));
    acc03 = Q6_Vsf_equals_Vqf32(hvx_vec_qf32_reduce_sum(acc03));

    acc10 = Q6_Vsf_equals_Vqf32(hvx_vec_qf32_reduce_sum(acc10));
    acc11 = Q6_Vsf_equals_Vqf32(hvx_vec_qf32_reduce_sum(acc11));
    acc12 = Q6_Vsf_equals_Vqf32(hvx_vec_qf32_reduce_sum(acc12));
    acc13 = Q6_Vsf_equals_Vqf32(hvx_vec_qf32_reduce_sum(acc13));

    acc20 = Q6_Vsf_equals_Vqf32(hvx_vec_qf32_reduce_sum(acc20));
    acc21 = Q6_Vsf_equals_Vqf32(hvx_vec_qf32_reduce_sum(acc21));
    acc22 = Q6_Vsf_equals_Vqf32(hvx_vec_qf32_reduce_sum(acc22));
    acc23 = Q6_Vsf_equals_Vqf32(hvx_vec_qf32_reduce_sum(acc23));

    acc30 = Q6_Vsf_equals_Vqf32(hvx_vec_qf32_reduce_sum(acc30));
    acc31 = Q6_Vsf_equals_Vqf32(hvx_vec_qf32_reduce_sum(acc31));
    acc32 = Q6_Vsf_equals_Vqf32(hvx_vec_qf32_reduce_sum(acc32));
    acc33 = Q6_Vsf_equals_Vqf32(hvx_vec_qf32_reduce_sum(acc33));

    // accRC: Row R, Col C
    // dC: Col C pointer
    // dC[R]: Row R element

    hvx_vec_store_u(&d0[0], 4, acc00);
    hvx_vec_store_u(&d1[0], 4, acc01);
    hvx_vec_store_u(&d2[0], 4, acc02);
    hvx_vec_store_u(&d3[0], 4, acc03);

    hvx_vec_store_u(&d0[1], 4, acc10);
    hvx_vec_store_u(&d1[1], 4, acc11);
    hvx_vec_store_u(&d2[1], 4, acc12);
    hvx_vec_store_u(&d3[1], 4, acc13);

    hvx_vec_store_u(&d0[2], 4, acc20);
    hvx_vec_store_u(&d1[2], 4, acc21);
    hvx_vec_store_u(&d2[2], 4, acc22);
    hvx_vec_store_u(&d3[2], 4, acc23);

    hvx_vec_store_u(&d0[3], 4, acc30);
    hvx_vec_store_u(&d1[3], 4, acc31);
    hvx_vec_store_u(&d2[3], 4, acc32);
    hvx_vec_store_u(&d3[3], 4, acc33);
}

static void vec_dot_f16_f32(const int n, float * restrict s, const void * restrict x, const void * restrict y) {
    const HVX_UVector * restrict vx = (const HVX_UVector * restrict) x;
    const HVX_UVector * restrict vy = (const HVX_UVector * restrict) y;

    uint32_t nvec = n / VLEN_FP16; // num full fp16 hvx vectors
    uint32_t nloe = n % VLEN_FP16; // leftover elements

    const HVX_Vector zero = Q6_V_vsplat_R(0);

    HVX_Vector       rsum = Q6_V_vsplat_R(0);

    uint32_t i = 0;

    #pragma unroll(2)
    for (i = 0; i < nvec; i++) {
        // Load y (fp32) and convert into fp16
        HVX_Vector y0_qf = Q6_Vqf32_vsub_VsfVsf(vy[i*2+0], zero);  // 32 elements
        HVX_Vector y1_qf = Q6_Vqf32_vsub_VsfVsf(vy[i*2+1], zero);  // 32 elements
        HVX_Vector y_hf  = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(y1_qf, y0_qf));

        // Load x (fp16)
        HVX_Vector x_hf = vx[i];

        HVX_VectorPair xy_qf = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(x_hf), y_hf);

        rsum = Q6_Vqf32_vadd_Vqf32Vqf32(rsum, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(xy_qf),  Q6_V_hi_W(xy_qf)));
    }

    if (nloe) {
        // Load y (fp32) and convert into fp16
        HVX_Vector y0_qf = Q6_Vqf32_vsub_VsfVsf(vy[i*2+0], zero);  // 32 elements
        HVX_Vector y1_qf = Q6_Vqf32_vsub_VsfVsf(vy[i*2+1], zero);  // 32 elements
        HVX_Vector y_hf  = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(y1_qf, y0_qf));

        // Load x (fp16) and zero-out unused elements
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 2);
        HVX_Vector      x_hf = Q6_V_vand_QV(bmask, vx[i]);

        HVX_VectorPair xy_qf = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(x_hf), y_hf);

        rsum = Q6_Vqf32_vadd_Vqf32Vqf32(rsum, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(xy_qf),  Q6_V_hi_W(xy_qf)));
    }

    rsum = Q6_Vsf_equals_Vqf32(hvx_vec_qf32_reduce_sum(rsum));
    hvx_vec_store_u(&s[0], 4, rsum);
}

static inline void quantize_row_fp32_to_fp16_vtcm(const float * restrict x, void * restrict y, uint32_t n) {
    assert((unsigned long) x % 128 == 0);
    assert((unsigned long) y % 128 == 0);

    const HVX_Vector * restrict vx = (const HVX_Vector *) x;
    HVX_Vector * restrict vy       = (HVX_Vector *) y;

    const HVX_Vector zero = Q6_V_vsplat_R(0);
    uint32_t nvec = n / 64;
    uint32_t nloe = n % 64;

    for (uint32_t i = 0; i < nvec; i++) {
        HVX_Vector x0 = vx[i * 2 + 0];
        HVX_Vector x1 = vx[i * 2 + 1];

        HVX_Vector x0_qf = Q6_Vqf32_vsub_VsfVsf(x0, zero);
        HVX_Vector x1_qf = Q6_Vqf32_vsub_VsfVsf(x1, zero);

        HVX_Vector y_hf = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(x1_qf, x0_qf));
        vy[i]           = y_hf;
    }

    if (nloe) {
        HVX_Vector x0 = vx[nvec * 2 + 0];
        HVX_Vector x1 = (nloe > 32) ? vx[nvec * 2 + 1] : zero;

        HVX_Vector x0_qf = Q6_Vqf32_vsub_VsfVsf(x0, zero);
        HVX_Vector x1_qf = Q6_Vqf32_vsub_VsfVsf(x1, zero);

        HVX_Vector y_hf = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(x1_qf, x0_qf));
        hvx_vec_store_u(&vy[nvec], nloe * 2, y_hf);
    }
}

#define htp_matmul_preamble            \
    const uint32_t ne00 = src0->ne[0]; \
    const uint32_t ne01 = src0->ne[1]; \
    const uint32_t ne02 = src0->ne[2]; \
    const uint32_t ne03 = src0->ne[3]; \
                                       \
    const uint32_t ne10 = src1->ne[0]; \
    const uint32_t ne11 = src1->ne[1]; \
    const uint32_t ne12 = src1->ne[2]; \
    const uint32_t ne13 = src1->ne[3]; \
                                       \
    const uint32_t ne0 = dst->ne[0];   \
    const uint32_t ne1 = dst->ne[1];   \
    const uint32_t ne2 = dst->ne[2];   \
    const uint32_t ne3 = dst->ne[3];   \
                                       \
    const uint32_t nb00 = src0->nb[0]; \
    const uint32_t nb01 = src0->nb[1]; \
    const uint32_t nb02 = src0->nb[2]; \
    const uint32_t nb03 = src0->nb[3]; \
                                       \
    const uint32_t nb10 = src1->nb[0]; \
    const uint32_t nb11 = src1->nb[1]; \
    const uint32_t nb12 = src1->nb[2]; \
    const uint32_t nb13 = src1->nb[3]; \
                                       \
    const uint32_t nb0 = dst->nb[0];   \
    const uint32_t nb1 = dst->nb[1];   \
    const uint32_t nb2 = dst->nb[2];   \
    const uint32_t nb3 = dst->nb[3];

// q8x4 src1 tensor is already in VTCM spad
static void matmul(struct htp_matmul_type * mt,
                   struct htp_tensor * restrict src0,
                   struct htp_tensor * restrict src1,
                   struct htp_tensor * restrict dst,
                   struct htp_spad * restrict src0_spad,
                   struct htp_spad * restrict src1_spad,
                   struct htp_spad * restrict dst_spad,
                   uint32_t    nth,
                   uint32_t    ith,
                   uint32_t    src0_nrows_per_thread,
                   dma_queue * dma_queue) {
    htp_matmul_preamble;

    const uint32_t src0_nrows = ne01 * ne02 * ne03;  // src0 rows
    const uint32_t src1_nrows = ne11 * ne12 * ne13;  // src1 rows

    const uint32_t src0_start_row  = src0_nrows_per_thread * ith;
    const uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, src0_nrows);
    const uint32_t src0_end_row_x2 = src0_start_row + ((src0_end_row - src0_start_row) & ~1U);

    // no work for this thread
    if (src0_start_row >= src0_end_row) {
        return;
    }

    const size_t dst_row_size  = nb1;
    const size_t src0_row_size = nb01;
    const size_t src1_row_size = q8x4x2_row_size(ne10);

    const size_t src0_row_size_padded = htp_round_up(src0_row_size, 128);

    // Per-thread VTCM scratchpads for all tensors
    // Note that the entire src1 tensor is already in VTCM
    // For other tensors we allocate N rows per thread, padded to HVX vector size
    uint8_t * restrict spad_dst  = dst_spad->data + dst_spad->size_per_thread * ith;
    uint8_t * restrict spad_src0 = src0_spad->data + src0_spad->size_per_thread * ith;
    uint8_t * restrict src1_data = src1_spad->data;

    volatile uint64_t t1, t2;
    t1 = HAP_perf_get_qtimer_count();

    const uint8_t * restrict src0_row = (const uint8_t *) src0->data;

    // Prefill spad with src0 rows
    #pragma unroll(4)
    for (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
        const int is0 = (ir0 - src0_start_row);
        if (is0 >= HTP_SPAD_SRC0_NROWS) {
            break;
        }
        dma_queue_push_ddr_to_vtcm(dma_queue, dma_make_ptr(spad_src0 + is0 * src0_row_size_padded, src0_row + ir0 * src0_row_size),
                       src0_row_size_padded, src0_row_size, 2);
    }

    // Process src0 rows
    for (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
        const uint8_t * ss0 = dma_queue_pop(dma_queue).dst;

        #pragma unroll(2)
        for (uint32_t ir1 = 0; ir1 < src1_nrows; ++ir1) {
            const uint8_t * restrict src1_col = (const uint8_t *) (src1_data + ir1 * src1_row_size);
            float * restrict dst_row          = (float *) (dst->data + (ir1 * dst_row_size));
            mt->vec_dot_rx2(ne00, &dst_row[ir0], ss0, src0_row_size_padded, src1_col);
        }

        // Prefetch next (n + spad_nrows) row
        const int pr0 = (ir0 + HTP_SPAD_SRC0_NROWS);
        const int is0 = (pr0 - src0_start_row) % HTP_SPAD_SRC0_NROWS;
        if (pr0 < src0_end_row_x2) {
            dma_queue_push_ddr_to_vtcm(dma_queue, dma_make_ptr(spad_src0 + is0 * src0_row_size_padded, src0_row + pr0 * src0_row_size),
                           src0_row_size_padded, src0_row_size, 2);
        }
    }

    // Process the last row (if any)
    if (src0_end_row != src0_end_row_x2) {
        uint32_t  ir0 = src0_end_row_x2;
        const int is0 = (ir0 - src0_start_row);
        dma_queue_push_ddr_to_vtcm(dma_queue, dma_make_ptr(spad_src0 + is0 * src0_row_size_padded, src0_row + ir0 * src0_row_size),
                       src0_row_size_padded, src0_row_size, 1);
        const uint8_t * ss0 = dma_queue_pop(dma_queue).dst;

        #pragma unroll(2)
        for (uint32_t ir1 = 0; ir1 < src1_nrows; ++ir1) {
            const uint8_t * restrict src1_col = (const uint8_t *) (src1_data + ir1 * src1_row_size);
            float * restrict dst_row          = (float *) (dst->data + (ir1 * dst_row_size));
            mt->vec_dot(ne00, &dst_row[ir0], ss0, src1_col);
        }
    }

    t2 = HAP_perf_get_qtimer_count();

    FARF(HIGH, "matmul-%s %d/%d: %ux%ux%ux%u (%u:%u) * %ux%ux%ux%u -> %ux%ux%ux%u usec %u\n", mt->type, ith, nth,
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src0_start_row, src0_end_row, src1->ne[0], src1->ne[1],
         src1->ne[2], src1->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         (unsigned) HAP_perf_qtimer_count_to_us(t2 - t1));
}

// q8x4x2 src1 tensor is already in VTCM spad
static void matvec(struct htp_matmul_type * mt,
                   struct htp_tensor * restrict src0,
                   struct htp_tensor * restrict src1,
                   struct htp_tensor * restrict dst,
                   struct htp_spad * restrict src0_spad,
                   struct htp_spad * restrict src1_spad,
                   struct htp_spad * restrict dst_spad,
                   uint32_t    nth,
                   uint32_t    ith,
                   uint32_t    src0_nrows_per_thread,
                   dma_queue * dma_queue) {
    htp_matmul_preamble;

    const uint32_t src0_nrows = ne01;

    const uint32_t src0_start_row  = src0_nrows_per_thread * ith;
    const uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, src0_nrows);
    const uint32_t src0_end_row_x2 = src0_start_row + ((src0_end_row - src0_start_row) & ~1U);

    // no work for this thread
    if (src0_start_row >= src0_end_row) {
        return;
    }

    const size_t dst_row_size  = nb1;
    const size_t src0_row_size = nb01;
    const size_t src1_row_size = q8x4x2_row_size(ne10);

    const size_t src0_row_size_padded = htp_round_up(src0_row_size, 128);

    // Per-thread VTCM scratchpads for all tensors
    // Note that the entire src1 tensor is already in VTCM
    // For other tensors we allocate N rows per thread, padded to HVX vector size
    uint8_t * spad_dst  = dst_spad->data + dst_spad->size_per_thread * ith;
    uint8_t * spad_src0 = src0_spad->data + src0_spad->size_per_thread * ith;
    uint8_t * src1_data = src1_spad->data;

    uint64_t t1, t2;
    t1 = HAP_perf_get_qtimer_count();

    float * tmp = (float *) spad_dst;

    const uint8_t * restrict src0_row = (const uint8_t *) src0->data;
    const uint8_t * restrict src1_col = (const uint8_t *) src1_data;
    float * restrict dst_col          = (float *) dst->data;

    // Prefill spad with 2x src0 rows
    #pragma unroll(2)
    for (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
        const uint32_t is0 = (ir0 - src0_start_row);
        if (is0 >= HTP_SPAD_SRC0_NROWS) {
            break;
        }
        dma_queue_push_ddr_to_vtcm(dma_queue, dma_make_ptr(spad_src0 + is0 * src0_row_size_padded, src0_row + ir0 * src0_row_size),
                       src0_row_size_padded, src0_row_size, 2);
    }

    // Process src0 rows
    for (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
        const uint8_t * ss0 = dma_queue_pop(dma_queue).dst;
        mt->vec_dot_rx2(ne00, &tmp[ir0 - src0_start_row], ss0, src0_row_size_padded, src1_col);

        // Prefetch next (n + spad_nrows) row
        const uint32_t pr0 = (ir0 + HTP_SPAD_SRC0_NROWS);
        const uint32_t is0 = (pr0 - src0_start_row) % HTP_SPAD_SRC0_NROWS;
        if (pr0 < src0_end_row_x2) {
            dma_queue_push_ddr_to_vtcm(dma_queue, dma_make_ptr(spad_src0 + is0 * src0_row_size_padded, src0_row + pr0 * src0_row_size),
                           src0_row_size_padded, src0_row_size, 2);
        }
    }

    // Process the last row (if any)
    if (src0_end_row != src0_end_row_x2) {
        const uint32_t ir0 = src0_end_row_x2;
        const uint32_t is0 = (ir0 - src0_start_row);
        dma_queue_push_ddr_to_vtcm(dma_queue, dma_make_ptr(spad_src0 + is0 * src0_row_size_padded, src0_row + ir0 * src0_row_size),
                       src0_row_size_padded, src0_row_size, 1);
        const uint8_t * ss0 = dma_queue_pop(dma_queue).dst;
        mt->vec_dot(ne00, &tmp[ir0 - src0_start_row], ss0, src1_col);
    }

    hvx_copy_fp32_ua((uint8_t *) &dst_col[src0_start_row], (uint8_t *) tmp, src0_end_row - src0_start_row);

    t2 = HAP_perf_get_qtimer_count();

    FARF(HIGH, "matvec-%s %u/%u: %ux%ux%ux%u (%u:%u) * %ux%ux%ux%u -> %ux%ux%ux%u usec %u\n", mt->type, ith, nth,
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src0_start_row, src0_end_row, src1->ne[0], src1->ne[1],
         src1->ne[2], src1->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         (unsigned) HAP_perf_qtimer_count_to_us(t2 - t1));
}

#define MMID_MATRIX_ROW(row_id, i1) matrix_rows[(row_id) * ids->ne[0] * ids->ne[1] + (i1)]

struct mmid_row_mapping {
    uint32_t i1;
    uint32_t i2;
};

// q8x4 src1 tensor is already in VTCM spad
static void matmul_id(struct htp_matmul_type * mt,
                      struct htp_tensor * restrict src0,
                      struct htp_tensor * restrict src1,
                      struct htp_tensor * restrict ids,
                      struct htp_tensor * restrict dst,
                      struct htp_spad * restrict src0_spad,
                      struct htp_spad * restrict src1_spad,
                      struct htp_spad * restrict src2_spad,
                      struct htp_spad * restrict dst_spad,
                      uint32_t    nth,
                      uint32_t    ith,
                      uint32_t    src0_nrows_per_thread,
                      dma_queue * dma_queue) {
    htp_matmul_preamble;

    uint64_t t1, t2;
    t1 = HAP_perf_get_qtimer_count();

    const uint32_t src0_nrows = ne01;  // src0 rows per expert
    const uint32_t src1_nrows = ne11;

    const uint32_t src0_start_row  = src0_nrows_per_thread * ith;
    const uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, src0_nrows);
    const uint32_t src0_end_row_x2 = src0_start_row + ((src0_end_row - src0_start_row) & ~1U);

    // no work for this thread
    if (src0_start_row >= src0_end_row) {
        return;
    }

    const uint32_t n_ids = ids->ne[0];  // n_expert_used
    const uint32_t n_as  = ne02;        // n_expert

    const size_t matrix_row_counts_size = n_as * sizeof(uint32_t);
    const size_t matrix_row_map_size    = n_as * ids->ne[0] * ids->ne[1] * sizeof(struct mmid_row_mapping);

    const uint32_t *                matrix_row_counts = (const uint32_t *) src2_spad->data + 0;
    const struct mmid_row_mapping * matrix_rows       = (const void *) src2_spad->data + matrix_row_counts_size;

    const size_t dst_row_size  = nb1;
    const size_t src0_row_size = nb01;
    const size_t src1_row_size = q8x4x2_row_size(ne10);

    const size_t src0_row_size_padded = htp_round_up(src0_row_size, 128);

    // Per-thread VTCM scratchpads for all tensors
    // Note that the entire src1 tensor is already in VTCM
    // For other tensors we allocate N rows per thread, padded to HVX vector size
    uint8_t * restrict spad_dst  = dst_spad->data + dst_spad->size_per_thread * ith;
    uint8_t * restrict spad_src0 = src0_spad->data + src0_spad->size_per_thread * ith;
    uint8_t * restrict src1_data = src1_spad->data;

    for (uint32_t cur_a = 0; cur_a < n_as; ++cur_a) {
        const int32_t cne1 = matrix_row_counts[cur_a];

        if (cne1 == 0) {
            continue;
        }

        const uint8_t * src0_row = (const uint8_t *) src0->data + (0 + cur_a * nb02 + 0);

        // Prefill spad with src0 rows
        #pragma unroll(4)
        for (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
            const int is0 = (ir0 - src0_start_row);
            if (is0 >= HTP_SPAD_SRC0_NROWS) {
                break;
            }
            dma_queue_push_ddr_to_vtcm(dma_queue, dma_make_ptr(spad_src0 + is0 * src0_row_size_padded, src0_row + ir0 * src0_row_size),
                           src0_row_size_padded, src0_row_size, 2);
        }

        // Process src0 rows
        for (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
            const uint8_t * ss0 = dma_queue_pop(dma_queue).dst;

            for (uint32_t cid = 0; cid < cne1; ++cid) {
                struct mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, cid);
                const int               rm1         = row_mapping.i1;  // expert idx
                const int               rm2         = row_mapping.i2;  // token idx

                const uint32_t ir1 = src1_nrows == 1 ? 0 : rm1;        // src1 row idx
                const uint8_t * restrict src1_col =
                    (const uint8_t *) (src1_data + (ir1 + rm2 * ne11 + 0) * src1_row_size);
                float * dst_row = (float *) (dst->data + (rm1 * nb1 + rm2 * nb2 + 0));

                mt->vec_dot_rx2(ne00, &dst_row[ir0], ss0, src0_row_size_padded, src1_col);
            }

            // Prefetch next (n + spad_nrows) row
            const int pr0 = (ir0 + HTP_SPAD_SRC0_NROWS);
            const int is0 = (pr0 - src0_start_row) % HTP_SPAD_SRC0_NROWS;
            if (pr0 < src0_end_row_x2) {
                dma_queue_push_ddr_to_vtcm(dma_queue, dma_make_ptr(spad_src0 + is0 * src0_row_size_padded, src0_row + pr0 * src0_row_size),
                               src0_row_size_padded, src0_row_size, 2);
            }
        }

        // Process the last row (if any)
        if (src0_end_row != src0_end_row_x2) {
            uint32_t       ir0 = src0_end_row_x2;
            const uint32_t is0 = (ir0 - src0_start_row);
            dma_queue_push_ddr_to_vtcm(dma_queue, dma_make_ptr(spad_src0 + is0 * src0_row_size_padded, src0_row + ir0 * src0_row_size),
                           src0_row_size_padded, src0_row_size, 1);
            const uint8_t * ss0 = dma_queue_pop(dma_queue).dst;

            for (uint32_t cid = 0; cid < cne1; ++cid) {
                struct mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, cid);
                const int               rm1         = row_mapping.i1;  // expert idx
                const int               rm2         = row_mapping.i2;  // token idx

                const uint32_t ir1 = src1_nrows == 1 ? 0 : rm1;        // src1 row idx
                const uint8_t * restrict src1_col =
                    (const uint8_t *) (src1_data + (ir1 + rm2 * ne11 + 0) * src1_row_size);
                float * dst_row = (float *) (dst->data + (rm1 * nb1 + rm2 * nb2 + 0));

                mt->vec_dot(ne00, &dst_row[ir0], ss0, src1_col);
            }
        }
    }

    t2 = HAP_perf_get_qtimer_count();

    FARF(HIGH, "matmul-id-%s %d/%d: %ux%ux%ux%u (%u:%u) * %ux%ux%ux%u (%ux%ux%ux%u) -> %ux%ux%ux%u usec %u\n", mt->type,
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src0_start_row, src0_end_row, src1->ne[0],
         src1->ne[1], src1->ne[2], src1->ne[3], ids->ne[0], ids->ne[1], ids->ne[2], ids->ne[3], dst->ne[0], dst->ne[1],
         dst->ne[2], dst->ne[3], (unsigned) HAP_perf_qtimer_count_to_us(t2 - t1));
}

// q8x4 src1 tensor is already in VTCM spad
static void matvec_id(struct htp_matmul_type * mt,
                      struct htp_tensor * restrict src0,
                      struct htp_tensor * restrict src1,
                      struct htp_tensor * restrict src2,
                      struct htp_tensor * restrict dst,
                      struct htp_spad * restrict src0_spad,
                      struct htp_spad * restrict src1_spad,
                      struct htp_spad * restrict src2_spad,
                      struct htp_spad * restrict dst_spad,
                      uint32_t    nth,
                      uint32_t    ith,
                      uint32_t    src0_nrows_per_thread,
                      dma_queue * dma_queue) {
    htp_matmul_preamble;

    uint64_t t1, t2;
    t1 = HAP_perf_get_qtimer_count();

    const uint32_t src0_nrows = ne01;  // src0 rows per expert

    const uint32_t src0_start_row  = src0_nrows_per_thread * ith;
    const uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, src0_nrows);
    const uint32_t src0_end_row_x2 = src0_start_row + ((src0_end_row - src0_start_row) & ~1U);

    // no work for this thread
    if (src0_start_row >= src0_end_row) {
        return;
    }

    assert(ne13 % ne03 == 0);

    const size_t dst_row_size  = nb1;
    const size_t src0_row_size = nb01;
    const size_t src1_row_size = q8x4x2_row_size(ne10);

    const size_t src0_row_size_padded = htp_round_up(src0_row_size, 128);

    const uint32_t n_aids = src2->ne[0];  // num activated experts
    const uint32_t n_ids  = ne02;         // num experts

    // Per-thread VTCM scratchpads for all tensors
    // Note that the entire src1 tensor is already in VTCM
    // For other tensors we allocate N rows per thread, padded to HVX vector size
    uint8_t * restrict spad_dst  = dst_spad->data + dst_spad->size_per_thread * ith;
    uint8_t * restrict spad_src0 = src0_spad->data + src0_spad->size_per_thread * ith;
    uint8_t * restrict src1_data = src1_spad->data;

    for (uint32_t ie1 = 0; ie1 < n_aids; ++ie1) {  // for each expert
        const uint32_t eid = *(const int32_t *) ((const uint8_t *) src2->data + ie1 * src2->nb[0]);
        assert(eid < n_ids);

        const uint8_t * restrict src0_row = (const uint8_t *) src0->data + eid * nb02;
        const uint8_t * restrict src1_col = (const uint8_t *) src1_data;
        float * restrict dst_row          = (float *) (dst->data + ie1 * nb1);

        // Prefill spad with src0 rows
        #pragma unroll(4)
        for (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
            const int is0 = (ir0 - src0_start_row);
            if (is0 >= HTP_SPAD_SRC0_NROWS) {
                break;
            }
            dma_queue_push_ddr_to_vtcm(dma_queue, dma_make_ptr(spad_src0 + is0 * src0_row_size_padded, src0_row + ir0 * src0_row_size),
                           src0_row_size_padded, src0_row_size, 2);
        }

        // Process src0 rows
        for (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
            const uint8_t * ss0 = dma_queue_pop(dma_queue).dst;
            mt->vec_dot_rx2(ne00, &dst_row[ir0], ss0, src0_row_size_padded, src1_col);

            // Prefetch next (n + spad_nrows) row
            const int pr0 = (ir0 + HTP_SPAD_SRC0_NROWS);
            const int is0 = (pr0 - src0_start_row) % HTP_SPAD_SRC0_NROWS;
            if (pr0 < src0_end_row_x2) {
                dma_queue_push_ddr_to_vtcm(dma_queue, dma_make_ptr(spad_src0 + is0 * src0_row_size_padded, src0_row + pr0 * src0_row_size),
                               src0_row_size_padded, src0_row_size, 2);
            }
        }

        // Process the last row (if any)
        if (src0_end_row != src0_end_row_x2) {
            uint32_t       ir0 = src0_end_row_x2;
            const uint32_t is0 = (ir0 - src0_start_row);
            dma_queue_push_ddr_to_vtcm(dma_queue, dma_make_ptr(spad_src0 + is0 * src0_row_size_padded, src0_row + ir0 * src0_row_size),
                           src0_row_size_padded, src0_row_size, 1);
            const uint8_t * ss0 = dma_queue_pop(dma_queue).dst;
            mt->vec_dot(ne00, &dst_row[ir0], ss0, src1_col);
        }
    }

    t2 = HAP_perf_get_qtimer_count();

    FARF(HIGH, "matvec-id-%s %d/%d: %ux%ux%ux%u (%u:%u) * %ux%ux%ux%u (%ux%ux%ux%u) -> %ux%ux%ux%u usec %u\n", mt->type,
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src0_start_row, src0_end_row, src1->ne[0],
         src1->ne[1], src1->ne[2], src1->ne[3], src2->ne[0], src2->ne[1], src2->ne[2], src2->ne[3], dst->ne[0],
         dst->ne[1], dst->ne[2], dst->ne[3], (unsigned) HAP_perf_qtimer_count_to_us(t2 - t1));
}

// *** matmul in fp16

static void matmul_f16_f32(struct htp_ops_context * octx, uint32_t nth, uint32_t ith) {
    struct htp_tensor * restrict src0    = &octx->src0;
    struct htp_tensor * restrict src1    = &octx->src1;
    struct htp_tensor * restrict dst     = &octx->dst;
    struct htp_spad * restrict src0_spad = &octx->src0_spad;
    struct htp_spad * restrict src1_spad = &octx->src1_spad;
    struct htp_spad * restrict dst_spad  = &octx->dst_spad;

    dma_queue * dma_queue = octx->ctx->dma[ith];

    htp_matmul_preamble;

    uint64_t t1, t2;
    t1 = HAP_perf_get_qtimer_count();

    assert(ne12 % ne02 == 0);
    assert(ne13 % ne03 == 0);

    // This is the size of the first dimension of the result, so we can iterate that way. (see the ASSERT above, these
    // are the same numbers)
    const uint32_t nr0 = ne0;

    // This is the size of the rest of the dimensions of the result
    const uint32_t nr1 = ne1 * ne2 * ne3;

    // distribute the thread work across the inner or outer loop based on which one is larger
    uint32_t nchunk0 = nr0 > nr1 ? nth : 1;  // parallelize by src0 rows
    uint32_t nchunk1 = nr0 > nr1 ? 1 : nth;  // parallelize by src1 rows

    // The number of elements in each chunk
    const uint32_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
    const uint32_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

    uint32_t current_chunk = ith;

    const uint32_t ith0 = current_chunk % nchunk0;
    const uint32_t ith1 = current_chunk / nchunk0;

    const uint32_t ir0_start = dr0 * ith0;
    const uint32_t ir0_end   = MIN(ir0_start + dr0, nr0);

    const uint32_t ir1_start = dr1 * ith1;
    const uint32_t ir1_end   = MIN(ir1_start + dr1, nr1);

    // no work for this thread
    if (ir0_start >= ir0_end || ir1_start >= ir1_end) {
        return;
    }

    const uint32_t blck_0 = 64;
    const uint32_t blck_1 = 64;

    const size_t src1_row_size_padded      = htp_round_up(nb11, 128);
    const size_t src1_fp16_row_size_padded = htp_round_up(nb11 / 2, 128);

    const size_t src1_fp32_size = htp_round_up(blck_1 * src1_row_size_padded, 256);
    const size_t src1_fp16_size = htp_round_up(blck_1 * src1_fp16_row_size_padded, 256);

    uint8_t * src1_fp32_buf = src1_spad->data + src1_spad->size_per_thread * ith;
    uint8_t * src1_fp16_buf = src1_fp32_buf + src1_fp32_size;

    uint8_t * spad_src0 = src0_spad->data + src0_spad->size_per_thread * ith;

    // Loop over src1 blocks
    // Ensure we don't cross broadcast boundaries (ne1)
    // If broadcasting is involved (src0 depends on i12/i13), we must ensure i12/i13 are constant for the block.
    // i12/i13 change when ir1 crosses ne1 (column index).
    // So next boundary is multiples of ne1.
    for (uint32_t iir1 = ir1_start; iir1 < ir1_end;) {
        uint32_t next_boundary = (iir1 / ne1 + 1) * ne1;
        uint32_t blck_1_actual = MIN(blck_1, ir1_end - iir1);
        blck_1_actual          = MIN(blck_1_actual, next_boundary - iir1);

        // Fetch src1 block (FP32)
        // We iterate and push to DMA.
        // We will store them contiguously in VTCM buffer with aligned stride.

        for (uint32_t ir1_offset = 0; ir1_offset < blck_1_actual; ++ir1_offset) {
            uint32_t       ir1 = iir1 + ir1_offset;
            const uint32_t i13 = fastdiv(ir1, &octx->mm_div_ne12_ne1);
            const uint32_t i12 = fastdiv(ir1 - i13 * ne12 * ne1, &octx->mm_div_ne1);
            const uint32_t i11 = (ir1 - i13 * ne12 * ne1 - i12 * ne1);

            const uint8_t * restrict src1_ptr = (const uint8_t *) src1->data + (i11 * nb11 + i12 * nb12 + i13 * nb13);
            uint8_t *                dst_ptr  = src1_fp32_buf + ir1_offset * src1_row_size_padded;  // Padded Stride

            dma_queue_push_ddr_to_vtcm(dma_queue, dma_make_ptr(dst_ptr, src1_ptr), src1_row_size_padded, nb11, 1);
        }
        dma_queue_wait(dma_queue);

        // Convert FP32 -> FP16 in VTCM
        for (uint32_t ir1_offset = 0; ir1_offset < blck_1_actual; ++ir1_offset) {
            float *  src = (float *) (src1_fp32_buf + ir1_offset * src1_row_size_padded);
            void *   dst = (void *) (src1_fp16_buf + ir1_offset * src1_fp16_row_size_padded);
            uint32_t n   = ne10;  // K dimension
            quantize_row_fp32_to_fp16_vtcm(src, dst, n);
        }

        const size_t src0_row_size_padded = htp_round_up(nb01, 128);

        // Loop over src0 blocks
        // We will double buffer src0 fetching.
        // Actually, we can just process rows.
        // We have `ir0_start` to `ir0_end`.
        // We can process in chunks of 4 rows.

        // Prefill src0
        uint32_t iir0 = ir0_start;

        // Process src0 rows in chunks of 4
        for (; iir0 < ir0_end; iir0 += 4) {
            uint32_t rows_to_process = MIN(4, ir0_end - iir0);

            // Fetch these rows
            for (uint32_t r = 0; r < rows_to_process; ++r) {
                uint32_t       ir0 = iir0 + r;
                const uint32_t i13 = fastdiv(iir1, &octx->mm_div_ne12_ne1);  // Use iir1's broadcast indices?
                // Wait, src0 broadcast logic depends on ir1?
                // No, src0 broadcasting depends on i12, i13 derived from ir1.
                // If we block on ir1, i12 and i13 might change if block crosses boundary?
                // Usually blocked on inner dimension.
                // Let's re-evaluate broadcasting.
                // i12, i13 are derived from ir1.
                // i02, i03 are derived from i12, i13.
                // If we process a block of ir1, we might have multiple i02/i03?
                // If we reuse src1 block against src0 rows, we assume src0 rows are independent of src1?
                // But `src0_base` depends on `i02`, `i03`.
                // `i02` depends on `i12` (from `ir1`).
                // So for different `ir1`, we might need different `src0` row!
                // This breaks the "reuse src1 block against src0 block" optimization IF src0 depends on src1 index.
                // This happens if we broadcast src0 along batch dimensions that match src1 batch dimensions.
                // E.g. Batched MatMul.
                // If we have standard MatMul (M,K) x (K,N) -> (M,N), no broadcast dependency.
                // But GGML supports broadcasting.

                // If `i02` and `i03` are constant for the whole `blck_1`, we are good.
                // `blck_1` is 64. `ne1` (N) is typically large.
                // If we iterate over N, `i12` and `i13` (batch indices) change only when N wraps?
                // `ir1` covers `ne1 * ne12 * ne13`.
                // `i11 = ir1 % (ne12*ne1) % ne1`. No.
                // `i13 = ir1 / (ne12*ne1)`.
                // `i12 = (ir1 / ne1) % ne12`.
                // `i11 = ir1 % ne1`.
                // So `i11` (column index) varies fastest.
                // As long as `blck_1` stays within one batch (i.e. we don't cross `ne1` boundary), `i12` and `i13` are constant.
                // `ne1` is usually large (vocab size, or hidden size).
                // So `blck_1 = 64` is likely within same batch.
                // We should ensure `blck_1` does not cross batch boundaries to optimize src0 reuse.
                // Or we just handle it?
                // If `src0_base` changes, we need to reload src0.
                // But we want to load src0 ONCE and reuse for all src1 in block?
                // No, we want to load SRC1 (weights) once, and reuse for all SRC0 (activations).
                // SRC0 (M, K) usually doesn't depend on N (ir1).
                // EXCEPT for broadcasting where src0 is smaller batch than src1.
                // `i02 = i12 % ...`
                // If `i12` is constant, `i02` is constant.
                // So if we process a block of `ir1` that shares same `i12`, `i13`, then `src0` row selection is constant for that block.
                // Then for each `ir0` (M dimension), we pick `src0` row.
            }
            // Okay, let's keep it simple for now.
            // We load `src1` block.
            // We iterate `ir0`.
            // For each `ir0`, we need `src0` row.
            // `src0_base` depends on `ir1`.
            // If `ir1` block has mixed `i12/i13`, we have mixed `src0_base`.
            // But `vec_dot` accumulates into `dst`.
            // `dst[ir0]` (row ir0) for column `ir1`.
            // The accumulation is over K.
            // Result is `dst[ir0, ir1]`.
            // We want to compute `dst[ir0..ir0+4, iir1..iir1+4]`.
            // We need `src0[ir0..ir0+4]` and `src1[iir1..iir1+4]`.
            // Does `src0` depend on `iir1`?
            // Yes, via `src0_base`.
            // But if `src0` is broadcasted, `src0_base` might point to same data for different `iir1`.
            // If `src0` is (M,K,1,1) and `src1` is (N,K,B,B), then `src0` is reused for all B.
            // If `src0` is (M,K,B,B) and `src1` is (N,K,B,B), then `src0` matches `src1`.
            // In standard case, we assume `ne1` is large enough that 64 columns are in same batch.
            // So `src0_base` is constant for the whole `src1` block.
            // We can assert or handle this.
            // Let's compute `src0_base` for the first element of `src1` block and assume it's same?
            // Or better, check if block crosses boundary.
            // If we split loops such that we iterate `i12`, `i13` in outer loops, we avoid this.
            // But `matmul_f16_f32` iterates flat `ir1`.
            // I will implement a check: if `src0_base` changes within `src1` block, we fallback or handle it (unlikely with small block size compared to ne1).
            // Actually, we can just compute `src0_base` inside the `vec_dot` call?
            // No, we want to prefetch `src0`.

            // Let's fetch `src0` rows into VTCM.
            // `src0` rows depend on `ir0` (varying) and `src0_base` (constant for block usually).
            // We can fetch 4 rows of `src0`.

            // Wait, `vec_dot_f16_f16_4x4` computes 4x4 block of `dst`.
            // dst[r:r+4, c:c+4].
            // It needs src0[r:r+4] and src1[c:c+4].
            // We need to ensure src0[r] is the correct row corresponding to src1[c].
            // If src0 depends on c (batch), then for different c in 4x4 block, we might need different src0?
            // Only if batch changes within 4 columns. Very unlikely (ne1 >= 4 usually).

            // So, for the inner loop, we assume `src0_base` is constant.
            // We calculate `src0_base` from `iir1`.

            uint32_t       first_ir1 = iir1;
            const uint32_t i13       = fastdiv(first_ir1, &octx->mm_div_ne12_ne1);
            const uint32_t i12       = fastdiv(first_ir1 - i13 * ne12 * ne1, &octx->mm_div_ne1);
            const uint32_t i03       = fastdiv(i13, &octx->mm_div_r3);
            const uint32_t i02       = fastdiv(i12, &octx->mm_div_r2);
            const uint8_t * restrict src0_base = (const uint8_t *) src0->data + (0 + i02 * nb02 + i03 * nb03);

            // Fetch `src0` rows [iir0, iir0+4).
            // Double buffer?
            // For simplicity, just fetch and process. 4 rows is small (4 * 2 * K bytes). 8KB for K=1024.

            uint8_t * src0_vtcm_ptr = spad_src0;
            for (uint32_t r = 0; r < rows_to_process; ++r) {
                uint32_t ir0 = iir0 + r;
                dma_queue_push_ddr_to_vtcm(dma_queue, dma_make_ptr(src0_vtcm_ptr + r * src0_row_size_padded, src0_base + ir0 * nb01), src0_row_size_padded, nb01, 1);
            }
            dma_queue_wait(dma_queue);

            // Now we have src0 rows in VTCM (FP16).
            // We have src1 rows in VTCM (FP16 converted).
            // We can compute 4xN blocks.
            // Iterate over `src1` chunks of 4.

            for (uint32_t c = 0; c < blck_1_actual; c += 4) {
                uint32_t cols_to_process = MIN(4, blck_1_actual - c);

                // Pointers to VTCM
                const void * vx = src0_vtcm_ptr;
                const void * vy = src1_fp16_buf + c * src1_fp16_row_size_padded;

                // Output pointers (DDR)
                // dst is (ir0, ir1).
                // dst_col = dst->data + (i1*nb1 + ...)
                // We need to calculate dst pointer carefully.
                // dst[iir0 + r, iir1 + c + cc]
                // We need to pass `dst` pointer to kernel.
                // Kernel takes `dst` and `dst_stride`.
                // `dst` varies by `nb1` for each column (c).
                // `dst` varies by `4 bytes` (float) for each row (r). (assuming contiguous inner dim 0).

                // We need to compute `dst` address for (iir0, iir1 + c).
                // And `dst_stride` corresponds to increment in `ir1`.

                // Recompute dst address
                // dst indices for (iir0, iir1+c)
                uint32_t cur_ir1 = iir1 + c;
                // indices i1, i2, i3
                uint32_t ci13 = fastdiv(cur_ir1, &octx->mm_div_ne12_ne1);
                uint32_t ci12 = fastdiv(cur_ir1 - ci13 * ne12 * ne1, &octx->mm_div_ne1);
                uint32_t ci11 = (cur_ir1 - ci13 * ne12 * ne1 - ci12 * ne1);

                float * dst_ptr = (float *) ((uint8_t *) dst->data + (ci11 * nb1 + ci12 * nb2 + ci13 * nb3)) + iir0;

                // If cols_to_process < 4, we need to handle boundaries or pad?
                // The kernel processes 4x4.
                // We should add boundary checks in kernel or here.
                // Or just use `vec_dot_f16_f16_4x4` which assumes valid pointers.
                // We can point valid dummy buffers for out of bound reads.
                // For writes, `dst` must be valid.
                // If we are at edge, maybe fallback to scalar?
                // Or just loop 1 by 1.

                if (rows_to_process == 4 && cols_to_process == 4) {
                    vec_dot_f16_f16_4x4(ne00, dst_ptr, nb1, vx, src0_row_size_padded, vy, src1_fp16_row_size_padded);
                } else {
                    // Fallback using vec_dot_f16_f16
                    for (uint32_t r = 0; r < rows_to_process; ++r) {
                        for (uint32_t cc = 0; cc < cols_to_process; ++cc) {
                            // Recompute dst for this specific cell
                            uint32_t cell_ir1 = iir1 + c + cc;
                            uint32_t k13      = fastdiv(cell_ir1, &octx->mm_div_ne12_ne1);
                            uint32_t k12      = fastdiv(cell_ir1 - k13 * ne12 * ne1, &octx->mm_div_ne1);
                            uint32_t k11      = (cell_ir1 - k13 * ne12 * ne1 - k12 * ne1);
                            float *  d        = (float *) ((uint8_t *) dst->data +
                                                  (k11 * nb1 + k12 * nb2 + k13 * nb3)) +
                                     (iir0 + r);

                            const void * x_row = (const uint8_t *) vx + r * src0_row_size_padded;
                            const void * y_row = (const uint8_t *) vy + cc * src1_fp16_row_size_padded;

                            vec_dot_f16_f16(ne00, d, x_row, y_row);
                        }
                    }
                }
            }
        }
        iir1 += blck_1_actual;
    }

    t2 = HAP_perf_get_qtimer_count();

    FARF(HIGH, "matmul-f16-f32 %d/%d: %ux%ux%ux%u (%u:%u %u:%u) * %ux%ux%ux%u -> %ux%ux%ux%u usec %u\n", ith, nth,
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ir0_start, ir0_end, ir1_start, ir1_end, src1->ne[0],
         src1->ne[1], src1->ne[2], src1->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         (unsigned) HAP_perf_qtimer_count_to_us(t2 - t1));
}

// *** dynamic quant

static inline void quantize_block_fp32_q8x1(float * restrict x, uint8_t * restrict y_q, uint8_t * restrict y_d) {
    assert((unsigned long) x % 128 == 0);
    assert((unsigned long) y_q % 128 == 0);

    HVX_Vector * vx = (HVX_Vector *) x;
    HVX_Vector zero   = Q6_V_vsplat_R(0);

    // Use reduce max fp32 to find max(abs(e)) first
    HVX_Vector vmax0_sf = hvx_vec_reduce_max_fp32(hvx_vec_abs_fp32(vx[0]));
    HVX_Vector vmax1_sf = hvx_vec_reduce_max_fp32(hvx_vec_abs_fp32(vx[1]));
    HVX_Vector vmax2_sf = hvx_vec_reduce_max_fp32(hvx_vec_abs_fp32(vx[2]));
    HVX_Vector vmax3_sf = hvx_vec_reduce_max_fp32(hvx_vec_abs_fp32(vx[3]));
    // Load and convert into QF32
    HVX_Vector vx0_qf = Q6_Vqf32_vsub_VsfVsf(vx[0], zero);  // 32 elements
    HVX_Vector vx1_qf = Q6_Vqf32_vsub_VsfVsf(vx[1], zero);  // 32 elements
    HVX_Vector vx2_qf = Q6_Vqf32_vsub_VsfVsf(vx[2], zero);  // 32 elements
    HVX_Vector vx3_qf = Q6_Vqf32_vsub_VsfVsf(vx[3], zero);  // 32 elements

    // Convert to QF32
    HVX_Vector vmax0_qf = Q6_Vqf32_vsub_VsfVsf(vmax0_sf, zero);
    HVX_Vector vmax1_qf = Q6_Vqf32_vsub_VsfVsf(vmax1_sf, zero);
    HVX_Vector vmax2_qf = Q6_Vqf32_vsub_VsfVsf(vmax2_sf, zero);
    HVX_Vector vmax3_qf = Q6_Vqf32_vsub_VsfVsf(vmax3_sf, zero);

    // Combine and convert to fp16
    HVX_Vector vmax01_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vmax1_qf, vmax0_qf)));
    HVX_Vector vmax23_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vmax3_qf, vmax2_qf)));

    // Convert into fp16
    HVX_Vector vx01_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vx1_qf, vx0_qf)));
    HVX_Vector vx23_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vx3_qf, vx2_qf)));

    // Replicate first fp16 scale across all lanes
    HVX_Vector ctrl = *(const HVX_Vector *) repl_2x_fp16;
    vmax01_hf         = Q6_V_vdelta_VV(vmax01_hf, ctrl);
    vmax23_hf         = Q6_V_vdelta_VV(vmax23_hf, ctrl);

    HVX_Vector vd01_qf16 = Q6_Vqf16_vmpy_VhfVhf(vmax01_hf, Q6_Vh_vsplat_R(0x2008));  // 1.0 / 127.0
    HVX_Vector vd23_qf16 = Q6_Vqf16_vmpy_VhfVhf(vmax23_hf, Q6_Vh_vsplat_R(0x2008));  // 1.0 / 127.0
    HVX_Vector vd01_hf   = Q6_Vhf_equals_Vqf16(vd01_qf16);
    HVX_Vector vd23_hf   = Q6_Vhf_equals_Vqf16(vd23_qf16);

    hvx_vec_store_u(y_d + 0, 2, vd01_hf);
    HVX_Vector rotated_vd_hf = Q6_V_vror_VR(vd01_hf, 64);
    hvx_vec_store_u(y_d + 2, 2, rotated_vd_hf);

    hvx_vec_store_u(y_d + 4, 2, vd23_hf);
    rotated_vd_hf = Q6_V_vror_VR(vd23_hf, 64);
    hvx_vec_store_u(y_d + 6, 2, rotated_vd_hf);

    // Divide input by the scale
    HVX_Vector vd01_inv_hf = hvx_vec_inverse_fp16(vd01_hf);
    HVX_Vector vd23_inv_hf = hvx_vec_inverse_fp16(vd23_hf);
    vx01_hf              = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(vx01_hf, vd01_inv_hf));
    vx23_hf              = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(vx23_hf, vd23_inv_hf));

    // Convert to int8
    HVX_Vector vx01_i16 = hvx_vec_i16_from_hf_rnd_sat(vx01_hf);
    HVX_Vector vx23_i16 = hvx_vec_i16_from_hf_rnd_sat(vx23_hf);
    HVX_Vector vx_i8    = Q6_Vb_vpack_VhVh_sat(vx23_i16, vx01_i16);

    *(HVX_Vector *) y_q = vx_i8;
}

static inline void quantize_block_fp32_q8x2(float * restrict x, uint8_t * restrict y_q, uint8_t * restrict y_d) {
    assert((unsigned long) x % 128 == 0);
    assert((unsigned long) y_q % 128 == 0);

    HVX_Vector * vx = (HVX_Vector *) x;

    // Load and convert into QF32
    HVX_Vector zero   = Q6_V_vsplat_R(0);
    HVX_Vector vx0_qf = Q6_Vqf32_vsub_VsfVsf(vx[0], zero);  // 32 elements
    HVX_Vector vx1_qf = Q6_Vqf32_vsub_VsfVsf(vx[1], zero);  // 32 elements
    HVX_Vector vx2_qf = Q6_Vqf32_vsub_VsfVsf(vx[2], zero);  // 32 elements
    HVX_Vector vx3_qf = Q6_Vqf32_vsub_VsfVsf(vx[3], zero);  // 32 elements

    // Convert into fp16
    HVX_Vector vx01_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vx1_qf, vx0_qf)));
    HVX_Vector vx23_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vx3_qf, vx2_qf)));

    // Compute max and scale
    HVX_Vector vmax01_hf = hvx_vec_reduce_max_fp16(hvx_vec_abs_fp16(vx01_hf));
    HVX_Vector vmax23_hf = hvx_vec_reduce_max_fp16(hvx_vec_abs_fp16(vx23_hf));

    // Replicate first fp16 scale across all lanes
    HVX_Vector ctrl = *(const HVX_Vector *) repl_1x_fp16;
    vmax01_hf         = Q6_V_vdelta_VV(vmax01_hf, ctrl);
    vmax23_hf         = Q6_V_vdelta_VV(vmax23_hf, ctrl);

    HVX_Vector vd01_qf16 = Q6_Vqf16_vmpy_VhfVhf(vmax01_hf, Q6_Vh_vsplat_R(0x2008));  // 1.0 / 127.0
    HVX_Vector vd23_qf16 = Q6_Vqf16_vmpy_VhfVhf(vmax23_hf, Q6_Vh_vsplat_R(0x2008));  // 1.0 / 127.0
    HVX_Vector vd01_hf   = Q6_Vhf_equals_Vqf16(vd01_qf16);
    HVX_Vector vd23_hf   = Q6_Vhf_equals_Vqf16(vd23_qf16);

    hvx_vec_store_u(y_d + 0, 4, vd01_hf);
    hvx_vec_store_u(y_d + 4, 4, vd23_hf);

    // Divide input by the scale
    HVX_Vector vd01_inv_hf = hvx_vec_inverse_fp16(vd01_hf);
    HVX_Vector vd23_inv_hf = hvx_vec_inverse_fp16(vd23_hf);
    vx01_hf              = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(vx01_hf, vd01_inv_hf));
    vx23_hf              = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(vx23_hf, vd23_inv_hf));

    // Convert to int8
    HVX_Vector vx01_i16 = hvx_vec_i16_from_hf_rnd_sat(vx01_hf);
    HVX_Vector vx23_i16 = hvx_vec_i16_from_hf_rnd_sat(vx23_hf);
    HVX_Vector vx_i8    = Q6_Vb_vpack_VhVh_sat(vx23_i16, vx01_i16);

    *(HVX_Vector *) y_q = vx_i8;
}

static inline void quantize_block_fp32_q8x4(float * restrict x, uint8_t * restrict y_q, uint8_t * restrict y_d) {
    assert((unsigned long) x % 128 == 0);
    assert((unsigned long) y_q % 128 == 0);

    HVX_Vector * vx = (HVX_Vector *) x;

    // Load and convert into QF32
    HVX_Vector zero   = Q6_V_vsplat_R(0);
    HVX_Vector vx0_qf = Q6_Vqf32_vsub_VsfVsf(vx[0], zero);  // 32 elements
    HVX_Vector vx1_qf = Q6_Vqf32_vsub_VsfVsf(vx[1], zero);  // 32 elements
    HVX_Vector vx2_qf = Q6_Vqf32_vsub_VsfVsf(vx[2], zero);  // 32 elements
    HVX_Vector vx3_qf = Q6_Vqf32_vsub_VsfVsf(vx[3], zero);  // 32 elements

    // Convert into fp16
    HVX_Vector vx01_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vx1_qf, vx0_qf)));
    HVX_Vector vx23_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vx3_qf, vx2_qf)));

    // Compute max and scale
    HVX_Vector vmax_hf = hvx_vec_reduce_max_fp16(hvx_vec_abs_fp16(vx01_hf));
    vmax_hf            = hvx_vec_reduce_max2_fp16(hvx_vec_abs_fp16(vx23_hf), vmax_hf);

    // Replicate first fp16 scale across all lanes
    HVX_Vector ctrl = *(const HVX_Vector *) repl_1x_fp16;
    vmax_hf         = Q6_V_vdelta_VV(vmax_hf, ctrl);

    HVX_Vector vd_qf16 = Q6_Vqf16_vmpy_VhfVhf(vmax_hf, Q6_Vh_vsplat_R(0x2008));  // 1.0 / 127.0
    HVX_Vector vd_hf   = Q6_Vhf_equals_Vqf16(vd_qf16);

    *(HVX_UVector *) y_d = vd_hf;

    // Divide input by the scale
    HVX_Vector vd_inv_hf = hvx_vec_inverse_fp16(vd_hf);
    vx01_hf              = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(vx01_hf, vd_inv_hf));
    vx23_hf              = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(vx23_hf, vd_inv_hf));

    // Convert to int8
    HVX_Vector vx01_i16 = hvx_vec_i16_from_hf_rnd_sat(vx01_hf);
    HVX_Vector vx23_i16 = hvx_vec_i16_from_hf_rnd_sat(vx23_hf);
    HVX_Vector vx_i8    = Q6_Vb_vpack_VhVh_sat(vx23_i16, vx01_i16);

    *(HVX_Vector *) y_q = vx_i8;
}

// Overrides input x
static void quantize_row_fp32_q8x4x2(float * restrict x, uint8_t * restrict y, uint32_t k) {
    assert(k % 32 == 0);
    const uint32_t qk = QK_Q8_0x4x2;
    const uint32_t nb = (k + qk - 1) / qk;

    const uint32_t qrow_size = k;              // int8

    const uint32_t dblk_size = 8 * 2;          // 8x __fp16
    const uint32_t qblk_size = QK_Q8_0x4x2;    // int8

    uint8_t * restrict y_q = (y + 0);          // quants first
    uint8_t * restrict y_d = (y + qrow_size);  // then scales

    // Temp scales override input since we're working off of the aligned temp buffer in VTCM
    uint8_t * restrict t_d = (uint8_t *) x;

    for (uint32_t i = 0; i < nb; i++) {
#if FP32_QUANTIZE_GROUP_SIZE == 32
        quantize_block_fp32_q8x1(x + (i * 2 + 0) * qk / 2, y_q + (i * 2 + 0) * qblk_size / 2,
                                 t_d + (i * 2 + 0) * dblk_size / 2);
        quantize_block_fp32_q8x1(x + (i * 2 + 1) * qk / 2, y_q + (i * 2 + 1) * qblk_size / 2,
                                 t_d + (i * 2 + 1) * dblk_size / 2);
#elif FP32_QUANTIZE_GROUP_SIZE == 64
        quantize_block_fp32_q8x2(x + (i * 2 + 0) * qk / 2, y_q + (i * 2 + 0) * qblk_size / 2,
                                 t_d + (i * 2 + 0) * dblk_size / 2);
        quantize_block_fp32_q8x2(x + (i * 2 + 1) * qk / 2, y_q + (i * 2 + 1) * qblk_size / 2,
                                 t_d + (i * 2 + 1) * dblk_size / 2);
#elif FP32_QUANTIZE_GROUP_SIZE == 128
        quantize_block_fp32_q8x4(x + (i * 2 + 0) * qk / 2, y_q + (i * 2 + 0) * qblk_size / 2,
                                 t_d + (i * 2 + 0) * dblk_size / 2);
        quantize_block_fp32_q8x4(x + (i * 2 + 1) * qk / 2, y_q + (i * 2 + 1) * qblk_size / 2,
                                 t_d + (i * 2 + 1) * dblk_size / 2);
#else
#error "FP32_QUANTIZE_GROUP_SIZE must be 32, 64, or 128"
#endif
    }

    // now copy the scales into final location
    hvx_copy_fp16_ua(y_d, t_d, nb * 8);
}

static void quantize_fp32_q8x4x2(const struct htp_tensor * src,
                                 uint8_t * restrict dst,
                                 struct htp_spad * spad,
                                 uint32_t          nth,
                                 uint32_t          ith,
                                 uint32_t          nrows_per_thread) {

    uint64_t t1 = HAP_perf_get_qtimer_count();

    const uint32_t ne0 = src->ne[0];
    const uint32_t ne1 = src->ne[1];
    const uint32_t ne2 = src->ne[2];
    const uint32_t ne3 = src->ne[3];

    const uint32_t nrows = ne1 * ne2 * ne3;                             // total n_rows

    const uint32_t ir_first = nrows_per_thread * ith;                   // first row
    const uint32_t ir_last  = MIN(ir_first + nrows_per_thread, nrows);  // last row

    const size_t src_row_size = src->nb[1];
    const size_t dst_row_size = q8x4x2_row_size(ne0);

    uint8_t * restrict src_data = (uint8_t *) src->data + (src_row_size * ir_first);
    uint8_t * restrict dst_data = (uint8_t *) dst + (dst_row_size * ir_first);
    uint8_t * restrict tmp_data = (uint8_t *) spad->data + (spad->size_per_thread * ith);

    const size_t src_row_size_padded = htp_round_up(src_row_size, QK_Q8_0x4x2 * sizeof(float));
    memset(tmp_data, 0, src_row_size_padded);  // zero-out temp row data for padding

    for (uint32_t i = ir_first; i < ir_last; ++i) {
        htp_l2fetch(src_data, 2, src_row_size, src_row_size);
        hvx_copy_fp32_aa(tmp_data, src_data, ne0);

        // FARF(HIGH, "quantize-q8x4-row: %u\n", i);
        quantize_row_fp32_q8x4x2((float *) tmp_data, dst_data, ne0);
        dst_data += dst_row_size;
        src_data += src_row_size;
    }

    uint64_t t2 = HAP_perf_get_qtimer_count();

    FARF(HIGH, "quantize-fp32-q8x4: %u/%u : n-rows %u (%u:%u) row-size %u -> %u usec %u\n", ith, nth, nrows, ir_first,
         ir_last, src_row_size, dst_row_size, (unsigned) HAP_perf_qtimer_count_to_us(t2 - t1));
}

static void htp_quantize_fp32_q8x4x2(unsigned int n, unsigned int i, void * data) {
    struct htp_ops_context * octx = data;
    quantize_fp32_q8x4x2(&octx->src1, octx->src1_spad.data, &octx->src0_spad, n, i, octx->src1_nrows_per_thread);
}

// ** matmul callbacks for worker_pool

static void htp_matvec_q4x4x2_q8x4x2(unsigned int n, unsigned int i, void * data) {
    struct htp_ops_context * octx = data;

    struct htp_matmul_type mt;
    mt.type        = "q4x4x2-q8x4x2";
    mt.vec_dot     = vec_dot_q4x4x2_q8x4x2;
    mt.vec_dot_rx2 = vec_dot_q4x4x2_q8x4x2_rx2;

    matvec(&mt, &octx->src0, &octx->src1, &octx->dst, &octx->src0_spad, &octx->src1_spad, &octx->dst_spad, n, i,
           octx->src0_nrows_per_thread, octx->ctx->dma[i]);
}

static void htp_matmul_q4x4x2_q8x4x2(unsigned int n, unsigned int i, void * data) {
    struct htp_ops_context * octx = data;

    struct htp_matmul_type mt;
    mt.type        = "q4x4x2-q8x4x2";
    mt.vec_dot     = vec_dot_q4x4x2_q8x4x2;
    mt.vec_dot_rx2 = vec_dot_q4x4x2_q8x4x2_rx2;

    matmul(&mt, &octx->src0, &octx->src1, &octx->dst, &octx->src0_spad, &octx->src1_spad, &octx->dst_spad, n, i,
           octx->src0_nrows_per_thread, octx->ctx->dma[i]);
}

static void htp_matvec_q8x4x2_q8x4x2(unsigned int n, unsigned int i, void * data) {
    struct htp_ops_context * octx = data;

    struct htp_matmul_type mt;
    mt.type        = "q8x4x2-q8x4x2";
    mt.vec_dot     = vec_dot_q8x4x2_q8x4x2;
    mt.vec_dot_rx2 = vec_dot_q8x4x2_q8x4x2_rx2;

    matvec(&mt, &octx->src0, &octx->src1, &octx->dst, &octx->src0_spad, &octx->src1_spad, &octx->dst_spad, n, i,
           octx->src0_nrows_per_thread, octx->ctx->dma[i]);
}

static void htp_matmul_q8x4x2_q8x4x2(unsigned int n, unsigned int i, void * data) {
    struct htp_ops_context * octx = data;

    struct htp_matmul_type mt;
    mt.type        = "q8x4x2-q8x4x2";
    mt.vec_dot     = vec_dot_q8x4x2_q8x4x2;
    mt.vec_dot_rx2 = vec_dot_q8x4x2_q8x4x2_rx2;

    matmul(&mt, &octx->src0, &octx->src1, &octx->dst, &octx->src0_spad, &octx->src1_spad, &octx->dst_spad, n, i,
           octx->src0_nrows_per_thread, octx->ctx->dma[i]);
}

static void htp_matvec_mxfp4x4x2_q8x4x2(unsigned int n, unsigned int i, void * data) {
    struct htp_ops_context * octx = data;

    struct htp_matmul_type mt;
    mt.type        = "mxfp4x4x2-q8x4x2";
    mt.vec_dot     = vec_dot_mxfp4x4x2_q8x4x2;
    mt.vec_dot_rx2 = vec_dot_mxfp4x4x2_q8x4x2_rx2;

    matvec(&mt, &octx->src0, &octx->src1, &octx->dst, &octx->src0_spad, &octx->src1_spad, &octx->dst_spad, n, i,
           octx->src0_nrows_per_thread, octx->ctx->dma[i]);
}

static void htp_matmul_mxfp4x4x2_q8x4x2(unsigned int n, unsigned int i, void * data) {
    struct htp_ops_context * octx = data;

    struct htp_matmul_type mt;
    mt.type        = "mxfp4x4x2-q8x4x2";
    mt.vec_dot     = vec_dot_mxfp4x4x2_q8x4x2;
    mt.vec_dot_rx2 = vec_dot_mxfp4x4x2_q8x4x2_rx2;

    matmul(&mt, &octx->src0, &octx->src1, &octx->dst, &octx->src0_spad, &octx->src1_spad, &octx->dst_spad, n, i,
           octx->src0_nrows_per_thread, octx->ctx->dma[i]);
}

static void htp_matmul_f16_f32(unsigned int n, unsigned int i, void * data) {
    struct htp_ops_context * octx = data;
    matmul_f16_f32(octx, n, i);
}

// ** matmul-id callbacks for worker_pool

static void htp_matvec_id_q4x4x2_q8x4x2(unsigned int n, unsigned int i, void * data) {
    struct htp_ops_context * octx = data;

    struct htp_matmul_type mt;
    mt.type        = "q4x4x2-q8x4x2";
    mt.vec_dot     = vec_dot_q4x4x2_q8x4x2;
    mt.vec_dot_rx2 = vec_dot_q4x4x2_q8x4x2_rx2;

    matvec_id(&mt, &octx->src0, &octx->src1, &octx->src2, &octx->dst, &octx->src0_spad, &octx->src1_spad,
              &octx->src2_spad, &octx->dst_spad, n, i, octx->src0_nrows_per_thread, octx->ctx->dma[i]);
}

static void htp_matmul_id_q4x4x2_q8x4x2(unsigned int n, unsigned int i, void * data) {
    struct htp_ops_context * octx = data;

    struct htp_matmul_type mt;
    mt.type        = "q4x4x2-q8x4x2";
    mt.vec_dot     = vec_dot_q4x4x2_q8x4x2;
    mt.vec_dot_rx2 = vec_dot_q4x4x2_q8x4x2_rx2;

    matmul_id(&mt, &octx->src0, &octx->src1, &octx->src2, &octx->dst, &octx->src0_spad, &octx->src1_spad,
              &octx->src2_spad, &octx->dst_spad, n, i, octx->src0_nrows_per_thread, octx->ctx->dma[i]);
}

static void htp_matvec_id_q8x4x2_q8x4x2(unsigned int n, unsigned int i, void * data) {
    struct htp_ops_context * octx = data;

    struct htp_matmul_type mt;
    mt.type        = "q8x4x2-q8x4x2";
    mt.vec_dot     = vec_dot_q8x4x2_q8x4x2;
    mt.vec_dot_rx2 = vec_dot_q8x4x2_q8x4x2_rx2;

    matvec_id(&mt, &octx->src0, &octx->src1, &octx->src2, &octx->dst, &octx->src0_spad, &octx->src1_spad,
              &octx->src2_spad, &octx->dst_spad, n, i, octx->src0_nrows_per_thread, octx->ctx->dma[i]);
}

static void htp_matmul_id_q8x4x2_q8x4x2(unsigned int n, unsigned int i, void * data) {
    struct htp_ops_context * octx = data;

    struct htp_matmul_type mt;
    mt.type        = "q8x4x2-q8x4x2";
    mt.vec_dot     = vec_dot_q8x4x2_q8x4x2;
    mt.vec_dot_rx2 = vec_dot_q8x4x2_q8x4x2_rx2;

    matmul_id(&mt, &octx->src0, &octx->src1, &octx->src2, &octx->dst, &octx->src0_spad, &octx->src1_spad,
              &octx->src2_spad, &octx->dst_spad, n, i, octx->src0_nrows_per_thread, octx->ctx->dma[i]);
}

static void htp_matvec_id_mxfp4x4x2_q8x4x2(unsigned int n, unsigned int i, void * data) {
    struct htp_ops_context * octx = data;

    struct htp_matmul_type mt;
    mt.type        = "mxfp4x4x2-q8x4x2";
    mt.vec_dot     = vec_dot_mxfp4x4x2_q8x4x2;
    mt.vec_dot_rx2 = vec_dot_mxfp4x4x2_q8x4x2_rx2;

    matvec_id(&mt, &octx->src0, &octx->src1, &octx->src2, &octx->dst, &octx->src0_spad, &octx->src1_spad,
              &octx->src2_spad, &octx->dst_spad, n, i, octx->src0_nrows_per_thread, octx->ctx->dma[i]);
}

static void htp_matmul_id_mxfp4x4x2_q8x4x2(unsigned int n, unsigned int i, void * data) {
    struct htp_ops_context * octx = data;

    struct htp_matmul_type mt;
    mt.type        = "mxfp4x4x2-q8x4x2";
    mt.vec_dot     = vec_dot_mxfp4x4x2_q8x4x2;
    mt.vec_dot_rx2 = vec_dot_mxfp4x4x2_q8x4x2_rx2;

    matmul_id(&mt, &octx->src0, &octx->src1, &octx->src2, &octx->dst, &octx->src0_spad, &octx->src1_spad,
              &octx->src2_spad, &octx->dst_spad, n, i, octx->src0_nrows_per_thread, octx->ctx->dma[i]);
}

// ** main matmul entry point

int op_matmul(struct htp_ops_context * octx) {
    const struct htp_tensor * src0 = &octx->src0;
    const struct htp_tensor * src1 = &octx->src1;
    struct htp_tensor *       dst  = &octx->dst;

    htp_matmul_preamble;

    const char * op_type;

    const uint32_t src0_nrows = ne01 * ne02 * ne03;
    const uint32_t src1_nrows = ne11 * ne12 * ne13;

    const size_t src0_row_size = nb01;
    const size_t dst_row_size  = nb1;
    size_t       src1_row_size = nb11;

    const size_t src0_row_size_padded = htp_round_up(src0_row_size, 128);
    size_t       src1_row_size_padded;

    worker_callback_t quant_job_func;
    worker_callback_t matmul_job_func;

    bool need_quant = !(octx->flags & HTP_OPFLAGS_SKIP_QUANTIZE);

    switch (src0->type) {
        case HTP_TYPE_Q4_0:
            op_type        = "q4x4x2-fp32";
            quant_job_func = htp_quantize_fp32_q8x4x2;
            if (src1_nrows > 1) {
                matmul_job_func = htp_matmul_q4x4x2_q8x4x2;
            } else {
                matmul_job_func = htp_matvec_q4x4x2_q8x4x2;
            }

            src1_row_size = q8x4x2_row_size(ne10);  // row size post quantization

            // Entire src1 tensor is placed into the VTCM
            // For other tensors we allocate N rows per thread, padded to HVX vector size

            octx->dst_spad.size_per_thread  = htp_round_up(HTP_SPAD_DST_NROWS * dst_row_size, 256);
            octx->src0_spad.size_per_thread = htp_round_up(HTP_SPAD_SRC0_NROWS * src0_row_size_padded, 256);
            octx->src1_spad.size_per_thread = htp_round_up(src1_row_size * src1_nrows, 256);

            // src0 spad is also used in dynamic quantizer to store padded src1 rows
            src1_row_size_padded = htp_round_up(src1_row_size, QK_Q8_0x4x2 * sizeof(float));
            if (octx->src0_spad.size_per_thread < src1_row_size_padded) {
                octx->src0_spad.size_per_thread = src1_row_size_padded;
            }

            octx->src1_spad.size = octx->src1_spad.size_per_thread;
            octx->src0_spad.size = octx->src0_spad.size_per_thread * octx->n_threads;
            octx->dst_spad.size  = octx->dst_spad.size_per_thread * octx->n_threads;
            break;

        case HTP_TYPE_Q8_0:
            op_type        = "q8x4x2-fp32";
            quant_job_func = htp_quantize_fp32_q8x4x2;
            if (src1_nrows > 1) {
                matmul_job_func = htp_matmul_q8x4x2_q8x4x2;
            } else {
                matmul_job_func = htp_matvec_q8x4x2_q8x4x2;
            }

            src1_row_size = q8x4x2_row_size(ne10);  // row size post quantization

            // Entire src1 tensor is placed into the VTCM
            // For other tensors we allocate N rows per thread, padded to HVX vector size

            octx->dst_spad.size_per_thread  = htp_round_up(HTP_SPAD_DST_NROWS * dst_row_size, 256);
            octx->src0_spad.size_per_thread = htp_round_up(HTP_SPAD_SRC0_NROWS * src0_row_size_padded, 256);
            octx->src1_spad.size_per_thread = htp_round_up(src1_row_size * src1_nrows, 256);

            // src0 spad is also used in dynamic quantizer to store padded src1 rows
            src1_row_size_padded = htp_round_up(src1_row_size, QK_Q8_0x4x2 * sizeof(float));
            if (octx->src0_spad.size_per_thread < src1_row_size_padded) {
                octx->src0_spad.size_per_thread = src1_row_size_padded;
            }

            octx->src1_spad.size = octx->src1_spad.size_per_thread;
            octx->src0_spad.size = octx->src0_spad.size_per_thread * octx->n_threads;
            octx->dst_spad.size  = octx->dst_spad.size_per_thread * octx->n_threads;
            break;

        case HTP_TYPE_MXFP4:
            op_type        = "mxfp4x4x2-f32";
            quant_job_func = htp_quantize_fp32_q8x4x2;
            if (src1_nrows > 1) {
                matmul_job_func = htp_matmul_mxfp4x4x2_q8x4x2;
            } else {
                matmul_job_func = htp_matvec_mxfp4x4x2_q8x4x2;
            }

            src1_row_size = q8x4x2_row_size(ne10);  // row size post quantization

            // Entire src1 tensor is placed into the VTCM
            // For other tensors we allocate N rows per thread, padded to HVX vector size

            octx->dst_spad.size_per_thread  = htp_round_up(HTP_SPAD_DST_NROWS * dst_row_size, 256);
            octx->src0_spad.size_per_thread = htp_round_up(HTP_SPAD_SRC0_NROWS * src0_row_size_padded, 256);
            octx->src1_spad.size_per_thread = htp_round_up(src1_row_size * src1_nrows, 256);

            // src0 spad is also used in dynamic quantizer to store padded src1 rows
            src1_row_size_padded = htp_round_up(src1_row_size, QK_Q8_0x4x2 * sizeof(float));
            if (octx->src0_spad.size_per_thread < src1_row_size_padded) {
                octx->src0_spad.size_per_thread = src1_row_size_padded;
            }

            octx->src1_spad.size = octx->src1_spad.size_per_thread;
            octx->src0_spad.size = octx->src0_spad.size_per_thread * octx->n_threads;
            octx->dst_spad.size  = octx->dst_spad.size_per_thread * octx->n_threads;
            break;

        case HTP_TYPE_F16:
            op_type         = "f16-f32";
            quant_job_func  = NULL;  // htp_quantize_f32_f16;
            matmul_job_func = htp_matmul_f16_f32;

            // For all tensors we allocate N rows per thread, padded to HVX vector size
            octx->dst_spad.size_per_thread  = htp_round_up(HTP_SPAD_DST_NROWS * dst_row_size, 256);
            octx->src0_spad.size_per_thread = htp_round_up(HTP_SPAD_SRC0_NROWS * src0_row_size, 256);

            // Allocate enough space for FP32 weights chunk AND FP16 converted weights chunk.
            // We use src1_spad for both if possible, or double allocate.
            // Let's allocate enough for the block size we want to process.
            // src1_row_size is FP32.
            // We process blocks of 'src1_nrows' (tiled).
            // Let's assume block size of 64 rows (HTP_SPAD_SRC1_NROWS is 16, maybe too small for tiling?)
            // We will define block size in matmul function, but here we need allocation.
            // Let's reserve enough for 64 rows of FP32 + 64 rows of FP16.
            // 64 is a good chunk size.
            // src1_row_size is in bytes (FP32).
            {
                size_t src1_block_rows      = 64;
                size_t src1_row_size_padded = htp_round_up(src1_row_size, 128);
                // We allocate enough for padded rows to ensure alignment in VTCM
                size_t src1_fp32_size = htp_round_up(src1_block_rows * src1_row_size_padded, 256);
                // For FP16, we also want aligned stride
                size_t src1_fp16_row_size_padded = htp_round_up(src1_row_size / 2, 128);
                size_t src1_fp16_size            = htp_round_up(src1_block_rows * src1_fp16_row_size_padded, 256);
                octx->src1_spad.size_per_thread  = src1_fp32_size + src1_fp16_size;
            }

            octx->src0_spad.size = octx->src0_spad.size_per_thread * octx->n_threads;
            octx->src1_spad.size = octx->src1_spad.size_per_thread * octx->n_threads;
            octx->dst_spad.size  = octx->dst_spad.size_per_thread * octx->n_threads;

            // FP16 matmul supports broadcast and needs fastdiv
            octx->mm_div_ne12_ne1 = init_fastdiv_values(src1->ne[2] * dst->ne[1]);
            octx->mm_div_ne1      = init_fastdiv_values(dst->ne[1]);
            octx->mm_div_r2       = init_fastdiv_values(src1->ne[2] / src0->ne[2]);
            octx->mm_div_r3       = init_fastdiv_values(src1->ne[3] / src0->ne[3]);

            need_quant = false;
            break;

        default:
            return HTP_STATUS_NO_SUPPORT;
    }

    // VTCM scratchpads for all tensors
    size_t spad_size = octx->src1_spad.size + octx->src0_spad.size + octx->dst_spad.size;

    FARF(HIGH, "matmul-%s : src0-spad-size %u src1-spad-size %u dst-spad-size %u (%zu)\n", op_type,
         octx->src0_spad.size, octx->src1_spad.size, octx->dst_spad.size, spad_size);

    FARF(HIGH, "matmul-%s : %ux%ux%ux%u * %ux%ux%ux%u-> %ux%ux%ux%u (0x%p, 0x%p, 0x%p)\n", op_type, src0->ne[0],
         src0->ne[1], src0->ne[2], src0->ne[3], src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3], dst->ne[0],
         dst->ne[1], dst->ne[2], dst->ne[3], src0->data, src1->data, dst->data);

    // Make sure the reserved vtcm size is sufficient
    if (octx->ctx->vtcm_size < spad_size) {
        FARF(ERROR, "matmul-%s : current VTCM reservation %zu is too small, needed %zu\n", op_type,
             octx->ctx->vtcm_size, spad_size);
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    octx->src0_spad.data = octx->ctx->vtcm_base;
    octx->src1_spad.data = octx->src0_spad.data + octx->src0_spad.size;
    octx->dst_spad.data  = octx->src1_spad.data + octx->src1_spad.size;

    octx->src0_nrows_per_thread = (src0_nrows + octx->n_threads - 1) / octx->n_threads;
    octx->src0_nrows_per_thread += (octx->src0_nrows_per_thread & 1);  // round up to even

    if (need_quant) {
        // Run quant jobs
        const uint32_t n_quant_jobs = MIN(src1_nrows, octx->n_threads);
        octx->src1_nrows_per_thread = (src1_nrows + n_quant_jobs - 1) / n_quant_jobs;
        worker_pool_run_func(octx->ctx->worker_pool, quant_job_func, octx, n_quant_jobs);
    }

    if (!(octx->flags & HTP_OPFLAGS_SKIP_COMPUTE)) {
        // Run matmul jobs
        const uint32_t n_matmul_jobs = octx->n_threads;
        worker_pool_run_func(octx->ctx->worker_pool, matmul_job_func, octx, n_matmul_jobs);
    }

    return HTP_STATUS_OK;
}

// ** main matmul-id entry point

int op_matmul_id(struct htp_ops_context * octx) {
    const struct htp_tensor * src0 = &octx->src0;
    const struct htp_tensor * src1 = &octx->src1;
    const struct htp_tensor * ids  = &octx->src2;
    struct htp_tensor *       dst  = &octx->dst;

    htp_matmul_preamble;

    const char * op_type;

    worker_callback_t quant_job_func;
    worker_callback_t matmul_id_job_func;

    const size_t src0_row_size = nb01;
    const size_t dst_row_size  = nb1;

    const size_t src0_row_size_padded = htp_round_up(src0_row_size, 128);

    const uint32_t src0_nrows = ne01;  // per expert
    const uint32_t src1_nrows = ne11 * ne12 * ne13;

    size_t src1_row_size;
    size_t src1_row_size_padded;

    // row groups
    const int n_ids = ids->ne[0];  // n_expert_used
    const int n_as  = ne02;        // n_expert

    size_t matrix_row_counts_size = n_as * sizeof(uint32_t);
    size_t matrix_row_map_size    = n_as * ids->ne[0] * ids->ne[1] * sizeof(struct mmid_row_mapping);

    switch (src0->type) {
        case HTP_TYPE_Q4_0:
            op_type        = "q4x2x2-f32";
            quant_job_func = htp_quantize_fp32_q8x4x2;
            src1_row_size  = q8x4x2_row_size(ne10);  // row size post quantization
            if (src1_nrows > 1) {
                matmul_id_job_func = htp_matmul_id_q4x4x2_q8x4x2;
            } else {
                matmul_id_job_func = htp_matvec_id_q4x4x2_q8x4x2;
            }

            // Entire src1 tensor is placed into the VTCM
            // For other tensors we allocate N rows per thread, padded to HVX vector size
            octx->dst_spad.size_per_thread  = htp_round_up(HTP_SPAD_DST_NROWS * dst_row_size, 256);
            octx->src0_spad.size_per_thread = htp_round_up(HTP_SPAD_SRC0_NROWS * src0_row_size_padded, 256);
            octx->src1_spad.size_per_thread = htp_round_up(src1_row_size * src1_nrows, 256);
            octx->src2_spad.size_per_thread = htp_round_up(matrix_row_counts_size + matrix_row_map_size, 256);

            // src0 spad is also used in dynamic quantizer to store padded src1 rows
            src1_row_size_padded = htp_round_up(src1_row_size, QK_Q8_0x4x2 * sizeof(float));
            if (octx->src0_spad.size_per_thread < src1_row_size_padded) {
                octx->src0_spad.size_per_thread = src1_row_size_padded;
            }

            octx->src2_spad.size = octx->src2_spad.size_per_thread;
            octx->src1_spad.size = octx->src1_spad.size_per_thread;
            octx->src0_spad.size = octx->src0_spad.size_per_thread * octx->n_threads;
            octx->dst_spad.size  = octx->dst_spad.size_per_thread * octx->n_threads;
            break;

        case HTP_TYPE_Q8_0:
            op_type        = "q8x2x2-f32";
            quant_job_func = htp_quantize_fp32_q8x4x2;
            src1_row_size  = q8x4x2_row_size(ne10);  // row size post quantization
            if (src1_nrows > 1) {
                matmul_id_job_func = htp_matmul_id_q8x4x2_q8x4x2;
            } else {
                matmul_id_job_func = htp_matvec_id_q8x4x2_q8x4x2;
            }

            // Entire src1 tensor is placed into the VTCM
            // For other tensors we allocate N rows per thread, padded to HVX vector size
            octx->dst_spad.size_per_thread  = htp_round_up(HTP_SPAD_DST_NROWS * dst_row_size, 256);
            octx->src0_spad.size_per_thread = htp_round_up(HTP_SPAD_SRC0_NROWS * src0_row_size_padded, 256);
            octx->src1_spad.size_per_thread = htp_round_up(src1_row_size * src1_nrows, 256);
            octx->src2_spad.size_per_thread = htp_round_up(matrix_row_counts_size + matrix_row_map_size, 256);

            // src0 spad is also used in dynamic quantizer to store padded src1 rows
            src1_row_size_padded = htp_round_up(src1_row_size, QK_Q8_0x4x2 * sizeof(float));
            if (octx->src0_spad.size_per_thread < src1_row_size_padded) {
                octx->src0_spad.size_per_thread = src1_row_size_padded;
            }

            octx->src2_spad.size = octx->src2_spad.size_per_thread;
            octx->src1_spad.size = octx->src1_spad.size_per_thread;
            octx->src0_spad.size = octx->src0_spad.size_per_thread * octx->n_threads;
            octx->dst_spad.size  = octx->dst_spad.size_per_thread * octx->n_threads;
            break;

        case HTP_TYPE_MXFP4:
            op_type        = "mxfp4x2x2-f32";
            quant_job_func = htp_quantize_fp32_q8x4x2;
            src1_row_size  = q8x4x2_row_size(ne10);  // row size post quantization
            if (src1_nrows > 1) {
                matmul_id_job_func = htp_matmul_id_mxfp4x4x2_q8x4x2;
            } else {
                matmul_id_job_func = htp_matvec_id_mxfp4x4x2_q8x4x2;
            }

            // Entire src1 tensor is placed into the VTCM
            // For other tensors we allocate N rows per thread, padded to HVX vector size
            octx->dst_spad.size_per_thread  = htp_round_up(HTP_SPAD_DST_NROWS * dst_row_size, 256);
            octx->src0_spad.size_per_thread = htp_round_up(HTP_SPAD_SRC0_NROWS * src0_row_size_padded, 256);
            octx->src1_spad.size_per_thread = htp_round_up(src1_row_size * src1_nrows, 256);
            octx->src2_spad.size_per_thread = htp_round_up(matrix_row_counts_size + matrix_row_map_size, 256);

            // src0 spad is also used in dynamic quantizer to store padded src1 rows
            src1_row_size_padded = htp_round_up(src1_row_size, QK_Q8_0x4x2 * sizeof(float));
            if (octx->src0_spad.size_per_thread < src1_row_size_padded) {
                octx->src0_spad.size_per_thread = src1_row_size_padded;
            }

            octx->src2_spad.size = octx->src2_spad.size_per_thread;
            octx->src1_spad.size = octx->src1_spad.size_per_thread;
            octx->src0_spad.size = octx->src0_spad.size_per_thread * octx->n_threads;
            octx->dst_spad.size  = octx->dst_spad.size_per_thread * octx->n_threads;
            break;

        default:
            return HTP_STATUS_NO_SUPPORT;
    }

    size_t spad_size = octx->src2_spad.size + octx->src1_spad.size + octx->src0_spad.size + octx->dst_spad.size;

    FARF(HIGH, "matmul-id-%s : src0-spad-size %u src1-spad-size %u src2-spad-size %u dst-spad-size %u (%zu)\n", op_type,
         octx->src0_spad.size, octx->src1_spad.size, octx->src2_spad.size, octx->dst_spad.size, spad_size);

    FARF(HIGH, "matmul-id-%s : %ux%ux%ux%u * %ux%ux%ux%u (%ux%ux%ux%u) -> %ux%ux%ux%u (0x%p, 0x%p, 0x%p)\n", op_type,
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
         ids->ne[0], ids->ne[1], ids->ne[2], ids->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], src0->data,
         src1->data, dst->data);

    // Make sure the reserved vtcm size is sufficient
    if (octx->ctx->vtcm_size < spad_size) {
        FARF(ERROR, "matmul-id-%s : current VTCM reservation %zu is too small, needed %zu\n", op_type,
             octx->ctx->vtcm_size, spad_size);
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    octx->src0_spad.data = octx->ctx->vtcm_base;
    octx->src1_spad.data = octx->src0_spad.data + octx->src0_spad.size;
    octx->src2_spad.data = octx->src1_spad.data + octx->src1_spad.size;
    octx->dst_spad.data  = octx->src2_spad.data + octx->src2_spad.size;

    octx->src0_nrows_per_thread = (src0_nrows + octx->n_threads - 1) / octx->n_threads;
    octx->src0_nrows_per_thread += (octx->src0_nrows_per_thread & 1);  // round up to even

    if (src1_nrows > 1) {
        // initialize matrix_row_counts and map
        uint32_t *                matrix_row_counts = (uint32_t *) octx->src2_spad.data + 0;
        struct mmid_row_mapping * matrix_rows       = (void *) octx->src2_spad.data + matrix_row_counts_size;

        memset(matrix_row_counts, 0, n_as * sizeof(uint32_t));

        // group rows by src0 matrix
        for (uint32_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {  // token idx
            for (uint32_t id = 0; id < n_ids; ++id) {         // expert idx
                const uint32_t i02 =
                    *(const uint32_t *) ((const uint8_t *) ids->data + iid1 * ids->nb[1] + id * ids->nb[0]);

                assert(i02 >= 0 && i02 < n_as);

                MMID_MATRIX_ROW(i02, matrix_row_counts[i02]) = (struct mmid_row_mapping) { id, iid1 };
                matrix_row_counts[i02] += 1;
            }
        }
    }

    // Setup worker pool callbacks
    if (!(octx->flags & HTP_OPFLAGS_SKIP_QUANTIZE)) {
        // Run quant jobs
        const uint32_t n_quant_jobs = MIN(src1_nrows, octx->n_threads);
        octx->src1_nrows_per_thread = (src1_nrows + n_quant_jobs - 1) / n_quant_jobs;
        worker_pool_run_func(octx->ctx->worker_pool, quant_job_func, octx, n_quant_jobs);
    }

    if (!(octx->flags & HTP_OPFLAGS_SKIP_COMPUTE)) {
        // Run matmul-id jobs
        const uint32_t n_matmul_jobs = octx->n_threads;
        worker_pool_run_func(octx->ctx->worker_pool, matmul_id_job_func, octx, n_matmul_jobs);
    }

    return HTP_STATUS_OK;
}
