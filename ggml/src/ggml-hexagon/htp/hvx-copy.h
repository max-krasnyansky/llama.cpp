#ifndef HVX_COPY_H
#define HVX_COPY_H

#include <hexagon_types.h>
#include <hexagon_protos.h>

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

static inline HVX_Vector hvx_vec_splat_fp32(float v) {
    union { float  f; uint32_t i; } u = { .f = v };
    return Q6_V_vsplat_R(u.i);
}

static inline HVX_Vector hvx_vec_splat_fp16(float v) {
    union { __fp16 f; uint16_t i; } u = { .f = v };
    return Q6_Vh_vsplat_R(u.i);
}

static inline void hvx_vec_store_u(void * restrict dst, uint32_t n, HVX_Vector v) {
    // Rotate as needed.
    v = Q6_V_vlalign_VVR(v, v, (size_t) dst);

    uint32_t left_off  = (size_t) dst & 127;
    uint32_t right_off = left_off + n;

    HVX_VectorPred ql_not = Q6_Q_vsetq_R((size_t) dst);
    HVX_VectorPred qr     = Q6_Q_vsetq2_R(right_off);

    if (right_off > 128) {
        Q6_vmem_QRIV(qr, (HVX_Vector *) dst + 1, v);
        // all 1's
        qr = Q6_Q_vcmp_eq_VbVb(v, v);
    }

    ql_not = Q6_Q_or_QQn(ql_not, qr);
    Q6_vmem_QnRIV(ql_not, (HVX_Vector *) dst, v);
}

static inline void hvx_vec_store_a(void * restrict dst, uint32_t n, HVX_Vector v) {
    assert((unsigned long) dst % 128 == 0);
    HVX_VectorPred m = Q6_Q_vsetq_R(n);
    Q6_vmem_QnRIV(m, (HVX_Vector *) dst, v);
}

#define hvx_copy_loop_body(dst_type, src_type, vec_store)            \
    do {                                                             \
        dst_type * restrict vdst = (dst_type *) dst;                 \
        src_type * restrict vsrc = (src_type *) src;                 \
                                                                     \
        uint32_t epv  = 128 / elem_size;                             \
        uint32_t nvec = n / epv;                                     \
        uint32_t nloe = n % epv;                                     \
                                                                     \
        uint32_t i = 0;                                              \
                                                                     \
        #pragma unroll(4)                                            \
        for (; i < nvec; i++) { vdst[i] = vsrc[i]; }                 \
                                                                     \
        if (nloe) {                                                  \
            vec_store((void *) &vdst[i], nloe * elem_size, vsrc[i]); \
        }                                                            \
    } while(0)

// Generic copy routines
static inline void hvx_copy_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n, uint32_t elem_size) {
    assert((unsigned long) dst % 128 == 0);
    assert((unsigned long) src % 128 == 0);
    hvx_copy_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

static inline void hvx_copy_ua(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n, uint32_t elem_size) {
    assert((unsigned long) src % 128 == 0);
    hvx_copy_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

static inline void hvx_copy_au(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n, uint32_t elem_size) {
    assert((unsigned long) dst % 128 == 0);
    hvx_copy_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

static inline void hvx_copy_uu(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n, uint32_t elem_size) {
    hvx_copy_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

// copy n fp16 elements : source and destination are aligned to HVX Vector (128)
static inline void hvx_copy_fp16_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    hvx_copy_aa(dst, src, n, sizeof(__fp16));
}

// copy n fp16 elements : source is aligned, destination is potentially unaligned
static inline void hvx_copy_fp16_ua(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    hvx_copy_ua(dst, src, n, sizeof(__fp16));
}

// copy n fp16 elements : source is aligned, destination is potentially unaligned
static inline void hvx_copy_fp16_au(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    hvx_copy_au(dst, src, n, sizeof(__fp16));
}

// copy n fp16 elements : source is aligned, destination is potentially unaligned
static inline void hvx_copy_fp16_au(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    hvx_copy_uu(dst, src, n, sizeof(__fp16));
}

// copy n fp32 elements : source and destination are aligned to HVX Vector (128)
static inline void hvx_copy_fp32_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    hvx_copy_aa(dst, src, n, sizeof(float));
}

// copy n fp32 elements : source is aligned, destination is unaligned
static inline void hvx_copy_fp32_ua(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    hvx_copy_ua(dst, src, n, sizeof(float));
}

// copy n fp32 elements : source is unaligned, destination is aligned
static inline void hvx_copy_fp32_au(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    hvx_copy_au(dst, src, n, sizeof(float));
}

// copy n fp32 elements : source is unaligned, destination unaligned
static inline void hvx_copy_fp32_uu(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    hvx_copy_uu(dst, src, n, sizeof(float));
}

#define hvx_copy_fp16_fp32_loop_body(dst_type, src_type, vec_store)                    \
    do {                                                                               \
        dst_type * restrict vdst = (dst_type *) dst;                                   \
        src_type * restrict vsrc = (src_type *) src;                                   \
                                                                                       \
        const HVX_Vector zero = Q6_V_vsplat_R(0);                                      \
                                                                                       \
        uint32_t nvec = n / 64;                                                        \
        uint32_t nloe = n % 64;                                                        \
                                                                                       \
        uint32_t i = 0;                                                                \
                                                                                       \
        #pragma unroll(4)                                                              \
        for (; i < nvec; i++) {                                                        \
            // Load src (fp32) and convert into fp16                                   \
            HVX_Vector s0_qf = Q6_Vqf32_vsub_VsfVsf(vsrc[i*2+0], zero);                \
            HVX_Vector s1_qf = Q6_Vqf32_vsub_VsfVsf(vsrc[i*2+1], zero);                \
            HVX_Vector s_hf  = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(s1_qf, s0_qf));    \
            vdst[i] = Q6_Vh_vdeal_Vh(s_hf);                                            \
        }                                                                              \
                                                                                       \
        if (nloe) {                                                                    \
            // Load src (fp32) and convert into fp16                                   \
            HVX_Vector s0_qf = Q6_Vqf32_vsub_VsfVsf(vsrc[i*2+0], zero);                \
            HVX_Vector s1_qf = Q6_Vqf32_vsub_VsfVsf(vsrc[i*2+1], zero);                \
            HVX_Vector s_hf  = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(s1_qf, s0_qf));    \
            vec_store((void *) &vdst[i], nloe * sizeof(__fp16), Q6_Vh_vdeal_Vh(s_hf)); \
        }                                                                              \
    } while(0)

// copy/convert n fp32 elements into n fp16 elements : source is aligned, destination is aligned
static inline void hvx_copy_fp16_fp32_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    assert((unsigned long) dst % 128 == 0);
    assert((unsigned long) src % 128 == 0);
    hvx_copy_fp16_fp32_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_u);
}

// copy/convert n fp32 elements into n fp16 elements : source is unaligned, destination is aligned
static inline void hvx_copy_fp16_fp32_au(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    assert((unsigned long) dst % 128 == 0);
    hvx_copy_fp16_fp32_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_u);
}

// copy/convert n fp32 elements into n fp16 elements : source is aligned, destination is unaligned
static inline void hvx_copy_fp16_fp32_ua(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    assert((unsigned long) src % 128 == 0);
    hvx_copy_fp16_fp32_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

// copy/convert n fp32 elements into n fp16 elements : source is unaligned, destination is unaligned
static inline void hvx_copy_fp16_fp32_uu(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    hvx_copy_fp16_fp32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

static inline void hvx_set_fp32_a(uint8_t * restrict dst, float elem, uint32_t n) {
    HVX_Vector * restrict vdst = (HVX_Vector *) dst;

    HVX_Vector velem = hvx_vec_splat_fp32(elem);

    assert((unsigned long) dst % 128 == 0);

    uint32_t nvec = n / 32;
    uint32_t nloe = n % 32;

    uint32_t i = 0;

    #pragma unroll(4)
    for (; i < nvec; i++) {
        vdst[i] = velem;
    }

    if (nloe) {
        hvx_vec_store_u((void *) &vdst[i], nloe * sizeof(float), velem);
    }
}

static inline void hvx_set_fp32_u(uint8_t * restrict dst, float elem, uint32_t n) {
    HVX_UVector * restrict vdst = (HVX_UVector *) dst;

    HVX_Vector velem = hvx_vec_splat_fp32(elem);

    assert((unsigned long) dst % 128 == 0);

    uint32_t nvec = n / 32;
    uint32_t nloe = n % 32;

    uint32_t i = 0;

    #pragma unroll(4)
    for (; i < nvec; i++) {
        vdst[i] = velem;
    }

    if (nloe) {
        hvx_vec_store_u((void *) &vdst[i], nloe * sizeof(float), velem);
    }
}

#endif // HVX_COPY_H
