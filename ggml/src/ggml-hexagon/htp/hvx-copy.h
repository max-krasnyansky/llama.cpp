#ifndef HVX_COPY_H
#define HVX_COPY_H

#include <hexagon_types.h>
#include <hexagon_protos.h>

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

static inline HVX_Vector hvx_vec_splat_fp32(float v) {
    union {
        float    f;
        uint32_t i;
    } fp32 = { .f = v };

    return Q6_V_vsplat_R(fp32.i);
}

static inline void hvx_vec_store_u(void * addr, uint32_t n, HVX_Vector v) {
    // Rotate as needed.
    v = Q6_V_vlalign_VVR(v, v, (size_t) addr);

    uint32_t left_off  = (size_t) addr & 127;
    uint32_t right_off = left_off + n;

    HVX_VectorPred ql_not = Q6_Q_vsetq_R((size_t) addr);
    HVX_VectorPred qr     = Q6_Q_vsetq2_R(right_off);

    if (right_off > 128) {
        Q6_vmem_QRIV(qr, (HVX_Vector *) addr + 1, v);
        // all 1's
        qr = Q6_Q_vcmp_eq_VbVb(v, v);
    }

    ql_not = Q6_Q_or_QQn(ql_not, qr);
    Q6_vmem_QnRIV(ql_not, (HVX_Vector *) addr, v);
}

static inline void hvx_vec_store_a(void * ptr, size_t n, HVX_Vector v) {
    assert((unsigned long) ptr % 128 == 0);

    HVX_VectorPred ql_not = Q6_Q_vsetq_R((size_t) ptr);
    HVX_VectorPred qr     = Q6_Q_vsetq2_R(n);
    ql_not                = Q6_Q_or_QQn(ql_not, qr);
    Q6_vmem_QnRIV(ql_not, (HVX_Vector *) ptr, v);
}

// Common copy routines
static inline void hvx_copy_common_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n, uint32_t elem_size) {
    HVX_Vector * restrict vdst = (HVX_Vector *) dst;
    HVX_Vector * restrict vsrc = (HVX_Vector *) src;

    assert((unsigned long) dst % 128 == 0);
    assert((unsigned long) src % 128 == 0);

    uint32_t elems_per_vec = 128 / elem_size;
    uint32_t nvec = n / elems_per_vec;
    uint32_t nloe = n % elems_per_vec;

    uint32_t i = 0;

    #pragma unroll(4)
    for (; i < nvec; i++) {
        vdst[i] = vsrc[i];
    }

    if (nloe) {
        hvx_vec_store_u((void *) &vdst[i], nloe * elem_size, vsrc[i]);
    }
}

static inline void hvx_copy_common_ua(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n, uint32_t elem_size) {
    HVX_UVector * restrict vdst = (HVX_UVector *) dst;
    HVX_Vector * restrict vsrc  = (HVX_Vector *) src;

    assert((unsigned long) src % 128 == 0);

    uint32_t elems_per_vec = 128 / elem_size;
    uint32_t nvec = n / elems_per_vec;
    uint32_t nloe = n % elems_per_vec;

    uint32_t i = 0;

    #pragma unroll(4)
    for (; i < nvec; i++) {
        vdst[i] = vsrc[i];
    }

    if (nloe) {
        hvx_vec_store_u((void *) &vdst[i], nloe * elem_size, vsrc[i]);
    }
}

static inline void hvx_copy_common_au(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n, uint32_t elem_size) {
    HVX_Vector * restrict vdst  = (HVX_Vector *) dst;
    HVX_UVector * restrict vsrc = (HVX_UVector *) src;

    assert((unsigned long) dst % 128 == 0);

    uint32_t elems_per_vec = 128 / elem_size;
    uint32_t nvec = n / elems_per_vec;
    uint32_t nloe = n % elems_per_vec;

    uint32_t i = 0;

    #pragma unroll(4)
    for (; i < nvec; i++) {
        vdst[i] = vsrc[i];
    }

    if (nloe) {
        hvx_vec_store_u((void *) &vdst[i], nloe * elem_size, vsrc[i]);
    }
}

static inline void hvx_copy_common_uu(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n, uint32_t elem_size) {
    HVX_UVector * restrict vdst = (HVX_UVector *) dst;
    HVX_UVector * restrict vsrc = (HVX_UVector *) src;

    uint32_t elems_per_vec = 128 / elem_size;
    uint32_t nvec = n / elems_per_vec;
    uint32_t nloe = n % elems_per_vec;

    uint32_t i = 0;

    #pragma unroll(4)
    for (; i < nvec; i++) {
        vdst[i] = vsrc[i];
    }

    if (nloe) {
        hvx_vec_store_u((void *) &vdst[i], nloe * elem_size, vsrc[i]);
    }
}

// copy n fp16 elements : source and destination are aligned to HVX Vector (128)
static inline void hvx_copy_fp16_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    hvx_copy_common_aa(dst, src, n, sizeof(__fp16));
}

// copy n fp16 elements : source is aligned, destination is potentially unaligned
static inline void hvx_copy_fp16_ua(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    hvx_copy_common_ua(dst, src, n, sizeof(__fp16));
}

// copy n fp16 elements : source is aligned, destination is potentially unaligned
static inline void hvx_copy_fp16_au(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    hvx_copy_common_au(dst, src, n, sizeof(__fp16));
}

// copy n fp32 elements : source and destination are aligned to HVX Vector (128)
static inline void hvx_copy_fp32_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    hvx_copy_common_aa(dst, src, n, sizeof(float));
}

// copy n fp32 elements : source is aligned, destination is unaligned
static inline void hvx_copy_fp32_ua(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    hvx_copy_common_ua(dst, src, n, sizeof(float));
}

// copy n fp32 elements : source is unaligned, destination is aligned
static inline void hvx_copy_fp32_au(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    hvx_copy_common_au(dst, src, n, sizeof(float));
}

// copy n fp32 elements : source is unaligned, destination unaligned
static inline void hvx_copy_fp32_uu(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    hvx_copy_common_uu(dst, src, n, sizeof(float));
}

// copy/convert n fp32 elements into n fp16 elements : source is unaligned, destination is unaligned
static inline void hvx_copy_fp16_fp32_uu(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    HVX_UVector * restrict vdst = (HVX_UVector *) dst; // fp16
    HVX_UVector * restrict vsrc = (HVX_UVector *) src; // fp32

    const HVX_Vector zero = Q6_V_vsplat_R(0);

    uint32_t nvec = n / 64;
    uint32_t nloe = n % 64;

    uint32_t i = 0;

    #pragma unroll(4)
    for (; i < nvec; i++) {
        // Load y (fp32) and convert into fp16
        HVX_Vector s0_qf = Q6_Vqf32_vsub_VsfVsf(vsrc[i*2+0], zero); // 32 elements
        HVX_Vector s1_qf = Q6_Vqf32_vsub_VsfVsf(vsrc[i*2+1], zero); // 32 elements
        HVX_Vector s_hf  = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(s1_qf, s0_qf));
        vdst[i] = Q6_Vh_vdeal_Vh(s_hf);
    }

    if (nloe) {
        // Load y (fp32) and convert into fp16
        HVX_Vector s0_qf = Q6_Vqf32_vsub_VsfVsf(vsrc[i*2+0], zero); // 32 elements
        HVX_Vector s1_qf = Q6_Vqf32_vsub_VsfVsf(vsrc[i*2+1], zero); // 32 elements
        HVX_Vector s_hf  = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(s1_qf, s0_qf));
        hvx_vec_store_u((void *) &vdst[i], nloe * sizeof(__fp16), Q6_Vh_vdeal_Vh(s_hf));
    }
}

// copy/convert n fp32 elements into n fp16 elements : source is aligned, destination is unaligned
static inline void hvx_copy_fp16_fp32_ua(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    HVX_UVector * restrict vdst = (HVX_UVector *) dst; // fp16
    HVX_Vector  * restrict vsrc = (HVX_Vector *)  src; // fp32

    const HVX_Vector zero = Q6_V_vsplat_R(0);

    uint32_t nvec = n / 64;
    uint32_t nloe = n % 64;

    uint32_t i = 0;

    #pragma unroll(4)
    for (; i < nvec; i++) {
        // Load y (fp32) and convert into fp16
        HVX_Vector s0_qf = Q6_Vqf32_vsub_VsfVsf(vsrc[i*2+0], zero); // 32 elements
        HVX_Vector s1_qf = Q6_Vqf32_vsub_VsfVsf(vsrc[i*2+1], zero); // 32 elements
        HVX_Vector s_hf  = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(s1_qf, s0_qf));
        vdst[i] = Q6_Vh_vdeal_Vh(s_hf);
    }

    if (nloe) {
        // Load y (fp32) and convert into fp16
        HVX_Vector s0_qf = Q6_Vqf32_vsub_VsfVsf(vsrc[i*2+0], zero); // 32 elements
        HVX_Vector s1_qf = Q6_Vqf32_vsub_VsfVsf(vsrc[i*2+1], zero); // 32 elements
        HVX_Vector s_hf  = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(s1_qf, s0_qf));
        hvx_vec_store_u((void *) &vdst[i], nloe * sizeof(__fp16), Q6_Vh_vdeal_Vh(s_hf));
    }
}

// copy/convert n fp32 elements into n fp16 elements : source is unaligned, destination is aligned
static inline void hvx_copy_fp16_fp32_au(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    HVX_Vector  * restrict vdst = (HVX_Vector *)  dst; // fp16
    HVX_UVector * restrict vsrc = (HVX_UVector *) src; // fp32

    const HVX_Vector zero = Q6_V_vsplat_R(0);

    uint32_t nvec = n / 64;
    uint32_t nloe = n % 64;

    uint32_t i = 0;

    #pragma unroll(4)
    for (; i < nvec; i++) {
        // Load y (fp32) and convert into fp16
        HVX_Vector s0_qf = Q6_Vqf32_vsub_VsfVsf(vsrc[i*2+0], zero); // 32 elements
        HVX_Vector s1_qf = Q6_Vqf32_vsub_VsfVsf(vsrc[i*2+1], zero); // 32 elements
        HVX_Vector s_hf  = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(s1_qf, s0_qf));
        vdst[i] = Q6_Vh_vdeal_Vh(s_hf);
    }

    if (nloe) {
        // Load y (fp32) and convert into fp16
        HVX_Vector s0_qf = Q6_Vqf32_vsub_VsfVsf(vsrc[i*2+0], zero); // 32 elements
        HVX_Vector s1_qf = Q6_Vqf32_vsub_VsfVsf(vsrc[i*2+1], zero); // 32 elements
        HVX_Vector s_hf  = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(s1_qf, s0_qf));
        hvx_vec_store_u((void *) &vdst[i], nloe * sizeof(__fp16), Q6_Vh_vdeal_Vh(s_hf));
    }
}

// bcast 1 fp32 element from source to n fp32 elements in destination : destination is aligned
static inline void hvx_bcast_fp32_a(uint8_t * restrict dst, float elem, uint32_t n) {
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

#endif // HVX_COPY_H
