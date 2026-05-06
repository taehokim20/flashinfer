#ifndef BGMV_MOE_VEC_DTYPES_CUH_
#define BGMV_MOE_VEC_DTYPES_CUH_

/*
 * Vectorized load/store helpers for BGMV MoE kernels.
 * Supports half, nv_bfloat16, and float at vec_size = 1, 2, 4, 8.
 *
 * Ported from vLLM's bgmv_moe_cuda/vec_dtypes.cuh for FlashInfer.
 * Copyright (c) 2025 by FlashInfer team.
 * Licensed under the Apache License, Version 2.0.
 */

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <type_traits>

#define BGMV_MOE_INLINE inline __attribute__((always_inline)) __device__ __host__

template <typename float_t, size_t vec_size>
struct vec_t {
  BGMV_MOE_INLINE float_t& operator[](size_t i);
  BGMV_MOE_INLINE const float_t& operator[](size_t i) const;
  BGMV_MOE_INLINE void fill(float_t val);
  BGMV_MOE_INLINE void load(const float_t* ptr);
  BGMV_MOE_INLINE void store(float_t* ptr) const;
  template <typename T>
  BGMV_MOE_INLINE void cast_from(const vec_t<T, vec_size>& src);
  template <typename T>
  BGMV_MOE_INLINE void cast_load(const T* ptr);
  template <typename T>
  BGMV_MOE_INLINE void cast_store(T* ptr) const;
  BGMV_MOE_INLINE static void memcpy(float_t* dst, const float_t* src);
};

template <typename src_float_t, typename tgt_float_t, size_t vec_size>
BGMV_MOE_INLINE void cast_from_impl(const vec_t<src_float_t, vec_size>& src,
                                    vec_t<tgt_float_t, vec_size>& dst) {
#pragma unroll
  for (size_t i = 0; i < vec_size; ++i) {
    dst[i] = tgt_float_t(src[i]);
  }
}

template <typename src_float_t, typename tgt_float_t, size_t vec_size>
BGMV_MOE_INLINE void cast_load_impl(const src_float_t* src_ptr, vec_t<tgt_float_t, vec_size>& dst) {
  if constexpr (std::is_same<src_float_t, tgt_float_t>::value) {
    dst.load(src_ptr);
  } else {
    vec_t<src_float_t, vec_size> tmp;
    tmp.load(src_ptr);
    dst.cast_from(tmp);
  }
}

template <typename src_float_t, typename tgt_float_t, size_t vec_size>
BGMV_MOE_INLINE void cast_store_impl(const vec_t<src_float_t, vec_size>& src,
                                     tgt_float_t* dst_ptr) {
  if constexpr (std::is_same<src_float_t, tgt_float_t>::value) {
    src.store(dst_ptr);
  } else {
    vec_t<tgt_float_t, vec_size> tmp;
    tmp.cast_from(src);
    tmp.store(dst_ptr);
  }
}

/******************* vec_t<half> *******************/

// half x 1
template <>
struct vec_t<half, 1> {
  half data;
  BGMV_MOE_INLINE half& operator[](size_t i) { return ((half*)(&data))[i]; }
  BGMV_MOE_INLINE const half& operator[](size_t i) const { return ((const half*)(&data))[i]; }
  BGMV_MOE_INLINE void fill(half val) { data = val; }
  BGMV_MOE_INLINE void load(const half* ptr) { data = *ptr; }
  BGMV_MOE_INLINE void store(half* ptr) const { *ptr = data; }
  template <typename T>
  BGMV_MOE_INLINE void cast_from(const vec_t<T, 1>& src) {
    cast_from_impl(src, *this);
  }
  template <typename T>
  BGMV_MOE_INLINE void cast_load(const T* ptr) {
    cast_load_impl(ptr, *this);
  }
  template <typename T>
  BGMV_MOE_INLINE void cast_store(T* ptr) const {
    cast_store_impl(*this, ptr);
  }
  BGMV_MOE_INLINE static void memcpy(half* dst, const half* src) { *dst = *src; }
};

// half x 2
template <>
struct vec_t<half, 2> {
  half2 data;
  BGMV_MOE_INLINE half& operator[](size_t i) { return ((half*)(&data))[i]; }
  BGMV_MOE_INLINE const half& operator[](size_t i) const { return ((const half*)(&data))[i]; }
  BGMV_MOE_INLINE void fill(half val) { data = make_half2(val, val); }
  BGMV_MOE_INLINE void load(const half* ptr) { data = *((half2*)ptr); }
  BGMV_MOE_INLINE void store(half* ptr) const { *((half2*)ptr) = data; }
  template <typename T>
  BGMV_MOE_INLINE void cast_from(const vec_t<T, 2>& src) {
    cast_from_impl(src, *this);
  }
  template <typename T>
  BGMV_MOE_INLINE void cast_load(const T* ptr) {
    cast_load_impl(ptr, *this);
  }
  template <typename T>
  BGMV_MOE_INLINE void cast_store(T* ptr) const {
    cast_store_impl(*this, ptr);
  }
  BGMV_MOE_INLINE static void memcpy(half* dst, const half* src) {
    *((half2*)dst) = *((half2*)src);
  }
};

// half x 4
template <>
struct vec_t<half, 4> {
  uint2 data;
  BGMV_MOE_INLINE half& operator[](size_t i) { return ((half*)(&data))[i]; }
  BGMV_MOE_INLINE const half& operator[](size_t i) const { return ((const half*)(&data))[i]; }
  BGMV_MOE_INLINE void fill(half val) {
    *(half2*)(&data.x) = make_half2(val, val);
    *(half2*)(&data.y) = make_half2(val, val);
  }
  BGMV_MOE_INLINE void load(const half* ptr) { data = *((uint2*)ptr); }
  BGMV_MOE_INLINE void store(half* ptr) const { *((uint2*)ptr) = data; }
  template <typename T>
  BGMV_MOE_INLINE void cast_from(const vec_t<T, 4>& src) {
    cast_from_impl(src, *this);
  }
  template <typename T>
  BGMV_MOE_INLINE void cast_load(const T* ptr) {
    cast_load_impl(ptr, *this);
  }
  template <typename T>
  BGMV_MOE_INLINE void cast_store(T* ptr) const {
    cast_store_impl(*this, ptr);
  }
  BGMV_MOE_INLINE static void memcpy(half* dst, const half* src) {
    *((uint2*)dst) = *((uint2*)src);
  }
};

// half x 8 or more
template <size_t vec_size>
struct vec_t<half, vec_size> {
  uint4 data[vec_size / 8];
  BGMV_MOE_INLINE half& operator[](size_t i) { return ((half*)data)[i]; }
  BGMV_MOE_INLINE const half& operator[](size_t i) const { return ((const half*)data)[i]; }
  BGMV_MOE_INLINE void fill(half val) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 8; ++i) {
      *(half2*)(&(data[i].x)) = make_half2(val, val);
      *(half2*)(&(data[i].y)) = make_half2(val, val);
      *(half2*)(&(data[i].z)) = make_half2(val, val);
      *(half2*)(&(data[i].w)) = make_half2(val, val);
    }
  }
  BGMV_MOE_INLINE void load(const half* ptr) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 8; ++i) {
      data[i] = ((uint4*)ptr)[i];
    }
  }
  BGMV_MOE_INLINE void store(half* ptr) const {
#pragma unroll
    for (size_t i = 0; i < vec_size / 8; ++i) {
      ((uint4*)ptr)[i] = data[i];
    }
  }
  template <typename T>
  BGMV_MOE_INLINE void cast_from(const vec_t<T, vec_size>& src) {
    cast_from_impl(src, *this);
  }
  template <typename T>
  BGMV_MOE_INLINE void cast_load(const T* ptr) {
    cast_load_impl(ptr, *this);
  }
  template <typename T>
  BGMV_MOE_INLINE void cast_store(T* ptr) const {
    cast_store_impl(*this, ptr);
  }
  BGMV_MOE_INLINE static void memcpy(half* dst, const half* src) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 8; ++i) {
      ((uint4*)dst)[i] = ((uint4*)src)[i];
    }
  }
};

/******************* vec_t<nv_bfloat16> *******************/

// nv_bfloat16 x 1
template <>
struct vec_t<nv_bfloat16, 1> {
  nv_bfloat16 data;
  BGMV_MOE_INLINE nv_bfloat16& operator[](size_t i) { return ((nv_bfloat16*)(&data))[i]; }
  BGMV_MOE_INLINE const nv_bfloat16& operator[](size_t i) const {
    return ((const nv_bfloat16*)(&data))[i];
  }
  BGMV_MOE_INLINE void fill(nv_bfloat16 val) { data = val; }
  BGMV_MOE_INLINE void load(const nv_bfloat16* ptr) { data = *ptr; }
  BGMV_MOE_INLINE void store(nv_bfloat16* ptr) const { *ptr = data; }
  template <typename T>
  BGMV_MOE_INLINE void cast_from(const vec_t<T, 1>& src) {
    cast_from_impl(src, *this);
  }
  template <typename T>
  BGMV_MOE_INLINE void cast_load(const T* ptr) {
    cast_load_impl(ptr, *this);
  }
  template <typename T>
  BGMV_MOE_INLINE void cast_store(T* ptr) const {
    cast_store_impl(*this, ptr);
  }
  BGMV_MOE_INLINE static void memcpy(nv_bfloat16* dst, const nv_bfloat16* src) { *dst = *src; }
};

// nv_bfloat16 x 2
template <>
struct vec_t<nv_bfloat16, 2> {
  nv_bfloat162 data;
  BGMV_MOE_INLINE nv_bfloat16& operator[](size_t i) { return ((nv_bfloat16*)(&data))[i]; }
  BGMV_MOE_INLINE const nv_bfloat16& operator[](size_t i) const {
    return ((const nv_bfloat16*)(&data))[i];
  }
  BGMV_MOE_INLINE void fill(nv_bfloat16 val) { data = make_bfloat162(val, val); }
  BGMV_MOE_INLINE void load(const nv_bfloat16* ptr) { data = *((nv_bfloat162*)ptr); }
  BGMV_MOE_INLINE void store(nv_bfloat16* ptr) const { *((nv_bfloat162*)ptr) = data; }
  template <typename T>
  BGMV_MOE_INLINE void cast_from(const vec_t<T, 2>& src) {
    cast_from_impl(src, *this);
  }
  template <typename T>
  BGMV_MOE_INLINE void cast_load(const T* ptr) {
    cast_load_impl(ptr, *this);
  }
  template <typename T>
  BGMV_MOE_INLINE void cast_store(T* ptr) const {
    cast_store_impl(*this, ptr);
  }
  BGMV_MOE_INLINE static void memcpy(nv_bfloat16* dst, const nv_bfloat16* src) {
    *((nv_bfloat162*)dst) = *((nv_bfloat162*)src);
  }
};

// nv_bfloat16 x 4
template <>
struct vec_t<nv_bfloat16, 4> {
  uint2 data;
  BGMV_MOE_INLINE nv_bfloat16& operator[](size_t i) { return ((nv_bfloat16*)(&data))[i]; }
  BGMV_MOE_INLINE const nv_bfloat16& operator[](size_t i) const {
    return ((const nv_bfloat16*)(&data))[i];
  }
  BGMV_MOE_INLINE void fill(nv_bfloat16 val) {
    *(nv_bfloat162*)(&data.x) = make_bfloat162(val, val);
    *(nv_bfloat162*)(&data.y) = make_bfloat162(val, val);
  }
  BGMV_MOE_INLINE void load(const nv_bfloat16* ptr) { data = *((uint2*)ptr); }
  BGMV_MOE_INLINE void store(nv_bfloat16* ptr) const { *((uint2*)ptr) = data; }
  template <typename T>
  BGMV_MOE_INLINE void cast_from(const vec_t<T, 4>& src) {
    cast_from_impl(src, *this);
  }
  template <typename T>
  BGMV_MOE_INLINE void cast_load(const T* ptr) {
    cast_load_impl(ptr, *this);
  }
  template <typename T>
  BGMV_MOE_INLINE void cast_store(T* ptr) const {
    cast_store_impl(*this, ptr);
  }
  BGMV_MOE_INLINE static void memcpy(nv_bfloat16* dst, const nv_bfloat16* src) {
    *((uint2*)dst) = *((uint2*)src);
  }
};

// nv_bfloat16 x 8 or more
template <size_t vec_size>
struct vec_t<nv_bfloat16, vec_size> {
  uint4 data[vec_size / 8];
  BGMV_MOE_INLINE nv_bfloat16& operator[](size_t i) { return ((nv_bfloat16*)data)[i]; }
  BGMV_MOE_INLINE const nv_bfloat16& operator[](size_t i) const {
    return ((const nv_bfloat16*)data)[i];
  }
  BGMV_MOE_INLINE void fill(nv_bfloat16 val) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 8; ++i) {
      *(nv_bfloat162*)(&(data[i].x)) = make_bfloat162(val, val);
      *(nv_bfloat162*)(&(data[i].y)) = make_bfloat162(val, val);
      *(nv_bfloat162*)(&(data[i].z)) = make_bfloat162(val, val);
      *(nv_bfloat162*)(&(data[i].w)) = make_bfloat162(val, val);
    }
  }
  BGMV_MOE_INLINE void load(const nv_bfloat16* ptr) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 8; ++i) {
      data[i] = ((uint4*)ptr)[i];
    }
  }
  BGMV_MOE_INLINE void store(nv_bfloat16* ptr) const {
#pragma unroll
    for (size_t i = 0; i < vec_size / 8; ++i) {
      ((uint4*)ptr)[i] = data[i];
    }
  }
  template <typename T>
  BGMV_MOE_INLINE void cast_from(const vec_t<T, vec_size>& src) {
    cast_from_impl(src, *this);
  }
  template <typename T>
  BGMV_MOE_INLINE void cast_load(const T* ptr) {
    cast_load_impl(ptr, *this);
  }
  template <typename T>
  BGMV_MOE_INLINE void cast_store(T* ptr) const {
    cast_store_impl(*this, ptr);
  }
  BGMV_MOE_INLINE static void memcpy(nv_bfloat16* dst, const nv_bfloat16* src) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 8; ++i) {
      ((uint4*)dst)[i] = ((uint4*)src)[i];
    }
  }
};

/******************* vec_t<float> *******************/

// float x 1
template <>
struct vec_t<float, 1> {
  float data;
  BGMV_MOE_INLINE float& operator[](size_t i) { return ((float*)(&data))[i]; }
  BGMV_MOE_INLINE const float& operator[](size_t i) const { return ((const float*)(&data))[i]; }
  BGMV_MOE_INLINE void fill(float val) { data = val; }
  BGMV_MOE_INLINE void load(const float* ptr) { data = *ptr; }
  BGMV_MOE_INLINE void store(float* ptr) const { *ptr = data; }
  template <typename T>
  BGMV_MOE_INLINE void cast_from(const vec_t<T, 1>& src) {
    cast_from_impl(src, *this);
  }
  template <typename T>
  BGMV_MOE_INLINE void cast_load(const T* ptr) {
    cast_load_impl(ptr, *this);
  }
  template <typename T>
  BGMV_MOE_INLINE void cast_store(T* ptr) const {
    cast_store_impl(*this, ptr);
  }
  BGMV_MOE_INLINE static void memcpy(float* dst, const float* src) { *dst = *src; }
};

// float x 2
template <>
struct vec_t<float, 2> {
  float2 data;
  BGMV_MOE_INLINE float& operator[](size_t i) { return ((float*)(&data))[i]; }
  BGMV_MOE_INLINE const float& operator[](size_t i) const { return ((const float*)(&data))[i]; }
  BGMV_MOE_INLINE void fill(float val) { data = make_float2(val, val); }
  BGMV_MOE_INLINE void load(const float* ptr) { data = *((float2*)ptr); }
  BGMV_MOE_INLINE void store(float* ptr) const { *((float2*)ptr) = data; }
  template <typename T>
  BGMV_MOE_INLINE void cast_from(const vec_t<T, 2>& src) {
    cast_from_impl(src, *this);
  }
  template <typename T>
  BGMV_MOE_INLINE void cast_load(const T* ptr) {
    cast_load_impl(ptr, *this);
  }
  template <typename T>
  BGMV_MOE_INLINE void cast_store(T* ptr) const {
    cast_store_impl(*this, ptr);
  }
  BGMV_MOE_INLINE static void memcpy(float* dst, const float* src) {
    *((float2*)dst) = *((float2*)src);
  }
};

// float x 4 or more
template <size_t vec_size>
struct vec_t<float, vec_size> {
  float4 data[vec_size / 4];
  BGMV_MOE_INLINE float& operator[](size_t i) { return ((float*)(data))[i]; }
  BGMV_MOE_INLINE const float& operator[](size_t i) const { return ((const float*)(data))[i]; }
  BGMV_MOE_INLINE void fill(float val) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 4; ++i) {
      data[i] = make_float4(val, val, val, val);
    }
  }
  BGMV_MOE_INLINE void load(const float* ptr) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 4; ++i) {
      data[i] = ((float4*)ptr)[i];
    }
  }
  BGMV_MOE_INLINE void store(float* ptr) const {
#pragma unroll
    for (size_t i = 0; i < vec_size / 4; ++i) {
      ((float4*)ptr)[i] = data[i];
    }
  }
  template <typename T>
  BGMV_MOE_INLINE void cast_from(const vec_t<T, vec_size>& src) {
    cast_from_impl(src, *this);
  }
  template <typename T>
  BGMV_MOE_INLINE void cast_load(const T* ptr) {
    cast_load_impl(ptr, *this);
  }
  template <typename T>
  BGMV_MOE_INLINE void cast_store(T* ptr) const {
    cast_store_impl(*this, ptr);
  }
  BGMV_MOE_INLINE static void memcpy(float* dst, const float* src) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 4; ++i) {
      ((float4*)dst)[i] = ((float4*)src)[i];
    }
  }
};

/******************* Type cast specializations *******************/

template <size_t vec_size>
BGMV_MOE_INLINE void cast_from_impl(const vec_t<half, vec_size>& src, vec_t<float, vec_size>& dst) {
  if constexpr (vec_size == 1) {
    dst.data = float(src.data);
  } else {
#pragma unroll
    for (size_t i = 0; i < vec_size / 2; ++i) {
      ((float2*)(&dst.data))[i] = __half22float2(((half2*)(&src.data))[i]);
    }
  }
}

template <size_t vec_size>
BGMV_MOE_INLINE void cast_from_impl(const vec_t<float, vec_size>& src, vec_t<half, vec_size>& dst) {
  if constexpr (vec_size == 1) {
    dst.data = half(src.data);
  } else {
#pragma unroll
    for (size_t i = 0; i < vec_size / 2; ++i) {
      ((half2*)(&dst.data))[i] = __float22half2_rn(((float2*)(&src.data))[i]);
    }
  }
}

template <size_t vec_size>
BGMV_MOE_INLINE void cast_from_impl(const vec_t<nv_bfloat16, vec_size>& src,
                                    vec_t<float, vec_size>& dst) {
  if constexpr (vec_size == 1) {
    dst.data = float(src.data);
  } else {
#pragma unroll
    for (size_t i = 0; i < vec_size / 2; ++i) {
      ((float2*)(&dst.data))[i] = __bfloat1622float2(((nv_bfloat162*)(&src.data))[i]);
    }
  }
}

template <size_t vec_size>
BGMV_MOE_INLINE void cast_from_impl(const vec_t<float, vec_size>& src,
                                    vec_t<nv_bfloat16, vec_size>& dst) {
  if constexpr (vec_size == 1) {
    dst.data = nv_bfloat16(src.data);
  } else {
#pragma unroll
    for (size_t i = 0; i < vec_size / 2; ++i) {
      ((nv_bfloat162*)(&dst.data))[i] = __float22bfloat162_rn(((float2*)(&src.data))[i]);
    }
  }
}

#endif  // BGMV_MOE_VEC_DTYPES_CUH_
