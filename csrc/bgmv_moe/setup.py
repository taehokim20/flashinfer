"""
Build script for the BGMV MoE CUDA extension.

Usage:
    cd flashinfer/csrc/bgmv_moe
    python setup.py install  # or: pip install -e .

This builds the extension as a standalone module `flashinfer_bgmv_moe_cuda`
that can be imported directly for testing.

Copyright (c) 2025 by FlashInfer team.
Licensed under the Apache License, Version 2.0.
"""

import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Source files
cuda_sources = [
    "moe_bgmv_ops.cu",
    "moe_bgmv_binding.cu",
    "moe_bgmv_bf16_bf16_bf16.cu",
    "moe_bgmv_bf16_fp32_bf16.cu",
    "moe_bgmv_fp16_fp16_fp16.cu",
    "moe_bgmv_fp16_fp32_fp16.cu",
    "moe_bgmv_fp32_bf16_bf16.cu",
    "moe_bgmv_fp32_fp16_fp16.cu",
]

# Compute capabilities: SM70+ (V100, A100, H100, B200)
# For faster compilation during development, reduce this list
nvcc_flags = [
    "-O3",
    "--use_fast_math",
    "-std=c++17",
    # Target architectures
    "-gencode=arch=compute_70,code=sm_70",
    "-gencode=arch=compute_80,code=sm_80",
    "-gencode=arch=compute_89,code=sm_89",
    "-gencode=arch=compute_90,code=sm_90",
]

# For faster dev iteration, only compile for current GPU:
# nvcc_flags = ["-O3", "--use_fast_math", "-std=c++17"]

setup(
    name="flashinfer_bgmv_moe_cuda",
    version="0.1.0",
    description="Multi-LoRA MoE BGMV CUDA kernels for FlashInfer",
    ext_modules=[
        CUDAExtension(
            name="flashinfer_bgmv_moe_cuda",
            sources=cuda_sources,
            include_dirs=[os.path.dirname(os.path.abspath(__file__))],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": nvcc_flags,
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
