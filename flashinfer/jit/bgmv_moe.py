"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import functools
from pathlib import Path

from . import env as jit_env
from .core import JitSpec, gen_jit_spec, logger


def _get_bgmv_moe_csrc_dir() -> Path:
    """Get the path to the BGMV MoE CUDA source directory.

    Handles both installed package (data/csrc/bgmv_moe) and
    development checkout (../../csrc/bgmv_moe relative to this file).
    """
    # Standard path via FlashInfer's data directory
    standard_path = jit_env.FLASHINFER_CSRC_DIR / "bgmv_moe"
    if standard_path.exists():
        return standard_path

    # Development fallback: relative to this file
    dev_path = Path(__file__).parent.parent.parent / "csrc" / "bgmv_moe"
    if dev_path.exists():
        return dev_path

    raise FileNotFoundError(
        f"BGMV MoE CUDA sources not found. Checked:\n"
        f"  - {standard_path}\n"
        f"  - {dev_path}\n"
        f"Please ensure the csrc/bgmv_moe/ directory exists."
    )


@functools.cache
def gen_bgmv_moe_module() -> JitSpec:
    """
    Generate the JIT compilation spec for the BGMV MoE CUDA kernels.

    This compiles the multi-LoRA MoE BGMV shrink/expand kernel pair.
    Supports SM70+ (V100, A100, H100, B200).

    Returns:
        JitSpec that can be built and loaded.
    """
    csrc_dir = _get_bgmv_moe_csrc_dir()

    sources = [
        csrc_dir / "moe_bgmv_ops.cu",
        csrc_dir / "moe_bgmv_binding.cu",
        csrc_dir / "moe_bgmv_bf16_bf16_bf16.cu",
        csrc_dir / "moe_bgmv_bf16_fp32_bf16.cu",
        csrc_dir / "moe_bgmv_fp16_fp16_fp16.cu",
        csrc_dir / "moe_bgmv_fp16_fp32_fp16.cu",
        csrc_dir / "moe_bgmv_fp32_bf16_bf16.cu",
        csrc_dir / "moe_bgmv_fp32_fp16_fp16.cu",
    ]

    # Verify sources exist
    for src in sources:
        if not src.exists():
            raise FileNotFoundError(
                f"BGMV MoE source file not found: {src}. Expected at: {csrc_dir}"
            )

    nvcc_flags = [
        "-gencode=arch=compute_70,code=sm_70",
        "-gencode=arch=compute_80,code=sm_80",
        "-gencode=arch=compute_89,code=sm_89",
        "-gencode=arch=compute_90,code=sm_90",
    ]

    spec = gen_jit_spec(
        name="flashinfer_bgmv_moe",
        sources=sources,
        extra_cuda_cflags=nvcc_flags,
        extra_include_paths=[str(csrc_dir)],
    )

    logger.info(f"Generated BGMV MoE JIT spec: {spec.name}")
    return spec


@functools.cache
def load_bgmv_moe_module():
    """
    Build and load the BGMV MoE CUDA extension via FlashInfer's JIT system.

    Returns the loaded module with `bgmv_moe_shrink` and `bgmv_moe_expand` functions.
    """
    spec = gen_bgmv_moe_module()
    module = spec.build_and_load()
    logger.info("BGMV MoE module loaded successfully")
    return module
