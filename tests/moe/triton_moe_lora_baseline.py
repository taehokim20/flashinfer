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

Standalone Triton MoE LoRA kernel for benchmarking.

This is a simplified extraction of vLLM's fused_moe_lora Triton kernel,
adapted to run without any vLLM dependencies. It uses the
"naive_block_assignment" path for simplicity.

Original source: vllm/vllm/lora/ops/triton_ops/fused_moe_lora_op.py
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _moe_lora_shrink_kernel(
    a_ptr,  # input hidden states [num_tokens, hidden_dim]
    b_ptr,  # lora_a weight pointers (uint64 tensor)
    c_ptr,  # output [num_slices, num_pairs, rank]
    expert_ids_ptr,  # [num_pairs]
    token_lora_mapping_ptr,  # [num_tokens]
    N,  # rank (output dim)
    K,  # hidden_dim (input dim)
    num_pairs,
    top_k_num,
    num_experts,
    max_loras,
    stride_am,
    stride_ak,
    stride_bl,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """MoE LoRA shrink: x @ lora_a^T for each (token, expert) pair."""
    pid = tl.program_id(axis=0)
    slice_id = tl.program_id(axis=1)

    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Each element in offs_m is a pair index
    # token_idx = pair_idx // top_k_num
    token_ids = offs_m // top_k_num
    pair_mask = offs_m < num_pairs

    # Load lora_id for each token in the block
    lora_ids = tl.load(token_lora_mapping_ptr + token_ids, mask=pair_mask, other=-1)

    # Load expert_id for each pair
    _ = tl.load(expert_ids_ptr + offs_m, mask=pair_mask, other=-1)

    # For simplicity, use the first valid pair's lora_id and expert_id
    # (This is a simplification — real kernel handles per-pair routing)
    lora_id = tl.min(tl.where(lora_ids >= 0, lora_ids, max_loras), axis=0)
    expert_id = tl.load(expert_ids_ptr + pid_m * BLOCK_SIZE_M)

    if lora_id >= max_loras:
        return
    if expert_id < 0:
        return

    # Load weight pointer for this slice
    cur_b_ptr = tl.load(b_ptr + slice_id).to(tl.pointer_type(tl.bfloat16))

    # Input: a_ptr[token_idx, :hidden_dim]
    a_ptrs = a_ptr + token_ids[:, None] * stride_am + offs_k[None, :] * stride_ak

    # Weight: lora_a[lora_id, expert_id, :rank, :hidden_dim]
    b_ptrs = (
        cur_b_ptr
        + lora_id * stride_bl
        + expert_id * stride_be
        + offs_k[:, None] * stride_bk
        + offs_n[None, :] * stride_bn
    )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for _k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_mask = offs_k < K
        a = tl.load(a_ptrs, mask=pair_mask[:, None] & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0)
        accumulator += tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16))
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        offs_k += BLOCK_SIZE_K

    # Mask out invalid pairs and lora_ids
    valid_mask = pair_mask[:, None] & (offs_n[None, :] < N) & (lora_ids[:, None] >= 0)

    # Store output
    c_ptrs = (
        c_ptr
        + slice_id * num_pairs * stride_cm
        + offs_m[:, None] * stride_cm
        + offs_n[None, :] * stride_cn
    )
    tl.store(c_ptrs, accumulator.to(tl.bfloat16), mask=valid_mask)


@triton.jit
def _moe_lora_expand_kernel(
    a_ptr,  # shrink output [num_slices, num_pairs, rank]
    b_ptr,  # lora_b weight pointers (uint64 tensor)
    c_ptr,  # output [num_tokens, feat_out * num_slices]
    expert_ids_ptr,
    topk_weights_ptr,
    token_lora_mapping_ptr,
    N,  # feat_out
    K,  # rank
    num_pairs,
    num_tokens,
    top_k_num,
    num_experts,
    max_loras,
    stride_am,
    stride_ak,
    stride_bl,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """MoE LoRA expand: shrink_out @ lora_b^T * topk_weight."""
    pid = tl.program_id(axis=0)
    slice_id = tl.program_id(axis=1)

    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    pair_mask = offs_m < num_pairs
    token_ids = offs_m // top_k_num

    lora_ids = tl.load(token_lora_mapping_ptr + token_ids, mask=pair_mask, other=-1)
    expert_id = tl.load(expert_ids_ptr + pid_m * BLOCK_SIZE_M)
    lora_id = tl.min(tl.where(lora_ids >= 0, lora_ids, max_loras), axis=0)

    if lora_id >= max_loras:
        return
    if expert_id < 0:
        return

    cur_b_ptr = tl.load(b_ptr + slice_id).to(tl.pointer_type(tl.bfloat16))

    # Input: shrink_out[slice_id, pair_idx, :rank]
    a_ptrs = (
        a_ptr
        + slice_id * num_pairs * stride_am
        + offs_m[:, None] * stride_am
        + offs_k[None, :] * stride_ak
    )

    # Weight: lora_b[lora_id, expert_id, :feat_out, :rank]
    b_ptrs = (
        cur_b_ptr
        + lora_id * stride_bl
        + expert_id * stride_be
        + offs_k[:, None] * stride_bk
        + offs_n[None, :] * stride_bn
    )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for _k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_mask = offs_k < K
        a = tl.load(a_ptrs, mask=pair_mask[:, None] & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0)
        accumulator += tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16))
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        offs_k += BLOCK_SIZE_K

    # Apply topk weights
    topk_w = tl.load(topk_weights_ptr + offs_m, mask=pair_mask, other=0.0)
    accumulator = accumulator * topk_w[:, None]

    # Store: accumulate into output[token_idx, slice_id*N + col]
    valid_mask = pair_mask[:, None] & (offs_n[None, :] < N) & (lora_ids[:, None] >= 0)
    c_ptrs = (
        c_ptr
        + token_ids[:, None] * stride_cm
        + (slice_id * N + offs_n[None, :]) * stride_cn
    )
    tl.atomic_add(c_ptrs, accumulator.to(tl.float32), mask=valid_mask)


def triton_moe_lora(
    x: torch.Tensor,  # [num_tokens, hidden_dim]
    lora_a_weights: list,  # list of [max_loras, num_experts, rank, hidden_dim]
    lora_b_weights: list,  # list of [max_loras, num_experts, feat_out, rank]
    expert_ids: torch.Tensor,  # [num_pairs]
    lora_indices: torch.Tensor,  # [num_tokens]
    topk_weights: torch.Tensor,  # [num_pairs]
    top_k: int,
) -> torch.Tensor:
    """
    Standalone Triton MoE LoRA implementation for benchmarking.
    """
    num_tokens = x.size(0)
    hidden_dim = x.size(1)
    num_slices = len(lora_a_weights)
    rank = lora_a_weights[0].size(2)
    num_experts = lora_a_weights[0].size(1)
    max_loras = lora_a_weights[0].size(0)
    num_pairs = num_tokens * top_k
    feat_out = lora_b_weights[0].size(2)
    device = x.device
    dtype = x.dtype

    # Build pointer tensors
    a_ptrs = torch.tensor(
        [w.data_ptr() for w in lora_a_weights], device=device, dtype=torch.uint64
    )
    b_ptrs = torch.tensor(
        [w.data_ptr() for w in lora_b_weights], device=device, dtype=torch.uint64
    )

    # Shrink
    BLOCK_M = 16
    BLOCK_N = max(16, min(64, rank))
    BLOCK_K = max(16, min(128, hidden_dim))

    # Pad rank to BLOCK_N if needed
    shrink_out = torch.zeros(num_slices, num_pairs, rank, dtype=dtype, device=device)

    grid_shrink = (
        triton.cdiv(num_pairs, BLOCK_M) * triton.cdiv(rank, BLOCK_N),
        num_slices,
    )

    _moe_lora_shrink_kernel[grid_shrink](
        x,
        a_ptrs,
        shrink_out,
        expert_ids,
        lora_indices,
        N=rank,
        K=hidden_dim,
        num_pairs=num_pairs,
        top_k_num=top_k,
        num_experts=num_experts,
        max_loras=max_loras,
        stride_am=x.stride(0),
        stride_ak=x.stride(1),
        stride_bl=lora_a_weights[0].stride(0),
        stride_be=lora_a_weights[0].stride(1),
        stride_bk=lora_a_weights[0].stride(3),  # hidden dim stride
        stride_bn=lora_a_weights[0].stride(2),  # rank stride
        stride_cm=shrink_out.stride(1),
        stride_cn=shrink_out.stride(2),
        BLOCK_SIZE_M=BLOCK_M,
        BLOCK_SIZE_N=BLOCK_N,
        BLOCK_SIZE_K=BLOCK_K,
    )

    # Expand
    BLOCK_N_E = max(16, min(128, feat_out))
    BLOCK_K_E = max(16, min(32, rank))

    output = torch.zeros(
        num_tokens, feat_out * num_slices, dtype=torch.float32, device=device
    )

    grid_expand = (
        triton.cdiv(num_pairs, BLOCK_M) * triton.cdiv(feat_out, BLOCK_N_E),
        num_slices,
    )

    _moe_lora_expand_kernel[grid_expand](
        shrink_out,
        b_ptrs,
        output,
        expert_ids,
        topk_weights,
        lora_indices,
        N=feat_out,
        K=rank,
        num_pairs=num_pairs,
        num_tokens=num_tokens,
        top_k_num=top_k,
        num_experts=num_experts,
        max_loras=max_loras,
        stride_am=shrink_out.stride(1),
        stride_ak=shrink_out.stride(2),
        stride_bl=lora_b_weights[0].stride(0),
        stride_be=lora_b_weights[0].stride(1),
        stride_bk=lora_b_weights[0].stride(3),  # rank stride
        stride_bn=lora_b_weights[0].stride(2),  # feat_out stride
        stride_cm=output.stride(0),
        stride_cn=output.stride(1),
        BLOCK_SIZE_M=BLOCK_M,
        BLOCK_SIZE_N=BLOCK_N_E,
        BLOCK_SIZE_K=BLOCK_K_E,
    )

    return output.to(dtype)
