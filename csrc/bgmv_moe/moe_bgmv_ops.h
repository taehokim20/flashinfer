#pragma once

/*
 * Public C++ interface for BGMV MoE kernels.
 *
 * Copyright (c) 2025 by FlashInfer team.
 * Licensed under the Apache License, Version 2.0.
 */

#include <torch/all.h>

void dispatch_bgmv_moe_shrink(torch::Tensor y, torch::Tensor x,
                               torch::Tensor w_ptr,
                               torch::Tensor sorted_token_ids,
                               torch::Tensor expert_ids,
                               torch::Tensor lora_indices,
                               int64_t lora_stride);

void dispatch_bgmv_moe_expand(torch::Tensor y, torch::Tensor x,
                               torch::Tensor w_ptr,
                               torch::Tensor sorted_token_ids,
                               torch::Tensor expert_ids,
                               torch::Tensor topk_weights,
                               torch::Tensor lora_indices,
                               torch::Tensor slice_start_loc,
                               std::vector<int64_t> output_slices,
                               int64_t lora_stride);
