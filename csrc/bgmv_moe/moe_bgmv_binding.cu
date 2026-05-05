/*
 * PyTorch C++ extension binding for BGMV MoE kernels.
 *
 * Registers two ops:
 *   - flashinfer_bgmv_moe::bgmv_moe_shrink
 *   - flashinfer_bgmv_moe::bgmv_moe_expand
 *
 * Copyright (c) 2025 by FlashInfer team.
 * Licensed under the Apache License, Version 2.0.
 */

#include <torch/extension.h>
#include "moe_bgmv_ops.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bgmv_moe_shrink", &dispatch_bgmv_moe_shrink,
        "BGMV MoE shrink kernel (multi-LoRA, expert-routed)",
        py::arg("y"), py::arg("x"), py::arg("w_ptr"),
        py::arg("sorted_token_ids"), py::arg("expert_ids"),
        py::arg("lora_indices"), py::arg("lora_stride"));

  m.def("bgmv_moe_expand", &dispatch_bgmv_moe_expand,
        "BGMV MoE expand kernel (multi-LoRA, expert-routed)",
        py::arg("y"), py::arg("x"), py::arg("w_ptr"),
        py::arg("sorted_token_ids"), py::arg("expert_ids"),
        py::arg("topk_weights"), py::arg("lora_indices"),
        py::arg("slice_start_loc"), py::arg("output_slices"),
        py::arg("lora_stride"));
}
