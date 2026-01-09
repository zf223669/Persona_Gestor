
#include <torch/extension.h>

void transpose_pad_permute_cuda(
                torch::Tensor input,
                torch::Tensor output,
                int tensor_layout);

void scale_fuse_quant_cuda(
                torch::Tensor input,
                torch::Tensor output,
                torch::Tensor scale,
                int num_tokens,
                float scale_max,
                int tensor_layout);