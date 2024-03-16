#ifndef CUDA__HOST__IMAGE_PROCESSING_CUH_
#define CUDA__HOST__IMAGE_PROCESSING_CUH_

#include <cuda_runtime.h>

namespace jhj
{
namespace visible
{
namespace host
{
cudaError_t apply_adaptive_gamma_correction(
  void * intensity_data,
  const float gamma,
  const float mean,
  const bool is_bright_image,
  const int image_height,
  const int image_width,
  cudaStream_t stream);
}  // namespace host
}  // namespace visible
}  // namespace jhj
#endif  // CUDA__HOST__IMAGE_PROCESSING_CUH_
