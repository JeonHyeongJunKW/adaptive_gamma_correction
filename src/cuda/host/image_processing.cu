#include "cuda/host/image_processing.cuh"


namespace jhj
{
namespace visible
{
namespace device
{
__global__ void apply_adaptive_gamma_correction(
  float * intensity_data,
  const float gamma,
  const float mean,
  const bool is_bright_image,
  const int2 size)
{
  const int2 position =
    make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

  if (position.x < 0 || position.x >= size.x || position.y < 0 || position.y >= size.y) {
    return;
  }

  int target_index = position.y * size.x + position.x;

  float input_value = intensity_data[target_index];
  float gamma_applied_value = powf(input_value, gamma);
  float c;
  if (is_bright_image) {
    c = 1.0f;
  } else {
    c = 1.0f / (gamma_applied_value + (1.0f - gamma_applied_value) * powf(mean, gamma));
  }
  intensity_data[target_index] = c * gamma_applied_value;

}
} // namespace device
namespace host
{
cudaError_t apply_adaptive_gamma_correction(
  void * intensity_data,
  const float gamma,
  const float mean,
  const bool is_bright_image,
  const int image_height,
  const int image_width,
  cudaStream_t stream)
{

  const dim3 block_dim(32, 32);
  const dim3 grid_dim(
    std::ceil(static_cast<float>(image_width) / 32.0f),
    std::ceil(static_cast<float>(image_height) / 32.0f));

  device::apply_adaptive_gamma_correction<<<grid_dim, block_dim, 0, stream>>>(
    static_cast<float *>(intensity_data),
    gamma,
    mean,
    is_bright_image,
    make_int2(image_width, image_height));

  return cudaGetLastError();
}
} // namespace host
} // namespace visible
} // namespace jhj
