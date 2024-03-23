#include "cuda/host/image_processing.cuh"


namespace jhj
{
namespace visible
{
namespace device
{
__global__ void apply_adaptive_gamma_correction(
  uchar3 * input_image,
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

  float3 hsv_value = change_bgr_to_hsv(input_image[target_index]);
  float intensity_value = hsv_value.z;
  float gamma_applied_value = powf(intensity_value, gamma);
  float c;
  if (is_bright_image) {
    c = 1.0f;
  } else {
    c = 1.0f / (gamma_applied_value + (1.0f - gamma_applied_value) * powf(mean, gamma));
  }
  hsv_value.z = c * gamma_applied_value;

  input_image[target_index] = change_hsv_to_bgr(hsv_value);

}
} // namespace device
namespace host
{
cudaError_t apply_adaptive_gamma_correction(
  void * input_image,
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
    static_cast<uchar3 *>(input_image),
    gamma,
    mean,
    is_bright_image,
    make_int2(image_width, image_height));

  return cudaGetLastError();
}
} // namespace host
} // namespace visible
} // namespace jhj
