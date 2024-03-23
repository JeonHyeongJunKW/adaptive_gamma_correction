#ifndef CUDA__HOST__IMAGE_PROCESSING_CUH_
#define CUDA__HOST__IMAGE_PROCESSING_CUH_

#include <stdint.h>

#include <cuda_runtime.h>

namespace jhj
{
namespace visible
{
namespace device
{
__device__ __forceinline__ float3 change_bgr_to_hsv(uchar3 bgr_value)
{
  // 0.0f <= H < 360.0f, 0.0f <= S <= 1.0f, 0.0f <= V <=1.0f
  float blue = static_cast<float>(bgr_value.x) / 255.0f;
  float green = static_cast<float>(bgr_value.y) / 255.0f;
  float red = static_cast<float>(bgr_value.z) / 255.0f;

  float value = fmaxf(blue, fmaxf(green, red));
  float min_bgr = fminf(blue, fminf(green, red));

  if (value == min_bgr) {
    return make_float3(0.0f, 0.0f, value);
  }

  float saturation = value != 0.0f ? (value - min_bgr) / value : 0.0f;
  float hue;
  if (red == value) {
    hue = 60.0f * (green - blue) / (value - min_bgr);
  } else if (green == value) {
    hue = 120.0f + 60.0f * (blue - red) / (value - min_bgr);
  } else {
    hue = 240.0f + 60.0f * (red - green) / (value - min_bgr);
  }
  hue = hue < 0.0f ? hue + 360.0f : hue;

  return make_float3(hue, saturation, value);
}

__device__ __forceinline__ uchar3 change_hsv_to_bgr(float3 hsv_value)
{
  float hue = hsv_value.x;
  float saturation = hsv_value.y;
  float value = hsv_value.z;

  float chroma = value * saturation;
  float x = chroma * (1.0f - fabsf(fmodf(hue / 60.0f, 2.0f) - 1.0f));
  float m = value - chroma;

  int hue_case = static_cast<int>(floorf(hue / 60.0f));

  float3 temp_bgr;
  switch (hue_case)
  {
    case 0:
      temp_bgr = make_float3(0.0f, x, chroma);
      break;
    case 1:
      temp_bgr = make_float3(0.0f, chroma, x);
      break;
    case 2:
      temp_bgr = make_float3(x, chroma, 0.0f);
      break;
    case 3:
      temp_bgr = make_float3(chroma, x, 0.0f);
      break;
    case 4:
      temp_bgr = make_float3(chroma, 0.0f, x);
      break;
    default:
      temp_bgr = make_float3(x, 0.0f, chroma);
  }
  uchar3 output_bgr =
    make_uchar3(
      static_cast<uint8_t>(255.0f * (temp_bgr.x + m)),
      static_cast<uint8_t>(255.0f * (temp_bgr.y + m)),
      static_cast<uint8_t>(255.0f * (temp_bgr.z + m)));
  return output_bgr;
}
}
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
