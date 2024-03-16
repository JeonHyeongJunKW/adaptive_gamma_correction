#ifndef IMAGE_PROCESSING__FILTER__COLOR__GAMMA_CORRECTOR_HPP_
#define IMAGE_PROCESSING__FILTER__COLOR__GAMMA_CORRECTOR_HPP_

#include <vector>
#include <cmath>

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include "cuda/host/image_processing.cuh"

namespace jhj
{
namespace visible
{
namespace filter
{
namespace color
{
class GammaCorrector
{
public:
  GammaCorrector(cudaStream_t stream, cv::Size2i image_size);
  ~GammaCorrector();
  cv::Mat apply_filter(cudaStream_t stream, cv::Mat input_image);

private:
  const float THRESHOLD = 3.0f;
  cv::Size2i image_size_;
  void * intensity_data_;

  cv::Mat correct_color_gamma(cudaStream_t stream, cv::Mat input_image);
  cv::Mat correct_gray_gamma(cudaStream_t stream, cv::Mat input_image);

  void apply_adaptive_gamma_correct(cudaStream_t stream, cv::Mat & value_image);
};
} // namespace color
} // namespace filter
} // namespace visible
} // namespace jhj
#endif  // IMAGE_PROCESSING__FILTER__COLOR__GAMMA_CORRECTOR_HPP_
