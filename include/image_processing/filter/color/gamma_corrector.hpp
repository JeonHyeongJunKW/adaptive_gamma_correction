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
  void analyze_color_image(cv::Mat & input_image, float & mean, float & standard_deviation);
  void apply_filter(
    cudaStream_t stream,
    cv::Mat & output_image,
    const cv::Mat & input_image,
    const float & mean,
    const float & standard_deviation);

private:
  const float THRESHOLD = 3.0f;
  cv::Size2i image_size_;
  void * corrected_image_;

  void correct_color_gamma(
    cudaStream_t stream,
    cv::Mat & output_image,
    const cv::Mat & input_image,
    const float & gamma,
    const float & mean,
    const bool & is_bright_image);
};
} // namespace color
} // namespace filter
} // namespace visible
} // namespace jhj
#endif  // IMAGE_PROCESSING__FILTER__COLOR__GAMMA_CORRECTOR_HPP_
