#ifndef IMAGE_PROCESSING__FILTER__COLOR__GAMMA_CORRECTOR_HPP_
#define IMAGE_PROCESSING__FILTER__COLOR__GAMMA_CORRECTOR_HPP_

#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

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
  GammaCorrector(cv::Size2i image_size);
  ~GammaCorrector();
  cv::Mat apply_filter(cv::Mat input_image);

private:
  cv::Size2i image_size_;

  cv::Mat correct_color_gamma(cv::Mat input_image);
  cv::Mat correct_gray_gamma(cv::Mat input_image);

  void apply_adaptive_gamma_correct(cv::Mat & value_image);
};
} // namespace color
} // namespace filter
} // namespace visible
} // namespace jhj
#endif  // IMAGE_PROCESSING__FILTER__COLOR__GAMMA_CORRECTOR_HPP_
