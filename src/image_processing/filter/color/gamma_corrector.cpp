#include <chrono>

#include "image_processing/filter/color/gamma_corrector.hpp"


namespace jhj
{
namespace visible
{
filter::color::GammaCorrector::GammaCorrector(cv::Size2i image_size)
: image_size_(image_size)
{

}

filter::color::GammaCorrector::~GammaCorrector()
{

}

cv::Mat filter::color::GammaCorrector::apply_filter(cv::Mat input_image)
{
  const int channel_size = input_image.channels();
  return channel_size == 1 ? correct_gray_gamma(input_image) : correct_color_gamma(input_image);
}

cv::Mat filter::color::GammaCorrector::correct_color_gamma(cv::Mat input_image)
{
  cv::Mat hsv_image;
  cv::Mat value_image;
  cv::Mat output_image;
  std::vector<cv::Mat> channel_images(3);

  cv::cvtColor(input_image, hsv_image, cv::COLOR_BGR2HSV);
  cv::split(hsv_image, channel_images);
  channel_images[2].convertTo(value_image, CV_32FC1, 1.0f / 255.0f);

  this->apply_adaptive_gamma_correct(value_image);

  value_image.convertTo(channel_images[2], CV_8UC1, 255.0f);
  cv::merge(channel_images, output_image);
  cv::cvtColor(output_image, output_image, cv::COLOR_HSV2BGR);
  return output_image;
}

cv::Mat filter::color::GammaCorrector::correct_gray_gamma(cv::Mat input_image)
{
  cv::Mat value_image;
  cv::Mat output_image;
  input_image.convertTo(value_image, CV_32FC1, 1.0f / 255.0f);

  this->apply_adaptive_gamma_correct(value_image);

  value_image.convertTo(output_image, CV_8UC1, 255.0f);
  return output_image;
}

void filter::color::GammaCorrector::apply_adaptive_gamma_correct(cv::Mat & value_image)
{
  // value_image : 0.0f ~ 1.0f
  // {
  //   auto start = std::chrono::high_resolution_clock::now();
  //   auto end = std::chrono::high_resolution_clock::now();
  //   double out_time = static_cast<double>(
  //     std::chrono::duration_cast<std::chrono::nanoseconds>(
  //       end - start).count()) / 1e6;
  // }

  cv::Scalar mean, standard_deviation;
  cv::meanStdDev(value_image, mean, standard_deviation);
}
} // namespace visible
} // namespace jhj
