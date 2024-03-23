#include <chrono>

#include "image_processing/filter/color/gamma_corrector.hpp"


namespace jhj
{
namespace visible
{
filter::color::GammaCorrector::GammaCorrector(cudaStream_t stream, cv::Size2i image_size)
: image_size_(image_size)
{
  cudaMallocAsync(&corrected_image_, image_size_.area() * sizeof(uchar3), stream);
  cudaStreamSynchronize(stream);
}

filter::color::GammaCorrector::~GammaCorrector()
{
  cudaFree(corrected_image_);
}

void filter::color::GammaCorrector::analyze_color_image(
  cv::Mat & input_image, float & mean, float & standard_deviation)
{
  cv::Mat hsv_image;
  cv::Mat value_image;
  std::vector<cv::Mat> channel_images(3);
  cv::cvtColor(input_image, hsv_image, cv::COLOR_BGR2HSV);
  cv::split(hsv_image, channel_images);
  channel_images[2].convertTo(value_image, CV_32FC1, 1.0f / 255.0f);

  // value_image : 0.0f ~ 1.0f
  cv::Scalar value_mean, value_standard_deviation;
  cv::meanStdDev(value_image, value_mean, value_standard_deviation);

  mean = value_mean(0);
  standard_deviation = value_standard_deviation(0);
}

void filter::color::GammaCorrector::apply_filter(
  cudaStream_t stream,
  cv::Mat & output_image,
  const cv::Mat & input_image,
  const float & mean,
  const float & standard_deviation)
{
  const bool is_low_contrast = 4.0f * standard_deviation <= 1.0f / THRESHOLD;
  float gamma;
  if (is_low_contrast) {
    gamma = -1.0f * std::log2f(standard_deviation);
  } else {
    gamma = std::exp(0.5f * (1 - (mean + standard_deviation)));
  }
  const bool is_bright_image = mean > 0.5f;

  correct_color_gamma(stream, output_image, input_image, gamma, mean, is_bright_image);
}

void filter::color::GammaCorrector::correct_color_gamma(
  cudaStream_t stream,
  cv::Mat & output_image,
  const cv::Mat & input_image,
  const float & gamma,
  const float & mean,
  const bool & is_bright_image)
{
  cudaMemcpyAsync(
    corrected_image_,
    input_image.data,
    image_size_.area() * sizeof(uchar3),
    cudaMemcpyHostToDevice,
    stream);

  host::apply_adaptive_gamma_correction(
    corrected_image_,
    gamma,
    mean,
    is_bright_image,
    image_size_.height,
    image_size_.width,
    stream);

  cudaMemcpyAsync(
    output_image.data,
    corrected_image_,
    image_size_.area() * sizeof(uchar3),
    cudaMemcpyDeviceToHost,
    stream);
}
} // namespace visible
} // namespace jhj
