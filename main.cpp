#include <iostream>
#include <memory>
#include <vector>

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include "image_processing/filter/color/gamma_corrector.hpp"


int main(int argc, char* argv[]) {
  cudaStream_t stream;
	std::vector<cv::Mat> input_images;
	std::vector<cv::Mat> output_images;
	cv::Size target_size(500, 500);
  int target_image_count = argc - 1;
	for (int i = 1; i <= target_image_count; i++) {
  	cv::Mat input_image = cv::imread(argv[i]);
		cv::resize(input_image, input_image, target_size, 0, 0, cv::INTER_NEAREST);
		input_images.push_back(input_image.clone());
		output_images.push_back(input_image.clone());
	}
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
	std::cout << "size : " << target_size << std::endl;

	std::shared_ptr<jhj::visible::filter::color::GammaCorrector> corrector =
	  std::make_shared<jhj::visible::filter::color::GammaCorrector>(stream, target_size);

	cv::Mat concat_input_image = input_images[0].clone();
	for (int i = 1; i < target_image_count; i++) {
		cv::hconcat(concat_input_image, input_images[i], concat_input_image);
	}
	cv::imshow("input image", concat_input_image);
	std::vector<float> means(target_image_count, 0.0f);
	std::vector<float> standard_deviations(target_image_count, 0.0f);
  for (int i = 0; i < target_image_count; i++) {
		corrector->analyze_color_image(input_images[i], means[i], standard_deviations[i]);
	}
	for (int i = 0; i < target_image_count; i++) {
		corrector->apply_filter(
			stream,
			output_images[i],
			input_images[i],
			means[i],
			standard_deviations[i]);
	}
	cudaStreamSynchronize(stream);
	cv::Mat concat_output_image = output_images[0].clone();
	for (int i = 1; i < target_image_count; i++) {
		cv::hconcat(concat_output_image, output_images[i], concat_output_image);
	}
	cv::imshow("output image", concat_output_image);
	cv::waitKey(0);
  cudaStreamDestroy(stream);
  return 0;
}
