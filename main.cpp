#include <iostream>
#include <memory>

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include "image_processing/filter/color/gamma_corrector.hpp"


int main(int argc, char* argv[]) {
  cudaStream_t stream;
  cv::Mat input_image = cv::imread(argv[1]);
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
	std::cout << "size : " << input_image.size() << std::endl;

	std::shared_ptr<jhj::visible::filter::color::GammaCorrector> corrector =
	  std::make_shared<jhj::visible::filter::color::GammaCorrector>(stream, input_image.size());

	while (true) {
		cv::imshow("input image", input_image);
		cv::Mat output_image = input_image.clone();
		float mean, standard_deviation;
		corrector->analyze_color_image(input_image, mean, standard_deviation);
		corrector->apply_filter(stream, output_image, input_image, mean, standard_deviation);
		cudaStreamSynchronize(stream);
		cv::imshow("color output image", output_image);
		if (cv::waitKey(1) == 27)
			break;
	}
  cudaStreamDestroy(stream);
  return 0;
}
