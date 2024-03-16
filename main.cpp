#include <iostream>
#include <memory>

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>

#include "image_processing/filter/color/gamma_corrector.hpp"


int main(int argc, char* argv[]) {
  cv::VideoCapture cap(0);
  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (!cap.isOpened())
	{
		std::cout << "Can't open the camera" << std::endl;
		return -1;
	}
  cap.set(cv::CAP_PROP_FRAME_HEIGHT, 240);
  cap.set(cv::CAP_PROP_FRAME_WIDTH, 320);
  cap.set(cv::CAP_PROP_FPS, 30);

	std::shared_ptr<jhj::visible::filter::color::GammaCorrector> corrector =
	  std::make_shared<jhj::visible::filter::color::GammaCorrector>(stream, cv::Size2i(320, 240));

  cv::Mat input_image;

	while (true) {
		cap >> input_image;
		cv::imshow("camera input image", input_image);
		cv::Mat color_output_image = corrector->apply_filter(stream, input_image);
		cv::imshow("color output image", color_output_image);
		cv::Mat gray_image;
		cv::cvtColor(input_image, gray_image, cv::COLOR_BGR2GRAY);
		cv::Mat gray_output_image = corrector->apply_filter(stream, gray_image);
		cv::imshow("gray output image", gray_output_image);
		if (cv::waitKey(1) == 27)
			break;
	}
  cap.release();
  cudaStreamDestroy(stream);
  return 0;
}
