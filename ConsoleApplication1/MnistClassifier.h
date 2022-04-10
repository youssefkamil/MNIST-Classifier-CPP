#pragma once
#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;


class MnistClassifier
{
private:
	int width, height;
	Ort::Env env;
	Ort::Session session{ nullptr };
	Ort::RunOptions runOptions;
	
	std::array<float, 10> results_{};
	int result_{ 0 };
	static constexpr const int width_ = 28;
	static constexpr const int height_ = 28;

	std::array<float, width_ * height_> input_image_{};
	Ort::Value input_tensor_{ nullptr };
	std::array<int64_t, 4> input_shape_{ 1, 1, width_, height_ };

	Ort::Value output_tensor_{ nullptr };
	std::array<int64_t, 2> output_shape_{ 1, 10 };

public:
	MnistClassifier(const wchar_t*);
	vector<float> Preprocess(Mat);
	int64 Postprocess(std::array<float,10>);
	int64 predict(string);
	template <typename T>
	static T softmax(T input);

};

