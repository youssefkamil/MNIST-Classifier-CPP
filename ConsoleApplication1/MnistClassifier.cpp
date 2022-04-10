#include "MnistClassifier.h"

MnistClassifier::MnistClassifier(const wchar_t* ModelPath) {


	session={ env,ModelPath, Ort::SessionOptions{} };
	
};

vector<float> MnistClassifier::Preprocess(Mat image) {
	resize(image, image, Size(28, 28));
	cvtColor(image, image, cv::COLOR_BGR2GRAY);
	image = image.reshape(1, 1);
	vector<float> vec;
	image.convertTo(vec, CV_32FC1, 1. / 255.0);

	return image;
}

int64 MnistClassifier::Postprocess(std::array<float, 10> results_)
{
	
	array<float, 10> ResSoft;
	ResSoft=softmax(results_);
	int64 res = 0;
	res = std::distance(ResSoft.begin(), std::max_element(ResSoft.begin(), ResSoft.end()));
	
	
	return res;
}

int64 MnistClassifier::predict(string imagePath) {

	
	
	Mat image = imread(imagePath);
	vector<float> PreprocessedImg;
	PreprocessedImg=Preprocess(image);


	//get input shape
	auto inputshape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
	//std::cout << "input shape : " << inputshape[0] << ", " << inputshape[1] << ", " << inputshape[2] << ", " << inputshape[3] << endl;

	//get output shape
	auto outputshape = session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
	//std::cout << "output shape : " << outputshape[0] << ", " << outputshape[1] << endl;


	auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	std::array<float, width_ * height_> inputImage{};
	
	input_tensor_ = Ort::Value::CreateTensor<float>(memoryInfo, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());
	output_tensor_ = Ort::Value::CreateTensor<float>(memoryInfo, results_.data(), results_.size(), output_shape_.data(), output_shape_.size());

	input_tensor_ = Ort::Value::CreateTensor<float>(memoryInfo, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());
	output_tensor_ = Ort::Value::CreateTensor<float>(memoryInfo, results_.data(), results_.size(), output_shape_.data(), output_shape_.size());
	Ort::AllocatorWithDefaultOptions allocator;
	// get inputs and outputs names
	//const char* input_names[] = { "Input3" };
	//const char* output_names[] = { "Plus214_Output_0" };

	const char* inputname[] = {session.GetInputName(0, allocator)};
	const char* outputname[] = { session.GetOutputName(0, allocator) };


	copy(PreprocessedImg.begin(), PreprocessedImg.end(), input_image_.begin());

	
	session.Run(Ort::RunOptions{ nullptr }, inputname, &input_tensor_, 1, outputname, &output_tensor_, 1);
	// confidence https://github.com/leimao/ONNX-Runtime-Inference/blob/main/src/inference.cpp
	
	int64 result;

	result=Postprocess(results_);

	return result;
}

template <typename T>
static T MnistClassifier::softmax(T input) {
	float rowmax = *std::max_element(input.begin(), input.end());
	std::vector<float> y(input.size());
	float sum = 0.0f;
	for (size_t i = 0; i != input.size(); ++i) {
		sum += y[i] = std::exp(input[i] - rowmax);
	}
	for (size_t i = 0; i != input.size(); ++i) {
		input[i] = y[i] / sum;
	}
	return input;
}
