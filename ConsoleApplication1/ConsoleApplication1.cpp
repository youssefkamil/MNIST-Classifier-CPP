// ConsoleApplication1.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "MnistClassifier.h"

using namespace std;
using namespace cv;
int main()
{
	MnistClassifier M(LR"(model.onnx)");


	Mat image = imread("5.png");
	resize(image, image, Size(100, 100));
	imshow(" ", image);
	waitKey();


	cout<< endl<<endl<<"Predicted Number = "<<M.predict("5.png")<<endl;


	image = imread("3.jpg");
	resize(image, image, Size(100, 100));
	imshow(" ", image);
	waitKey();


	cout << endl << endl << "Predicted Number = " << M.predict("3.jpg") << endl;

	
	return 0;
}
