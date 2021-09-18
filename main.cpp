#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
//#include <highgui/highgui_c.h>
#include <iostream>
#include "yolov5_best.h"
using namespace cv;
using namespace cv::dnn;
using namespace std;


int main()
//{
//	cv::VideoCapture capture;
//	capture.open(0);  //修改这个参数可以选择打开想要用的摄像头
//
//	cv::Mat frame;
//	while (true)
//	{
//		capture >> frame;
//		cv::Mat m = frame;
//		double start = GetTickCount();
//		std::vector<Object> objects;
//		detect_yolov5(frame, objects);
//		double end = GetTickCount();
//		fprintf(stderr, "cost time:  %.5f ms \n", (end - start));
//
//		// imshow("外接摄像头", m);	//remember, imshow() needs a window name for its first parameter
//		draw_objects(m, objects);
//
//		if (cv::waitKey(30) >= 0)
//			break;
//	}

{
	int flag = 0;
	//cv::VideoCapture capture;
	//capture.open(0);  //修改这个参数可以选择打开想要用的摄像头
	//cv::Mat frame;
	string img_path = "test.png";
	string model_path = "best.onnx";
	YOLO test;
	Net net;
	Mat img = imread(img_path);
	net = readNetFromONNX(model_path);
	if (test.readModel(net, model_path, false)) {
		cout << "read net ok!" << endl;
	}
	else {
		cout << flag << endl;
		return -1;
	}
	//生成随机颜色
	vector<Scalar> color;
	srand(time(0));
	for (int i = 0; i < 80; i++) {
		int b = rand() % 256;
		int g = rand() % 256;
		int r = rand() % 256;
		color.push_back(Scalar(b, g, r));
	}
	vector<Output> result;
	//VideoCapture capture;
	//capture.open("yolo_test.mp4");
	//Mat frame;
	//capture >> frame;
	if (test.Detect(img, net, result)) {
		test.drawPred(img, result, color);
		flag = 1;

	}
	else {
		cout << "Detect Failed!" << endl;
		flag = 2;
	}
	system("pause");

	return 0;
}

