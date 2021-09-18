#pragma once
#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include<time.h>
using namespace cv;
using namespace dnn;
using namespace std;
struct Output {
	int id;//结果类别id
	float confidence;//结果置信度
	cv::Rect box;//矩形框
};

class YOLO
{
public:
	YOLO() {};
	bool readModel(Net& net, string& netPath, bool isCuda = false);
	bool Detect(cv::Mat& SrcImg, cv::dnn::Net& net, std::vector<Output>& output);
	void drawPred(cv::Mat& img, std::vector<Output> result, std::vector<cv::Scalar> color);
private:
	//计算归一化函数
	float Sigmoid(float x) {
		return static_cast<float>(1.f / (1.f + exp(-x)));
	}
	const float netAnchors[3][6] = { { 19.0,24.0, 25.0,34.0, 33.0,46.0 },{ 42.0,61.0, 53.0,86.0, 71.0,119.0 },{ 113.0,95.0, 99.0,168.0, 163.0,319.0 } };
	const float netStride[3] = { 8.0, 16.0, 32.0 };  //根据你的anchors修改
	const int netWidth = 640;
	const int netHeight = 640;
	float nmsThreshold = 0.2;
	float boxThreshold = 0.5;
	float classThreshold = 0.5;
	std::vector<std::string> className = { "no_mask", "mask" };
};

