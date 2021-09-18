#include "yolov5_best.h"
#include<iostream>
#include<time.h>
using namespace cv;
using namespace std;
bool YOLO::readModel(Net& net, string& netPath, bool isCuda)
{
	try {
		net = readNetFromONNX(netPath);
	}
	catch (const std::exception&) {
		return false;
	}
	//cuda
	/*
	if (isCuda) {
		net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
		net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
	}
	//cpu*/

	net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
	net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

	return true;
}

bool YOLO::Detect(cv::Mat& SrcImg, cv::dnn::Net& net, std::vector<Output>& output)
{
	Mat blob;
	blobFromImage(SrcImg, blob, 1 / 255.0, cv::Size(netWidth, netHeight), Scalar(0, 0, 0), true, false);
	net.setInput(blob);
	vector<Mat> netOutputImg;
	net.forward(netOutputImg, net.getUnconnectedOutLayersNames());
	vector<int> classIds;//结果id数组
	vector<float> confidences;//结果每个id对应置信度数组
	vector<Rect> boxes;//每个id矩形框
	float ratio_h = (float)SrcImg.rows / netHeight;
	float ratio_w = (float)SrcImg.cols / netWidth;
	int net_width = className.size() + 5;  //输出的网络宽度是类别数+5
	float* pdata = (float*)netOutputImg[0].data;
	for (int stride = 0; stride < 3; stride++) {    //stride
		int grid_x = (int)(netWidth / netStride[stride]);
		int grid_y = (int)(netHeight / netStride[stride]);
		for (int anchor = 0; anchor < 3; anchor++) { //anchors
			const float anchor_w = netAnchors[stride][anchor * 2];
			const float anchor_h = netAnchors[stride][anchor * 2 + 1];
			for (int i = 0; i < grid_y; i++) {
				for (int j = 0; j < grid_y; j++) {
					float box_score = Sigmoid(pdata[4]);//获取每一行的box框中含有某个物体的概率
					if (box_score > boxThreshold) {
						cv::Mat scores(1, className.size(), CV_32FC1, pdata + 5);
						Point classIdPoint;
						double max_class_socre;
						minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
						max_class_socre = Sigmoid((float)max_class_socre);
						if (max_class_socre > boxThreshold) {
							//rect [x,y,w,h]
							float x = (Sigmoid(pdata[0]) * 2.f - 0.5f + j) * netStride[stride];  //x
							float y = (Sigmoid(pdata[1]) * 2.f - 0.5f + i) * netStride[stride];   //y
							float w = powf(Sigmoid(pdata[2]) * 2.f, 2.f) * anchor_w;   //w
							float h = powf(Sigmoid(pdata[3]) * 2.f, 2.f) * anchor_h;  //h
							int left = (x - 0.5 * w) * ratio_w;
							int top = (y - 0.5 * h) * ratio_h;
							classIds.push_back(classIdPoint.x);
							confidences.push_back(max_class_socre);
							boxes.push_back(Rect(left, top, int(w * ratio_w), int(h * ratio_h)));
						}
					}
					pdata += net_width;//指针移到下一行
				}
			}
		}
	}
	vector<int> nms_result;
	NMSBoxes(boxes, confidences, classThreshold, nmsThreshold, nms_result);
	for (int i = 0; i < nms_result.size(); i++) {
		int idx = nms_result[i];
		Output result;
		result.id = classIds[idx];
		result.confidence = confidences[idx];
		result.box = boxes[idx];
		output.push_back(result);
	}
	if (output.size())
		return true;
	else
		return false;
}

void YOLO::drawPred(Mat& img, vector<Output> result, vector<Scalar> color) {
	for (int i = 0; i < result.size(); i++)
	{
		int left, top;
		left = result[i].box.x;
		top = result[i].box.y;
		int color_num = i;
		rectangle(img, result[i].box, color[result[i].id], 2, 8);
		string label = className[result[i].id] + ":" + to_string(result[i].confidence);
		int baseLine;
		Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
		top = max(top, labelSize.height);
		putText(img, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 1, color[result[i].id], 1);
	}
	cv::resize(img, img, cv::Size(1400, 800));
	imshow("test", img);
	waitKey(1);
}








