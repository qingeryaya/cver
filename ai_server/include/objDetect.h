#ifndef __OBJ_DETECT_H__
#define __OBJ_DETECT_H__
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <cstdlib> // For system() function

struct targetAttributes
{
        cv::Rect box;
        float score;
        int idx;
        std::string name;
};

class objDetectYOLOV5
{
public:
        std::shared_ptr<cv::dnn::Net> net = std::make_shared<cv::dnn::Net>();
        std::shared_ptr<cv::Mat> blob = std::make_shared<cv::Mat>();
        cv::Size imgSize;
        float confTh;
        float NMSTh;
        int classNum;
        std::vector<std::string> classNames;
        std::vector<cv::String> outLayerNames;

public:
        objDetectYOLOV5(std::string modelPath, cv::Size imgSize, float confTh, int classNum, std::vector<std::string> classNames);
        void generateBlob(cv::Mat &image, float &rate, cv::Rect &originalImgRect);
        void inference(std::vector<cv::Mat> &outputs);
        std::vector<targetAttributes> postProcess(std::vector<cv::Mat> &outputs,cv::Rect &originalImgRect, float &rate);

        std::vector<targetAttributes> detector(cv::Mat &image);
};

#endif