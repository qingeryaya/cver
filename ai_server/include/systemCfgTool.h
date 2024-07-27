#ifndef __SYSTEM_CFG_TOOL_H__
#define __SYSTEM_CFG_TOOL_H__
#include <iostream>
#include <opencv2/opencv.hpp>

class SystemCfgTool
{
private:
    /* data */
public:
    // 手
    std::string handModelPath;
    float handModelConfTh;
    int handModelClassNum;
    std::vector<std::string> classNames_Hands;

    // 人
    std::string personModelPath;
    float personModelConfTh;
    int personModelClassNum;
    std::vector<std::string> classNames_Person;

    // 脸
    std::string faceModelPath;
    float faceModelConfTh;
    int faceModelClassNum;
    std::vector<std::string> classNames_Face;

    int imageH;
    int imageW;

    SystemCfgTool();
};

#endif