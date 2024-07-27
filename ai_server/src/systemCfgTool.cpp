#include "systemCfgTool.h"

SystemCfgTool::SystemCfgTool()
{

        cv::FileStorage fs("/home/demo/Desktop/aiLearning/cver/yolov5ForBudDetection/ai_server/cfg/system.yaml", cv::FileStorage::READ);
        if (!fs.isOpened())
        {
                std::cerr << "Failed to open file." << std::endl;
                return;
        }
        fs["imageH"] >> imageH;
        fs["imageW"] >> imageW;

        // hand
        fs["handModelPath"] >> handModelPath;
        fs["handModelConfTh"] >> handModelConfTh;
        fs["handModelClassNum"] >> handModelClassNum;
        cv::FileNode classNamesNode_Hands = fs["classNames_Hands"];
        if (classNamesNode_Hands.type() != cv::FileNode::SEQ)
        {
                std::cerr << "Error: classNames_Hands is not a sequence." << std::endl;
                fs.release();
                return;
        }
        for (cv::FileNodeIterator it = classNamesNode_Hands.begin(); it != classNamesNode_Hands.end(); ++it)
        {
                std::string className = (std::string)*it;
                classNames_Hands.push_back(className);
        }

        // person
        fs["personModelPath"] >> personModelPath;
        fs["personModelConfTh"] >> personModelConfTh;
        fs["personModelClassNum"] >> personModelClassNum;
        cv::FileNode classNamesNode_Person = fs["classNames_Person"];
        if (classNamesNode_Person.type() != cv::FileNode::SEQ)
        {
                std::cerr << "Error: classNames_Person is not a sequence." << std::endl;
                fs.release();
                return;
        }
        for (cv::FileNodeIterator it = classNamesNode_Person.begin(); it != classNamesNode_Person.end(); ++it)
        {
                std::string className = (std::string)*it;
                classNames_Person.push_back(className);
        }

        // face
        fs["faceModelPath"] >> faceModelPath;
        fs["faceModelConfTh"] >> faceModelConfTh;
        fs["faceModelClassNum"] >> faceModelClassNum;
        cv::FileNode classNamesNode_Face = fs["classNames_Face"];
        if (classNamesNode_Face.type() != cv::FileNode::SEQ)
        {
                std::cerr << "Error: classNames_Face is not a sequence." << std::endl;
                fs.release();
                return;
        }
        for (cv::FileNodeIterator it = classNamesNode_Face.begin(); it != classNamesNode_Face.end(); ++it)
        {
                std::string className = (std::string)*it;
                classNames_Face.push_back(className);
        }

        std::cout << "参数加载完毕！" << std::endl;
}