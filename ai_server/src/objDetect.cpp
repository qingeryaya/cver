#include "objDetect.h"

objDetectYOLOV5::objDetectYOLOV5(std::string modelPath, cv::Size imgSize, float confTh, int classNum, std::vector<std::string> classNames_Hands)
{
        this->imgSize = imgSize;
        this->confTh = confTh;
        this->classNum = classNum;
        this->classNames = classNames_Hands;
        *net = cv::dnn::readNetFromONNX(modelPath);
        if ((*net).empty())
        {
                std::cerr << "Failed to load model" << std::endl;
                return;
        }
        auto availableBackends = cv::dnn::getAvailableBackends();
        for (auto &item : availableBackends)
        {
                if (item.first == cv::dnn::Backend::DNN_BACKEND_CUDA && item.second == cv::dnn::Target::DNN_TARGET_CUDA)
                {
                        std::cout << "cuda env is available!" << std::endl;
                        (*net).setPreferableBackend(item.first);
                        (*net).setPreferableTarget(item.second);
                }
        }
        outLayerNames = (*net).getUnconnectedOutLayersNames();
}

void objDetectYOLOV5::generateBlob(cv::Mat &image, float &rate, cv::Rect &originalImgRect)
{
        int orih = image.size[0];
        int oriw = image.size[1];

        int maxEdge = oriw > orih ? oriw : orih;
        rate = (float)(this->imgSize.height) / (float)maxEdge;
        cv::Mat resizedImg;
        if (oriw > orih)
                cv::resize(image, resizedImg, cv::Size(this->imgSize.width, (int)(this->imgSize.width * orih / ((float)oriw))));
        else
                cv::resize(image, resizedImg, cv::Size((int)(this->imgSize.height * oriw / ((float)orih)), this->imgSize.height));

        float wR = (float)maxEdge / (float)oriw;
        float hR = (float)maxEdge / (float)orih;
        cv::Mat blackImage(this->imgSize.height, this->imgSize.width, CV_8UC3, cv::Scalar(144, 144, 144)); // blackImage为加了padding后的图片
        cv::Rect roiRect(0, 0, resizedImg.size[1], resizedImg.size[0]);
        originalImgRect = roiRect;
        cv::Mat roiblackImage = blackImage(roiRect);
        resizedImg.copyTo(roiblackImage);
        cv::dnn::blobFromImage(blackImage, *(this->blob), 1 / 255.0, this->imgSize, cv::Scalar(0, 0, 0), true, false);
}

void objDetectYOLOV5::inference(std::vector<cv::Mat> &outputs)
{
        (*net).setInput(*blob);
        (*net).forward(outputs, outLayerNames);
}

std::vector<targetAttributes> objDetectYOLOV5::postProcess(std::vector<cv::Mat> &outputs, cv::Rect &originalImgRect, float &rate)
{
        std::vector<int> classIds;
        std::vector<float> confidences;
        std::vector<float> scores;
        std::vector<cv::Rect> boxes;
        float xFactor = rate;
        float yFactor = rate;

        // 存储模型预测结果
        float *data = (float *)outputs[0].data;
        int bigThanThNum = 0;
        for (int i = 0; i < outputs[0].size[1]; i++)
        {
                // std::cout << "判断第" << i << "次" << std::endl;
                float confidence = 0.f;
                confidence = data[4];

                if (confidence > confTh)
                {
                        std::vector<float> classScores;
                        cv::Rect box;
                        float x = data[0];
                        float y = data[1];
                        float w = data[2];
                        float h = data[3];
                        int left = int((x - 0.5 * w) / xFactor);
                        int top = int((y - 0.5 * h) / yFactor);
                        int width = int(w / xFactor);
                        int height = int(h / yFactor);
                        boxes.push_back(cv::Rect(left, top, width, height));
                        classScores.assign(data + 5, data + 5 + this->classNum);

                        auto maxElementIter = std::max_element(classScores.begin(), classScores.end());
                        scores.push_back(*maxElementIter);
                        // 计算最大元素的下标
                        int maxIndex = std::distance(classScores.begin(), maxElementIter);
                        classIds.push_back(maxIndex);
                        confidences.push_back(confidence);
                        bigThanThNum++;
                }
                data += 5 + this->classNum;
        }
        std::cout << "大于检测框阈值的检测框个数：" << bigThanThNum << std::endl;
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, this->confTh, this->NMSTh, indices);
        std::vector<targetAttributes> res;
        for (auto &item : indices)
        {
                targetAttributes TAtt;
                TAtt.box = boxes[item];
                TAtt.score = scores[item];
                TAtt.idx = classIds[item];
                TAtt.name = this->classNames[TAtt.idx];
                res.push_back(TAtt);
        }
        return res;
}

std::vector<targetAttributes> objDetectYOLOV5::detector(cv::Mat &image)
{
        std::vector<cv::Mat> outputs;
        float rate;
        cv::Rect originalImgRect;

        this->generateBlob(image, rate, originalImgRect);
        std::cout << "blob 生成完毕" << std::endl;

        this->inference(outputs);
        std::cout << "推理完毕" << std::endl;

        auto res = this->postProcess(outputs, originalImgRect, rate);

        std::cout << "后处理完毕" << std::endl;
        return res;
}
