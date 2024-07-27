#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <chrono>
#include "systemCfgTool.h"
#include "objDetect.h"
#include <random>
#include <iostream>
#include <cstdlib> // For system()
#include "commonTools.h"

using namespace cv;
using namespace dnn;
using namespace std;

int main()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, 255);
    auto systemCfg = SystemCfgTool();
    objDetectYOLOV5 obj_hand(systemCfg.handModelPath, cv::Size(systemCfg.imageW, systemCfg.imageH), systemCfg.handModelConfTh, systemCfg.handModelClassNum, systemCfg.classNames_Hands);
    objDetectYOLOV5 obj_person(systemCfg.personModelPath, cv::Size(systemCfg.imageW, systemCfg.imageH), systemCfg.personModelConfTh, systemCfg.personModelClassNum, systemCfg.classNames_Person);
    objDetectYOLOV5 obj_face(systemCfg.faceModelPath, cv::Size(systemCfg.imageW, systemCfg.imageH), systemCfg.faceModelConfTh, systemCfg.faceModelClassNum, systemCfg.classNames_Face);

    std::string videoFilename = "../demo.MP4";
    cv::VideoCapture cap(videoFilename);
    if (!cap.isOpened())
    {
        std::cerr << "Error: Could not open video file." << std::endl;
        return 1;
    }

    // Get video properties
    int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    int totalFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

    std::string outputVideoFilenameNoAudio = "output_video_no_audio.mp4";
    cv::VideoWriter videoWriterNoAudio(outputVideoFilenameNoAudio, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(frameWidth, frameHeight));

    if (!videoWriterNoAudio.isOpened())
    {
        std::cerr << "Error: Could not create output video file." << std::endl;
        return 1;
    }

    // Read and write each frame
    cv::Mat frame;
    while (cap.read(frame))
    {
        auto start = std::chrono::high_resolution_clock::now();
        auto detRes_hand = obj_hand.detector(frame);
        auto detRes_person = obj_person.detector(frame);
        auto detRes_face = obj_face.detector(frame);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;

        //hand
        for (auto &item : detRes_hand)
        {
            cv::rectangle(frame, item.box, cv::Scalar(distrib(gen), distrib(gen), distrib(gen)), 2, cv::LINE_AA);
            cv::Point textOrg(item.box.x, item.box.y - 10); // Adjust this point for text position
            cv::putText(frame, item.name + " " + to_string_with_truncation(item.score, 2), textOrg, cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
        }

        //person
        for (auto &item : detRes_person)
        {
            cv::rectangle(frame, item.box, cv::Scalar(distrib(gen), distrib(gen), distrib(gen)), 2, cv::LINE_AA);
            cv::Point textOrg(item.box.x, item.box.y - 10); // Adjust this point for text position
            cv::putText(frame, item.name + " " + to_string_with_truncation(item.score, 2), textOrg, cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
        }

        //face
        for (auto &item : detRes_face)
        {
            cv::rectangle(frame, item.box, cv::Scalar(distrib(gen), distrib(gen), distrib(gen)), 2, cv::LINE_AA);
            cv::Point textOrg(item.box.x, item.box.y - 10); // Adjust this point for text position
            cv::putText(frame, item.name + " " + to_string_with_truncation(item.score, 2), textOrg, cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
        }

        cv::Point textOrg(10, 30); // Adjust this point for text position
        cv::putText(frame, "FPS " + std::to_string(1000 / duration.count()), textOrg, cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);

        videoWriterNoAudio.write(frame);
    }

    // Release resources
    cap.release();
    videoWriterNoAudio.release();
    cv::destroyAllWindows();

    // Use FFmpeg to add audio to the output video
    std::string outputVideoFilenameWithAudio = "output_video_with_audio.mp4";
    std::string ffmpegCmd = "ffmpeg -i " + outputVideoFilenameNoAudio + " -i " + videoFilename + " -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 -strict experimental " + outputVideoFilenameWithAudio;

    int ret = system(ffmpegCmd.c_str());
    if (ret != 0)
    {
        std::cerr << "Error executing FFmpeg command" << std::endl;
        return 1;
    }

    std::cout << "Output video with audio created successfully!" << std::endl;

    return 0;
}
