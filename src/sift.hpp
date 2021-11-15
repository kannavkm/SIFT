#ifndef SIFT_H
#define SIFT_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <vector>

namespace sift
{

    class sift_handler
    {
    public:
        sift_handler(cv::Mat &&_base, int32_t _scales);
        cv::Mat get();
        ~sift_handler();
    private:
    private:
        std::vector<cv::Mat> img;

    public:
        cv::Mat base;
        int32_t scales;
        int32_t octaves;
    };

}

#endif