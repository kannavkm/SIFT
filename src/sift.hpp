#ifndef SIFT_H
#define SIFT_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>

namespace sift {

#include <vector>

class sift_handler {
    int32_t octaves;
    int32_t scales;
    std::vector<cv::Mat> img;
};

}

#endif