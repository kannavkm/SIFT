#include "sift.hpp"
#include <iostream>

namespace sift
{
    // converts 8bit image to 32b float for maths and all.
    sift_handler::sift_handler(cv::Mat &&_base, int32_t _scales) : scales(_scales)
    {
        _base.convertTo(base, CV_32F);
        _base.release();
    }

    // returns the base. converted back to 8 bit
    cv::Mat sift_handler::get()
    {
        cv::Mat image;
        base.convertTo(image, CV_8U);
        return image;
    }

    // destructor releases all the memory
    sift_handler::~sift_handler()
    {
        base.release();
    }

}
