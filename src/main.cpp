#include <cstdio>
#include <opencv2/opencv.hpp>

#include "sift.hpp"

using namespace cv;

int main(int argc, char **argv) {
    if (argc != 2) {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return -1;
    }
    Mat image = imread(argv[1], cv::IMREAD_GRAYSCALE);
    if (!image.data) {
        printf("No image data \n");
        return -1;
    }
    // move the image to conserve memory no need to release it now
    sift::sift_handler ss(std::move(image));
    namedWindow("Display Image", WINDOW_AUTOSIZE);
    ss.exec();

    // waitKey(0);
    return 0;
}
