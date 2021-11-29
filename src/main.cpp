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
    std::string str(argv[1]);
    std::size_t found = str.find_last_of("/");
    auto file = str.substr(found + 1);
    found = file.find_last_of(".");
    auto name = file.substr(0, found);
    sift::sift_handler ss(name, std::move(image));
    // namedWindow("Display Image", WINDOW_AUTOSIZE);
    ss.exec();

    // waitKey(0);
    return 0;
}
