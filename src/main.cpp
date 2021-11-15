#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "sift.hpp"

using namespace cv;

int main(int argc, char** argv ) {
    if ( argc != 2 ) {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return -1;
    }
    Mat image;
    image = imread( argv[1], 1 );
    if ( !image.data ) {
        printf("No image data \n");
        return -1;
    }
    // move the image to conserve memory
    sift::sift_handler ss(std::move(image), 3);
    namedWindow("Display Image", WINDOW_AUTOSIZE );
    imshow("Display Image", ss.base);
    waitKey(0);
    return 0;
}
    