#include <cstdio>
#include <opencv2/opencv.hpp>

#include "opencv2/xfeatures2d.hpp"
#include "sift.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

string getFileName(string str) {
    std::size_t found = str.find_last_of("/");
    auto file = str.substr(found + 1);
    found = file.find_last_of(".");
    auto name = file.substr(0, found);
    return name;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        printf("usage: DisplayImage.out <Image_Path> <Image_Path>\n");
        return -1;
    }
    Mat image1 = imread(argv[1], cv::IMREAD_GRAYSCALE);
    if (!image1.data) {
        printf("No image data \n");
        return -1;
    }
    // move the image to conserve memory no need to release it now
    std::string name = getFileName(argv[1]);
    std::string name2 = getFileName(argv[2]);

    sift::sift_handler s1(name, std::move(image1));
    // namedWindow("Display Image", WINDOW_AUTOSIZE);
    s1.exec();
    auto keypoints1 = s1.keypoints;
    cv::Mat descriptors1 = s1.get();

    Mat image2 = imread(argv[2], cv::IMREAD_GRAYSCALE);
    if (!image2.data) {
        printf("No image data \n");
        return -1;
    }

    sift::sift_handler s2(name2, std::move(image2));
    // namedWindow("Display Image", WINDOW_AUTOSIZE);
    s2.exec();

    auto keypoints2 = s2.keypoints;
    cv::Mat descriptors2 = s2.get();

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    std::vector<std::vector<DMatch> > knn_matches;
    matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);

    const float ratio_thresh = 0.7f;
    std::vector<DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++) {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance) {
            good_matches.push_back(knn_matches[i][0]);
        }
    }
    //-- Draw matches
    Mat img_matches;
    drawMatches(image1, keypoints1, image2, keypoints2, good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    //-- Show detected matches
    imshow("Good Matches", img_matches);
    waitKey();

    // waitKey(0);
    return 0;
}
