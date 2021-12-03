#include "sift.hpp"

#include <cstdio>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

string getFileName(const string& str) {
    std::size_t found = str.find_last_of('/');
    auto file = str.substr(found + 1);
    found = file.find_last_of('.');
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

    Mat image2 = imread(argv[2], cv::IMREAD_GRAYSCALE);
    if (!image2.data) {
        printf("No image data \n");
        return -1;
    }

    // move the image to conserve memory no need to release it now
    std::string name = getFileName(argv[1]);
    std::string name2 = getFileName(argv[2]);

    auto cp_img1 = image1.clone();
    auto cp_img2 = image2.clone();

    sift::sift_handler s1(name, std::move(cp_img1));
    // namedWindow("Display Image", WINDOW_AUTOSIZE);
    s1.exec();
    auto keypoints1 = s1.keypoints;
    auto descriptors = s1.descriptors;
    cv::Mat descriptors1 = s1.get();

    sift::sift_handler s2(name2, std::move(cp_img2));
    // namedWindow("Display Image", WINDOW_AUTOSIZE);
    s2.exec();
    auto keypoints2 = s2.keypoints;
    cv::Mat descriptors2 = s2.get();
    const Ptr<flann::IndexParams>& indexParams=new flann::KDTreeIndexParams(4);
    const Ptr<flann::SearchParams>& searchParams=new flann::SearchParams(64);

    FlannBasedMatcher matcher(indexParams, searchParams);
    std::vector<std::vector<DMatch> > knn_matches;
    matcher.knnMatch(descriptors1, descriptors2, knn_matches, 2);

    const float ratio_thresh = 0.7f;
    std::vector<DMatch> good_matches;
    for (auto & knn_match : knn_matches) {
        if (knn_match[0].distance < ratio_thresh * knn_match[1].distance) {
            good_matches.push_back(knn_match[0]);
        }
    }
    //-- Draw matches
    Mat img_matches;
    drawMatches(image1, keypoints1, image2, keypoints2, good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    //    -- Show detected matches
    imshow("Good Matches", img_matches);
    waitKey(0);

    return 0;
}
