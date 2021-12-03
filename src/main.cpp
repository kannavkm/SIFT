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

    // Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    // std::vector< std::vector<DMatch> > knn_matches;
    // matcher->knnMatch( descriptors1, descriptors2, knn_matches, 2 );
    // //-- Filter matches using the Lowe's ratio test
    // const float ratio_thresh = 0.7f;
    // std::vector<DMatch> matches;
    // for (size_t i = 0; i < knn_matches.size(); i++)
    // {
    //     if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
    //     {
    //         matches.push_back(knn_matches[i][0]);
    //     }
    // }
    //-- Draw matches
    // Mat img_matches;
    // drawMatches( image1, keypoints1, image2, keypoints2, good_matches, img_matches, Scalar::all(-1),
    //              Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );



/*--------------------------------------------*/

    FlannBasedMatcher matcher;
    std::vector< DMatch > matches;
    matcher.match( descriptors1, descriptors2, matches );

    double max_dist = 0; double min_dist = 100;
 
    //-- Quick calculation of max and min distances between keypoints
    for( int i = 0; i < descriptors1.rows; i++ )
    { double dist = matches[i].distance;
    if( dist < min_dist ) min_dist = dist;
    if( dist > max_dist ) max_dist = dist;
    }
    
    printf("-- Max dist : %f \n", max_dist );
    printf("-- Min dist : %f \n", min_dist );
    
    //-- Use only "good" matches (i.e. whose distance is less than 3*min_dist )
    std::vector< DMatch > good_matches;
    
    for( int i = 0; i < descriptors1.rows; i++ )
    { if( matches[i].distance < 3*min_dist )
    { good_matches.push_back( matches[i]); }
    }
    std::vector< Point2f > obj;
    std::vector< Point2f > scene;
    
    for( int i = 0; i < good_matches.size(); i++ )
    {
    //-- Get the keypoints from the good matches
    obj.push_back( keypoints1[ good_matches[i].queryIdx ].pt );
    scene.push_back( keypoints2[ good_matches[i].trainIdx ].pt );
    }
    
    // Find the Homography Matrix
    Mat H = findHomography( obj, scene, RANSAC );
    // Use the Homography Matrix to warp the images
    cv::Mat result;
    warpPerspective(image1,result,H,cv::Size(image1.cols+image2.cols,image1.rows));
    cv::Mat half(result,cv::Rect(0,0,image2.cols,image2.rows));
    image2.copyTo(half);
    imshow( "Result", result );
    
    waitKey(0);
    return 0;


    // const Ptr<flann::IndexParams>& indexParams=new flann::KDTreeIndexParams(4);
    // const Ptr<flann::SearchParams>& searchParams=new flann::SearchParams(64);

    // FlannBasedMatcher matcher(indexParams, searchParams);
    // std::vector<std::vector<DMatch> > knn_matches;
    // matcher.knnMatch(descriptors1, descriptors2, knn_matches, 2);

    // const float ratio_thresh = 0.7f;
    // std::vector<DMatch> good_matches;
    // for (auto & knn_match : knn_matches) {
    //     if (knn_match[0].distance < ratio_thresh * knn_match[1].distance) {
    //         good_matches.push_back(knn_match[0]);
    //     }
    // }
    // //-- Draw matches
    // Mat img_matches;
    // drawMatches(image1, keypoints1, image2, keypoints2, good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
    //             std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    //    -- Show detected matches
    // imwrite("./Final" + name2 + "-" + name + ".png", img_matches);
    // waitKey(0);

    return 0;
}
