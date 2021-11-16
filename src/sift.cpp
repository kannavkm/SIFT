//
// Created by exclowd on 16/11/21.
//

#include "sift.hpp"
#include <chrono>

#define TIMEIT(f) {\
    auto start = std::chrono::steady_clock::now(); \
    f(); \
    auto end = std::chrono::steady_clock::now(); \
    std::chrono::duration<double> duration = end - start; \
    std::cout << "elapsed time: " << #f << ": " << duration.count() << "s\n";}

namespace sift {

    sift_handler::sift_handler(cv::Mat &&_base) {
        cv::Mat temp, interpolated, blurred_image;
        _base.convertTo(temp, CV_32F);
        _base.release();
        // compute the number of octaves
        cv::Size sz = temp.size();
        octaves = (size_t) std::round(std::log2((double) std::min(sz.width, sz.height))) - 1;
        // interpolate and blur base image
        cv::resize(temp, interpolated, sz * 2, 0, 0, cv::INTER_LINEAR);
        double diff = std::max(std::sqrt(pow(SIGMA, 2) - 4 * pow(assumed_blur, 2)), 0.1);
        cv::GaussianBlur(interpolated, blurred_image, cv::Size(0, 0), diff, diff);
        base = blurred_image;
    }

    void sift_handler::exec() {
        TIMEIT(gen_gaussian_images);
        TIMEIT(gen_dog_images);
        TIMEIT(gen_scale_space_extrema);
        for (auto &octave: images) {
            for (auto &img: octave) {
                cv::imshow("Display Image", img);
                cv::waitKey(0);
            }
        }
    }

    cv::Mat sift_handler::get() const {
        cv::Mat image;
        base.convertTo(image, CV_8U);
        return image;
    }

    sift_handler::~sift_handler() {
        base.release();
        for (auto &octave: images) {
            octave.clear();
        }
        images.clear();
    }

    cv::Mat sift_handler::getImg(const cv::Mat &mat) {
        cv::Mat image;
        mat.convertTo(image, CV_8U);
        return image;
    }

    void sift_handler::gen_gaussian_images() {
        // first generate all gaussian kernels
        double k = std::pow(2, 1.0 / SCALES);
        std::array<double, IMAGES> kernel{};
        double prev = SIGMA;
        for (int i = 1; i < (int) IMAGES; i++) {
            double now = prev * k;
            kernel[i] = std::sqrt(std::pow(now, 2) - std::pow(prev, 2));
            prev = now;
        }
        // Now do gaussian blurring
        cv::Mat temp = base.clone();
        images.reserve(octaves);
        for (int i = 0; i < octaves; i++) {
            std::vector<cv::Mat> octave_images(IMAGES);
            // the base image for each octave is just interpolated base image
            octave_images[0] = temp;
            for (int j = 1; j < (int) IMAGES; j++) {
                cv::GaussianBlur(octave_images[j - 1], octave_images[j], cv::Size(), kernel[j], kernel[j]);
            }
            size_t baseid = octave_images.size() - 3;
            cv::resize(octave_images[baseid], temp, cv::Size(), 0.5, 0.5, cv::INTER_NEAREST);
            images.push_back(std::move(octave_images));
        }
    }

    void sift_handler::gen_dog_images() {
        // dog would result vector of size IMAGES - 1
        for (auto &octave: images) {
            std::vector<cv::Mat> dog_images(IMAGES - 1);
            for (int j = 1; j < (int) IMAGES; j++) {
                dog_images[j - 1] = octave[j] - octave[j - 1];
            }
            octave.clear();
            octave = std::move(dog_images);
        }
    }

    void sift_handler::gen_scale_space_extrema() {
        int cnt = false;
        for (int oct = 0; oct < octaves; oct++) {
            for (int img = 1; img < IMAGES - 2; img++) {
                cv::Mat first_image = images[oct][img - 1];
                cv::Mat second_image = images[oct][img];
                cv::Mat third_image = images[oct][img + 1];
                cv::Size size = second_image.size();
                for (int i = BORDER; i < (int) (size.height - BORDER); i++) {
                    for (int j = BORDER; j < (int) (size.width - BORDER); j++) {
                        cv::Rect r(i - 1, j - 1, 3, 3);
                        std::vector<cv::Mat> pixel_cube{first_image(r), second_image(r), third_image(r)};
                        bool ans = is_pixel_extremum(pixel_cube);
                        cnt += (int) ans;
                    }
                }
            }
        }
        std::cout << cnt << std::endl;
    }

    // TODO do this on the gpu
    bool sift_handler::is_pixel_extremum(const std::vector<cv::Mat> &pixel_cube) {
        bool is_maximum = true, is_minimum = true;
        for (int k = 0; k < 3; k++) {
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    is_maximum &= pixel_cube[1].at<float>(1, 1) >= pixel_cube[k].at<float>(i, j);
                    is_minimum &= pixel_cube[1].at<float>(1, 1) <= pixel_cube[k].at<float>(i, j);
                }
            }
        }
        return is_minimum || is_maximum;
    }

}