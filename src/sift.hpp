#ifndef SIFT_H
#define SIFT_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <vector>

namespace sift {
    // lets us declare array of size SCALES

    template<size_t SCALES>
    class sift_handler {
    public:
        explicit sift_handler(cv::Mat &&_base) {
            cv::Mat temp, interpolated, blurred_image;
            _base.convertTo(temp, CV_32F);
            _base.release();

            // compute the number of octaves
            cv::Size sz = temp.size();
            octaves = (size_t) std::round(std::log2((double) std::min(sz.width, sz.height))) - 1;

            // interpolate and blur base image
            cv::resize(temp, interpolated, sz * 2, 0, 0, cv::INTER_LINEAR);
            double sigmadiff = std::max(std::sqrt(pow(sigma, 2) - 4 * pow(assumed_blur, 2)), 0.1);
            cv::GaussianBlur(interpolated, blurred_image, cv::Size(0, 0), sigmadiff, sigmadiff);

            // base image
            base = blurred_image;
        }

        void exec() {
            auto start = std::chrono::steady_clock::now();
            gen_gaussian_images();
            auto after_gauss = std::chrono::steady_clock::now();
            gen_dog_images();
            auto end = std::chrono::steady_clock::now();
            std::chrono::duration<double> for_gauss = after_gauss - start;
            std::chrono::duration<double> for_dog = end - after_gauss;
            std::cout << "elapsed time: gen gass" << for_gauss.count() << "s\n";
            std::cout << "elapsed time: " << for_dog.count() << "s\n";
            for (auto &octave: images) {
                for (auto &img: octave) {
                    cv::imshow("Display Image", img);
                    cv::waitKey(0);
                }
            }
        }

        cv::Mat get() {
            cv::Mat image;
            base.convertTo(image, CV_8U);
            return image;
        }

        ~sift_handler() {
            base.release();
            for (auto &octave: images) {
                octave.clear();
            }
            images.clear();
        }

    private:

        cv::Mat getImg(const cv::Mat &mat) {
            cv::Mat image;
            mat.convertTo(image, CV_8U);
            return image;
        }

        void gen_gaussian_images() {
            // first generate all gaussian kernels and do gaussian
            double k = std::pow(2, 1.0 / SCALES);
            std::array<double, IMAGES> kernel; 
            double prev = sigma;
            for (int i = 1; i < (int) IMAGES; i++) {
                double now = prev * k;
                kernel[i] = std::sqrt(now*now - prev*prev);
                prev = now;
            }
            cv::Mat temp = base.clone();
            images.reserve(octaves);
            for (int i = 0; i < octaves; i++) {
                std::vector<cv::Mat> octave_images(IMAGES);
                // the base image for each octave is just interpolated base image
                octave_images[0] = temp;
                for (int j = 1; j < (int) IMAGES; j++) {
                    cv::GaussianBlur(octave_images[j-1], octave_images[j], cv::Size(), kernel[j], kernel[j]);
                }
                size_t baseid = octave_images.size() - 3;
                cv::resize(octave_images[baseid], temp, cv::Size(), 0.5, 0.5, cv::INTER_NEAREST);
                images.push_back(std::move(octave_images));
            }
        }   

        void gen_dog_images() {
            for (auto &octave: images) {
                std::vector<cv::Mat> dog_images(IMAGES - 1);
                for (int j = 1; j < (int) IMAGES; j++) {
                    dog_images[j - 1] = octave[j] - octave[j - 1];
                }
                octave.clear();
                octave = std::move(dog_images);
            }
        }

        void find_scale_space_extrema() {
            
        }

    public:
        cv::Mat base;
        int32_t octaves;
    private:
        static constexpr double sigma = 1.6;
        static constexpr double assumed_blur = 0.5;
        static constexpr size_t IMAGES = SCALES + 3;
        std::vector<std::vector<cv::Mat>> images;
    };

}

#endif

/*

base image 

k = 2 * (1/N)

sigma sigma*k sigma*k^2 ..... 2*sigma






*/


