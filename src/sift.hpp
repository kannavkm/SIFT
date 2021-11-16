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
            std::cout << "elapsed time for gen gauss: " << for_gauss.count() << "s\n";
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
                kernel[i] = std::sqrt(std::pow(now, 2) - std::pow(prev, 2));
                prev = now;
            }
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
            for (int oct = 0; oct < octaves; oct++) {
                for (int img = 1; img < IMAGES - 1; img++) {
                    auto &first_image = images[oct][img - 1];
                    auto &second_image = images[oct][img];
                    auto &third_image = images[oct][img + 1];
                    cv::Size size = second_image.size();
                    for (int i = image_border_width; i < size.width - image_border_width; i++) {
                        for (int j = image_border_width; i < size.height - image_border_width; i++) {
                            cv::Rect r(i - 1, j - 1, 3, 3);
                            std::vector<cv::Mat> pixel_cube{first_image(r), second_image(r), third_image(r)};
                            auto ans = is_pixel_extremum(pixel_cube);
                        }
                    }
                }
            }
        }

        bool is_pixel_extremum(const std::vector<cv::Mat> &pixel_cube) {

            bool is_maximum = true, is_minimum = true;
            for (int k = 0; k < 3; k++) {
                for (int i = 0; i < 3; i++) {
                    for (int j = 0; j < 3; j++) {
                        is_maximum &= pixel_cube[1].at<float>(1, 1) >= pixel_cube[k].at<float>(i, j);
                        is_minimum &= pixel_cube[1].at<float>(1, 1) <= pixel_cube[k].at<float>(i, j);
                    }
                }
            }
            return is_minimum | is_maximum;
        }

        // def Gradient(pixel_array):
        //     dx = (pixel_array[1, 1, 2] - pixel_array[1, 1, 0]) / 2
        //     dy = (pixel_array[1, 2, 1] - pixel_array[1, 0, 1]) / 2
        //     ds = (pixel_array[2, 1, 1] - pixel_array[0, 1, 1]) / 2
        //     return np.array([dx, dy, ds])


        // # Approximate Hessian at center pixel [1, 1, 1] of 3x3x3 array
        // def Hessian(pixel_array):
        //     dxx = pixel_array[1, 1, 2] + \
        //         pixel_array[1, 1, 0] - 2 * pixel_array[1, 1, 1]
        //     dyy = pixel_array[1, 2, 1] + \
        //         pixel_array[1, 0, 1] - 2 * pixel_array[1, 1, 1]
        //     dss = pixel_array[2, 1, 1] + \
        //         pixel_array[0, 1, 1] - 2 * pixel_array[1, 1, 1]
        //     dxy = (pixel_array[1, 2, 2] - pixel_array[1, 2, 0] -
        //         pixel_array[1, 0, 2] + pixel_array[1, 0, 0]) / 4
        //     dxs = (pixel_array[2, 1, 2] - pixel_array[2, 1, 0] -
        //         pixel_array[0, 1, 2] + pixel_array[0, 1, 0]) / 4
        //     dys = (pixel_array[2, 2, 1] - pixel_array[2, 0, 1] -
        //         pixel_array[0, 2, 1] + pixel_array[0, 0, 1]) / 4
        //     return np.array([[dxx, dxy, dxs],
        //                     [dxy, dyy, dys],
        //                     [dxs, dys, dss]])


    public:
        cv::Mat base;
        int32_t octaves;
    private:
        static constexpr size_t image_border_width = 3;
        static constexpr double contrast_threshhold = 0.04;
        static constexpr double threshhold = (0.5 * contrast_threshhold / SCALES * 255);
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


