//
// Created by exclowd on 16/11/21.
//

#include "sift.hpp"
#include <chrono>

namespace sift {

#define TIMEIT(f) {\
auto start = std::chrono::steady_clock::now(); \
f(); \
auto end = std::chrono::steady_clock::now(); \
std::chrono::duration<double> duration = end - start; \
std::cout << "elapsed time: " << #f << ": " << duration.count() << "s\n";}

#define G(x, a, b) ((x).at<double>(a, b))
#define EPS ((double) 1e-15)

    sift_handler::sift_handler(cv::Mat &&_base) {
        cv::Mat temp, interpolated, blurred_image;
        _base.convertTo(temp, CV_64F);
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

    std::vector<cv::Mat> sift_handler::get_pixel_cube(int oct, int img, size_t i, size_t j) {
        cv::Mat first_image = images[oct][img - 1];
        cv::Mat second_image = images[oct][img];
        cv::Mat third_image = images[oct][img + 1];
        cv::Rect r(i - 1, j - 1, 3, 3);
        std::vector<cv::Mat> pixel_cube{first_image(r), second_image(r), third_image(r)};
        return pixel_cube;
    }

    void sift_handler::gen_scale_space_extrema() {
        for (int oct = 0; oct < octaves; oct++) {
            for (int img = 1; img < IMAGES - 2; img++) {
                cv::Size size = images[oct][img].size();
                for (int i = BORDER; i < (int) (size.height - BORDER); i++) {
                    for (int j = BORDER; j < (int) (size.width - BORDER); j++) {
                        std::vector<cv::Mat> pixel_cube = get_pixel_cube(oct, img, i, j);
                        if (is_pixel_extremum(pixel_cube)) {
                            cv::KeyPoint kpt;
                            auto image_index = localize_extrema(oct, img, i, j, kpt);
                            if (image_index < 0) {
                                continue;
                            }
                            std::cout << "#" << oct << ":" << image_index << "->" << i << " " << j << "\n";
                        }
                    }
                }
            }
        }
    }

    // TODO do this on the gpu
    bool sift_handler::is_pixel_extremum(const std::vector<cv::Mat> &pixel_cube) {
        bool is_maximum = true, is_minimum = true;
        for (int k = 0; k < 3; k++) {
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    is_maximum &= G(pixel_cube[1], 1, 1) >= G(pixel_cube[k], i, j);
                    is_minimum &= G(pixel_cube[1], 1, 1) <= G(pixel_cube[k], i, j);
                }
            }
        }
        return (is_minimum | is_maximum);
    }

    cv::Mat sift_handler::get_gradient(const std::vector<cv::Mat> &pixel_cube) {
        cv::Mat grad(3, 1, CV_64F);
        G(grad, 0, 0) = 0.5 * (G(pixel_cube[1], 1, 2) - G(pixel_cube[1], 1, 0));
        G(grad, 1, 0) = 0.5 * (G(pixel_cube[1], 2, 1) - G(pixel_cube[1], 0, 1));
        G(grad, 2, 0) = 0.5 * (G(pixel_cube[2], 1, 1) - G(pixel_cube[0], 1, 1));
        return grad;
    }

    cv::Mat sift_handler::get_hessian(const std::vector<cv::Mat> &pixel_cube) {
        cv::Mat hess(3, 3, CV_64F);
        G(hess, 0, 0) = G(pixel_cube[1], 1, 2) - 2 * G(pixel_cube[1], 1, 1) + G(pixel_cube[1], 1, 0);
        G(hess, 1, 1) = G(pixel_cube[1], 2, 1) - 2 * G(pixel_cube[1], 1, 1) + G(pixel_cube[1], 0, 1);
        G(hess, 2, 2) = G(pixel_cube[2], 1, 1) - 2 * G(pixel_cube[1], 1, 1) + G(pixel_cube[0], 1, 1);

        G(hess, 0, 1) = G(hess, 1, 0) = 0.25 * (G(pixel_cube[1], 2, 2) - G(pixel_cube[1], 2, 0) -
                                                G(pixel_cube[1], 0, 2) + G(pixel_cube[1], 0, 0));
        G(hess, 0, 2) = G(hess, 2, 0) = 0.25 * (G(pixel_cube[2], 1, 2) - G(pixel_cube[2], 1, 0) -
                                                G(pixel_cube[0], 1, 2) + G(pixel_cube[0], 1, 0));
        G(hess, 1, 2) = G(hess, 2, 1) = 0.25 * (G(pixel_cube[2], 2, 1) - G(pixel_cube[2], 0, 1) -
                                                G(pixel_cube[0], 2, 1) + G(pixel_cube[0], 0, 1));
        return hess;
    }

    int sift_handler::localize_extrema(int oct, int img, int i, int j, cv::KeyPoint &kpt) {
        constexpr int attempts = 5;
        cv::Size sz = images[oct][0].size();
        int attempt;
        std::vector<cv::Mat> pixel_cube;
        cv::Mat grad, hess, res;
        for (attempt = 0; attempt < attempts; attempt++) {
            pixel_cube.clear();
            pixel_cube = get_pixel_cube(oct, img, i, j);
            // gradient
            grad = get_gradient(pixel_cube);
            // hessian
            hess = get_hessian(pixel_cube);
            // solve the equation
            bool temp = cv::solve(hess, grad, res, cv::DECOMP_NORMAL);
            if (!temp) {
                return 0;
            }
            res *= -1;
            // std::cout << temp << " " << hess << "*" << res << "=" << grad << std::endl;
            // only way to get convergence
            if (std::abs(G(res, 0, 0)) < 0.5 && std::abs(G(res, 1, 0)) < 0.5 && std::abs(G(res, 2, 0)) < 0.5) {
                break;
            }
            j += (int) std::round(G(res, 0, 0));
            i += (int) std::round(G(res, 1, 0));
            img += (int) std::round(G(res, 2, 0));

            // extremum is outside search zone
            if (i < BORDER || i >= sz.width - BORDER || j < BORDER || j >= sz.height - BORDER || img < 1 ||
                img > SCALES) {
                return -1;
            }
            grad.release();
            hess.release();
            res.release();
        }
        // nahi mila
        if (attempt == 5) {
            return -1;
        }
        double value = G(pixel_cube[1], 1, 1) + 0.5 * grad.dot(res);

        if (std::abs(value) * SCALES >= contrast_threshold) {
            cv::Mat hess2 = hess(cv::Rect(1, 1, 2, 2));
            double hess_trace = cv::trace(hess)[0];
            double hess_det = cv::determinant(hess);
            if (hess_det <= EPS) {
                return -1;
            }
            double ratio = (hess_trace * hess_trace) / hess_det;

            if (ratio < THRESHOLD_EIGEN_RATIO) {

                // octave_index + image_index * (2 ** 8) + int(round((extremum_update[2] + 0.5) * 255)) * (2 ** 16)
                double keypt_octave = oct + 256 * img + 65536 * std::round((G(res, 2, 0) + 0.5) * 255);
                double keypt_pt_x = (j + G(res, 0, 0)) * double(1LL << oct);
                double keypt_pt_y = (i + G(res, 1, 0)) * double(1LL << oct);
                double keypt_size =
                        SIGMA * (std::pow(2, img + G(res, 2, 0)) / (1.0 * SCALES)) * double(1LL << (oct + 1));
                double keypt_response = std::abs(value);
                kpt = cv::KeyPoint(
                        keypt_pt_x,
                        keypt_pt_y,
                        keypt_size,
                        -1,
                        keypt_response,
                        keypt_octave
                );
                return img;
            }
        }

        return -1;
    }


}
/*
keypoint = KeyPoint()
keypoint.pt = ((j + extremum_update[0]) * (2 ** octave_index), (i + extremum_update[1]) * (2 ** octave_index))
keypoint.octave = octave_index + image_index * (2 ** 8) + int(round((extremum_update[2] + 0.5) * 255)) * (2 ** 16)
keypoint.size = sigma * (2 ** ((image_index + extremum_update[2]) / float32(num_intervals))) * (2 ** (octave_index + 1))  # octave_index + 1 because the input image was doubled
keypoint.response = abs(functionValueAtUpdatedExtremum)
return keypoint, image_index
*/