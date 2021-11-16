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
                            std::cout << "#" << i << " " << j << "\n";
                            localize_extrema(oct, img, i, j);
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
                    is_maximum &= pixel_cube[1].at<double>(1, 1) >= pixel_cube[k].at<double>(i, j);
                    is_minimum &= pixel_cube[1].at<double>(1, 1) <= pixel_cube[k].at<double>(i, j);
                }
            }
        }
        return (is_minimum | is_maximum);
    }

#define G(x, a, b) ((x).at<double>(a, b))

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

    int sift_handler::localize_extrema(int oct, int img, int i, int j) {
        constexpr int attempts = 5;
        cv::Size sz = images[oct][0].size();
        int attempt;
        for (attempt = 0; attempt < attempts; attempt++) {
            auto pixel_cube = get_pixel_cube(oct, img, i, j);
            // gradient
            cv::Mat grad = get_gradient(pixel_cube);
            // hessian
            cv::Mat hess = get_hessian(pixel_cube);
            // solve the equation
            cv::Mat temp;
            bool res = cv::solve(hess, grad, temp, cv::DECOMP_NORMAL);
            if (!res) continue;
            temp *= -1;
            std::cout << res << " " << hess << "*" << temp << "=" << grad << std::endl;
            // only way to do it ok, got convergence
            if (std::abs(G(temp, 0, 0)) < 0.5 && std::abs(G(temp, 1, 0)) < 0.5 && std::abs(G(temp, 2, 0)) < 0.5) {
                break;
            }
            j += (int) std::round(G(temp, 0, 0));
            i += (int) std::round(G(temp, 1, 0));
            img += (int) std::round(G(temp, 2, 0));

            if (i < BORDER || i >= sz.width - BORDER || j < BORDER || j >= sz.height - BORDER || img < 1 ||
                img > SCALES) {
                return 0;
            }
        }
        if (attempt == 5) {
            return 0;
        }
        return 1;
    }


    /*
    def localizeExtremumViaQuadraticFit(i, j, image_index, octave_index, num_intervals, dog_images_in_octave, sigma, contrast_threshold, image_border_width, eigenvalue_ratio=10, num_attempts_until_convergence=5):
    """Iteratively refine pixel positions of scale-space extrema via quadratic fit around each extremum's neighbors
    """
    logger.debug('Localizing scale-space extrema...')
    extremum_is_outside_image = False
    image_shape = dog_images_in_octave[0].shape
    for attempt_index in range(num_attempts_until_convergence):
        # need to convert from uint8 to float32 to compute derivatives and need to rescale pixel values to [0, 1] to apply Lowe's thresholds
        first_image, second_image, third_image = dog_images_in_octave[image_index-1:image_index+2]
        pixel_cube = stack([first_image[i-1:i+2, j-1:j+2],
                            second_image[i-1:i+2, j-1:j+2],
                            third_image[i-1:i+2, j-1:j+2]]).astype('float32') / 255.
        gradient = computeGradientAtCenterPixel(pixel_cube)
        hessian = computeHessianAtCenterPixel(pixel_cube)
        extremum_update = -lstsq(hessian, gradient, rcond=None)[0]
        if abs(extremum_update[0]) < 0.5 and abs(extremum_update[1]) < 0.5 and abs(extremum_update[2]) < 0.5:
            break
        j += int(round(extremum_update[0]))
        i += int(round(extremum_update[1]))
        image_index += int(round(extremum_update[2]))
        # make sure the new pixel_cube will lie entirely within the image
        if i < image_border_width or i >= image_shape[0] - image_border_width or j < image_border_width or j >= image_shape[1] - image_border_width or image_index < 1 or image_index > num_intervals:
            extremum_is_outside_image = True
            break
    if extremum_is_outside_image:
        logger.debug('Updated extremum moved outside of image before reaching convergence. Skipping...')
        return None
    if attempt_index >= num_attempts_until_convergence - 1:
        logger.debug('Exceeded maximum number of attempts without reaching convergence for this extremum. Skipping...')
        return None
    functionValueAtUpdatedExtremum = pixel_cube[1, 1, 1] + 0.5 * dot(gradient, extremum_update)
    if abs(functionValueAtUpdatedExtremum) * num_intervals >= contrast_threshold:
        xy_hessian = hessian[:2, :2]
        xy_hessian_trace = trace(xy_hessian)
        xy_hessian_det = det(xy_hessian)
        if xy_hessian_det > 0 and eigenvalue_ratio * (xy_hessian_trace ** 2) < ((eigenvalue_ratio + 1) ** 2) * xy_hessian_det:
            # Contrast check passed -- construct and return OpenCV KeyPoint object
            keypoint = KeyPoint()
            keypoint.pt = ((j + extremum_update[0]) * (2 ** octave_index), (i + extremum_update[1]) * (2 ** octave_index))
            keypoint.octave = octave_index + image_index * (2 ** 8) + int(round((extremum_update[2] + 0.5) * 255)) * (2 ** 16)
            keypoint.size = sigma * (2 ** ((image_index + extremum_update[2]) / float32(num_intervals))) * (2 ** (octave_index + 1))  # octave_index + 1 because the input image was doubled
            keypoint.response = abs(functionValueAtUpdatedExtremum)
            return keypoint, image_index
    return None
    */

}