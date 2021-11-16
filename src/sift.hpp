#ifndef SIFT_H
#define SIFT_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <vector>

namespace sift {
    class sift_handler {
    public:
        explicit sift_handler(cv::Mat &&_base);

        void exec();

        [[nodiscard]] cv::Mat get() const;

        ~sift_handler();

    private:

        static cv::Mat getImg(const cv::Mat &mat);

        void gen_gaussian_images();

        void gen_dog_images();

        void gen_scale_space_extrema();

        static bool is_pixel_extremum(const std::vector<cv::Mat> &pixel_cube);

        void localize_extrema(int i, int j, const std::vector<cv::Mat> &pixel_cube);

    public:
        cv::Mat base;
        int32_t octaves;
    private:
        static constexpr size_t SCALES = 3;
        static constexpr size_t BORDER = 3;
        static constexpr double contrast_threshold = 0.04;
        static constexpr double threshold = (0.5 * contrast_threshold / SCALES * 255);
        static constexpr double SIGMA = 1.6;
        static constexpr double assumed_blur = 0.5;
        static constexpr size_t IMAGES = SCALES + 3;
        std::vector<std::vector<cv::Mat>> images;
    };

}

#endif


