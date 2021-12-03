#ifndef SIFT_H
#define SIFT_H

#include <opencv2/core/core.hpp>
#include <opencv2/core/utils/tls.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

#define TIMEIT(f)                                                                 \
    {                                                                             \
        auto start = std::chrono::steady_clock::now();                            \
        f();                                                                      \
        auto end = std::chrono::steady_clock::now();                              \
        std::chrono::duration<double> duration = end - start;                     \
        std::cerr << "elapsed time: " << #f << ": " << duration.count() << "s\n"; \
    }

namespace sift {
class sift_handler {
   public:
    explicit sift_handler(std::string _name, cv::Mat &&_base);

    void exec();

    [[nodiscard]] cv::Mat get() const;

    ~sift_handler();

   private:
    class get_descriptors_parallel : public cv::ParallelLoopBody {
       public:
        get_descriptors_parallel(std::vector<cv::KeyPoint> &_keypoints,
                                 std::vector<std::vector<cv::Mat>> &_gauss_images,
                                 cv::TLSData<std::vector<std::pair<int, std::vector<double>>>> &_tls_data_struct)
            : keypts(_keypoints), gauss_images(_gauss_images), tls_data_struct(_tls_data_struct){};

        void operator()(const cv::Range &range) const override;

        const std::vector<cv::KeyPoint> keypts;
        const std::vector<std::vector<cv::Mat>> &gauss_images;
        cv::TLSData<std::vector<std::pair<int, std::vector<double>>>> &tls_data_struct;
    };

    class scale_space_extrema_parallel : public cv::ParallelLoopBody {
       public:
        scale_space_extrema_parallel(std::vector<std::vector<cv::Mat>> &_images,
                                     std::vector<std::vector<cv::Mat>> &_gauss_images, int _oct, int _img,
                                     cv::TLSData<std::vector<cv::KeyPoint>> &_tls_data_struct)
            : images(_images), gauss_images(_gauss_images), oct(_oct), img(_img), tls_data_struct(_tls_data_struct){};

        void operator()(const cv::Range &range) const override;

        [[nodiscard]] std::vector<cv::Mat> get_pixel_cube(int _oct, int _img, int i, int j) const;

        static cv::Mat get_gradient(const std::vector<cv::Mat> &pixel_cube);

        static cv::Mat get_hessian(const std::vector<cv::Mat> &pixel_cube);

        static bool is_pixel_extremum(const std::vector<cv::Mat> &pixel_cube);

        int localize_extrema(int _oct, int _img, int i, int j, cv::KeyPoint &) const;

        void get_keypoint_orientations(int _oct, int _img, cv::KeyPoint &kpt,
                                       std::vector<cv::KeyPoint> &_keypoints) const;

        const std::vector<std::vector<cv::Mat>> &images, &gauss_images;
        int oct;
        int img;
        cv::TLSData<std::vector<cv::KeyPoint>> &tls_data_struct;
    };

    static cv::Mat getImg(const cv::Mat &mat);

    void gen_gaussian_images();

    void gen_dog_images();

    void gen_scale_space_extrema();

    void clean_keypoints();

    void get_descriptors();

   public:
    cv::Mat base, onex;
    int32_t octaves;
    std::string name;
    std::vector<cv::KeyPoint> keypoints;
    std::vector<std::vector<double>> descriptors;

   private:
    static constexpr int SCALES = 3;
    static constexpr int BORDER = 5;
    static constexpr double contrast_threshold = 0.04;
    static constexpr double SIGMA = 1.6;
    static constexpr double assumed_blur = 0.5;
    static constexpr int IMAGES = SCALES + 3;
    static constexpr double EIGEN_VALUE_RATIO = 10.;
    static constexpr int BINS = 36;
    static constexpr double PEAK_RATIO = .8;
    static constexpr double SCALE_FACTOR = 1.5;
    static constexpr double RADIUS_FACTOR = 3;
    static constexpr double PI = 3.14159265358979323846;

    static constexpr double SCALE_MULTIPLIER = 3;
    static constexpr int WINDOW_WIDTH = 4;
    static constexpr double DESCRIPTOR_MAX = 0.2;

    std::vector<std::vector<cv::Mat>> images;
    std::vector<std::vector<cv::Mat>> gauss_images;
};

}  // namespace sift

#endif
