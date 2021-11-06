#include <iostream>
#include <jpeglib.h>
#include <jerror.h>
#include <png.h>

JSAMPLE * image_buffer;
size_t image_width;
size_t image_height;

int main() {
    std::cout << "Hello, World!" << std::endl;
    struct jpeg_decompress_struct info{};
    struct jpeg_error_mgr err{};

    FILE * file = fopen("~/Pictures/", "rb");
    info.err = jpeg_std_error(&err);
    jpeg_create_decompress(&info);
    if (!file) {
        std::cerr << "Error loading jpeg files";
    }
    jpeg_stdio_src(&info, file);
    jpeg_read_header(&info, true);
    jpeg_start_decompress(&info);
    size_t x = info.output_width;
    size_t y = info.output_height;
    int channels = info.num_components;
    size_t data_size = x * y * 3;
    image_buffer = new JSAMPLE[data_size];

    return 0;
}
