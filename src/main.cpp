#include <iostream>
#include "img.h"

struct image_t img;

int main() {
    read_JPEG_file("../res/prof.jpeg");
    write_JPEG_file("../res/prof2.jpeg", 1);
    return 0;
}
