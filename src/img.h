//
// Created by kannav
//

#ifndef SMAI_IMAGE_H
#define SMAI_IMAGE_H

#include <cstdio>
#include <jpeglib.h>

struct image_t {
    JSAMPLE * buffer;
    int width;
    int height;
};

extern image_t img;

extern int write_JPEG_file(const char * filename, int quality);

extern int read_JPEG_file(const char * filename);

#endif // SMAI_IMAGE_H