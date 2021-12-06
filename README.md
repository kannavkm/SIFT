# SMAI project

## Objective
Write a fast no-dependency image stitching program in C++ from scratch.

## Algorithms
+ Features: [SIFT](http://en.wikipedia.org/wiki/Scale-invariant_feature_transform)
+ Feature Matching: [KNN](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) in O(n log(n)) time by using a k-d tree to find approximate nearest neighbours.
+ Transformation: use [RANSAC](http://en.wikipedia.org/wiki/RANSAC) to estimate a homography or affine transformation.
+ Optimization: focal estimation, [bundle adjustment](https://en.wikipedia.org/wiki/Bundle_adjustment), and some straightening tricks.

## Libraries
+ [libpng](http://www.libpng.org/pub/png/libpng.html) - for working with png files
+ [libjpeg](http://libjpeg.sourceforge.net/) - for working with jpeg files

## setup
```bash
sudo apt install \ 
libpng-dev \ # for png can be skipped for now
libjpeg-dev \ # for jpeg can be skipped
ocl-icd-opencl-dev \ # for opencl can be skipped for now
libopencv-dev
```
## How to run
you will need `cmake` to build
```bash
mkdir build
cd build
cmake ..
make
./smai ../res/Lenna.jpg
./smai <Images you want to stitch in order from left to right>
```

## Optional
+ Do matrix multiplication on GPU(CUDA || OpenCL || OpenMP)
+ Create a webapp to upload images
