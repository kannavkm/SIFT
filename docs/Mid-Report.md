<div style="text-align:center;">
    <h1>Scale-Invariant Feature Transform</h1>
    <h2>Mid-Term Progress Report</h2>
</div>

### Team 4: Drowning in Inefficiency
- Kannav Mehta   (2019101044)
- Raj Maheshwari (2019101039)
- Triansh Sharma (2019101006)
- Aashwin Vaish  (2019114014)

[Github link](https://github.com/exclowd/smai-team4-sift)


## Code

### How to run:
Although the instructions are clearly written in the [README](), let us rewrite them here.
1. Create an empty build directory.
```mkdir build && cd build```
2. Call cmake inside it.
```cmake ..```
3. Now make.
```make```
4. Finally run the executable along with the path of your image as an argument. 
```./smai <path>```


### Code structure
The code is written in <b>C++</b> .
The sift implementation consists of primarily 3 files - the main.cpp, the sift.cpp and the sift.hpp

1. **main.cpp** -- Here we read the image file from the path passed in as input argument. The image is converted to GRAYSCALE and passed to the sift handler. A window is opened to display the output image.
2. **sift.cpp** -- All the major code resides here. Everything from core logic to algorithm implemenation lies here.
3. **sift.hpp** -- This is the header file which separates all the variable and function declarations from the main logic. We define all the critical constants here so tweaks can easily be made while analysing performance.

Here are some of the constants that we used:

```c++
SCALES = 3 // number of scales per octave
BORDER = 5 // boundary region where no keypoints will be detected
contrast_threshold = 0.04 
threshold = (0.5 * contrast_threshold / SCALES * 255)
SIGMA = 1.6
assumed_blur = 0.5
IMAGES = SCALES + 3
EIGEN_VALUE_RATIO = 10.0 // To eliminate edge responses
THRESHOLD_EIGEN_RATIO = pow((EIGEN_VALUE_RATIO + 1), 2) / EIGEN_VALUE_RATIO
BINS = 36 // Number of bins in histogram 1 per 10 degress
PEAK_RATIO = 0.8
SCALE_FACTOR = 1.5
RADIUS_FACTOR = 3
```
These hyperparameters were chosen by the discretion of the authors in the research paper by Lowe, and we trust them.

Although we are using OpenCV, we restricted its usage to only the following:
* Reading and writing images
* Mathematical classes and functions (`cv::Mat` classes, `cv::Range` and `cv::parallel_for` functions etc.)
* Gaussian Blur
* Interpolation

## ```sift_handler``` class

The sift handler class contains the various different methods that get sequentially called in the `exec` function. As a first step, we perform some preprocessing steps in the constructor of this class.
Number of octaves is calculated:
```c++
octaves = (size_t)std::round(std::log2((double)std::min(sz.width, sz.height)))-1
```
We compute the log base 2 of the minimum of the dimensions to find how times the image can be halved before it is reduced to a size of atmost 3 pixels.

A base image is created by resizing the original image to twice its size using `interlinear` interpolation. Then we apply gaussian blur on it so our final image has a blur of `SIGMA`.


### ```sift_handler::exec()```

This is basically the ***execute*** function which orderly calls all the different functions on the image. 

<ol>
    <li>Generate the <b>Gaussian</b> images</li>    
    <li>Generate the <b>Difference of Gaussian</b> images</li>
    <li>Generating the <b>Scale Space Extremas</b> </li>
    <li> <b>Cleaning</b> out the keypoints</li>
</ol>

Since we care about latency, we have <u>timed</u> these important functions and made efforts to reduce processing time. Finally, display the image upon which are nicely plotted the keypoints captured by our program.

### ```sift_handler::gen_gaussian_images()```

Here we generate gaussian kernels that successively blur the image. Each octave has `IMAGES` number of images. These images differ by a constant amount of blur. The third last image of the octave gets carried forward to the next octave after squeezing it to half its size.


### ```sift_handler::gen_dog_images()```

The DoGs are simply the differences between successive images in a octave. When plotted, they seem like edge maps.

### ```sift_handler::gen_scale_space_extrema()```

Since space scale extremum is an operation that needs to be carried out for all the pixels of each image in every octave, we decided to perform parallel processing on these pixels with the help of threading. This greatly improves the performance of our code and makes it much more efficient because of better use of computational resources.

### ```sift_handler::clean_keypoints()```

Upon running the algorithm, lots of keypoints are generated. Many of these of keypoints lie close to each other and are unnecceassry repetitions. In order to focus on different features of an image, these need to be cleaned and filtered out. We try to club different keypoints into small subsets by removing those keypoints which lie very close to one another. First we sort them using a custom comparator function based on their coordinate, size, octave etc, and then we find unique points by considering 2 points distanced by a single pixel as nearly equal. Finally, we scale these keypoints back by half to match the original image.

### ```sift_handler::scale_space_extrema_parallel``` class
We are using multi-threading, using opencv's builtin `parallel_for_` to speed up finding the scale space extrema points and computing keypoints. This class contains all the functions that find keypoints of a $3\times3\times3$ pixel cube centered on a pixel $(i,j)$.

### ```sift_handler::scale_space_extrema_parallel::is_pixel_extremum()``` 

A pixel cube is the 1 pixel neighbourhood of an pixel in an image. This is the layer of $3\times3$ pixels above and below it and $3+2+3$ pixels around it. A pixel extremum is one which is greater or lesser than all 26 pixels around it. For such pixel extremas, we perform keypoint localization and find the orientation of this keypoint.

### ```sift_handler::scale_space_extrema_parallel::localize_extrema()```

In order to localize a keypoint, a quadratic model is fit to the $3\times3\times3$ pixel cube centered on the keypoint pixel. `get_gradient()` and `get_hessian()` functions implement second-order central finite difference approximations in three dimensions.

The localization of the extremum is defined as:
$$ \hat{x} = -\frac{\partial^2 D}{\partial x^2}^{-1} \frac{\partial D}{\partial x}$$
In code, this resolves to:
```c++
bool temp = cv::solve(hess, grad, res, cv::DECOMP_NORMAL);
if (!temp) return 0;
res *= -1;
```
Here, `hess` is the hessian and `grad` is the gradient while the resultant x gets stored in `res`. `temp` denotes whether or the equation was solved correctly.

Following updates are made in each iteration.
```c++
j += (int)std::round(res[0][0]);
i += (int)std::round(res[1][0]);
img += (int)std::round(res[2][0]);
```

Convergence is achieved for following condition:
```c++
std::abs(res[0][0]) < 0.5 && 
std::abs(res[1][0]) < 0.5 && 
std::abs(res[2][0]) < 0.5
```

The extremum must be within the search zone:
```c++
BORDER < i and i <= sz.width - BORDER and
BORDER < j and j <= sz.height - BORDER and
1 < img and img <= IMAGES - 2
```

Threshold is defined as:
$$\tau =  \frac{Tr(H)^2}{Det(H)} < \frac{(r+1)^2}{r}$$
if $\tau$ is less than `THRESHOLD_EIGEN_RATIO` then we compute the following keypoint:
```c++
double keypt_octave = oct + (1 << 8) * img + (1 << 16) * std::round((res[2][0] + 0.5) * 255);
double keypt_pt_x = (j + G(res, 0, 0)) * (1 << oct);
double keypt_pt_y = (i + G(res, 1, 0)) * (1 << oct);
double keypt_size = SIGMA * (std::pow(2, img + res[2][0]) / (1.0 * SCALES)) * (1 << (oct + 1));
double keypt_response = std::abs(value);
kpt = cv::KeyPoint(keypt_pt_x, keypt_pt_y, keypt_size, -1, keypt_response, keypt_octave);
```


### ```sift_handler::scale_space_extrema_parallel::get_keypoint_orientations()```

To calculate the orientation(s) of a keypoint, we use the histograms of gradients for pixels in a square neighbourhood centered on the keypoint. For each pixel in this neighbourhood, we calculate the magnitude and orientation of the 2D gradient. Using a 36-bin histogram (each $10^{\circ}$). The value put into the bin is the gradient magnitude * gaussian weight for that pixel, so that farther pixels have less influence than central ones.
At last we smooth out the histogram using 5 point gaussian kernel.


## Statistics

We modified the `EIGEN_VALUE_RATIO` from `[1, 2, 3, ..., 10]` and calculated the number of keypoints obtained using it for image of size $512\times512$.
The below plot describes the nature of curve.

<center>
<img src = "https://i.imgur.com/tZvpSUg.jpg">
</center>

We also calculated DoG and found the keypoints in variety of images with different dimensions. Below are the results shown for a few test images.

<b> Note: </b> The time measured is highly dependent on configurations of system. Hence, the time might be not reproducible. The system configuration used to calculate the total time taken by program to run has <i>i7 9th Gen processor with 12 CPU cores and 16GB RAM</i>.

### Image 1

<b> Size: </b> 512 $\times$ 512
<b> Keypoints: </b> 2739
<b> Time taken: </b> 0.868 seconds

<center>
<table>
<tr>
<td>
<figure>
<img src = "https://i.imgur.com/kOXADoF.png">
<figcaption><center>Input image</center></figcaption>
</figure>
</td>
<td>
<figure>
<img src = "https://i.imgur.com/kuOcsOR.png">
<figcaption><center>Keypoints</center></figcaption>
</figure>
</td>
</tr>
</table>

</center>

#### DoG pyramid
<table>
    <tr>
        <th colspan=5 style="text-align:center">Octave 0</th>
    </tr>
    <tr>
        <td><img src="https://i.imgur.com/2ig8b5D.png"></td>
        <td><img src="https://i.imgur.com/hbGUAPN.png"></td>
        <td><img src="https://i.imgur.com/qT3fsez.png"></td>
        <td><img src="https://i.imgur.com/ZYYUxAI.png"></td>
        <td><img src="https://i.imgur.com/X0QaAYL.png"></td>
    </tr>
    <tr>
        <th colspan=5 style="text-align:center">Octave 1</th>
    </tr>
    <tr>
        <td><img src="https://i.imgur.com/QTATFy8.png"></td>
        <td><img src="https://i.imgur.com/g72wH5w.png"></td>
        <td><img src="https://i.imgur.com/1kB7Zge.png"></td>
        <td><img src="https://i.imgur.com/5PXBnoZ.png"></td>
        <td><img src="https://i.imgur.com/3fRbJn0.png"></td>
    </tr>
    <tr>
        <th colspan=5 style="text-align:center">Octave 2</th>
    </tr>
    <tr>
        <td><img src="https://i.imgur.com/FryAFBQ.png"></td>
        <td><img src="https://i.imgur.com/Dl4r79M.png"></td>
        <td><img src="https://i.imgur.com/6GJmSds.png"></td>
        <td><img src="https://i.imgur.com/v0SDf2V.png"></td>
        <td><img src="https://i.imgur.com/Q9fNo4G.png"></td>
    </tr>
    <tr>
        <th colspan=5 style="text-align:center">Octave 3</th>
    </tr>
    <tr>
        <td><img src="https://i.imgur.com/hYGulPm.png"></td>
        <td><img src="https://i.imgur.com/59kInUA.png"></td>
        <td><img src="https://i.imgur.com/eRLWkL8.png"></td>
        <td><img src="https://i.imgur.com/3JzmzhM.png"></td>
        <td><img src="https://i.imgur.com/jjlrxrs.png"></td>
    </tr>
    <tr>
        <th colspan=5 style="text-align:center">Octave 4</th>
    </tr>
    <tr>
        <td><img src="https://i.imgur.com/yts1YvK.png"></td>
        <td><img src="https://i.imgur.com/iKW46Qx.png"></td>
        <td><img src="https://i.imgur.com/vhsxmCH.png"></td>
        <td><img src="https://i.imgur.com/EJaV7PS.png"></td>
        <td><img src="https://i.imgur.com/hR6toFH.png"></td>
    </tr>
    <tr>
        <th colspan=5 style="text-align:center">Octave 5</th>
    </tr>
    <tr>
        <td><img src="https://i.imgur.com/apb1MMe.png"></td>
        <td><img src="https://i.imgur.com/xfU1fKS.png"></td>
        <td><img src="https://i.imgur.com/5EqSn9X.png"></td>
        <td><img src="https://i.imgur.com/pvpb5Ms.png"></td>
        <td><img src="https://i.imgur.com/yKlw6mg.png"></td>
    </tr>
    <tr>
        <th colspan=5 style="text-align:center">Octave 6</th>
    </tr>
    <tr>
        <td><img src="https://i.imgur.com/A7MSZD1.png"></td>
        <td><img src="https://i.imgur.com/sE6QrT9.png"></td>
        <td><img src="https://i.imgur.com/07BRSBw.png"></td>
        <td><img src="https://i.imgur.com/3BJ402h.png"></td>
        <td><img src="https://i.imgur.com/Q37GC5T.png"></td>
    </tr>

</table>
<hr>

### Image 2

<b> Size: </b> 910 $\times$ 683
<b> Keypoints: </b> 6030
<b> Time taken: </b> 2.175 seconds


<center>
<table>
<tr>
<td>
<figure>
<img src = "https://i.imgur.com/Q3fx8TD.jpg">
<figcaption><center>Input image</center></figcaption>
</figure>
</td>
<td>
<figure>
<img src = "https://i.imgur.com/PlVKN4e.jpg">
<figcaption><center>Keypoints</center></figcaption>
</figure>
</td>
</tr>
</table>
</center>


#### DoG pyramid
<table>
    <tr>
        <th colspan=5 style="text-align:center">Octave 0</th>
    </tr>
    <tr>
        <td><img src="https://i.imgur.com/tF4634U.png"></td>
        <td><img src="https://i.imgur.com/oWiVC1u.png"></td>
        <td><img src="https://i.imgur.com/d8YlkPr.png"></td>
        <td><img src="https://i.imgur.com/EEbPGcR.png"></td>
        <td><img src="https://i.imgur.com/F2UiAG6.png"></td>
    </tr>
    <tr>
        <th colspan=5 style="text-align:center">Octave 1</th>
    </tr>
    <tr>
        <td><img src="https://i.imgur.com/qo3DoAx.png"></td>
        <td><img src="https://i.imgur.com/cKD5POZ.png"></td>
        <td><img src="https://i.imgur.com/XvVCip6.png"></td>
        <td><img src="https://i.imgur.com/XyPtvcO.png"></td>
        <td><img src="https://i.imgur.com/wO3VV0k.png"></td>
    </tr>
    <tr>
        <th colspan=5 style="text-align:center">Octave 2</th>
    </tr>
    <tr>
        <td><img src="https://i.imgur.com/Q1zopEf.png"></td>
        <td><img src="https://i.imgur.com/9xJFREJ.png"></td>
        <td><img src="https://i.imgur.com/D6Qq7p0.png"></td>
        <td><img src="https://i.imgur.com/9y6Xuhj.png"></td>
        <td><img src="https://i.imgur.com/Byapxbd.png"></td>
    </tr>
    <tr>
        <th colspan=5 style="text-align:center">Octave 3</th>
    </tr>
    <tr>
        <td><img src="https://i.imgur.com/bk5g6tY.png"></td>
        <td><img src="https://i.imgur.com/U7O9rV0.png"></td>
        <td><img src="https://i.imgur.com/a4PQt1J.png"></td>
        <td><img src="https://i.imgur.com/Z47FA7C.png"></td>
        <td><img src="https://i.imgur.com/eQPpANC.png"></td>
    </tr>
    <tr>
        <th colspan=5 style="text-align:center">Octave 4</th>
    </tr>
    <tr>
        <td><img src="https://i.imgur.com/vPIm6G2.png"></td>
        <td><img src="https://i.imgur.com/svvExuG.png"></td>
        <td><img src="https://i.imgur.com/lJGTrLT.png"></td>
        <td><img src="https://i.imgur.com/T1E1LiT.png"></td>
        <td><img src="https://i.imgur.com/bzHFGNG.png"></td>
    </tr>
    <tr>
        <th colspan=5 style="text-align:center">Octave 5</th>
    </tr>
    <tr>
        <td><img src="https://i.imgur.com/ZFJ8tuo.png"></td>
        <td><img src="https://i.imgur.com/1kRbicm.png"></td>
        <td><img src="https://i.imgur.com/1oIv8bE.png"></td>
        <td><img src="https://i.imgur.com/Fpx3eei.png"></td>
        <td><img src="https://i.imgur.com/kroocEK.png"></td>
    </tr>
    <tr>
        <th colspan=5 style="text-align:center">Octave 6</th>
    </tr>
    <tr>
        <td><img src="https://i.imgur.com/ch3AemH.png"></td>
        <td><img src="https://i.imgur.com/NyoglAb.png"></td>
        <td><img src="https://i.imgur.com/GWjQ2dt.png"></td>
        <td><img src="https://i.imgur.com/1YHpzSc.png"></td>
        <td><img src="https://i.imgur.com/hEZyN9e.png"></td>
    </tr>
    <tr>
        <th colspan=5 style="text-align:center">Octave 7</th>
    </tr>
    <tr>
        <td><img src="https://i.imgur.com/JT7T2RU.png"></td>
        <td><img src="https://i.imgur.com/sQXnisY.png"></td>
        <td><img src="https://i.imgur.com/POUAIlL.png"></td>
        <td><img src="https://i.imgur.com/4O28Ehr.png"></td>
        <td><img src="https://i.imgur.com/03fdTCW.png"></td>
    </tr>

</table>
<hr>

### Image 3

<b> Size: </b> 3440 $\times$ 1440
<b> Keypoints: </b> 31370
<b> Time taken: </b> 17.411 seconds

<center>
<table>
<tr>
<td>
<figure>
<img src = "https://i.imgur.com/Te60t1o.jpg">
<figcaption><center>Input image</center></figcaption>
</figure>
</td>
</tr>
<tr>
<td>
<figure>
<img src = "https://i.imgur.com/wwvCMWQ.jpg">
<figcaption><center>Keypoints</center></figcaption>
</figure>
</td>
</tr>
</table>

</center>

#### DoG pyramid
<table>
    <tr>
        <th colspan=5 style="text-align:center">Octave 0</th>
    </tr>
    <tr>
        <td><img src="https://i.imgur.com/1sA66ye.jpg"></td>
        <td><img src="https://i.imgur.com/AFLhd3D.jpg"></td>
        <td><img src="https://i.imgur.com/QXOhFGo.jpg"></td>
        <td><img src="https://i.imgur.com/nPcJggR.jpg"></td>
        <td><img src="https://i.imgur.com/AOANGQz.jpg"></td>
    </tr>
    <tr>
        <th colspan=5 style="text-align:center">Octave 1</th>
    </tr>
    <tr>
        <td><img src="https://i.imgur.com/pFovig1.png"></td>
        <td><img src="https://i.imgur.com/pWKPAIe.png"></td>
        <td><img src="https://i.imgur.com/5iKWClO.png"></td>
        <td><img src="https://i.imgur.com/kDauqjV.png"></td>
        <td><img src="https://i.imgur.com/3rPVdET.png"></td>
    </tr>
    <tr>
        <th colspan=5 style="text-align:center">Octave 2</th>
    </tr>
    <tr>
        <td><img src="https://i.imgur.com/y8wWFjj.png"></td>
        <td><img src="https://i.imgur.com/ULHUZkr.png"></td>
        <td><img src="https://i.imgur.com/53tx2uW.png"></td>
        <td><img src="https://i.imgur.com/jX3g9uR.png"></td>
        <td><img src="https://i.imgur.com/z5OwKGr.png"></td>
    </tr>
    <tr>
        <th colspan=5 style="text-align:center">Octave 3</th>
    </tr>
    <tr>
        <td><img src="https://i.imgur.com/Ou6cvqu.png"></td>
        <td><img src="https://i.imgur.com/862t2LU.png"></td>
        <td><img src="https://i.imgur.com/rhPK1B3.png"></td>
        <td><img src="https://i.imgur.com/rVXJqDz.png"></td>
        <td><img src="https://i.imgur.com/0vKpPTL.png"></td>
    </tr>
    <tr>
        <th colspan=5 style="text-align:center">Octave 4</th>
    </tr>
    <tr>
        <td><img src="https://i.imgur.com/hz3xIps.png"></td>
        <td><img src="https://i.imgur.com/Ij2qdhz.png"></td>
        <td><img src="https://i.imgur.com/9pATn3O.png"></td>
        <td><img src="https://i.imgur.com/OMt3Xfg.png"></td>
        <td><img src="https://i.imgur.com/iKbkrG6.png"></td>
    </tr>
    <tr>
        <th colspan=5 style="text-align:center">Octave 5</th>
    </tr>
    <tr>
        <td><img src="https://i.imgur.com/6a9BuBY.png"></td>
        <td><img src="https://i.imgur.com/VEpElEA.png"></td>
        <td><img src="https://i.imgur.com/VoQu7yi.png"></td>
        <td><img src="https://i.imgur.com/rCD8THi.png"></td>
        <td><img src="https://i.imgur.com/dwS6XYv.png"></td>
    </tr>
    <tr>
        <th colspan=5 style="text-align:center">Octave 6</th>
    </tr>
    <tr>
        <td><img src="https://i.imgur.com/I32Zvni.png"></td>
        <td><img src="https://i.imgur.com/7lAGjhR.png"></td>
        <td><img src="https://i.imgur.com/0w8UqjU.png"></td>
        <td><img src="https://i.imgur.com/AdqTHPW.png"></td>
        <td><img src="https://i.imgur.com/M9ii0Ow.png"></td>
    </tr>
    <tr>
        <th colspan=5 style="text-align:center">Octave 7</th>
    </tr>
    <tr>
        <td><img src="https://i.imgur.com/WeiNnlY.png"></td>
        <td><img src="https://i.imgur.com/4XssPrL.png"></td>
        <td><img src="https://i.imgur.com/xpH0Siu.png"></td>
        <td><img src="https://i.imgur.com/v5yxFtU.png"></td>
        <td><img src="https://i.imgur.com/y4hmcvi.png"></td>
    </tr>
    <tr>
        <th colspan=5 style="text-align:center">Octave 8</th>
    </tr>
    <tr>
        <td><img src="https://i.imgur.com/LUpuLFl.png"></td>
        <td><img src="https://i.imgur.com/ucV3VB6.png"></td>
        <td><img src="https://i.imgur.com/Cn8yLPn.png"></td>
        <td><img src="https://i.imgur.com/QkgDb89.png"></td>
        <td><img src="https://i.imgur.com/JZzeiHk.png"></td>
    </tr>

</table>



## Further Plan

* We are still left with generating descriptors from the keypoints of the image. When done, we will complete the implementation of SIFT and then  implement feature matching across two images using [flann](https://github.com/flann-lib/flann) library to implement KNN.
* Further, we will try to implement the image stitching algorithm using the features detected by our SIFT implementation.