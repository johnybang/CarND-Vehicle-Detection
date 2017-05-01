## Vehicle Detection
John Bang - 4/28/17
### As a part of the Udacity Self Driving Car Engineer Nanodegree program, we use computer vision and supervised machine learning to detect the presence and location of vehicles on the road from the perspective of a moving car.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/train_images.jpg
[image2]: ./output_images/hog_example.jpg
[image3]: ./output_images/test1_grid.jpg
[image4]: ./output_images/test1_windowed.jpg
[image5]: ./output_images/test2_windowed.jpg
[image6]: ./output_images/test3_window.jpg
[image7]: ./output_images/test4_windowed.jpg
[image8]: ./output_images/test5_windowed.jpg
[image9]: ./output_images/test6_windowed.jpg
[image10]: ./output_images/test1_heat.jpg
[image11]: ./output_images/test2_heat.jpg
[image12]: ./output_images/test3_heat.jpg
[image13]: ./output_images/test4_heat.jpg
[image14]: ./output_images/test5_heat.jpg
[image15]: ./output_images/test6_heat.jpg
[image16]: ./examples/bboxes_and_heat.png
[image17]: ./examples/labels_map.png
[image18]: ./examples/output_bboxes.png
[video1]: ./project_video_out.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained [cells 1 through 4](https://github.com/johnybang/CarND-Vehicle-Detection/blob/master/VehicleDetection.ipynb#Prepare-Training-Image-Lists) of the jupyter notebook.  

In cell 1, I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like, but ultimately decided to do a parameter sweep using linear SVM classification performance as the figure of merit.

In cell 4, I plotted a visualization of the `RGB` and `YUV` and the extraction/visualization of HOG features with parameterization of `orientations=12`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

Since it showed top results in my parameter sweep, I ultimately used a concatenation of hog features for all three channels of the `YUV` color space. If I were to speculate from the visualization why `YUV` may be better than `RGB`, on thing that I might argue is that since I'm using all three channels of HOG features, it may be beneficial to the classifier that the channels show some independence from one another.  In the visualization above, `R`, `G`, and `B` HOG feature images look quite similar to one another, whereas the `Y`, `U` and `V` HOG feature images look a little bit more distinct from each other.

#### 2. Explain how you settled on your final choice of HOG parameters.

Using `itertools.product()`, I tried various combinations of parameters on a class-balanced 2250 sample subset of the full dataset over the course of a multi-hour automated test. The results of that experiment are at the end of this document. I then took the top ten from that test and ran them on the full dataset, yielding the results shown here. I chose the best outcome (in bold) out of these, sorted by test accuracy, prediction speed, and training speed. Here they are:

|   cell\_per_block | colorspace   | hog_channel   |   n_predict |   nfeatures |   orient |   pix\_per_cell |   predict_secs |   test_accy |   train_secs |
|-----------------:|:-------------|:--------------|------------:|------------:|---------:|---------------:|---------------:|------------:|-------------:|
|                **2** | **YUV**          | **ALL**           |          10 |        7056 |       **12** |              **8** |        0.00296 |      0.9909 |         3.73 |
|                2 | YCrCb        | ALL           |          10 |       32400 |       12 |              4 |        0.00268 |      0.99   |        15.64 |
|                3 | YCrCb        | ALL           |          10 |       14580 |       15 |              8 |        0.00276 |      0.9875 |         7.58 |
|                2 | YUV          | ALL           |          10 |        1296 |       12 |             16 |        0.00242 |      0.9855 |         2.16 |
|                2 | YUV          | ALL           |          10 |        3528 |        6 |              8 |        0.00265 |      0.9846 |         9.65 |
|                2 | YCrCb        | ALL           |          10 |        1620 |       15 |             16 |        0.00254 |      0.9835 |         2.68 |
|                3 | YUV          | ALL           |          10 |       31752 |        6 |              4 |        0.00299 |      0.9832 |        18.67 |
|                3 | YCrCb        | ALL           |          10 |        1458 |        6 |             12 |        0.0026  |      0.9809 |         2.9  |
|                2 | YUV          | ALL           |          10 |         972 |        9 |             16 |        0.0024  |      0.9801 |         1.52 |
|                2 | LUV          | ALL           |          10 |        1152 |        6 |             12 |        0.00255 |      0.9733 |         2.36 |

I also chose to include spatial and color histogram features. I used the top parameter set from a similar experiment. Here are the top 10 candidates of which I used the best YUV candidate (in bold):

| colorspace   |   histbin |   n_predict |   nfeatures |   predict_secs |   spatial |   test_accy |   train_secs |
|:-------------|----------:|------------:|------------:|---------------:|----------:|------------:|-------------:|
| YCrCb        |        36 |          10 |         540 |        0.00266 |        12 |      0.9741 |         4.81 |
| HSV          |        40 |          10 |         552 |        0.00273 |        12 |      0.9738 |         4.18 |
| HSV          |        36 |          10 |         300 |        0.00243 |         8 |      0.9727 |         4.37 |
| **YUV**          |        **28** |          10 |         276 |        0.00258 |         **8** |      0.9727 |         3.9  |
| YUV          |        36 |          10 |         540 |        0.00403 |        12 |      0.9727 |         4.95 |
| YCrCb        |        32 |          10 |         288 |        0.00204 |         8 |      0.9724 |         3.36 |
| YUV          |        32 |          10 |         288 |        0.00261 |         8 |      0.9724 |         3.55 |
| YUV          |        40 |          10 |         552 |        0.00263 |        12 |      0.9724 |         5.06 |
| YUV          |        40 |          10 |         312 |        0.00195 |         8 |      0.9721 |         3.41 |
| LUV          |        40 |          10 |         552 |        0.00263 |        12 |      0.9721 |         5.29 |


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

In cell 3, after extracting the HOG features I split the feature vectors into a 80/20 train/test split, and created a scaler which standarized each training feature.  I then trained a linear SVM classifier which I subsequently tested with the test data scaled in the same way.  After that, I run training again on the entire dataset (no train/test split) to be used by the vehicle detection system.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

In cells 5 and 6, I implemented a sliding window search. I used "hog-subsampling" which extracts hog features for the entire image which can then be sliced according to the window desired. This saves unnecessary repeated computations that would take place due to window overlap if hog features were extracted per window. One acceptable but resultant constraint of this approach is that the possible window overlaps have cell granularity rather than pixel granularity. My windowing was influenced by the following motivations:

* Don't search the sky
* Use smaller scale for vehicles further away
* Don't search the hood of the car

My final choices were determined by looking at the identified windows on test images and seeking to balance vehicle detection with false positives. I used two groups of windows:

* ystart=360, ystop=500, scale=1.0, cells\_per\_step=1
* ystart=400, ystop=600, scale=1.5, cells\_per\_step=1

And here is a visualization of the search grid:

![alt text][image3]


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YUV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a decent though admittedly imperfect result. (Though, I'm still motivated to pursue further improvements in the future as time permits.) I allowed some false positives knowing these could be filtered at the heatmapping stage. I originally searched the entire lower half of the image with a couple scales but constrained it to what I mentioned above when I found excessive falsing. I also tried thresholding the decision funciton on a value other than zero; however, I found that the score information was best kept as soft information going into the heatmapping stage explained in the next section.  I suspect this could be further optimized in the future. Here are some example images:

![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]
![alt text][image9]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
I favored avoidance of false positives over detection of the car at distance, so that it would be a smooth result. Swerving to avoid a false obstacle seemed more important than knowledge of a car a further distance away. In the future, I would like to pursue a strategy where I apply a dense detection grid around the likely future location of an already detected car, taking inspiration from the margin around the polynomials for the advanced lane finding project.
Here's a [link to my video result](https://github.com/johnybang/CarND-Vehicle-Detection/blob/master/project_video_out.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the bounding boxes of positive detections in each frame of the video.  From the positive detections I created a heatmap, in code cell 7, which accumulated classifier confidences scores (`svc.decision_function()`).  I did this instead of the overlap counting method to preserve the soft confidence information; this led to better performance.  I then thresholded that map to identify vehicle positions.  I used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap and constructed bounding boxes to cover the area of each blob (presumed vehicle) detected.  

Here's an example result showing the heatmap of several test images and their corresponding bounding boxes then overlaid:

![alt text][image10]
![alt text][image11]
![alt text][image12]
![alt text][image13]
![alt text][image14]
![alt text][image15]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

This project proved to be a rich problem with many interesting challenges.  Here are some thoughts of note:

* I found that my first attempt at the video produced very jittery bounding boxes and more false positives than I wanted. I turned at first to frame-wise exponential smoothing of the heatmap and found that the boxes lagged the car position a bit; I scaled back the time constant from 0.75 to 0.67 to increase responsiveness while still preserving its false positive filtering effect.
* I still found the false positive (and false negative) unsatisfactory on the video so I decided to play more with scale. I had originally scanned the entire lower part of the image with a couple scales with modest (50%) overlap.  I found that increasing overlap and reducing coverage, even varying coverage with scale helped improve both aspects of performance.
* I made an attempt at hard negative mining without initial success.  Since most of the bounding boxes on the left were false positives in the video, I was able to filter by bounding box position to extract the hard negative images.  However, when I introduced the new data, the performance degraded.  I suspect I had a bug and will have to revisit hard neg mining in the future.
* As mentioned above, I think the system could benefit from using knowledge of an already detected car's trajectory. My system seems pretty robust to close range cars but starts to fail as cars get further away. In the future I'd like to apply a dense grid to likely future car locations according to existing cars.  I also believe that my window scale groups could be even further optimized, but much further experimentation is required.
* The 0.67 exponential averaging constant could still pose a problem if a car were particularly fast-moving relative to the camera. Further robustness on each frame should be pursued further.
* Overall, the many components from the SVM to color feature design to hog feature design to sliding window design to heatmapping design could all benefit from their own deep dives and literature searches.


### Appendix Tables

#### Color feature experiment:

| colorspace   |   histbin |   n_predict |   nfeatures |   predict_secs |   spatial |   test_accy |   train_secs |
|:-------------|----------:|------------:|------------:|---------------:|----------:|------------:|-------------:|
| YCrCb        |        36 |          10 |         540 |        0.00266 |        12 |      0.9741 |         4.81 |
| HSV          |        40 |          10 |         552 |        0.00273 |        12 |      0.9738 |         4.18 |
| HSV          |        36 |          10 |         300 |        0.00243 |         8 |      0.9727 |         4.37 |
| YUV          |        28 |          10 |         276 |        0.00258 |         8 |      0.9727 |         3.9  |
| YUV          |        36 |          10 |         540 |        0.00403 |        12 |      0.9727 |         4.95 |
| YCrCb        |        32 |          10 |         288 |        0.00204 |         8 |      0.9724 |         3.36 |
| YUV          |        32 |          10 |         288 |        0.00261 |         8 |      0.9724 |         3.55 |
| YUV          |        40 |          10 |         552 |        0.00263 |        12 |      0.9724 |         5.06 |
| YUV          |        40 |          10 |         312 |        0.00195 |         8 |      0.9721 |         3.41 |
| LUV          |        40 |          10 |         552 |        0.00263 |        12 |      0.9721 |         5.29 |
| LUV          |        32 |          10 |         288 |        0.00221 |         8 |      0.9707 |         4.28 |
| YCrCb        |        36 |          10 |         300 |        0.00261 |         8 |      0.9704 |         4.12 |
| LUV          |        32 |          10 |         528 |        0.00263 |        12 |      0.9704 |         6.26 |
| YCrCb        |        24 |          10 |         504 |        0.0026  |        12 |      0.9702 |         6.57 |
| YCrCb        |        20 |          10 |         252 |        0.00254 |         8 |      0.9699 |         3.15 |
| YCrCb        |        32 |          10 |         528 |        0.00257 |        12 |      0.9696 |         5.5  |
| YCrCb        |        40 |          10 |         312 |        0.00264 |         8 |      0.9696 |         3.29 |
| LUV          |        40 |          10 |         312 |        0.00284 |         8 |      0.9693 |         4.19 |
| LUV          |        28 |          10 |         276 |        0.00207 |         8 |      0.969  |         4.36 |
| YUV          |        36 |          10 |         300 |        0.00417 |         8 |      0.969  |         3.79 |
| YCrCb        |        24 |          10 |         264 |        0.00232 |         8 |      0.9688 |         3.32 |
| YCrCb        |        28 |          10 |         516 |        0.00268 |        12 |      0.9688 |         5.52 |
| HSV          |        32 |          10 |         528 |        0.00272 |        12 |      0.9688 |         5.2  |
| HLS          |        32 |          10 |         288 |        0.00235 |         8 |      0.9685 |         5.21 |
| LUV          |        28 |          10 |         516 |        0.00263 |        12 |      0.9682 |         5.72 |
| YCrCb        |        20 |          10 |         492 |        0.00255 |        12 |      0.9679 |         5.45 |
| YUV          |        20 |          10 |         492 |        0.00263 |        12 |      0.9679 |         6.15 |
| HSV          |        28 |          10 |         516 |        0.0027  |        12 |      0.9676 |         5.43 |
| HSV          |        24 |          10 |         264 |        0.00263 |         8 |      0.9673 |         4.36 |
| YUV          |        28 |          10 |         516 |        0.00409 |        12 |      0.9673 |         5.32 |
| LUV          |        36 |          10 |         540 |        0.00291 |        12 |      0.9668 |         5.36 |
| LUV          |        36 |          10 |         300 |        0.00217 |         8 |      0.9665 |         4    |
| YUV          |        24 |          10 |         264 |        0.00234 |         8 |      0.9665 |         3.37 |
| YCrCb        |        40 |          10 |         552 |        0.00241 |        12 |      0.9665 |         4.9  |
| YUV          |        32 |          10 |         528 |        0.00261 |        12 |      0.9662 |         4.51 |
| HLS          |        36 |          10 |         300 |        0.00267 |         8 |      0.9662 |         4.81 |
| LUV          |        20 |          10 |         492 |        0.00215 |        12 |      0.9659 |         6.36 |
| HSV          |        28 |          10 |         276 |        0.00271 |         8 |      0.9657 |         3.88 |
| LUV          |        24 |          10 |         264 |        0.00205 |         8 |      0.9654 |         4.39 |
| RGB          |        36 |          10 |         300 |        0.0026  |         8 |      0.9654 |         5.75 |
| HSV          |        24 |          10 |         504 |        0.00269 |        12 |      0.9654 |         5.21 |
| RGB          |        40 |          10 |         312 |        0.00271 |         8 |      0.9654 |         5.82 |
| HSV          |        32 |          10 |         288 |        0.00271 |         8 |      0.9651 |         3.81 |
| LUV          |        24 |          10 |         504 |        0.00201 |        12 |      0.9648 |         6.24 |
| HSV          |        20 |          10 |         492 |        0.00208 |        12 |      0.9648 |         6.99 |
| YCrCb        |        32 |          10 |         864 |        0.00264 |        16 |      0.9648 |         6.15 |
| RGB          |        40 |          10 |         552 |        0.00262 |        12 |      0.9645 |         6.52 |
| LUV          |        20 |          10 |         252 |        0.00281 |         8 |      0.9645 |         4.29 |
| RGB          |        24 |          10 |         840 |        0.00266 |        16 |      0.9642 |        10.6  |
| YCrCb        |        36 |          10 |         876 |        0.0016  |        16 |      0.964  |         5.55 |
| YUV          |        20 |          10 |         252 |        0.00437 |         8 |      0.964  |         3.42 |
| RGB          |        40 |          10 |         888 |        0.00264 |        16 |      0.9637 |         8.12 |
| RGB          |        32 |          10 |         528 |        0.00261 |        12 |      0.9634 |         6.89 |
| YUV          |        24 |          10 |         504 |        0.0028  |        12 |      0.9634 |         6.27 |
| HSV          |        36 |          10 |         540 |        0.00263 |        12 |      0.9631 |         4.1  |
| YCrCb        |        40 |          10 |         888 |        0.00419 |        16 |      0.9628 |         4.98 |
| HLS          |        40 |          10 |         552 |        0.00253 |        12 |      0.9626 |         5.24 |
| HLS          |        24 |          10 |         504 |        0.00258 |        12 |      0.9626 |         7.8  |
| RGB          |        36 |          10 |        1308 |        0.00258 |        20 |      0.9626 |        11.88 |
| RGB          |        36 |          10 |         540 |        0.00262 |        12 |      0.9623 |         6.55 |
| RGB          |        32 |          10 |         288 |        0.00268 |         8 |      0.962  |         5.29 |
| LUV          |        24 |          10 |         840 |        0.0027  |        16 |      0.962  |         8.11 |
| HSV          |        40 |          10 |         312 |        0.00266 |         8 |      0.9617 |         4.89 |
| HLS          |        32 |          10 |         528 |        0.00257 |        12 |      0.9614 |         5.88 |
| YCrCb        |        28 |          10 |         276 |        0.00259 |         8 |      0.9614 |         3.2  |
| HLS          |        28 |          10 |         516 |        0.00261 |        12 |      0.9611 |         7.39 |
| RGB          |        28 |          10 |         516 |        0.00268 |        12 |      0.9611 |         7.77 |
| HLS          |        40 |          10 |         312 |        0.00242 |         8 |      0.9609 |         4.49 |
| YUV          |        32 |          10 |         864 |        0.00256 |        16 |      0.9606 |         5.3  |
| YUV          |        40 |          10 |         888 |        0.00263 |        16 |      0.9606 |         5.53 |
| HLS          |        24 |          10 |         264 |        0.00254 |         8 |      0.9603 |         4.86 |
| RGB          |        28 |          10 |         852 |        0.00285 |        16 |      0.96   |         9.92 |
| YCrCb        |        28 |          10 |         852 |        0.00297 |        16 |      0.96   |         7.06 |
| LUV          |        36 |          10 |         876 |        0.00255 |        16 |      0.9597 |         5.66 |
| LUV          |        32 |          10 |         864 |        0.00258 |        16 |      0.9595 |         6.58 |
| RGB          |        32 |          10 |         864 |        0.00261 |        16 |      0.9595 |         9.56 |
| RGB          |        20 |          10 |         828 |        0.0026  |        16 |      0.9592 |        11.32 |
| RGB          |        32 |          10 |        1296 |        0.0026  |        20 |      0.9592 |        12.08 |
| RGB          |        40 |          10 |        3192 |        0.00217 |        32 |      0.9589 |        25.48 |
| RGB          |        24 |          10 |         264 |        0.00258 |         8 |      0.9589 |         5.53 |
| YUV          |        28 |          10 |         852 |        0.0026  |        16 |      0.9589 |         6.3  |
| HLS          |        28 |          10 |         276 |        0.00261 |         8 |      0.9589 |         4.56 |
| RGB          |        36 |          10 |         876 |        0.00261 |        16 |      0.9589 |         8.86 |
| HSV          |        20 |          10 |         252 |        0.00263 |         8 |      0.9589 |         4.58 |
| RGB          |        28 |          10 |         276 |        0.00271 |         8 |      0.9589 |         5.32 |
| RGB          |        24 |          10 |        1272 |        0.00288 |        20 |      0.9583 |        13.14 |
| HSV          |        36 |          10 |        3180 |        0.00264 |        32 |      0.9581 |        19.39 |
| LUV          |        36 |          10 |        1308 |        0.00475 |        20 |      0.9575 |         8.52 |
| YUV          |        24 |          10 |         840 |        0.00256 |        16 |      0.9572 |         6.61 |
| RGB          |        24 |          10 |         504 |        0.00265 |        12 |      0.9572 |         6.82 |
| YUV          |        36 |          10 |        3180 |        0.00256 |        32 |      0.9569 |        21.16 |
| HSV          |        36 |          10 |        1308 |        0.00281 |        20 |      0.9569 |         5.79 |
| RGB          |        20 |          10 |         492 |        0.00255 |        12 |      0.9566 |         7.68 |
| YUV          |        40 |          10 |        2472 |        0.00236 |        28 |      0.9564 |        14.8  |
| RGB          |        36 |          10 |        1836 |        0.00242 |        24 |      0.9564 |        16.01 |
| YUV          |        20 |          10 |         828 |        0.00257 |        16 |      0.9564 |         7.92 |
| RGB          |        32 |          10 |        3168 |        0.00244 |        32 |      0.9561 |        25.44 |
| RGB          |        40 |          10 |        1848 |        0.00252 |        24 |      0.9561 |        14.9  |
| HLS          |        20 |          10 |         252 |        0.00258 |         8 |      0.9561 |         5.02 |
| RGB          |        28 |          10 |        1284 |        0.0026  |        20 |      0.9561 |        12.38 |
| HSV          |        24 |          10 |         840 |        0.00261 |        16 |      0.9561 |         4.61 |
| LUV          |        40 |          10 |         888 |        0.00262 |        16 |      0.9561 |         5.79 |
| YUV          |        36 |          10 |         876 |        0.00263 |        16 |      0.9561 |         5.28 |
| HSV          |        40 |          10 |         888 |        0.00269 |        16 |      0.9561 |         4.66 |
| YUV          |        32 |          10 |        2448 |        0.00243 |        28 |      0.9558 |        15.89 |
| HSV          |        28 |          10 |         852 |        0.00273 |        16 |      0.9558 |         4.49 |
| YUV          |        32 |          10 |        1296 |        0.00291 |        20 |      0.9558 |         7.11 |
| HLS          |        36 |          10 |         540 |        0.0026  |        12 |      0.9555 |         5.29 |
| YUV          |        40 |          10 |        1320 |        0.00261 |        20 |      0.9555 |         6.48 |
| YUV          |        32 |          10 |        3168 |        0.00271 |        32 |      0.9552 |        20.27 |
| YCrCb        |        24 |          10 |         840 |        0.00276 |        16 |      0.9552 |         7.33 |
| YUV          |        36 |          10 |        1308 |        0.00287 |        20 |      0.9552 |         6.93 |
| YCrCb        |        36 |          10 |        1308 |        0.00204 |        20 |      0.955  |         7.16 |
| HSV          |        36 |          10 |        1836 |        0.00238 |        24 |      0.955  |         9.5  |
| YUV          |        40 |          10 |        1848 |        0.00231 |        24 |      0.9547 |        10.24 |
| RGB          |        40 |          10 |        1320 |        0.00255 |        20 |      0.9547 |        11.21 |
| YUV          |        36 |          10 |        2460 |        0.00273 |        28 |      0.9547 |        16.2  |
| YCrCb        |        40 |          10 |        1848 |        0.00274 |        24 |      0.9547 |        10.79 |
| YCrCb        |        20 |          10 |         828 |        0.0028  |        16 |      0.9547 |         7.19 |
| RGB          |        28 |          10 |        1812 |        0.00233 |        24 |      0.9544 |        17.03 |
| YUV          |        36 |          10 |        1836 |        0.00248 |        24 |      0.9541 |        10.37 |
| LUV          |        40 |          10 |        3192 |        0.0029  |        32 |      0.9541 |        20.88 |
| YCrCb        |        32 |          10 |        1296 |        0.00222 |        20 |      0.9538 |         7.71 |
| HLS          |        28 |          10 |         852 |        0.00259 |        16 |      0.9538 |         6.7  |
| YCrCb        |        28 |          10 |        1812 |        0.00279 |        24 |      0.9535 |        11.69 |
| LUV          |        36 |          10 |        1836 |        0.00297 |        24 |      0.9535 |        12.36 |
| YCrCb        |        28 |          10 |        1284 |        0.00262 |        20 |      0.9533 |         8.66 |
| HSV          |        32 |          10 |        1824 |        0.00257 |        24 |      0.953  |         9.28 |
| HSV          |        32 |          10 |         864 |        0.00266 |        16 |      0.953  |         4.48 |
| LUV          |        40 |          10 |        2472 |        0.00312 |        28 |      0.953  |        15.97 |
| YUV          |        32 |          10 |        1824 |        0.00273 |        24 |      0.9527 |        10.78 |
| RGB          |        32 |          10 |        1824 |        0.00238 |        24 |      0.9524 |        16.01 |
| HSV          |        36 |          10 |         876 |        0.00276 |        16 |      0.9524 |         4.04 |
| HSV          |        32 |          10 |        3168 |        0.00284 |        32 |      0.9524 |        19.18 |
| LUV          |        20 |          10 |         828 |        0.00265 |        16 |      0.9521 |         8.71 |
| LUV          |        40 |          10 |        1848 |        0.00281 |        24 |      0.9521 |        11.28 |
| RGB          |        28 |          10 |        3156 |        0.00319 |        32 |      0.9521 |        27.42 |
| HSV          |        24 |          10 |        1800 |        0.00249 |        24 |      0.9519 |         9.71 |
| YCrCb        |        36 |          10 |        3180 |        0.00261 |        32 |      0.9519 |        21.06 |
| HSV          |        24 |          10 |        3144 |        0.00294 |        32 |      0.9519 |        19.21 |
| YCrCb        |        40 |          10 |        2472 |        0.00305 |        28 |      0.9519 |        15.67 |
| HLS          |        40 |          10 |        3192 |        0.00518 |        32 |      0.9519 |        20.04 |
| RGB          |        32 |          10 |        2448 |        0.00267 |        28 |      0.9516 |        20.83 |
| YCrCb        |        32 |          10 |        1824 |        0.00211 |        24 |      0.9513 |        11.64 |
| LUV          |        28 |          10 |         852 |        0.0026  |        16 |      0.9513 |         6.09 |
| YCrCb        |        32 |          10 |        3168 |        0.00387 |        32 |      0.9513 |        22.37 |
| HSV          |        28 |          10 |        1812 |        0.00253 |        24 |      0.951  |         9.55 |
| YUV          |        40 |          10 |        3192 |        0.00259 |        32 |      0.951  |        20.03 |
| HSV          |        28 |          10 |        3156 |        0.00207 |        32 |      0.9507 |        18.42 |
| RGB          |        40 |          10 |        2472 |        0.00257 |        28 |      0.9507 |        18.15 |
| LUV          |        32 |          10 |        3168 |        0.00262 |        32 |      0.9507 |        22.94 |
| RGB          |        20 |          10 |        1260 |        0.00263 |        20 |      0.9507 |        14.78 |
| HLS          |        20 |          10 |         828 |        0.00264 |        16 |      0.9507 |         9.2  |
| YCrCb        |        36 |          10 |        1836 |        0.0028  |        24 |      0.9507 |        10.74 |
| HSV          |        40 |          10 |        1320 |        0.00287 |        20 |      0.9507 |         5.65 |
| HSV          |        40 |          10 |        3192 |        0.00205 |        32 |      0.9505 |        18.34 |
| YUV          |        28 |          10 |        1812 |        0.00213 |        24 |      0.9505 |        11.17 |
| LUV          |        28 |          10 |        1284 |        0.00227 |        20 |      0.9505 |         8.8  |
| YCrCb        |        24 |          10 |        1800 |        0.0021  |        24 |      0.9502 |        12.51 |
| HLS          |        40 |          10 |         888 |        0.00258 |        16 |      0.9502 |         4.8  |
| YCrCb        |        20 |          10 |        1260 |        0.00268 |        20 |      0.9502 |         9.01 |
| HLS          |        32 |          10 |         864 |        0.00265 |        16 |      0.9499 |         5.19 |
| HSV          |        36 |          10 |        2460 |        0.00216 |        28 |      0.9496 |        14.15 |
| RGB          |        24 |          10 |        1800 |        0.0025  |        24 |      0.9496 |        18.79 |
| YUV          |        28 |          10 |        2436 |        0.00216 |        28 |      0.9493 |        15.21 |
| RGB          |        20 |          10 |        3132 |        0.00237 |        32 |      0.9493 |        29.43 |
| RGB          |        24 |          10 |        3144 |        0.00256 |        32 |      0.9493 |        26.58 |
| HLS          |        24 |          10 |         840 |        0.00261 |        16 |      0.9493 |         6.92 |
| RGB          |        36 |          10 |        3180 |        0.01171 |        32 |      0.9493 |        26.53 |
| YCrCb        |        32 |          10 |        2448 |        0.00208 |        28 |      0.949  |        16.49 |
| HLS          |        20 |          10 |         492 |        0.00263 |        12 |      0.949  |         7.5  |
| HSV          |        20 |          10 |         828 |        0.00269 |        16 |      0.949  |         6.36 |
| LUV          |        36 |          10 |        3180 |        0.00281 |        32 |      0.949  |        22.04 |
| HLS          |        36 |          10 |        1836 |        0.00287 |        24 |      0.949  |         9.15 |
| LUV          |        24 |          10 |        1272 |        0.00207 |        20 |      0.9488 |         8.66 |
| YCrCb        |        28 |          10 |        2436 |        0.00224 |        28 |      0.9488 |        16.66 |
| RGB          |        24 |          10 |        2424 |        0.00255 |        28 |      0.9488 |        21.88 |
| YUV          |        28 |          10 |        3156 |        0.00256 |        32 |      0.9488 |        21.37 |
| LUV          |        32 |          10 |        1296 |        0.00249 |        20 |      0.9485 |         8.56 |
| YUV          |        24 |          10 |        2424 |        0.00264 |        28 |      0.9485 |        16.22 |
| HSV          |        32 |          10 |        1296 |        0.00269 |        20 |      0.9485 |         5.98 |
| LUV          |        32 |          10 |        2448 |        0.00404 |        28 |      0.9485 |        17.19 |
| HSV          |        40 |          10 |        2472 |        0.00205 |        28 |      0.9482 |        13.57 |
| RGB          |        20 |          10 |        1788 |        0.00267 |        24 |      0.9482 |        18.68 |
| HSV          |        20 |          10 |        3132 |        0.00279 |        32 |      0.9479 |        21.02 |
| HLS          |        36 |          10 |        1308 |        0.00259 |        20 |      0.9476 |         6.12 |
| YCrCb        |        40 |          10 |        3192 |        0.00264 |        32 |      0.9476 |        21.19 |
| YUV          |        20 |          10 |        3132 |        0.00212 |        32 |      0.9474 |        23.84 |
| YCrCb        |        36 |          10 |        2460 |        0.00223 |        28 |      0.9474 |        15.15 |
| HLS          |        36 |          10 |        2460 |        0.00244 |        28 |      0.9474 |        14.71 |
| YCrCb        |        40 |          10 |        1320 |        0.00506 |        20 |      0.9474 |         6.85 |
| YUV          |        24 |          10 |        3144 |        0.00202 |        32 |      0.9471 |        21.82 |
| LUV          |        24 |          10 |        3144 |        0.00211 |        32 |      0.9471 |        24.19 |
| LUV          |        32 |          10 |        1824 |        0.00261 |        24 |      0.9471 |        11.81 |
| RGB          |        28 |          10 |        2436 |        0.00259 |        28 |      0.9468 |        19.72 |
| HSV          |        28 |          10 |        1284 |        0.0027  |        20 |      0.9468 |         6.2  |
| LUV          |        40 |          10 |        1320 |        0.00249 |        20 |      0.9465 |         7.7  |
| YCrCb        |        20 |          10 |        1788 |        0.00283 |        24 |      0.9465 |        12.5  |
| HLS          |        32 |          10 |        2448 |        0.0029  |        28 |      0.9462 |        14.35 |
| HSV          |        40 |          10 |        1848 |        0.00267 |        24 |      0.9459 |         9.39 |
| LUV          |        28 |          10 |        3156 |        0.00302 |        32 |      0.9459 |        23.81 |
| HSV          |        20 |          10 |        1788 |        0.00243 |        24 |      0.9457 |        10.21 |
| YCrCb        |        24 |          10 |        2424 |        0.00246 |        28 |      0.9457 |        17.29 |
| RGB          |        36 |          10 |        2460 |        0.00261 |        28 |      0.9457 |        19.41 |
| YUV          |        20 |          10 |        1260 |        0.00268 |        20 |      0.9457 |         9.77 |
| HSV          |        32 |          10 |        2448 |        0.00243 |        28 |      0.9454 |        14.11 |
| HSV          |        24 |          10 |        2424 |        0.00259 |        28 |      0.9454 |        14.49 |
| YCrCb        |        20 |          10 |        3132 |        0.00205 |        32 |      0.9451 |        24.21 |
| YCrCb        |        28 |          10 |        3156 |        0.0024  |        32 |      0.9451 |        23.24 |
| YUV          |        24 |          10 |        1272 |        0.00261 |        20 |      0.9451 |         8.31 |
| YCrCb        |        24 |          10 |        1272 |        0.00264 |        20 |      0.9448 |         8    |
| HLS          |        36 |          10 |         876 |        0.00253 |        16 |      0.9445 |         4.56 |
| LUV          |        20 |          10 |        2412 |        0.00263 |        28 |      0.9443 |        18.81 |
| RGB          |        20 |          10 |        2412 |        0.00266 |        28 |      0.9443 |        23.15 |
| HLS          |        36 |          10 |        3180 |        0.00293 |        32 |      0.944  |        19.8  |
| LUV          |        24 |          10 |        2424 |        0.00271 |        28 |      0.9437 |        17.62 |
| LUV          |        20 |          10 |        3132 |        0.00248 |        32 |      0.9434 |        25.86 |
| HSV          |        24 |          10 |        1272 |        0.00261 |        20 |      0.9434 |         6.32 |
| YUV          |        20 |          10 |        1788 |        0.00288 |        24 |      0.9434 |        12.02 |
| HLS          |        40 |          10 |        1848 |        0.00252 |        24 |      0.9431 |         9.3  |
| HSV          |        20 |          10 |        2412 |        0.00295 |        28 |      0.9431 |        15.31 |
| YCrCb        |        24 |          10 |        3144 |        0.009   |        32 |      0.9428 |        23.6  |
| YUV          |        20 |          10 |        2412 |        0.00268 |        28 |      0.9426 |        17.27 |
| HLS          |        24 |          10 |        3144 |        0.00285 |        32 |      0.9426 |        21.36 |
| LUV          |        20 |          10 |        1788 |        0.00259 |        24 |      0.9423 |        13.44 |
| LUV          |        24 |          10 |        1800 |        0.00295 |        24 |      0.9423 |        13.69 |
| LUV          |        36 |          10 |        2460 |        0.00216 |        28 |      0.942  |        15.75 |
| YUV          |        24 |          10 |        1800 |        0.00244 |        24 |      0.942  |        11.51 |
| LUV          |        28 |          10 |        2436 |        0.00258 |        28 |      0.9414 |        17.1  |
| RGB          |        20 |          10 |         252 |        0.00261 |         8 |      0.9414 |         6.11 |
| HLS          |        32 |          10 |        3168 |        0.00291 |        32 |      0.9414 |        19.89 |
| HLS          |        32 |          10 |        1824 |        0.00262 |        24 |      0.9406 |         9.55 |
| LUV          |        20 |          10 |        1260 |        0.00245 |        20 |      0.94   |         8.65 |
| HLS          |        40 |          10 |        2472 |        0.00288 |        28 |      0.94   |        14.55 |
| HLS          |        40 |          10 |        1320 |        0.00262 |        20 |      0.9395 |         5.98 |
| YUV          |        28 |          10 |        1284 |        0.00266 |        20 |      0.9392 |         7.38 |
| LUV          |        28 |          10 |        1812 |        0.00256 |        24 |      0.9389 |        11.76 |
| HLS          |        28 |          10 |        3156 |        0.00277 |        32 |      0.9389 |        20.09 |
| YCrCb        |        20 |          10 |        2412 |        0.00441 |        28 |      0.9389 |        18.5  |
| HSV          |        20 |          10 |        1260 |        0.00242 |        20 |      0.9381 |         6.94 |
| HLS          |        32 |          10 |        1296 |        0.00252 |        20 |      0.9381 |         5.98 |
| HLS          |        24 |          10 |        2424 |        0.00254 |        28 |      0.9375 |        15.01 |
| HLS          |        24 |          10 |        1800 |        0.00255 |        24 |      0.9372 |        10.46 |
| HLS          |        28 |          10 |        1812 |        0.00245 |        24 |      0.9367 |         9.44 |
| HSV          |        28 |          10 |        2436 |        0.00231 |        28 |      0.9361 |        13.75 |
| HLS          |        28 |          10 |        1284 |        0.00262 |        20 |      0.935  |         6.83 |
| HLS          |        28 |          10 |        2436 |        0.00255 |        28 |      0.9336 |        15.34 |
| HLS          |        24 |          10 |        1272 |        0.00259 |        20 |      0.9327 |         7.29 |
| HLS          |        20 |          10 |        1788 |        0.00252 |        24 |      0.9324 |        10.28 |
| HLS          |        20 |          10 |        1260 |        0.00259 |        20 |      0.9313 |         7.94 |
| HLS          |        20 |          10 |        3132 |        0.00259 |        32 |      0.9296 |        21.45 |
| HLS          |        20 |          10 |        2412 |        0.00259 |        28 |      0.9178 |        16.63 |


#### HOG feature experiment:


|   cell\_per_block | colorspace   | hog_channel   |   n_predict |   nfeatures |   orient |   pix\_per_cell |   predict_secs |   test_accy |   train_secs |
|-----------------:|:-------------|:--------------|------------:|------------:|---------:|---------------:|---------------:|------------:|-------------:|
|                2 | YUV          | ALL           |          10 |        1296 |       12 |             16 |        0.00237 |      1      |         0.07 |
|                3 | YUV          | ALL           |          10 |       31752 |        6 |              4 |        0.003   |      1      |         2.32 |
|                2 | YCrCb        | ALL           |          10 |        1620 |       15 |             16 |        0.00179 |      0.9978 |         0.11 |
|                2 | LUV          | ALL           |          10 |        1152 |        6 |             12 |        0.0019  |      0.9978 |         0.09 |
|                3 | YCrCb        | ALL           |          10 |        1458 |        6 |             12 |        0.00192 |      0.9978 |         0.11 |
|                2 | YCrCb        | ALL           |          10 |       32400 |       12 |              4 |        0.0021  |      0.9978 |         2.12 |
|                3 | YCrCb        | ALL           |          10 |       14580 |       15 |              8 |        0.00235 |      0.9978 |         1.02 |
|                2 | YUV          | ALL           |          10 |        7056 |       12 |              8 |        0.00239 |      0.9978 |         0.43 |
|                2 | YUV          | ALL           |          10 |         972 |        9 |             16 |        0.0025  |      0.9978 |         0.08 |
|                2 | YUV          | ALL           |          10 |        3528 |        6 |              8 |        0.0025  |      0.9978 |         0.24 |
|                3 | YCrCb        | ALL           |          10 |         972 |        9 |             16 |        0.00254 |      0.9978 |         0.07 |
|                3 | YUV          | ALL           |          10 |        8748 |        9 |              8 |        0.0026  |      0.9978 |         0.58 |
|                1 | HLS          | ALL           |          10 |        9216 |       12 |              4 |        0.00264 |      0.9978 |         0.59 |
|                2 | YUV          | ALL           |          10 |        2880 |       15 |             12 |        0.00268 |      0.9978 |         0.15 |
|                3 | LUV          | ALL           |          10 |       11664 |       12 |              8 |        0.0027  |      0.9978 |         0.83 |
|                4 | YUV          | ALL           |          10 |       10800 |        9 |              8 |        0.0029  |      0.9978 |         0.87 |
|                2 | YUV          | ALL           |          10 |       24300 |        9 |              4 |        0.00292 |      0.9978 |         1.68 |
|                2 | HLS          | ALL           |          10 |       40500 |       15 |              4 |        0.00309 |      0.9978 |         3.13 |
|                3 | YCrCb        | ALL           |          10 |       47628 |        9 |              4 |        0.00315 |      0.9978 |         4.25 |
|                4 | YCrCb        | ALL           |          10 |       48672 |        6 |              4 |        0.00324 |      0.9978 |         3.69 |
|                3 | YUV          | ALL           |          10 |       47628 |        9 |              4 |        0.00325 |      0.9978 |         3.27 |
|                3 | HSV          | ALL           |          10 |       63504 |       12 |              4 |        0.00344 |      0.9978 |         4.83 |
|                3 | YCrCb        | ALL           |          10 |       79380 |       15 |              4 |        0.00353 |      0.9978 |         6.28 |
|                3 | YUV          | ALL           |          10 |        3645 |       15 |             12 |        0.00455 |      0.9978 |         0.22 |
|                3 | LUV          | ALL           |          10 |       79380 |       15 |              4 |        0.00656 |      0.9978 |         6.12 |
|                3 | YUV          | ALL           |          10 |        1620 |       15 |             16 |        0.002   |      0.9956 |         0.1  |
|                2 | HLS          | ALL           |          10 |        2304 |       12 |             12 |        0.00201 |      0.9956 |         0.17 |
|                4 | LUV          | ALL           |          10 |        7200 |        6 |              8 |        0.00206 |      0.9956 |         0.61 |
|                4 | YUV          | ALL           |          10 |        1152 |        6 |             12 |        0.00211 |      0.9956 |         0.08 |
|                2 | YUV          | ALL           |          10 |        8820 |       15 |              8 |        0.00212 |      0.9956 |         0.58 |
|                2 | YUV          | ALL           |          10 |       16200 |        6 |              4 |        0.00226 |      0.9956 |         1.07 |
|                4 | YCrCb        | ALL           |          10 |         288 |        6 |             16 |        0.0023  |      0.9956 |         0.03 |
|                3 | LUV          | ALL           |          10 |        5832 |        6 |              8 |        0.00246 |      0.9956 |         0.51 |
|                4 | LUV          | ALL           |          10 |         576 |       12 |             16 |        0.00249 |      0.9956 |         0.04 |
|                2 | YUV          | ALL           |          10 |         648 |        6 |             16 |        0.00251 |      0.9956 |         0.05 |
|                4 | YUV          | ALL           |          10 |        2304 |       12 |             12 |        0.00252 |      0.9956 |         0.16 |
|                2 | HLS          | ALL           |          10 |        8820 |       15 |              8 |        0.00255 |      0.9956 |         0.57 |
|                4 | YCrCb        | ALL           |          10 |         720 |       15 |             16 |        0.00257 |      0.9956 |         0.05 |
|                1 | YUV          | ALL           |          10 |       11520 |       15 |              4 |        0.0027  |      0.9956 |         0.76 |
|                2 | YUV          | ALL           |          10 |        5292 |        9 |              8 |        0.00273 |      0.9956 |         0.32 |
|                4 | YCrCb        | ALL           |          10 |       10800 |        9 |              8 |        0.00276 |      0.9956 |         0.83 |
|                3 | HSV          | 0             |          10 |       21168 |       12 |              4 |        0.00278 |      0.9956 |         1.45 |
|                3 | YCrCb        | ALL           |          10 |       31752 |        6 |              4 |        0.00285 |      0.9956 |         2.55 |
|                3 | LUV          | ALL           |          10 |       47628 |        9 |              4 |        0.00309 |      0.9956 |         4.86 |
|                4 | HLS          | ALL           |          10 |       48672 |        6 |              4 |        0.0032  |      0.9956 |         4.1  |
|                3 | LUV          | ALL           |          10 |       63504 |       12 |              4 |        0.00332 |      0.9956 |         5.46 |
|                3 | HLS          | ALL           |          10 |       63504 |       12 |              4 |        0.00339 |      0.9956 |         4.73 |
|                4 | HLS          | ALL           |          10 |       97344 |       12 |              4 |        0.00381 |      0.9956 |         7.34 |
|                4 | HSV          | ALL           |          10 |       97344 |       12 |              4 |        0.00384 |      0.9956 |         7.3  |
|                4 | LUV          | ALL           |          10 |       97344 |       12 |              4 |        0.00393 |      0.9956 |         7.55 |
|                2 | HLS          | ALL           |          10 |       24300 |        9 |              4 |        0.0056  |      0.9956 |         2.16 |
|                3 | YCrCb        | ALL           |          10 |       11664 |       12 |              8 |        0.002   |      0.9933 |         0.78 |
|                4 | LUV          | ALL           |          10 |       10800 |        9 |              8 |        0.00211 |      0.9933 |         0.9  |
|                1 | HLS          | ALL           |          10 |        1125 |       15 |             12 |        0.00226 |      0.9933 |         0.06 |
|                3 | YUV          | ALL           |          10 |       14580 |       15 |              8 |        0.00231 |      0.9933 |         0.98 |
|                1 | YCrCb        | ALL           |          10 |        2880 |       15 |              8 |        0.00235 |      0.9933 |         0.14 |
|                4 | YUV          | ALL           |          10 |       18000 |       15 |              8 |        0.00235 |      0.9933 |         1.3  |
|                4 | YUV          | ALL           |          10 |        7200 |        6 |              8 |        0.00239 |      0.9933 |         0.61 |
|                1 | YCrCb        | ALL           |          10 |        1152 |        6 |              8 |        0.00243 |      0.9933 |         0.06 |
|                2 | HSV          | ALL           |          10 |        8820 |       15 |              8 |        0.00244 |      0.9933 |         0.63 |
|                2 | HSV          | ALL           |          10 |        7056 |       12 |              8 |        0.00245 |      0.9933 |         0.47 |
|                3 | YUV          | ALL           |          10 |        1458 |        6 |             12 |        0.00249 |      0.9933 |         0.1  |
|                1 | YUV          | ALL           |          10 |         432 |        9 |             16 |        0.0025  |      0.9933 |         0.03 |
|                1 | YUV          | ALL           |          10 |        6912 |        9 |              4 |        0.00252 |      0.9933 |         0.45 |
|                2 | YCrCb        | ALL           |          10 |        1296 |       12 |             16 |        0.00254 |      0.9933 |         0.07 |
|                2 | LUV          | ALL           |          10 |        7056 |       12 |              8 |        0.00254 |      0.9933 |         0.51 |
|                2 | YCrCb        | ALL           |          10 |        1152 |        6 |             12 |        0.00256 |      0.9933 |         0.07 |
|                1 | YUV          | ALL           |          10 |        2304 |       12 |              8 |        0.00257 |      0.9933 |         0.11 |
|                2 | HLS          | ALL           |          10 |        3528 |        6 |              8 |        0.0026  |      0.9933 |         0.29 |
|                1 | HLS          | ALL           |          10 |        6912 |        9 |              4 |        0.0026  |      0.9933 |         0.55 |
|                4 | YCrCb        | ALL           |          10 |        2880 |       15 |             12 |        0.00261 |      0.9933 |         0.16 |
|                4 | LUV          | ALL           |          10 |        2304 |       12 |             12 |        0.00264 |      0.9933 |         0.15 |
|                4 | YCrCb        | ALL           |          10 |        1152 |        6 |             12 |        0.00265 |      0.9933 |         0.1  |
|                1 | YCrCb        | ALL           |          10 |        9216 |       12 |              4 |        0.00265 |      0.9933 |         0.63 |
|                2 | YUV          | ALL           |          10 |        2304 |       12 |             12 |        0.00268 |      0.9933 |         0.13 |
|                3 | HLS          | ALL           |          10 |       31752 |        6 |              4 |        0.00268 |      0.9933 |         2.55 |
|                1 | LUV          | ALL           |          10 |        1125 |       15 |             12 |        0.00274 |      0.9933 |         0.06 |
|                1 | LUV          | ALL           |          10 |        2880 |       15 |              8 |        0.00274 |      0.9933 |         0.25 |
|                4 | YUV          | ALL           |          10 |       14400 |       12 |              8 |        0.00274 |      0.9933 |         1.14 |
|                2 | YCrCb        | ALL           |          10 |       24300 |        9 |              4 |        0.00274 |      0.9933 |         2.28 |
|                3 | LUV          | ALL           |          10 |       14580 |       15 |              8 |        0.00289 |      0.9933 |         1.17 |
|                2 | HSV          | ALL           |          10 |       24300 |        9 |              4 |        0.00289 |      0.9933 |         2.12 |
|                3 | YUV          | ALL           |          10 |        2916 |       12 |             12 |        0.0029  |      0.9933 |         0.2  |
|                2 | LUV          | ALL           |          10 |       32400 |       12 |              4 |        0.0029  |      0.9933 |         3.47 |
|                4 | HSV          | ALL           |          10 |       14400 |       12 |              8 |        0.00293 |      0.9933 |         1.29 |
|                2 | YCrCb        | ALL           |          10 |       40500 |       15 |              4 |        0.00319 |      0.9933 |         3.28 |
|                3 | HSV          | ALL           |          10 |       47628 |        9 |              4 |        0.00322 |      0.9933 |         4.05 |
|                3 | YUV          | ALL           |          10 |       63504 |       12 |              4 |        0.00347 |      0.9933 |         4.31 |
|                4 | YCrCb        | ALL           |          10 |       73008 |        9 |              4 |        0.0035  |      0.9933 |         5.19 |
|                4 | LUV          | ALL           |          10 |       73008 |        9 |              4 |        0.00353 |      0.9933 |         5.96 |
|                3 | YCrCb        | ALL           |          10 |        1296 |       12 |             16 |        0.00415 |      0.9933 |         0.07 |
|                4 | YCrCb        | ALL           |          10 |      121680 |       15 |              4 |        0.00417 |      0.9933 |         8.38 |
|                3 | YCrCb        | ALL           |          10 |        8748 |        9 |              8 |        0.00457 |      0.9933 |         0.69 |
|                1 | HLS          | ALL           |          10 |        1728 |        9 |              8 |        0.00535 |      0.9933 |         0.1  |
|                1 | HLS          | ALL           |          10 |        4608 |        6 |              4 |        0.01479 |      0.9933 |         0.28 |
|                4 | YCrCb        | ALL           |          10 |         432 |        9 |             16 |        0.00153 |      0.9911 |         0.04 |
|                4 | YUV          | ALL           |          10 |         720 |       15 |             16 |        0.00181 |      0.9911 |         0.05 |
|                2 | YCrCb        | ALL           |          10 |        3528 |        6 |              8 |        0.00187 |      0.9911 |         0.24 |
|                3 | HSV          | ALL           |          10 |        5832 |        6 |              8 |        0.00189 |      0.9911 |         0.44 |
|                3 | YUV          | ALL           |          10 |         972 |        9 |             16 |        0.0019  |      0.9911 |         0.07 |
|                4 | LUV          | ALL           |          10 |         720 |       15 |             16 |        0.00193 |      0.9911 |         0.04 |
|                4 | YCrCb        | 1             |          10 |         960 |       15 |             12 |        0.00199 |      0.9911 |         0.07 |
|                2 | LUV          | ALL           |          10 |        8820 |       15 |              8 |        0.00206 |      0.9911 |         0.66 |
|                1 | YCrCb        | ALL           |          10 |        6912 |        9 |              4 |        0.00209 |      0.9911 |         0.54 |
|                3 | HLS          | ALL           |          10 |        8748 |        9 |              8 |        0.00212 |      0.9911 |         0.74 |
|                1 | HSV          | ALL           |          10 |       11520 |       15 |              4 |        0.00216 |      0.9911 |         0.78 |
|                2 | YCrCb        | ALL           |          10 |       16200 |        6 |              4 |        0.00241 |      0.9911 |         1.26 |
|                3 | LUV          | ALL           |          10 |        2187 |        9 |             12 |        0.00245 |      0.9911 |         0.15 |
|                3 | HSV          | 0             |          10 |        3888 |       12 |              8 |        0.00245 |      0.9911 |         0.26 |
|                3 | YCrCb        | ALL           |          10 |        1620 |       15 |             16 |        0.00246 |      0.9911 |         0.1  |
|                1 | HSV          | ALL           |          10 |        4608 |        6 |              4 |        0.00247 |      0.9911 |         0.27 |
|                4 | YUV          | ALL           |          10 |         288 |        6 |             16 |        0.00249 |      0.9911 |         0.03 |
|                1 | YUV          | ALL           |          10 |        4608 |        6 |              4 |        0.00249 |      0.9911 |         0.26 |
|                3 | YUV          | 1             |          10 |        3888 |       12 |              8 |        0.00249 |      0.9911 |         0.39 |
|                1 | HSV          | ALL           |          10 |        2304 |       12 |              8 |        0.0025  |      0.9911 |         0.12 |
|                3 | HLS          | ALL           |          10 |        1620 |       15 |             16 |        0.0025  |      0.9911 |         0.15 |
|                4 | YUV          | 1             |          10 |       32448 |       12 |              4 |        0.0025  |      0.9911 |         3.73 |
|                1 | LUV          | ALL           |          10 |         900 |       12 |             12 |        0.00251 |      0.9911 |         0.05 |
|                2 | YUV          | ALL           |          10 |        1152 |        6 |             12 |        0.00252 |      0.9911 |         0.08 |
|                1 | HLS          | ALL           |          10 |         720 |       15 |             16 |        0.00253 |      0.9911 |         0.05 |
|                1 | HSV          | ALL           |          10 |        9216 |       12 |              4 |        0.00253 |      0.9911 |         0.61 |
|                2 | HSV          | ALL           |          10 |        3528 |        6 |              8 |        0.00254 |      0.9911 |         0.26 |
|                1 | HSV          | ALL           |          10 |        2880 |       15 |              8 |        0.00256 |      0.9911 |         0.16 |
|                4 | YCrCb        | ALL           |          10 |         576 |       12 |             16 |        0.00257 |      0.9911 |         0.03 |
|                3 | YUV          | ALL           |          10 |        2187 |        9 |             12 |        0.00257 |      0.9911 |         0.16 |
|                1 | LUV          | ALL           |          10 |       11520 |       15 |              4 |        0.00258 |      0.9911 |         0.88 |
|                2 | LUV          | ALL           |          10 |        1620 |       15 |             16 |        0.00259 |      0.9911 |         0.1  |
|                2 | LUV          | ALL           |          10 |         648 |        6 |             16 |        0.00263 |      0.9911 |         0.07 |
|                4 | YCrCb        | ALL           |          10 |        7200 |        6 |              8 |        0.00263 |      0.9911 |         0.56 |
|                3 | YCrCb        | 1             |          10 |         972 |       12 |             12 |        0.00265 |      0.9911 |         0.32 |
|                3 | YUV          | ALL           |          10 |       11664 |       12 |              8 |        0.00266 |      0.9911 |         0.79 |
|                3 | YCrCb        | ALL           |          10 |        3645 |       15 |             12 |        0.00267 |      0.9911 |         0.21 |
|                3 | YCrCb        | ALL           |          10 |         648 |        6 |             16 |        0.00268 |      0.9911 |         0.07 |
|                1 | YCrCb        | ALL           |          10 |       11520 |       15 |              4 |        0.00271 |      0.9911 |         0.79 |
|                2 | HSV          | ALL           |          10 |       40500 |       15 |              4 |        0.00274 |      0.9911 |         3.18 |
|                2 | YCrCb        | ALL           |          10 |        2304 |       12 |             12 |        0.00275 |      0.9911 |         0.15 |
|                3 | HLS          | ALL           |          10 |       14580 |       15 |              8 |        0.00275 |      0.9911 |         1.12 |
|                4 | YCrCb        | ALL           |          10 |       18000 |       15 |              8 |        0.00276 |      0.9911 |         1.43 |
|                3 | YUV          | 1             |          10 |       26460 |       15 |              4 |        0.00287 |      0.9911 |         2.58 |
|                2 | HSV          | ALL           |          10 |       32400 |       12 |              4 |        0.00289 |      0.9911 |         2.54 |
|                3 | HLS          | ALL           |          10 |       47628 |        9 |              4 |        0.00291 |      0.9911 |         3.76 |
|                2 | HLS          | ALL           |          10 |       32400 |       12 |              4 |        0.00294 |      0.9911 |         2.23 |
|                3 | YUV          | ALL           |          10 |       79380 |       15 |              4 |        0.00312 |      0.9911 |         5.45 |
|                4 | YUV          | ALL           |          10 |       48672 |        6 |              4 |        0.00318 |      0.9911 |         3.66 |
|                2 | HLS          | ALL           |          10 |       16200 |        6 |              4 |        0.00341 |      0.9911 |         1.2  |
|                3 | YCrCb        | ALL           |          10 |       63504 |       12 |              4 |        0.00349 |      0.9911 |         4.26 |
|                2 | YCrCb        | ALL           |          10 |         648 |        6 |             16 |        0.00353 |      0.9911 |         0.06 |
|                3 | HSV          | ALL           |          10 |       79380 |       15 |              4 |        0.00356 |      0.9911 |         6.15 |
|                3 | HLS          | ALL           |          10 |       79380 |       15 |              4 |        0.00361 |      0.9911 |         5.94 |
|                4 | YCrCb        | ALL           |          10 |       97344 |       12 |              4 |        0.0038  |      0.9911 |         6.93 |
|                4 | LUV          | ALL           |          10 |      121680 |       15 |              4 |        0.00401 |      0.9911 |         9.09 |
|                4 | LUV          | ALL           |          10 |       18000 |       15 |              8 |        0.00463 |      0.9911 |         1.45 |
|                4 | LUV          | 1             |          10 |        4800 |       12 |              8 |        0.00145 |      0.9889 |         0.34 |
|                3 | YCrCb        | 1             |          10 |        1215 |       15 |             12 |        0.00181 |      0.9889 |         0.09 |
|                1 | HSV          | ALL           |          10 |         900 |       12 |             12 |        0.00182 |      0.9889 |         0.04 |
|                1 | YCrCb        | ALL           |          10 |        2304 |       12 |              8 |        0.00186 |      0.9889 |         0.11 |
|                1 | LUV          | ALL           |          10 |         720 |       15 |             16 |        0.00191 |      0.9889 |         0.04 |
|                2 | YUV          | ALL           |          10 |        1620 |       15 |             16 |        0.00193 |      0.9889 |         0.1  |
|                4 | LUV          | ALL           |          10 |         432 |        9 |             16 |        0.00195 |      0.9889 |         0.04 |
|                1 | YUV          | ALL           |          10 |         900 |       12 |             12 |        0.00195 |      0.9889 |         0.04 |
|                4 | YCrCb        | 1             |          10 |         192 |       12 |             16 |        0.00199 |      0.9889 |         0.02 |
|                1 | YUV          | ALL           |          10 |        9216 |       12 |              4 |        0.00208 |      0.9889 |         0.6  |
|                4 | HLS          | ALL           |          10 |         720 |       15 |             16 |        0.00217 |      0.9889 |         0.06 |
|                2 | HLS          | 0             |          10 |       10800 |       12 |              4 |        0.00217 |      0.9889 |         0.69 |
|                2 | YCrCb        | 1             |          10 |        2352 |       12 |              8 |        0.00235 |      0.9889 |         0.16 |
|                1 | YCrCb        | ALL           |          10 |         720 |       15 |             16 |        0.00242 |      0.9889 |         0.04 |
|                1 | YCrCb        | ALL           |          10 |        1125 |       15 |             12 |        0.00242 |      0.9889 |         0.05 |
|                1 | YUV          | ALL           |          10 |         450 |        6 |             12 |        0.00243 |      0.9889 |         0.03 |
|                2 | LUV          | ALL           |          10 |        1296 |       12 |             16 |        0.00245 |      0.9889 |         0.08 |
|                1 | LUV          | ALL           |          10 |        9216 |       12 |              4 |        0.00245 |      0.9889 |         0.87 |
|                1 | YCrCb        | ALL           |          10 |         900 |       12 |             12 |        0.00247 |      0.9889 |         0.04 |
|                2 | HLS          | ALL           |          10 |        1296 |       12 |             16 |        0.00247 |      0.9889 |         0.11 |
|                2 | HSV          | ALL           |          10 |        2304 |       12 |             12 |        0.00249 |      0.9889 |         0.16 |
|                1 | LUV          | ALL           |          10 |        6912 |        9 |              4 |        0.00249 |      0.9889 |         0.58 |
|                2 | HLS          | ALL           |          10 |        1620 |       15 |             16 |        0.0025  |      0.9889 |         0.11 |
|                1 | YCrCb        | ALL           |          10 |         450 |        6 |             12 |        0.00251 |      0.9889 |         0.03 |
|                2 | YUV          | ALL           |          10 |        1728 |        9 |             12 |        0.00251 |      0.9889 |         0.1  |
|                3 | LUV          | ALL           |          10 |         972 |        9 |             16 |        0.00252 |      0.9889 |         0.07 |
|                1 | YUV          | ALL           |          10 |        1728 |        9 |              8 |        0.00252 |      0.9889 |         0.09 |
|                4 | YCrCb        | ALL           |          10 |        1728 |        9 |             12 |        0.00252 |      0.9889 |         0.13 |
|                1 | LUV          | ALL           |          10 |        1728 |        9 |              8 |        0.00253 |      0.9889 |         0.1  |
|                2 | HLS          | ALL           |          10 |        7056 |       12 |              8 |        0.00254 |      0.9889 |         0.47 |
|                3 | LUV          | ALL           |          10 |        8748 |        9 |              8 |        0.00254 |      0.9889 |         0.72 |
|                4 | YUV          | 2             |          10 |       16224 |        6 |              4 |        0.00265 |      0.9889 |         1.35 |
|                3 | HSV          | ALL           |          10 |       14580 |       15 |              8 |        0.00266 |      0.9889 |         1.19 |
|                1 | YUV          | ALL           |          10 |        2880 |       15 |              8 |        0.00269 |      0.9889 |         0.15 |
|                4 | YCrCb        | ALL           |          10 |       14400 |       12 |              8 |        0.00269 |      0.9889 |         1.07 |
|                4 | HSV          | ALL           |          10 |       18000 |       15 |              8 |        0.00272 |      0.9889 |         1.62 |
|                2 | LUV          | ALL           |          10 |       16200 |        6 |              4 |        0.00274 |      0.9889 |         1.45 |
|                3 | HSV          | 0             |          10 |       26460 |       15 |              4 |        0.00283 |      0.9889 |         2    |
|                3 | HLS          | 0             |          10 |       26460 |       15 |              4 |        0.00284 |      0.9889 |         1.99 |
|                3 | LUV          | ALL           |          10 |       31752 |        6 |              4 |        0.003   |      0.9889 |         2.25 |
|                4 | HSV          | ALL           |          10 |      121680 |       15 |              4 |        0.00365 |      0.9889 |        10    |
|                4 | HSV          | ALL           |          10 |        7200 |        6 |              8 |        0.00373 |      0.9889 |         0.69 |
|                3 | LUV          | ALL           |          10 |        3645 |       15 |             12 |        0.00395 |      0.9889 |         0.28 |
|                4 | YUV          | ALL           |          10 |      121680 |       15 |              4 |        0.00409 |      0.9889 |         8.28 |
|                3 | YCrCb        | ALL           |          10 |        5832 |        6 |              8 |        0.0054  |      0.9889 |         0.41 |
|                4 | YCrCb        | ALL           |          10 |        2304 |       12 |             12 |        0.0067  |      0.9889 |         0.14 |
|                2 | LUV          | ALL           |          10 |        2880 |       15 |             12 |        0.0016  |      0.9867 |         0.2  |
|                2 | YCrCb        | 1             |          10 |         768 |       12 |             12 |        0.00174 |      0.9867 |         0.06 |
|                4 | YUV          | ALL           |          10 |         576 |       12 |             16 |        0.00196 |      0.9867 |         0.04 |
|                1 | YUV          | ALL           |          10 |         288 |        6 |             16 |        0.00217 |      0.9867 |         0.03 |
|                2 | LUV          | 1             |          10 |       13500 |       15 |              4 |        0.00227 |      0.9867 |         1.54 |
|                3 | LUV          | ALL           |          10 |        1620 |       15 |             16 |        0.00233 |      0.9867 |         0.12 |
|                3 | YUV          | ALL           |          10 |        5832 |        6 |              8 |        0.00235 |      0.9867 |         0.47 |
|                4 | YCrCb        | 1             |          10 |        4800 |       12 |              8 |        0.00236 |      0.9867 |         2.76 |
|                1 | YCrCb        | 1             |          10 |         375 |       15 |             12 |        0.00237 |      0.9867 |         0.02 |
|                3 | YUV          | 1             |          10 |         432 |       12 |             16 |        0.00237 |      0.9867 |         0.12 |
|                1 | YCrCb        | ALL           |          10 |         288 |        6 |             16 |        0.00239 |      0.9867 |         0.03 |
|                3 | LUV          | 1             |          10 |       21168 |       12 |              4 |        0.0024  |      0.9867 |         2.15 |
|                1 | HSV          | ALL           |          10 |         576 |       12 |             16 |        0.00243 |      0.9867 |         0.04 |
|                1 | YUV          | ALL           |          10 |         576 |       12 |             16 |        0.00244 |      0.9867 |         0.03 |
|                4 | LUV          | 1             |          10 |        3600 |        9 |              8 |        0.00247 |      0.9867 |         0.33 |
|                1 | HSV          | ALL           |          10 |         720 |       15 |             16 |        0.00249 |      0.9867 |         0.05 |
|                2 | HSV          | ALL           |          10 |        1296 |       12 |             16 |        0.0025  |      0.9867 |         0.1  |
|                4 | YUV          | 1             |          10 |         960 |       15 |             12 |        0.00251 |      0.9867 |         0.07 |
|                1 | HLS          | ALL           |          10 |        1152 |        6 |              8 |        0.00251 |      0.9867 |         0.08 |
|                1 | YCrCb        | ALL           |          10 |        1728 |        9 |              8 |        0.00251 |      0.9867 |         0.09 |
|                1 | HLS          | ALL           |          10 |        2880 |       15 |              8 |        0.00251 |      0.9867 |         0.16 |
|                4 | LUV          | 1             |          10 |       32448 |       12 |              4 |        0.00251 |      0.9867 |         2.97 |
|                4 | HSV          | ALL           |          10 |        2304 |       12 |             12 |        0.00252 |      0.9867 |         0.25 |
|                1 | YCrCb        | ALL           |          10 |         576 |       12 |             16 |        0.00253 |      0.9867 |         0.03 |
|                3 | HSV          | ALL           |          10 |       31752 |        6 |              4 |        0.00253 |      0.9867 |         2.39 |
|                1 | YUV          | ALL           |          10 |        1152 |        6 |              8 |        0.00254 |      0.9867 |         0.06 |
|                1 | YUV          | 1             |          10 |        3072 |       12 |              4 |        0.00254 |      0.9867 |         0.59 |
|                2 | LUV          | ALL           |          10 |        2304 |       12 |             12 |        0.00256 |      0.9867 |         0.13 |
|                2 | HSV          | ALL           |          10 |        2880 |       15 |             12 |        0.00257 |      0.9867 |         0.19 |
|                1 | HLS          | ALL           |          10 |       11520 |       15 |              4 |        0.00257 |      0.9867 |         0.86 |
|                2 | YCrCb        | ALL           |          10 |        2880 |       15 |             12 |        0.00259 |      0.9867 |         0.18 |
|                1 | HSV          | ALL           |          10 |        6912 |        9 |              4 |        0.0026  |      0.9867 |         0.48 |
|                3 | HLS          | ALL           |          10 |        2916 |       12 |             12 |        0.00261 |      0.9867 |         0.22 |
|                1 | LUV          | ALL           |          10 |        2304 |       12 |              8 |        0.00269 |      0.9867 |         0.16 |
|                2 | YCrCb        | ALL           |          10 |        7056 |       12 |              8 |        0.0027  |      0.9867 |         0.46 |
|                3 | HSV          | ALL           |          10 |        2916 |       12 |             12 |        0.00272 |      0.9867 |         0.27 |
|                1 | HLS          | ALL           |          10 |        2304 |       12 |              8 |        0.0028  |      0.9867 |         0.13 |
|                4 | LUV          | ALL           |          10 |       14400 |       12 |              8 |        0.0028  |      0.9867 |         1.11 |
|                2 | HSV          | ALL           |          10 |       16200 |        6 |              4 |        0.00283 |      0.9867 |         1.15 |
|                1 | HLS          | 0             |          10 |        3840 |       15 |              4 |        0.00297 |      0.9867 |         0.25 |
|                2 | LUV          | ALL           |          10 |       40500 |       15 |              4 |        0.00308 |      0.9867 |         4.95 |
|                4 | YUV          | 1             |          10 |       40560 |       15 |              4 |        0.00313 |      0.9867 |         2.8  |
|                4 | YUV          | 2             |          10 |       40560 |       15 |              4 |        0.00315 |      0.9867 |         3.09 |
|                3 | RGB          | ALL           |          10 |       47628 |        9 |              4 |        0.00327 |      0.9867 |         5.13 |
|                4 | LUV          | ALL           |          10 |        1152 |        6 |             12 |        0.00338 |      0.9867 |         0.09 |
|                4 | YUV          | ALL           |          10 |       73008 |        9 |              4 |        0.00364 |      0.9867 |         5.72 |
|                4 | YUV          | ALL           |          10 |       97344 |       12 |              4 |        0.00392 |      0.9867 |         6.44 |
|                2 | YCrCb        | ALL           |          10 |        5292 |        9 |              8 |        0.00597 |      0.9867 |         0.33 |
|                3 | LUV          | 1             |          10 |        3888 |       12 |              8 |        0.01466 |      0.9867 |         0.28 |
|                2 | YUV          | 1             |          10 |        2940 |       15 |              8 |        0.00105 |      0.9844 |         0.16 |
|                1 | LUV          | ALL           |          10 |         675 |        9 |             12 |        0.00163 |      0.9844 |         0.05 |
|                4 | LUV          | ALL           |          10 |         288 |        6 |             16 |        0.0018  |      0.9844 |         0.03 |
|                2 | LUV          | ALL           |          10 |        5292 |        9 |              8 |        0.00185 |      0.9844 |         0.34 |
|                1 | YCrCb        | ALL           |          10 |        4608 |        6 |              4 |        0.00195 |      0.9844 |         0.29 |
|                2 | YCrCb        | ALL           |          10 |        8820 |       15 |              8 |        0.00197 |      0.9844 |         0.53 |
|                1 | HLS          | ALL           |          10 |         576 |       12 |             16 |        0.00198 |      0.9844 |         0.04 |
|                4 | YCrCb        | 1             |          10 |        6000 |       15 |              8 |        0.00202 |      0.9844 |         0.4  |
|                1 | LUV          | ALL           |          10 |        1152 |        6 |              8 |        0.00232 |      0.9844 |         0.09 |
|                4 | HLS          | ALL           |          10 |       18000 |       15 |              8 |        0.00232 |      0.9844 |         1.54 |
|                2 | HSV          | ALL           |          10 |         648 |        6 |             16 |        0.00239 |      0.9844 |         0.06 |
|                4 | YUV          | ALL           |          10 |         432 |        9 |             16 |        0.00241 |      0.9844 |         0.04 |
|                1 | HLS          | ALL           |          10 |         900 |       12 |             12 |        0.00241 |      0.9844 |         0.05 |
|                1 | YUV          | ALL           |          10 |         675 |        9 |             12 |        0.00243 |      0.9844 |         0.04 |
|                3 | YUV          | ALL           |          10 |        1296 |       12 |             16 |        0.00244 |      0.9844 |         0.06 |
|                1 | YUV          | ALL           |          10 |        1125 |       15 |             12 |        0.00245 |      0.9844 |         0.06 |
|                2 | LUV          | ALL           |          10 |         972 |        9 |             16 |        0.00246 |      0.9844 |         0.07 |
|                1 | HSV          | ALL           |          10 |         288 |        6 |             16 |        0.00247 |      0.9844 |         0.05 |
|                2 | YUV          | 1             |          10 |         576 |        9 |             12 |        0.00247 |      0.9844 |         0.08 |
|                4 | YUV          | 1             |          10 |         768 |       12 |             12 |        0.00247 |      0.9844 |         0.27 |
|                1 | HSV          | ALL           |          10 |        1152 |        6 |              8 |        0.00248 |      0.9844 |         0.05 |
|                1 | HLS          | ALL           |          10 |         432 |        9 |             16 |        0.0025  |      0.9844 |         0.05 |
|                3 | YUV          | ALL           |          10 |         648 |        6 |             16 |        0.0025  |      0.9844 |         0.06 |
|                2 | HLS          | ALL           |          10 |         972 |        9 |             16 |        0.0025  |      0.9844 |         0.1  |
|                4 | YUV          | 1             |          10 |        2400 |        6 |              8 |        0.0025  |      0.9844 |         1.07 |
|                4 | LUV          | ALL           |          10 |        2880 |       15 |             12 |        0.00251 |      0.9844 |         0.2  |
|                2 | LUV          | ALL           |          10 |        3528 |        6 |              8 |        0.00253 |      0.9844 |         0.25 |
|                4 | LUV          | 1             |          10 |         768 |       12 |             12 |        0.00253 |      0.9844 |         0.3  |
|                1 | HSV          | 0             |          10 |        3072 |       12 |              4 |        0.00255 |      0.9844 |         0.16 |
|                3 | LUV          | ALL           |          10 |        1458 |        6 |             12 |        0.00257 |      0.9844 |         0.11 |
|                2 | YUV          | ALL           |          10 |       32400 |       12 |              4 |        0.00259 |      0.9844 |         2.21 |
|                3 | LUV          | ALL           |          10 |        2916 |       12 |             12 |        0.0026  |      0.9844 |         0.22 |
|                4 | HLS          | ALL           |          10 |       10800 |        9 |              8 |        0.00263 |      0.9844 |         1.06 |
|                3 | HSV          | ALL           |          10 |        1296 |       12 |             16 |        0.00265 |      0.9844 |         0.11 |
|                2 | HSV          | 0             |          10 |       10800 |       12 |              4 |        0.00266 |      0.9844 |         0.73 |
|                4 | HLS          | ALL           |          10 |       14400 |       12 |              8 |        0.00266 |      0.9844 |         1.26 |
|                3 | HLS          | ALL           |          10 |       11664 |       12 |              8 |        0.00268 |      0.9844 |         0.91 |
|                2 | LUV          | 1             |          10 |        2940 |       15 |              8 |        0.0027  |      0.9844 |         0.19 |
|                3 | YCrCb        | 1             |          10 |       15876 |        9 |              4 |        0.00271 |      0.9844 |         1.6  |
|                1 | HLS          | 0             |          10 |        3072 |       12 |              4 |        0.00273 |      0.9844 |         0.15 |
|                4 | HSV          | 0             |          10 |       40560 |       15 |              4 |        0.00273 |      0.9844 |         3.5  |
|                2 | LUV          | ALL           |          10 |       24300 |        9 |              4 |        0.00284 |      0.9844 |         1.86 |
|                3 | HSV          | 0             |          10 |       10584 |        6 |              4 |        0.00286 |      0.9844 |         0.75 |
|                3 | HLS          | 0             |          10 |       21168 |       12 |              4 |        0.00295 |      0.9844 |         1.42 |
|                4 | YUV          | 1             |          10 |       24336 |        9 |              4 |        0.00296 |      0.9844 |         2.13 |
|                4 | LUV          | ALL           |          10 |       48672 |        6 |              4 |        0.00337 |      0.9844 |         3.81 |
|                4 | HLS          | ALL           |          10 |      121680 |       15 |              4 |        0.0042  |      0.9844 |         9.26 |
|                3 | YUV          | 1             |          10 |         540 |       15 |             16 |        0.00152 |      0.9822 |         0.06 |
|                2 | HLS          | ALL           |          10 |        2880 |       15 |             12 |        0.00155 |      0.9822 |         0.18 |
|                2 | YCrCb        | ALL           |          10 |        1728 |        9 |             12 |        0.00157 |      0.9822 |         0.12 |
|                1 | YCrCb        | 1             |          10 |         576 |        9 |              8 |        0.00178 |      0.9822 |         0.04 |
|                4 | HSV          | ALL           |          10 |         720 |       15 |             16 |        0.00181 |      0.9822 |         0.06 |
|                1 | YCrCb        | 1             |          10 |        3840 |       15 |              4 |        0.00193 |      0.9822 |         0.6  |
|                1 | HSV          | ALL           |          10 |         450 |        6 |             12 |        0.00197 |      0.9822 |         0.04 |
|                2 | HSV          | ALL           |          10 |        1152 |        6 |             12 |        0.00199 |      0.9822 |         0.08 |
|                4 | YCrCb        | 1             |          10 |         768 |       12 |             12 |        0.00201 |      0.9822 |         0.25 |
|                3 | HLS          | ALL           |          10 |        5832 |        6 |              8 |        0.00201 |      0.9822 |         0.48 |
|                2 | LUV          | ALL           |          10 |        1728 |        9 |             12 |        0.00225 |      0.9822 |         0.12 |
|                3 | YUV          | 1             |          10 |       15876 |        9 |              4 |        0.00237 |      0.9822 |         1.19 |
|                1 | LUV          | ALL           |          10 |         576 |       12 |             16 |        0.00239 |      0.9822 |         0.03 |
|                2 | YCrCb        | 1             |          10 |         432 |       12 |             16 |        0.0024  |      0.9822 |         0.04 |
|                3 | YUV          | 1             |          10 |        2916 |        9 |              8 |        0.00241 |      0.9822 |         0.33 |
|                2 | YUV          | 1             |          10 |        1764 |        9 |              8 |        0.00242 |      0.9822 |         0.1  |
|                1 | HSV          | ALL           |          10 |         432 |        9 |             16 |        0.00244 |      0.9822 |         0.04 |
|                4 | LUV          | ALL           |          10 |        1728 |        9 |             12 |        0.00246 |      0.9822 |         0.14 |
|                2 | YUV          | 0             |          10 |        5400 |        6 |              4 |        0.00246 |      0.9822 |         0.58 |
|                1 | HSV          | ALL           |          10 |        1125 |       15 |             12 |        0.00249 |      0.9822 |         0.06 |
|                2 | HSV          | 0             |          10 |        2940 |       15 |              8 |        0.00249 |      0.9822 |         0.19 |
|                4 | YUV          | 1             |          10 |        4800 |       12 |              8 |        0.0025  |      0.9822 |         2.42 |
|                1 | HSV          | 0             |          10 |        1536 |        6 |              4 |        0.00251 |      0.9822 |         0.09 |
|                2 | YCrCb        | ALL           |          10 |         972 |        9 |             16 |        0.00252 |      0.9822 |         0.06 |
|                2 | LUV          | 2             |          10 |         768 |       12 |             12 |        0.00252 |      0.9822 |         0.1  |
|                1 | YUV          | 0             |          10 |        2304 |        9 |              4 |        0.00252 |      0.9822 |         0.17 |
|                3 | LUV          | 1             |          10 |        4860 |       15 |              8 |        0.00252 |      0.9822 |         0.43 |
|                3 | YUV          | 1             |          10 |        1944 |        6 |              8 |        0.00255 |      0.9822 |         0.15 |
|                1 | HLS          | 0             |          10 |         768 |       12 |              8 |        0.00256 |      0.9822 |         0.04 |
|                1 | YCrCb        | ALL           |          10 |         675 |        9 |             12 |        0.00256 |      0.9822 |         0.04 |
|                3 | YUV          | 1             |          10 |         729 |        9 |             12 |        0.00256 |      0.9822 |         0.07 |
|                2 | HSV          | ALL           |          10 |        5292 |        9 |              8 |        0.00259 |      0.9822 |         0.37 |
|                3 | YCrCb        | ALL           |          10 |        2187 |        9 |             12 |        0.00264 |      0.9822 |         0.15 |
|                2 | YCrCb        | 2             |          10 |       13500 |       15 |              4 |        0.00268 |      0.9822 |         1.05 |
|                4 | HLS          | ALL           |          10 |        2880 |       15 |             12 |        0.00269 |      0.9822 |         0.23 |
|                3 | LUV          | 1             |          10 |        1215 |       15 |             12 |        0.00271 |      0.9822 |         0.12 |
|                4 | HLS          | ALL           |          10 |        7200 |        6 |              8 |        0.00272 |      0.9822 |         0.71 |
|                4 | HSV          | ALL           |          10 |       48672 |        6 |              4 |        0.00272 |      0.9822 |         3.68 |
|                3 | YCrCb        | ALL           |          10 |        2916 |       12 |             12 |        0.00274 |      0.9822 |         0.18 |
|                4 | YUV          | 1             |          10 |       16224 |        6 |              4 |        0.00274 |      0.9822 |         1.45 |
|                2 | YCrCb        | 1             |          10 |       13500 |       15 |              4 |        0.00275 |      0.9822 |         1.04 |
|                3 | YUV          | 1             |          10 |       21168 |       12 |              4 |        0.00286 |      0.9822 |         1.73 |
|                3 | YCrCb        | 2             |          10 |       21168 |       12 |              4 |        0.00288 |      0.9822 |         1.52 |
|                4 | YUV          | 2             |          10 |       32448 |       12 |              4 |        0.00294 |      0.9822 |         2.39 |
|                3 | YUV          | 2             |          10 |       26460 |       15 |              4 |        0.00295 |      0.9822 |         1.74 |
|                4 | HSV          | ALL           |          10 |       73008 |        9 |              4 |        0.00305 |      0.9822 |         5.93 |
|                4 | YCrCb        | 2             |          10 |       40560 |       15 |              4 |        0.00311 |      0.9822 |         3.03 |
|                2 | YUV          | ALL           |          10 |       40500 |       15 |              4 |        0.00312 |      0.9822 |         2.84 |
|                3 | RGB          | 2             |          10 |       26460 |       15 |              4 |        0.00331 |      0.9822 |         2.39 |
|                3 | RGB          | ALL           |          10 |       63504 |       12 |              4 |        0.00339 |      0.9822 |         6.09 |
|                4 | HLS          | ALL           |          10 |         288 |        6 |             16 |        0.00475 |      0.9822 |         0.05 |
|                3 | YCrCb        | 0             |          10 |       15876 |        9 |              4 |        0.00601 |      0.9822 |         2.02 |
|                4 | YUV          | ALL           |          10 |        2880 |       15 |             12 |        0.00157 |      0.98   |         0.21 |
|                3 | YCrCb        | 2             |          10 |        1215 |       15 |             12 |        0.00178 |      0.98   |         0.11 |
|                1 | HLS          | 0             |          10 |         960 |       15 |              8 |        0.00186 |      0.98   |         0.05 |
|                1 | YCrCb        | ALL           |          10 |         432 |        9 |             16 |        0.00187 |      0.98   |         0.03 |
|                2 | HLS          | 0             |          10 |        2352 |       12 |              8 |        0.00187 |      0.98   |         0.13 |
|                4 | HSV          | ALL           |          10 |         576 |       12 |             16 |        0.0019  |      0.98   |         0.06 |
|                4 | HLS          | ALL           |          10 |         576 |       12 |             16 |        0.00191 |      0.98   |         0.05 |
|                1 | HSV          | ALL           |          10 |         675 |        9 |             12 |        0.00193 |      0.98   |         0.04 |
|                2 | HLS          | 0             |          10 |        5400 |        6 |              4 |        0.00197 |      0.98   |         0.39 |
|                3 | LUV          | 2             |          10 |         432 |       12 |             16 |        0.002   |      0.98   |         0.18 |
|                2 | LUV          | 1             |          10 |         432 |       12 |             16 |        0.00206 |      0.98   |         0.09 |
|                2 | HLS          | 0             |          10 |        8100 |        9 |              4 |        0.00211 |      0.98   |         0.63 |
|                4 | YUV          | 1             |          10 |        6000 |       15 |              8 |        0.00216 |      0.98   |         0.37 |
|                3 | HLS          | ALL           |          10 |        1296 |       12 |             16 |        0.00221 |      0.98   |         0.1  |
|                4 | YUV          | ALL           |          10 |        1728 |        9 |             12 |        0.00224 |      0.98   |         0.11 |
|                3 | YCrCb        | 1             |          10 |        2916 |        9 |              8 |        0.00224 |      0.98   |         1.66 |
|                1 | LUV          | ALL           |          10 |         432 |        9 |             16 |        0.00243 |      0.98   |         0.03 |
|                3 | LUV          | ALL           |          10 |         648 |        6 |             16 |        0.00244 |      0.98   |         0.08 |
|                2 | YCrCb        | 1             |          10 |        2940 |       15 |              8 |        0.00244 |      0.98   |         0.16 |
|                1 | HLS          | ALL           |          10 |         675 |        9 |             12 |        0.00246 |      0.98   |         0.05 |
|                3 | YUV          | 2             |          10 |        2916 |        9 |              8 |        0.00246 |      0.98   |         0.33 |
|                4 | LUV          | 1             |          10 |        6000 |       15 |              8 |        0.0025  |      0.98   |         4.1  |
|                4 | YUV          | 1             |          10 |         576 |        9 |             12 |        0.00252 |      0.98   |         0.05 |
|                2 | HLS          | ALL           |          10 |         648 |        6 |             16 |        0.00252 |      0.98   |         0.1  |
|                2 | YCrCb        | 2             |          10 |       10800 |       12 |              4 |        0.00253 |      0.98   |         0.95 |
|                1 | HLS          | ALL           |          10 |         288 |        6 |             16 |        0.00257 |      0.98   |         0.02 |
|                2 | YUV          | 1             |          10 |       10800 |       12 |              4 |        0.00259 |      0.98   |         1    |
|                4 | LUV          | 1             |          10 |       40560 |       15 |              4 |        0.00263 |      0.98   |         4.21 |
|                3 | YCrCb        | 0             |          10 |        4860 |       15 |              8 |        0.00264 |      0.98   |         0.57 |
|                2 | HSV          | 0             |          10 |       13500 |       15 |              4 |        0.00268 |      0.98   |         1.19 |
|                3 | LUV          | 1             |          10 |       15876 |        9 |              4 |        0.00274 |      0.98   |         1.13 |
|                1 | LUV          | ALL           |          10 |        4608 |        6 |              4 |        0.00277 |      0.98   |         0.28 |
|                3 | YCrCb        | 1             |          10 |       26460 |       15 |              4 |        0.00283 |      0.98   |         1.82 |
|                4 | YCrCb        | 1             |          10 |       32448 |       12 |              4 |        0.00294 |      0.98   |         2.15 |
|                4 | RGB          | 1             |          10 |       40560 |       15 |              4 |        0.003   |      0.98   |         3.87 |
|                4 | LUV          | 2             |          10 |       32448 |       12 |              4 |        0.00301 |      0.98   |         2.48 |
|                1 | YUV          | 1             |          10 |        3840 |       15 |              4 |        0.00305 |      0.98   |         0.57 |
|                4 | LUV          | 2             |          10 |       40560 |       15 |              4 |        0.0031  |      0.98   |         3.38 |
|                4 | HSV          | 0             |          10 |       32448 |       12 |              4 |        0.00322 |      0.98   |         2.49 |
|                4 | RGB          | ALL           |          10 |       97344 |       12 |              4 |        0.00386 |      0.98   |         9.22 |
|                1 | YCrCb        | 0             |          10 |        2304 |        9 |              4 |        0.00433 |      0.98   |         0.17 |
|                2 | HSV          | ALL           |          10 |        1620 |       15 |             16 |        0.00161 |      0.9778 |         0.13 |
|                3 | YCrCb        | 1             |          10 |        1944 |        6 |              8 |        0.00189 |      0.9778 |         1.03 |
|                2 | HLS          | ALL           |          10 |        5292 |        9 |              8 |        0.00196 |      0.9778 |         0.36 |
|                1 | HSV          | 0             |          10 |        3840 |       15 |              4 |        0.00198 |      0.9778 |         0.23 |
|                3 | YCrCb        | 1             |          10 |        4860 |       15 |              8 |        0.00201 |      0.9778 |         0.28 |
|                2 | HSV          | 2             |          10 |       13500 |       15 |              4 |        0.00202 |      0.9778 |         1.28 |
|                4 | HLS          | 0             |          10 |       16224 |        6 |              4 |        0.00207 |      0.9778 |         1.27 |
|                1 | YUV          | 1             |          10 |         225 |        9 |             12 |        0.00242 |      0.9778 |         0.03 |
|                1 | LUV          | ALL           |          10 |         450 |        6 |             12 |        0.00242 |      0.9778 |         0.04 |
|                4 | YCrCb        | 1             |          10 |        3600 |        9 |              8 |        0.00242 |      0.9778 |         0.28 |
|                4 | YUV          | 1             |          10 |         144 |        9 |             16 |        0.00243 |      0.9778 |         0.05 |
|                1 | YUV          | 1             |          10 |         144 |        9 |             16 |        0.00243 |      0.9778 |         0.06 |
|                2 | YUV          | 1             |          10 |         432 |       12 |             16 |        0.00244 |      0.9778 |         0.04 |
|                1 | RGB          | 1             |          10 |         768 |       12 |              8 |        0.00244 |      0.9778 |         0.07 |
|                3 | LUV          | ALL           |          10 |        1296 |       12 |             16 |        0.00244 |      0.9778 |         0.08 |
|                3 | LUV          | 1             |          10 |         972 |       12 |             12 |        0.00245 |      0.9778 |         0.13 |
|                2 | YUV          | 1             |          10 |         960 |       15 |             12 |        0.00246 |      0.9778 |         0.06 |
|                3 | HSV          | ALL           |          10 |         648 |        6 |             16 |        0.00246 |      0.9778 |         0.09 |
|                3 | HSV          | ALL           |          10 |        3645 |       15 |             12 |        0.00246 |      0.9778 |         0.31 |
|                4 | YCrCb        | 1             |          10 |        2400 |        6 |              8 |        0.00246 |      0.9778 |         1.16 |
|                2 | YUV          | 1             |          10 |        5400 |        6 |              4 |        0.00247 |      0.9778 |         0.37 |
|                2 | YUV          | 1             |          10 |         384 |        6 |             12 |        0.00248 |      0.9778 |         0.05 |
|                3 | HLS          | ALL           |          10 |         972 |        9 |             16 |        0.00248 |      0.9778 |         0.14 |
|                3 | HLS          | ALL           |          10 |         648 |        6 |             16 |        0.00249 |      0.9778 |         0.15 |
|                4 | LUV          | 1             |          10 |         960 |       15 |             12 |        0.00249 |      0.9778 |         0.48 |
|                2 | HSV          | 0             |          10 |         768 |       12 |             12 |        0.00251 |      0.9778 |         0.05 |
|                4 | YUV          | 1             |          10 |         192 |       12 |             16 |        0.00252 |      0.9778 |         0.03 |
|                3 | LUV          | 1             |          10 |         540 |       15 |             16 |        0.00252 |      0.9778 |         0.13 |
|                1 | HLS          | 0             |          10 |         375 |       15 |             12 |        0.00253 |      0.9778 |         0.03 |
|                3 | HLS          | ALL           |          10 |        2187 |        9 |             12 |        0.00253 |      0.9778 |         0.23 |
|                3 | HSV          | ALL           |          10 |        1620 |       15 |             16 |        0.00255 |      0.9778 |         0.16 |
|                2 | LUV          | 1             |          10 |       10800 |       12 |              4 |        0.00255 |      0.9778 |         0.83 |
|                1 | HSV          | 2             |          10 |        3840 |       15 |              4 |        0.00256 |      0.9778 |         0.38 |
|                2 | LUV          | 2             |          10 |       10800 |       12 |              4 |        0.00258 |      0.9778 |         0.76 |
|                3 | YCrCb        | 1             |          10 |        3888 |       12 |              8 |        0.00261 |      0.9778 |         2.13 |
|                4 | YUV          | 1             |          10 |         384 |        6 |             12 |        0.00262 |      0.9778 |         0.11 |
|                3 | HSV          | ALL           |          10 |       11664 |       12 |              8 |        0.00262 |      0.9778 |         0.94 |
|                3 | YUV          | 2             |          10 |       10584 |        6 |              4 |        0.00263 |      0.9778 |         0.83 |
|                4 | HSV          | ALL           |          10 |       10800 |        9 |              8 |        0.00263 |      0.9778 |         1.12 |
|                3 | YCrCb        | 2             |          10 |        3888 |       12 |              8 |        0.0027  |      0.9778 |         0.27 |
|                2 | HLS          | 1             |          10 |        2940 |       15 |              8 |        0.00273 |      0.9778 |         0.29 |
|                3 | HLS          | ALL           |          10 |        3645 |       15 |             12 |        0.00275 |      0.9778 |         0.29 |
|                3 | LUV          | 2             |          10 |       15876 |        9 |              4 |        0.00278 |      0.9778 |         1.33 |
|                3 | YCrCb        | 1             |          10 |       21168 |       12 |              4 |        0.00278 |      0.9778 |         1.41 |
|                4 | HSV          | 0             |          10 |       16224 |        6 |              4 |        0.0028  |      0.9778 |         1.29 |
|                3 | LUV          | 0             |          10 |       21168 |       12 |              4 |        0.00286 |      0.9778 |         1.97 |
|                3 | YCrCb        | 0             |          10 |       26460 |       15 |              4 |        0.00298 |      0.9778 |         2.76 |
|                4 | HLS          | ALL           |          10 |       73008 |        9 |              4 |        0.00957 |      0.9778 |         6.01 |
|                2 | HLS          | ALL           |          10 |        1152 |        6 |             12 |        0.00159 |      0.9756 |         0.09 |
|                1 | YUV          | 2             |          10 |        3840 |       15 |              4 |        0.0016  |      0.9756 |         0.28 |
|                1 | YUV          | 1             |          10 |         576 |        9 |              8 |        0.00192 |      0.9756 |         0.21 |
|                4 | YCrCb        | 1             |          10 |         240 |       15 |             16 |        0.00202 |      0.9756 |         0.03 |
|                3 | HSV          | ALL           |          10 |        8748 |        9 |              8 |        0.00204 |      0.9756 |         0.73 |
|                3 | YCrCb        | 1             |          10 |       10584 |        6 |              4 |        0.00204 |      0.9756 |         0.83 |
|                3 | HLS          | ALL           |          10 |        1458 |        6 |             12 |        0.00213 |      0.9756 |         0.13 |
|                3 | RGB          | 1             |          10 |       26460 |       15 |              4 |        0.00233 |      0.9756 |         2.67 |
|                2 | YCrCb        | 1             |          10 |         960 |       15 |             12 |        0.00236 |      0.9756 |         0.05 |
|                1 | YUV          | 2             |          10 |         375 |       15 |             12 |        0.00239 |      0.9756 |         0.04 |
|                1 | YCrCb        | 1             |          10 |         300 |       12 |             12 |        0.0024  |      0.9756 |         0.02 |
|                4 | YCrCb        | 1             |          10 |       16224 |        6 |              4 |        0.0024  |      0.9756 |         1.65 |
|                2 | YUV          | 1             |          10 |         768 |       12 |             12 |        0.00241 |      0.9756 |         0.04 |
|                1 | YUV          | 1             |          10 |         768 |       12 |              8 |        0.00242 |      0.9756 |         0.04 |
|                1 | YUV          | ALL           |          10 |         720 |       15 |             16 |        0.00242 |      0.9756 |         0.04 |
|                2 | YCrCb        | 1             |          10 |         324 |        9 |             16 |        0.00243 |      0.9756 |         0.07 |
|                1 | LUV          | 2             |          10 |        3072 |       12 |              4 |        0.00244 |      0.9756 |         0.2  |
|                3 | LUV          | 1             |          10 |        1944 |        6 |              8 |        0.00248 |      0.9756 |         0.18 |
|                3 | YUV          | 1             |          10 |         972 |       12 |             12 |        0.00248 |      0.9756 |         0.27 |
|                1 | YUV          | 1             |          10 |        2304 |        9 |              4 |        0.00248 |      0.9756 |         0.69 |
|                1 | YUV          | 2             |          10 |        3072 |       12 |              4 |        0.0025  |      0.9756 |         0.16 |
|                4 | LUV          | 1             |          10 |        2400 |        6 |              8 |        0.0025  |      0.9756 |         0.23 |
|                3 | LUV          | 2             |          10 |       26460 |       15 |              4 |        0.0025  |      0.9756 |         3.06 |
|                2 | LUV          | 2             |          10 |        8100 |        9 |              4 |        0.00253 |      0.9756 |         0.64 |
|                4 | HSV          | ALL           |          10 |         288 |        6 |             16 |        0.00254 |      0.9756 |         0.07 |
|                2 | LUV          | 1             |          10 |         768 |       12 |             12 |        0.00255 |      0.9756 |         0.06 |
|                2 | RGB          | 1             |          10 |       13500 |       15 |              4 |        0.00259 |      0.9756 |         1.29 |
|                4 | YCrCb        | 1             |          10 |          96 |        6 |             16 |        0.00262 |      0.9756 |         0.03 |
|                2 | YUV          | 2             |          10 |       10800 |       12 |              4 |        0.00264 |      0.9756 |         0.73 |
|                2 | HLS          | 0             |          10 |       13500 |       15 |              4 |        0.00271 |      0.9756 |         1.12 |
|                2 | YCrCb        | 0             |          10 |       10800 |       12 |              4 |        0.00278 |      0.9756 |         0.94 |
|                3 | YUV          | 2             |          10 |       15876 |        9 |              4 |        0.00279 |      0.9756 |         1.16 |
|                3 | YCrCb        | 0             |          10 |       21168 |       12 |              4 |        0.00279 |      0.9756 |         2.03 |
|                3 | YUV          | 0             |          10 |       21168 |       12 |              4 |        0.00287 |      0.9756 |         2.26 |
|                3 | YUV          | 0             |          10 |       26460 |       15 |              4 |        0.00288 |      0.9756 |         2.5  |
|                4 | YCrCb        | 1             |          10 |       24336 |        9 |              4 |        0.00292 |      0.9756 |         1.88 |
|                4 | HSV          | 2             |          10 |       40560 |       15 |              4 |        0.00305 |      0.9756 |         4.07 |
|                4 | RGB          | ALL           |          10 |       48672 |        6 |              4 |        0.00326 |      0.9756 |         5.4  |
|                3 | RGB          | ALL           |          10 |       79380 |       15 |              4 |        0.00364 |      0.9756 |         7.44 |
|                2 | LUV          | 1             |          10 |        2352 |       12 |              8 |        0.00416 |      0.9756 |         0.16 |
|                3 | YUV          | 1             |          10 |        4860 |       15 |              8 |        0.00947 |      0.9756 |         0.26 |
|                1 | HLS          | ALL           |          10 |         450 |        6 |             12 |        0.00168 |      0.9733 |         0.03 |
|                2 | YCrCb        | 1             |          10 |         540 |       15 |             16 |        0.00185 |      0.9733 |         0.04 |
|                1 | LUV          | 1             |          10 |         768 |       12 |              8 |        0.00191 |      0.9733 |         0.06 |
|                1 | HLS          | 0             |          10 |        1536 |        6 |              4 |        0.00196 |      0.9733 |         0.09 |
|                1 | YUV          | 1             |          10 |         960 |       15 |              8 |        0.00199 |      0.9733 |         0.55 |
|                2 | YUV          | 0             |          10 |       10800 |       12 |              4 |        0.00201 |      0.9733 |         0.97 |
|                3 | YCrCb        | 2             |          10 |       10584 |        6 |              4 |        0.0021  |      0.9733 |         0.97 |
|                3 | YUV          | 1             |          10 |       10584 |        6 |              4 |        0.00214 |      0.9733 |         0.76 |
|                2 | YUV          | 1             |          10 |        8100 |        9 |              4 |        0.00215 |      0.9733 |         0.55 |
|                1 | HSV          | ALL           |          10 |        1728 |        9 |              8 |        0.00221 |      0.9733 |         0.1  |
|                2 | YCrCb        | 1             |          10 |       10800 |       12 |              4 |        0.00227 |      0.9733 |         0.95 |
|                4 | YCrCb        | 1             |          10 |         384 |        6 |             12 |        0.0023  |      0.9733 |         0.12 |
|                2 | YCrCb        | 1             |          10 |        1176 |        6 |              8 |        0.00234 |      0.9733 |         0.58 |
|                1 | YUV          | 1             |          10 |         375 |       15 |             12 |        0.00238 |      0.9733 |         0.13 |
|                1 | HLS          | 0             |          10 |         300 |       12 |             12 |        0.00239 |      0.9733 |         0.03 |
|                4 | LUV          | 1             |          10 |          96 |        6 |             16 |        0.00239 |      0.9733 |         0.06 |
|                4 | YCrCb        | 0             |          10 |       24336 |        9 |              4 |        0.0024  |      0.9733 |         3.07 |
|                1 | YCrCb        | 1             |          10 |        1536 |        6 |              4 |        0.00244 |      0.9733 |         0.13 |
|                1 | LUV          | ALL           |          10 |         288 |        6 |             16 |        0.00246 |      0.9733 |         0.03 |
|                1 | RGB          | ALL           |          10 |        9216 |       12 |              4 |        0.00246 |      0.9733 |         0.7  |
|                3 | LUV          | 1             |          10 |        2916 |        9 |              8 |        0.00247 |      0.9733 |         0.24 |
|                4 | LUV          | 2             |          10 |         768 |       12 |             12 |        0.00247 |      0.9733 |         0.34 |
|                1 | YCrCb        | 1             |          10 |         960 |       15 |              8 |        0.00248 |      0.9733 |         0.05 |
|                3 | HSV          | ALL           |          10 |         972 |        9 |             16 |        0.00249 |      0.9733 |         0.12 |
|                2 | RGB          | ALL           |          10 |        7056 |       12 |              8 |        0.00251 |      0.9733 |         0.88 |
|                1 | YCrCb        | 1             |          10 |         768 |       12 |              8 |        0.00252 |      0.9733 |         0.04 |
|                1 | LUV          | 0             |          10 |        3840 |       15 |              4 |        0.00252 |      0.9733 |         0.28 |
|                3 | YCrCb        | 1             |          10 |         432 |       12 |             16 |        0.00253 |      0.9733 |         0.04 |
|                1 | RGB          | ALL           |          10 |        1728 |        9 |              8 |        0.00253 |      0.9733 |         0.19 |
|                1 | LUV          | 2             |          10 |        3840 |       15 |              4 |        0.00253 |      0.9733 |         0.24 |
|                3 | HSV          | 0             |          10 |         972 |       12 |             12 |        0.00254 |      0.9733 |         0.1  |
|                1 | YCrCb        | 0             |          10 |        3840 |       15 |              4 |        0.00254 |      0.9733 |         0.3  |
|                1 | RGB          | ALL           |          10 |       11520 |       15 |              4 |        0.00258 |      0.9733 |         1.14 |
|                2 | LUV          | 0             |          10 |       13500 |       15 |              4 |        0.00261 |      0.9733 |         1.29 |
|                1 | HSV          | 0             |          10 |         768 |       12 |              8 |        0.00264 |      0.9733 |         0.04 |
|                2 | RGB          | 2             |          10 |       13500 |       15 |              4 |        0.00264 |      0.9733 |         1.21 |
|                2 | YUV          | 1             |          10 |        2352 |       12 |              8 |        0.00265 |      0.9733 |         0.12 |
|                2 | LUV          | 1             |          10 |        5400 |        6 |              4 |        0.00269 |      0.9733 |         0.38 |
|                4 | LUV          | 1             |          10 |       16224 |        6 |              4 |        0.0027  |      0.9733 |         1.82 |
|                2 | HLS          | 0             |          10 |        2940 |       15 |              8 |        0.00274 |      0.9733 |         0.2  |
|                4 | LUV          | 1             |          10 |       24336 |        9 |              4 |        0.00279 |      0.9733 |         1.93 |
|                4 | LUV          | 2             |          10 |       24336 |        9 |              4 |        0.00285 |      0.9733 |         2.26 |
|                2 | RGB          | ALL           |          10 |       40500 |       15 |              4 |        0.00303 |      0.9733 |         3.49 |
|                4 | YCrCb        | 1             |          10 |       40560 |       15 |              4 |        0.00307 |      0.9733 |         2.82 |
|                4 | HLS          | 0             |          10 |       40560 |       15 |              4 |        0.00316 |      0.9733 |         3.34 |
|                4 | LUV          | 0             |          10 |       40560 |       15 |              4 |        0.0033  |      0.9733 |         4.64 |
|                1 | RGB          | 2             |          10 |        3840 |       15 |              4 |        0.00159 |      0.9711 |         0.3  |
|                4 | HSV          | ALL           |          10 |         432 |        9 |             16 |        0.00172 |      0.9711 |         0.11 |
|                1 | YUV          | 2             |          10 |         768 |       12 |              8 |        0.00174 |      0.9711 |         0.04 |
|                1 | YCrCb        | 1             |          10 |         240 |       15 |             16 |        0.00182 |      0.9711 |         0.04 |
|                1 | YCrCb        | 0             |          10 |         768 |       12 |              8 |        0.00182 |      0.9711 |         0.08 |
|                4 | LUV          | 1             |          10 |         240 |       15 |             16 |        0.00189 |      0.9711 |         0.06 |
|                3 | YCrCb        | 1             |          10 |         216 |        6 |             16 |        0.0019  |      0.9711 |         0.06 |
|                3 | YCrCb        | 2             |          10 |         972 |       12 |             12 |        0.00195 |      0.9711 |         0.15 |
|                2 | HSV          | 2             |          10 |        5400 |        6 |              4 |        0.0022  |      0.9711 |         0.57 |
|                3 | LUV          | 1             |          10 |       10584 |        6 |              4 |        0.00226 |      0.9711 |         0.8  |
|                2 | YUV          | 1             |          10 |         540 |       15 |             16 |        0.00238 |      0.9711 |         0.05 |
|                3 | HLS          | 0             |          10 |        4860 |       15 |              8 |        0.00238 |      0.9711 |         0.36 |
|                4 | YUV          | 1             |          10 |        3600 |        9 |              8 |        0.00239 |      0.9711 |         1.81 |
|                2 | LUV          | 0             |          10 |        5400 |        6 |              4 |        0.00243 |      0.9711 |         0.5  |
|                3 | YUV          | 1             |          10 |        1215 |       15 |             12 |        0.00244 |      0.9711 |         0.07 |
|                4 | HSV          | ALL           |          10 |        1152 |        6 |             12 |        0.00246 |      0.9711 |         0.13 |
|                2 | RGB          | 1             |          10 |        1764 |        9 |              8 |        0.00249 |      0.9711 |         0.24 |
|                3 | HSV          | ALL           |          10 |        2187 |        9 |             12 |        0.00249 |      0.9711 |         0.25 |
|                2 | LUV          | 1             |          10 |         960 |       15 |             12 |        0.0025  |      0.9711 |         0.08 |
|                4 | HLS          | 0             |          10 |       24336 |        9 |              4 |        0.0025  |      0.9711 |         2.4  |
|                4 | YUV          | 1             |          10 |          96 |        6 |             16 |        0.00251 |      0.9711 |         0.03 |
|                3 | RGB          | ALL           |          10 |         972 |        9 |             16 |        0.00251 |      0.9711 |         0.55 |
|                2 | RGB          | ALL           |          10 |        3528 |        6 |              8 |        0.00252 |      0.9711 |         0.49 |
|                3 | RGB          | 1             |          10 |       10584 |        6 |              4 |        0.00258 |      0.9711 |         1.21 |
|                2 | YCrCb        | 0             |          10 |        8100 |        9 |              4 |        0.0026  |      0.9711 |         1.02 |
|                3 | YUV          | 2             |          10 |        4860 |       15 |              8 |        0.00263 |      0.9711 |         0.36 |
|                1 | HSV          | 0             |          10 |         960 |       15 |              8 |        0.00265 |      0.9711 |         0.06 |
|                2 | HSV          | 0             |          10 |        5400 |        6 |              4 |        0.00269 |      0.9711 |         0.34 |
|                2 | YUV          | 2             |          10 |       13500 |       15 |              4 |        0.00274 |      0.9711 |         0.96 |
|                3 | YUV          | 2             |          10 |       21168 |       12 |              4 |        0.00275 |      0.9711 |         1.4  |
|                2 | LUV          | 2             |          10 |       13500 |       15 |              4 |        0.00282 |      0.9711 |         2.04 |
|                4 | HLS          | 0             |          10 |       32448 |       12 |              4 |        0.00285 |      0.9711 |         2.36 |
|                4 | YCrCb        | 2             |          10 |       24336 |        9 |              4 |        0.00288 |      0.9711 |         2.25 |
|                3 | RGB          | 2             |          10 |       21168 |       12 |              4 |        0.00292 |      0.9711 |         2.08 |
|                2 | RGB          | ALL           |          10 |       24300 |        9 |              4 |        0.003   |      0.9711 |         2.07 |
|                4 | HLS          | 1             |          10 |       32448 |       12 |              4 |        0.003   |      0.9711 |         3.52 |
|                4 | RGB          | 2             |          10 |       40560 |       15 |              4 |        0.00309 |      0.9711 |         4.11 |
|                2 | RGB          | ALL           |          10 |       32400 |       12 |              4 |        0.00335 |      0.9711 |         2.91 |
|                4 | YCrCb        | 2             |          10 |       32448 |       12 |              4 |        0.00412 |      0.9711 |         2.48 |
|                1 | YCrCb        | 1             |          10 |        3072 |       12 |              4 |        0.00459 |      0.9711 |         0.25 |
|                4 | HLS          | ALL           |          10 |        2304 |       12 |             12 |        0.00593 |      0.9711 |         0.19 |
|                3 | LUV          | 0             |          10 |       15876 |        9 |              4 |        0.0097  |      0.9711 |         2.08 |
|                2 | HSV          | 0             |          10 |        1764 |        9 |              8 |        0.00141 |      0.9689 |         0.15 |
|                2 | LUV          | 1             |          10 |         540 |       15 |             16 |        0.00156 |      0.9689 |         0.07 |
|                1 | HSV          | 0             |          10 |         300 |       12 |             12 |        0.00169 |      0.9689 |         0.02 |
|                3 | LUV          | 2             |          10 |        2916 |        9 |              8 |        0.00182 |      0.9689 |         2.03 |
|                1 | YCrCb        | 2             |          10 |        1536 |        6 |              4 |        0.00191 |      0.9689 |         0.11 |
|                1 | LUV          | 1             |          10 |         960 |       15 |              8 |        0.00196 |      0.9689 |         0.07 |
|                2 | YUV          | 1             |          10 |         216 |        6 |             16 |        0.00198 |      0.9689 |         0.07 |
|                2 | YCrCb        | 2             |          10 |         960 |       15 |             12 |        0.00198 |      0.9689 |         0.09 |
|                2 | HSV          | 2             |          10 |        2940 |       15 |              8 |        0.00199 |      0.9689 |         0.35 |
|                2 | YUV          | 0             |          10 |         216 |        6 |             16 |        0.00203 |      0.9689 |         0.15 |
|                4 | YCrCb        | 0             |          10 |       16224 |        6 |              4 |        0.00207 |      0.9689 |         2.03 |
|                3 | HSV          | 0             |          10 |       15876 |        9 |              4 |        0.00218 |      0.9689 |         1.3  |
|                4 | YUV          | 1             |          10 |         240 |       15 |             16 |        0.00236 |      0.9689 |         0.04 |
|                3 | LUV          | 0             |          10 |       26460 |       15 |              4 |        0.00236 |      0.9689 |         2.65 |
|                4 | LUV          | 0             |          10 |       24336 |        9 |              4 |        0.00237 |      0.9689 |         2.92 |
|                3 | YCrCb        | 1             |          10 |         540 |       15 |             16 |        0.0024  |      0.9689 |         0.05 |
|                2 | HSV          | 0             |          10 |        1176 |        6 |              8 |        0.00243 |      0.9689 |         0.08 |
|                4 | HLS          | ALL           |          10 |         432 |        9 |             16 |        0.00244 |      0.9689 |         0.07 |
|                4 | YUV          | 0             |          10 |       24336 |        9 |              4 |        0.00244 |      0.9689 |         2.87 |
|                2 | YUV          | 2             |          10 |         540 |       15 |             16 |        0.00246 |      0.9689 |         0.13 |
|                3 | HLS          | 0             |          10 |        1944 |        6 |              8 |        0.00246 |      0.9689 |         0.22 |
|                4 | HLS          | 1             |          10 |        4800 |       12 |              8 |        0.00246 |      0.9689 |         0.66 |
|                2 | YCrCb        | 1             |          10 |         384 |        6 |             12 |        0.00247 |      0.9689 |         0.05 |
|                2 | YCrCb        | 2             |          10 |         768 |       12 |             12 |        0.00248 |      0.9689 |         0.09 |
|                2 | RGB          | ALL           |          10 |        1620 |       15 |             16 |        0.00248 |      0.9689 |         0.37 |
|                4 | HLS          | 0             |          10 |        4800 |       12 |              8 |        0.0025  |      0.9689 |         0.37 |
|                2 | RGB          | 1             |          10 |        8100 |        9 |              4 |        0.00251 |      0.9689 |         0.73 |
|                4 | YCrCb        | 2             |          10 |         768 |       12 |             12 |        0.00252 |      0.9689 |         0.1  |
|                1 | RGB          | 1             |          10 |        2304 |        9 |              4 |        0.00254 |      0.9689 |         0.16 |
|                2 | LUV          | 0             |          10 |        2940 |       15 |              8 |        0.00259 |      0.9689 |         0.36 |
|                2 | LUV          | 1             |          10 |        8100 |        9 |              4 |        0.0026  |      0.9689 |         0.65 |
|                3 | HLS          | 0             |          10 |       15876 |        9 |              4 |        0.0026  |      0.9689 |         1.4  |
|                3 | LUV          | 2             |          10 |        4860 |       15 |              8 |        0.0027  |      0.9689 |         0.4  |
|                2 | YUV          | 1             |          10 |       13500 |       15 |              4 |        0.00271 |      0.9689 |         1.51 |
|                2 | YCrCb        | 0             |          10 |       13500 |       15 |              4 |        0.00272 |      0.9689 |         1.87 |
|                2 | RGB          | ALL           |          10 |       16200 |        6 |              4 |        0.00279 |      0.9689 |         1.58 |
|                4 | YUV          | 2             |          10 |        3600 |        9 |              8 |        0.0028  |      0.9689 |         0.38 |
|                3 | HSV          | 2             |          10 |       15876 |        9 |              4 |        0.00294 |      0.9689 |         1.73 |
|                4 | YUV          | 0             |          10 |       32448 |       12 |              4 |        0.00301 |      0.9689 |         4.11 |
|                4 | RGB          | ALL           |          10 |       73008 |        9 |              4 |        0.00354 |      0.9689 |         7.64 |
|                1 | LUV          | 1             |          10 |        2304 |        9 |              4 |        0.0052  |      0.9689 |         0.14 |
|                3 | RGB          | ALL           |          10 |       11664 |       12 |              8 |        0.0064  |      0.9689 |         1.8  |
|                2 | HLS          | ALL           |          10 |        1728 |        9 |             12 |        0.0016  |      0.9667 |         0.14 |
|                2 | LUV          | 1             |          10 |        1764 |        9 |              8 |        0.00162 |      0.9667 |         0.13 |
|                1 | LUV          | 2             |          10 |        2304 |        9 |              4 |        0.00162 |      0.9667 |         0.15 |
|                1 | LUV          | 0             |          10 |        1536 |        6 |              4 |        0.00181 |      0.9667 |         0.12 |
|                2 | HLS          | 1             |          10 |        2352 |       12 |              8 |        0.00192 |      0.9667 |         0.23 |
|                1 | YUV          | 2             |          10 |          96 |        6 |             16 |        0.00193 |      0.9667 |         0.08 |
|                1 | RGB          | ALL           |          10 |        1152 |        6 |              8 |        0.002   |      0.9667 |         0.13 |
|                1 | YUV          | 2             |          10 |         960 |       15 |              8 |        0.00201 |      0.9667 |         0.05 |
|                1 | RGB          | ALL           |          10 |        2880 |       15 |              8 |        0.00205 |      0.9667 |         0.26 |
|                2 | YCrCb        | 1             |          10 |        8100 |        9 |              4 |        0.00223 |      0.9667 |         0.76 |
|                3 | HLS          | 0             |          10 |       10584 |        6 |              4 |        0.00235 |      0.9667 |         0.82 |
|                3 | HLS          | 1             |          10 |       21168 |       12 |              4 |        0.00235 |      0.9667 |         2.04 |
|                3 | LUV          | 1             |          10 |         432 |       12 |             16 |        0.00242 |      0.9667 |         0.08 |
|                2 | YCrCb        | 0             |          10 |        2940 |       15 |              8 |        0.00245 |      0.9667 |         0.3  |
|                1 | YUV          | 1             |          10 |         300 |       12 |             12 |        0.00246 |      0.9667 |         0.03 |
|                2 | YUV          | 2             |          10 |        2940 |       15 |              8 |        0.00246 |      0.9667 |         0.18 |
|                3 | YUV          | 2             |          10 |         972 |       12 |             12 |        0.00248 |      0.9667 |         0.13 |
|                2 | RGB          | 1             |          10 |        5400 |        6 |              4 |        0.00249 |      0.9667 |         0.49 |
|                2 | HSV          | ALL           |          10 |        1728 |        9 |             12 |        0.0025  |      0.9667 |         0.13 |
|                2 | HSV          | 0             |          10 |        2352 |       12 |              8 |        0.0025  |      0.9667 |         0.14 |
|                3 | YCrCb        | 1             |          10 |         486 |        6 |             12 |        0.0025  |      0.9667 |         0.15 |
|                1 | HLS          | 1             |          10 |        3072 |       12 |              4 |        0.00252 |      0.9667 |         0.24 |
|                2 | HSV          | 0             |          10 |        8100 |        9 |              4 |        0.00252 |      0.9667 |         0.66 |
|                4 | LUV          | 2             |          10 |         960 |       15 |             12 |        0.00253 |      0.9667 |         0.52 |
|                1 | HSV          | 0             |          10 |        2304 |        9 |              4 |        0.00256 |      0.9667 |         0.14 |
|                3 | YUV          | 2             |          10 |        3888 |       12 |              8 |        0.00256 |      0.9667 |         0.33 |
|                2 | YCrCb        | 0             |          10 |        1764 |        9 |              8 |        0.00257 |      0.9667 |         0.19 |
|                1 | RGB          | 1             |          10 |        3072 |       12 |              4 |        0.00257 |      0.9667 |         0.24 |
|                2 | YUV          | 2             |          10 |        8100 |        9 |              4 |        0.0026  |      0.9667 |         0.57 |
|                4 | HSV          | 0             |          10 |        6000 |       15 |              8 |        0.0026  |      0.9667 |         0.59 |
|                2 | LUV          | 2             |          10 |        2352 |       12 |              8 |        0.00266 |      0.9667 |         0.16 |
|                1 | LUV          | 0             |          10 |        2304 |        9 |              4 |        0.00277 |      0.9667 |         0.16 |
|                3 | RGB          | 2             |          10 |       15876 |        9 |              4 |        0.00279 |      0.9667 |         1.58 |
|                3 | YUV          | 0             |          10 |       15876 |        9 |              4 |        0.00282 |      0.9667 |         1.83 |
|                2 | YUV          | 0             |          10 |       13500 |       15 |              4 |        0.00285 |      0.9667 |         1.28 |
|                4 | YCrCb        | 0             |          10 |       40560 |       15 |              4 |        0.00296 |      0.9667 |         3.93 |
|                4 | RGB          | 1             |          10 |       32448 |       12 |              4 |        0.00298 |      0.9667 |         3.83 |
|                3 | HSV          | ALL           |          10 |        1458 |        6 |             12 |        0.00323 |      0.9667 |         0.16 |
|                3 | RGB          | 1             |          10 |       21168 |       12 |              4 |        0.00323 |      0.9667 |         1.95 |
|                1 | YCrCb        | 2             |          10 |        3072 |       12 |              4 |        0.00421 |      0.9667 |         0.21 |
|                2 | YCrCb        | 1             |          10 |        5400 |        6 |              4 |        0.00445 |      0.9667 |         0.41 |
|                4 | RGB          | ALL           |          10 |      121680 |       15 |              4 |        0.0047  |      0.9667 |        11.42 |
|                2 | HLS          | 1             |          10 |        8100 |        9 |              4 |        0.00537 |      0.9667 |         0.73 |
|                2 | RGB          | ALL           |          10 |        2880 |       15 |             12 |        0.00635 |      0.9667 |         0.39 |
|                2 | HSV          | 0             |          10 |         540 |       15 |             16 |        0.00101 |      0.9644 |         0.15 |
|                1 | LUV          | 2             |          10 |         960 |       15 |              8 |        0.00149 |      0.9644 |         0.06 |
|                1 | YUV          | 1             |          10 |        1536 |        6 |              4 |        0.00173 |      0.9644 |         0.23 |
|                1 | YCrCb        | 1             |          10 |         384 |        6 |              8 |        0.00189 |      0.9644 |         0.03 |
|                3 | HSV          | 0             |          10 |        4860 |       15 |              8 |        0.0019  |      0.9644 |         0.36 |
|                4 | LUV          | 1             |          10 |         192 |       12 |             16 |        0.00194 |      0.9644 |         0.06 |
|                3 | YCrCb        | 1             |          10 |         324 |        9 |             16 |        0.00196 |      0.9644 |         0.11 |
|                2 | LUV          | 1             |          10 |         324 |        9 |             16 |        0.00198 |      0.9644 |         0.09 |
|                2 | YCrCb        | 0             |          10 |         768 |       12 |             12 |        0.00229 |      0.9644 |         0.18 |
|                2 | YCrCb        | 2             |          10 |         540 |       15 |             16 |        0.0024  |      0.9644 |         0.09 |
|                3 | LUV          | 1             |          10 |         486 |        6 |             12 |        0.00241 |      0.9644 |         0.19 |
|                1 | HSV          | 0             |          10 |         240 |       15 |             16 |        0.00243 |      0.9644 |         0.03 |
|                1 | YCrCb        | 1             |          10 |         192 |       12 |             16 |        0.00243 |      0.9644 |         0.03 |
|                1 | HSV          | 0             |          10 |         384 |        6 |              8 |        0.00244 |      0.9644 |         0.03 |
|                4 | LUV          | 2             |          10 |        4800 |       12 |              8 |        0.00244 |      0.9644 |         0.39 |
|                3 | RGB          | ALL           |          10 |        1620 |       15 |             16 |        0.00244 |      0.9644 |         0.43 |
|                4 | HLS          | 0             |          10 |        6000 |       15 |              8 |        0.00244 |      0.9644 |         0.56 |
|                2 | YCrCb        | 1             |          10 |        1764 |        9 |              8 |        0.00244 |      0.9644 |         1    |
|                2 | YUV          | 2             |          10 |         432 |       12 |             16 |        0.00246 |      0.9644 |         0.1  |
|                3 | LUV          | 2             |          10 |         972 |       12 |             12 |        0.00246 |      0.9644 |         0.41 |
|                2 | YCrCb        | 2             |          10 |         432 |       12 |             16 |        0.00248 |      0.9644 |         0.07 |
|                2 | HSV          | 0             |          10 |         432 |       12 |             16 |        0.00248 |      0.9644 |         0.08 |
|                1 | YUV          | 0             |          10 |        3072 |       12 |              4 |        0.00248 |      0.9644 |         0.2  |
|                2 | RGB          | ALL           |          10 |        5292 |        9 |              8 |        0.00248 |      0.9644 |         0.61 |
|                4 | HLS          | 1             |          10 |        6000 |       15 |              8 |        0.00249 |      0.9644 |         0.81 |
|                1 | LUV          | 2             |          10 |         300 |       12 |             12 |        0.00254 |      0.9644 |         0.07 |
|                3 | LUV          | 2             |          10 |        1944 |        6 |              8 |        0.00256 |      0.9644 |         0.29 |
|                3 | HSV          | 2             |          10 |       10584 |        6 |              4 |        0.00256 |      0.9644 |         1.08 |
|                3 | YCrCb        | 2             |          10 |       26460 |       15 |              4 |        0.00256 |      0.9644 |         1.83 |
|                2 | YUV          | 1             |          10 |        1176 |        6 |              8 |        0.00257 |      0.9644 |         0.14 |
|                3 | LUV          | 2             |          10 |        3888 |       12 |              8 |        0.00257 |      0.9644 |         2.14 |
|                2 | RGB          | 2             |          10 |        8100 |        9 |              4 |        0.00258 |      0.9644 |         0.7  |
|                2 | YCrCb        | 1             |          10 |         216 |        6 |             16 |        0.0026  |      0.9644 |         0.07 |
|                1 | YUV          | 2             |          10 |        2304 |        9 |              4 |        0.0026  |      0.9644 |         0.14 |
|                2 | LUV          | 0             |          10 |       10800 |       12 |              4 |        0.00261 |      0.9644 |         0.94 |
|                3 | LUV          | 2             |          10 |       10584 |        6 |              4 |        0.00263 |      0.9644 |         0.9  |
|                4 | YUV          | 0             |          10 |       16224 |        6 |              4 |        0.00266 |      0.9644 |         2.31 |
|                2 | RGB          | 2             |          10 |       10800 |       12 |              4 |        0.00267 |      0.9644 |         0.89 |
|                3 | LUV          | 0             |          10 |       10584 |        6 |              4 |        0.00267 |      0.9644 |         1.34 |
|                1 | HLS          | 0             |          10 |         150 |        6 |             12 |        0.0027  |      0.9644 |         0.05 |
|                4 | LUV          | 2             |          10 |       16224 |        6 |              4 |        0.00279 |      0.9644 |         1.42 |
|                3 | LUV          | 2             |          10 |       21168 |       12 |              4 |        0.00285 |      0.9644 |         1.65 |
|                3 | RGB          | ALL           |          10 |       31752 |        6 |              4 |        0.00294 |      0.9644 |         3.42 |
|                4 | YCrCb        | 0             |          10 |       32448 |       12 |              4 |        0.00298 |      0.9644 |         3.31 |
|                4 | RGB          | 0             |          10 |       40560 |       15 |              4 |        0.003   |      0.9644 |         4.1  |
|                4 | RGB          | 2             |          10 |       32448 |       12 |              4 |        0.00304 |      0.9644 |         3.21 |
|                4 | HSV          | 2             |          10 |       32448 |       12 |              4 |        0.00317 |      0.9644 |         3.15 |
|                2 | LUV          | 2             |          10 |         540 |       15 |             16 |        0.00325 |      0.9644 |         0.09 |
|                3 | LUV          | 0             |          10 |        4860 |       15 |              8 |        0.00403 |      0.9644 |         0.67 |
|                2 | YCrCb        | 2             |          10 |        5400 |        6 |              4 |        0.00583 |      0.9644 |         0.42 |
|                1 | HLS          | 0             |          10 |         384 |        6 |              8 |        0.00146 |      0.9622 |         0.03 |
|                4 | HSV          | ALL           |          10 |        2880 |       15 |             12 |        0.00156 |      0.9622 |         0.24 |
|                4 | LUV          | 1             |          10 |         144 |        9 |             16 |        0.00179 |      0.9622 |         0.06 |
|                4 | RGB          | ALL           |          10 |        7200 |        6 |              8 |        0.00194 |      0.9622 |         1.38 |
|                1 | YCrCb        | 1             |          10 |         144 |        9 |             16 |        0.00195 |      0.9622 |         0.06 |
|                2 | LUV          | 0             |          10 |        8100 |        9 |              4 |        0.00196 |      0.9622 |         0.77 |
|                2 | HLS          | 0             |          10 |         432 |       12 |             16 |        0.00199 |      0.9622 |         0.06 |
|                4 | YUV          | 2             |          10 |         768 |       12 |             12 |        0.00199 |      0.9622 |         0.1  |
|                2 | YUV          | 2             |          10 |        5400 |        6 |              4 |        0.00199 |      0.9622 |         0.35 |
|                1 | HLS          | 0             |          10 |        2304 |        9 |              4 |        0.00203 |      0.9622 |         0.14 |
|                4 | HSV          | 0             |          10 |         768 |       12 |             12 |        0.00203 |      0.9622 |         0.17 |
|                2 | YUV          | 0             |          10 |        8100 |        9 |              4 |        0.0021  |      0.9622 |         0.84 |
|                4 | YCrCb        | 2             |          10 |       16224 |        6 |              4 |        0.00219 |      0.9622 |         1.61 |
|                3 | RGB          | ALL           |          10 |       14580 |       15 |              8 |        0.00221 |      0.9622 |         1.72 |
|                3 | HLS          | 0             |          10 |        1215 |       15 |             12 |        0.00228 |      0.9622 |         0.14 |
|                1 | YCrCb        | 1             |          10 |        2304 |        9 |              4 |        0.00238 |      0.9622 |         0.27 |
|                3 | RGB          | 0             |          10 |       21168 |       12 |              4 |        0.00238 |      0.9622 |         2.05 |
|                3 | HSV          | 2             |          10 |       26460 |       15 |              4 |        0.00239 |      0.9622 |         2.6  |
|                4 | YUV          | 2             |          10 |          96 |        6 |             16 |        0.00243 |      0.9622 |         0.07 |
|                4 | YCrCb        | 2             |          10 |          96 |        6 |             16 |        0.00243 |      0.9622 |         0.09 |
|                1 | YCrCb        | 2             |          10 |         375 |       15 |             12 |        0.00244 |      0.9622 |         0.04 |
|                3 | YUV          | 2             |          10 |         540 |       15 |             16 |        0.00244 |      0.9622 |         0.17 |
|                3 | LUV          | 2             |          10 |         729 |        9 |             12 |        0.00246 |      0.9622 |         0.35 |
|                2 | RGB          | 2             |          10 |        5400 |        6 |              4 |        0.00246 |      0.9622 |         0.5  |
|                4 | RGB          | ALL           |          10 |         720 |       15 |             16 |        0.00247 |      0.9622 |         0.15 |
|                4 | LUV          | 2             |          10 |         384 |        6 |             12 |        0.00247 |      0.9622 |         0.17 |
|                4 | YUV          | 2             |          10 |         576 |        9 |             12 |        0.00247 |      0.9622 |         0.2  |
|                3 | YUV          | 2             |          10 |        1215 |       15 |             12 |        0.00248 |      0.9622 |         0.12 |
|                3 | RGB          | 1             |          10 |        1944 |        6 |              8 |        0.00248 |      0.9622 |         0.52 |
|                2 | YUV          | 0             |          10 |        1176 |        6 |              8 |        0.0025  |      0.9622 |         0.17 |
|                1 | LUV          | 1             |          10 |        3840 |       15 |              4 |        0.0025  |      0.9622 |         0.49 |
|                1 | YUV          | 1             |          10 |         150 |        6 |             12 |        0.00251 |      0.9622 |         0.05 |
|                2 | RGB          | 2             |          10 |        2940 |       15 |              8 |        0.00252 |      0.9622 |         0.28 |
|                3 | YCrCb        | 0             |          10 |        1944 |        6 |              8 |        0.00252 |      0.9622 |         0.53 |
|                1 | RGB          | ALL           |          10 |        2304 |       12 |              8 |        0.00254 |      0.9622 |         0.18 |
|                4 | YUV          | 0             |          10 |        2400 |        6 |              8 |        0.00254 |      0.9622 |         0.77 |
|                1 | HLS          | 0             |          10 |         240 |       15 |             16 |        0.00265 |      0.9622 |         0.07 |
|                2 | LUV          | 2             |          10 |         576 |        9 |             12 |        0.00266 |      0.9622 |         0.2  |
|                3 | LUV          | 0             |          10 |         540 |       15 |             16 |        0.00266 |      0.9622 |         0.29 |
|                4 | YUV          | 2             |          10 |        4800 |       12 |              8 |        0.00271 |      0.9622 |         0.34 |
|                4 | RGB          | 2             |          10 |       16224 |        6 |              4 |        0.00274 |      0.9622 |         2.05 |
|                4 | HSV          | 2             |          10 |       24336 |        9 |              4 |        0.00274 |      0.9622 |         2.64 |
|                3 | HLS          | 1             |          10 |       15876 |        9 |              4 |        0.0028  |      0.9622 |         1.6  |
|                4 | HLS          | 1             |          10 |       24336 |        9 |              4 |        0.00283 |      0.9622 |         3.16 |
|                4 | YUV          | 0             |          10 |       40560 |       15 |              4 |        0.00304 |      0.9622 |         4.38 |
|                3 | HLS          | 0             |          10 |        3888 |       12 |              8 |        0.00315 |      0.9622 |         0.25 |
|                4 | HLS          | 1             |          10 |       40560 |       15 |              4 |        0.00319 |      0.9622 |         4.03 |
|                3 | YCrCb        | 2             |          10 |       15876 |        9 |              4 |        0.00428 |      0.9622 |         1.27 |
|                2 | LUV          | 0             |          10 |        1764 |        9 |              8 |        0.001   |      0.96   |         0.2  |
|                1 | HSV          | 0             |          10 |         375 |       15 |             12 |        0.00175 |      0.96   |         0.03 |
|                2 | LUV          | 1             |          10 |         576 |        9 |             12 |        0.00177 |      0.96   |         0.14 |
|                1 | YUV          | 2             |          10 |         225 |        9 |             12 |        0.00183 |      0.96   |         0.1  |
|                3 | YUV          | 1             |          10 |         486 |        6 |             12 |        0.0019  |      0.96   |         0.14 |
|                4 | LUV          | 2             |          10 |         144 |        9 |             16 |        0.00191 |      0.96   |         0.09 |
|                1 | YUV          | 0             |          10 |        3840 |       15 |              4 |        0.00197 |      0.96   |         0.29 |
|                4 | YUV          | 2             |          10 |        6000 |       15 |              8 |        0.002   |      0.96   |         0.46 |
|                3 | LUV          | 0             |          10 |        1215 |       15 |             12 |        0.00201 |      0.96   |         0.38 |
|                1 | HSV          | 0             |          10 |         150 |        6 |             12 |        0.00207 |      0.96   |         0.05 |
|                3 | YCrCb        | 2             |          10 |        4860 |       15 |              8 |        0.00211 |      0.96   |         0.35 |
|                2 | RGB          | 1             |          10 |       10800 |       12 |              4 |        0.00219 |      0.96   |         0.93 |
|                2 | RGB          | 0             |          10 |       13500 |       15 |              4 |        0.00219 |      0.96   |         1.21 |
|                3 | YCrCb        | 2             |          10 |         729 |        9 |             12 |        0.00236 |      0.96   |         0.3  |
|                2 | HSV          | 2             |          10 |        8100 |        9 |              4 |        0.0024  |      0.96   |         0.73 |
|                4 | YCrCb        | 2             |          10 |         192 |       12 |             16 |        0.00243 |      0.96   |         0.07 |
|                1 | RGB          | ALL           |          10 |         432 |        9 |             16 |        0.00243 |      0.96   |         0.23 |
|                1 | YUV          | 1             |          10 |         240 |       15 |             16 |        0.00244 |      0.96   |         0.04 |
|                1 | YCrCb        | 2             |          10 |         300 |       12 |             12 |        0.00244 |      0.96   |         0.04 |
|                1 | LUV          | 0             |          10 |         960 |       15 |              8 |        0.00244 |      0.96   |         0.08 |
|                2 | HSV          | 0             |          10 |         960 |       15 |             12 |        0.00244 |      0.96   |         0.09 |
|                3 | HLS          | 0             |          10 |        2916 |        9 |              8 |        0.00244 |      0.96   |         0.28 |
|                2 | LUV          | 1             |          10 |        1176 |        6 |              8 |        0.00245 |      0.96   |         0.09 |
|                3 | YCrCb        | 2             |          10 |         432 |       12 |             16 |        0.00245 |      0.96   |         0.1  |
|                2 | HSV          | ALL           |          10 |         972 |        9 |             16 |        0.00246 |      0.96   |         0.1  |
|                3 | YCrCb        | 2             |          10 |         540 |       15 |             16 |        0.00246 |      0.96   |         0.12 |
|                3 | HSV          | 2             |          10 |        3888 |       12 |              8 |        0.00247 |      0.96   |         0.43 |
|                3 | LUV          | 0             |          10 |        3888 |       12 |              8 |        0.00247 |      0.96   |         0.47 |
|                2 | HLS          | 0             |          10 |        1176 |        6 |              8 |        0.00248 |      0.96   |         0.11 |
|                1 | LUV          | 2             |          10 |         150 |        6 |             12 |        0.00248 |      0.96   |         0.13 |
|                2 | RGB          | ALL           |          10 |         972 |        9 |             16 |        0.00248 |      0.96   |         0.51 |
|                4 | YCrCb        | 1             |          10 |         576 |        9 |             12 |        0.00249 |      0.96   |         0.06 |
|                1 | YCrCb        | 0             |          10 |         960 |       15 |              8 |        0.00249 |      0.96   |         0.08 |
|                3 | RGB          | ALL           |          10 |        3645 |       15 |             12 |        0.00249 |      0.96   |         0.71 |
|                2 | YUV          | 2             |          10 |         768 |       12 |             12 |        0.0025  |      0.96   |         0.06 |
|                4 | LUV          | 1             |          10 |         384 |        6 |             12 |        0.0025  |      0.96   |         0.15 |
|                4 | HSV          | 0             |          10 |        2400 |        6 |              8 |        0.00251 |      0.96   |         0.24 |
|                2 | RGB          | 0             |          10 |        8100 |        9 |              4 |        0.00251 |      0.96   |         0.72 |
|                3 | RGB          | 1             |          10 |        2916 |        9 |              8 |        0.00256 |      0.96   |         0.43 |
|                2 | RGB          | ALL           |          10 |        8820 |       15 |              8 |        0.00257 |      0.96   |         1.07 |
|                2 | YUV          | 2             |          10 |        1764 |        9 |              8 |        0.00259 |      0.96   |         0.12 |
|                4 | RGB          | ALL           |          10 |        2304 |       12 |             12 |        0.00269 |      0.96   |         0.67 |
|                2 | YUV          | 2             |          10 |        2352 |       12 |              8 |        0.0027  |      0.96   |         0.15 |
|                2 | HSV          | 2             |          10 |         384 |        6 |             12 |        0.00272 |      0.96   |         0.23 |
|                3 | RGB          | 1             |          10 |       15876 |        9 |              4 |        0.00274 |      0.96   |         1.58 |
|                3 | RGB          | 0             |          10 |       15876 |        9 |              4 |        0.00275 |      0.96   |         1.72 |
|                2 | HLS          | 0             |          10 |        1764 |        9 |              8 |        0.0028  |      0.96   |         0.17 |
|                3 | RGB          | 0             |          10 |       26460 |       15 |              4 |        0.0028  |      0.96   |         2.48 |
|                1 | YUV          | 2             |          10 |         576 |        9 |              8 |        0.00282 |      0.96   |         0.04 |
|                4 | HSV          | 0             |          10 |       24336 |        9 |              4 |        0.00284 |      0.96   |         2.12 |
|                3 | HSV          | 2             |          10 |       21168 |       12 |              4 |        0.00287 |      0.96   |         2.2  |
|                4 | RGB          | 2             |          10 |       24336 |        9 |              4 |        0.00294 |      0.96   |         2.67 |
|                2 | YCrCb        | 2             |          10 |         384 |        6 |             12 |        0.00412 |      0.96   |         0.18 |
|                2 | LUV          | 2             |          10 |        2940 |       15 |              8 |        0.00432 |      0.96   |         0.22 |
|                1 | HLS          | 1             |          10 |          96 |        6 |             16 |        0.00505 |      0.96   |         0.13 |
|                4 | YCrCb        | 2             |          10 |         576 |        9 |             12 |        0.00522 |      0.96   |         0.32 |
|                3 | LUV          | 1             |          10 |       26460 |       15 |              4 |        0.00589 |      0.96   |         1.99 |
|                1 | HSV          | 2             |          10 |        3072 |       12 |              4 |        0.00157 |      0.9578 |         0.2  |
|                1 | RGB          | 1             |          10 |        3840 |       15 |              4 |        0.00158 |      0.9578 |         0.35 |
|                1 | RGB          | 1             |          10 |         576 |        9 |              8 |        0.00163 |      0.9578 |         0.08 |
|                2 | LUV          | 2             |          10 |        5400 |        6 |              4 |        0.00182 |      0.9578 |         0.4  |
|                2 | LUV          | 2             |          10 |         432 |       12 |             16 |        0.00183 |      0.9578 |         0.09 |
|                1 | LUV          | 1             |          10 |        3072 |       12 |              4 |        0.00183 |      0.9578 |         0.18 |
|                1 | LUV          | 2             |          10 |         768 |       12 |              8 |        0.00185 |      0.9578 |         0.05 |
|                2 | YCrCb        | 0             |          10 |        5400 |        6 |              4 |        0.0019  |      0.9578 |         0.5  |
|                4 | YUV          | 2             |          10 |         192 |       12 |             16 |        0.00191 |      0.9578 |         0.05 |
|                4 | YUV          | 0             |          10 |        3600 |        9 |              8 |        0.00191 |      0.9578 |         0.79 |
|                1 | LUV          | 0             |          10 |        3072 |       12 |              4 |        0.00192 |      0.9578 |         0.26 |
|                2 | YCrCb        | 1             |          10 |         576 |        9 |             12 |        0.00194 |      0.9578 |         0.05 |
|                4 | YUV          | 2             |          10 |         960 |       15 |             12 |        0.00196 |      0.9578 |         0.1  |
|                4 | RGB          | 1             |          10 |        4800 |       12 |              8 |        0.002   |      0.9578 |         0.8  |
|                1 | YCrCb        | 2             |          10 |         960 |       15 |              8 |        0.00208 |      0.9578 |         0.05 |
|                2 | YCrCb        | 2             |          10 |        2940 |       15 |              8 |        0.00219 |      0.9578 |         0.2  |
|                3 | YCrCb        | 1             |          10 |         729 |        9 |             12 |        0.00234 |      0.9578 |         0.1  |
|                1 | YCrCb        | 2             |          10 |         192 |       12 |             16 |        0.00241 |      0.9578 |         0.09 |
|                3 | HLS          | 1             |          10 |        1944 |        6 |              8 |        0.00244 |      0.9578 |         0.43 |
|                3 | YCrCb        | 0             |          10 |       10584 |        6 |              4 |        0.00244 |      0.9578 |         1.1  |
|                2 | YUV          | 1             |          10 |         324 |        9 |             16 |        0.00246 |      0.9578 |         0.08 |
|                1 | RGB          | 2             |          10 |         768 |       12 |              8 |        0.00247 |      0.9578 |         0.08 |
|                3 | YUV          | 0             |          10 |        3888 |       12 |              8 |        0.00247 |      0.9578 |         0.42 |
|                1 | HSV          | 0             |          10 |          96 |        6 |             16 |        0.00248 |      0.9578 |         0.05 |
|                1 | HSV          | 2             |          10 |        2304 |        9 |              4 |        0.00248 |      0.9578 |         0.17 |
|                1 | HLS          | 0             |          10 |         576 |        9 |              8 |        0.0025  |      0.9578 |         0.05 |
|                2 | YCrCb        | 0             |          10 |         216 |        6 |             16 |        0.0025  |      0.9578 |         0.17 |
|                4 | YUV          | 0             |          10 |        4800 |       12 |              8 |        0.0025  |      0.9578 |         0.7  |
|                2 | LUV          | 0             |          10 |        2352 |       12 |              8 |        0.00252 |      0.9578 |         0.2  |
|                1 | RGB          | ALL           |          10 |        4608 |        6 |              4 |        0.00252 |      0.9578 |         0.37 |
|                4 | HLS          | 1             |          10 |        3600 |        9 |              8 |        0.00254 |      0.9578 |         0.57 |
|                4 | LUV          | 2             |          10 |        2400 |        6 |              8 |        0.00255 |      0.9578 |         1.41 |
|                2 | YUV          | 0             |          10 |        1764 |        9 |              8 |        0.00259 |      0.9578 |         0.2  |
|                3 | RGB          | 0             |          10 |        1944 |        6 |              8 |        0.00259 |      0.9578 |         0.48 |
|                1 | LUV          | 1             |          10 |         192 |       12 |             16 |        0.0026  |      0.9578 |         0.08 |
|                3 | YUV          | 2             |          10 |         486 |        6 |             12 |        0.00262 |      0.9578 |         0.19 |
|                4 | YCrCb        | 2             |          10 |        4800 |       12 |              8 |        0.00262 |      0.9578 |         0.36 |
|                1 | YCrCb        | 2             |          10 |        3840 |       15 |              4 |        0.00267 |      0.9578 |         0.24 |
|                1 | YCrCb        | 0             |          10 |        3072 |       12 |              4 |        0.00268 |      0.9578 |         0.22 |
|                1 | RGB          | ALL           |          10 |         576 |       12 |             16 |        0.00269 |      0.9578 |         0.25 |
|                4 | LUV          | 0             |          10 |       16224 |        6 |              4 |        0.00269 |      0.9578 |         1.86 |
|                4 | HSV          | 2             |          10 |       16224 |        6 |              4 |        0.00271 |      0.9578 |         2    |
|                1 | HLS          | 1             |          10 |        1536 |        6 |              4 |        0.00274 |      0.9578 |         0.12 |
|                1 | HLS          | 1             |          10 |        2304 |        9 |              4 |        0.00279 |      0.9578 |         0.16 |
|                4 | HLS          | ALL           |          10 |        1152 |        6 |             12 |        0.0028  |      0.9578 |         0.18 |
|                3 | HLS          | 1             |          10 |       26460 |       15 |              4 |        0.0028  |      0.9578 |         2.46 |
|                4 | YUV          | 2             |          10 |       24336 |        9 |              4 |        0.00296 |      0.9578 |         1.97 |
|                4 | LUV          | 0             |          10 |       32448 |       12 |              4 |        0.00298 |      0.9578 |         3.31 |
|                3 | LUV          | 1             |          10 |         216 |        6 |             16 |        0.00181 |      0.9556 |         0.09 |
|                2 | RGB          | 0             |          10 |        2940 |       15 |              8 |        0.00189 |      0.9556 |         0.29 |
|                4 | LUV          | 2             |          10 |        3600 |        9 |              8 |        0.0019  |      0.9556 |         2.48 |
|                4 | YCrCb        | 2             |          10 |        6000 |       15 |              8 |        0.00199 |      0.9556 |         0.48 |
|                2 | RGB          | 2             |          10 |        2352 |       12 |              8 |        0.00209 |      0.9556 |         0.25 |
|                1 | YCrCb        | 2             |          10 |         576 |        9 |              8 |        0.00214 |      0.9556 |         0.05 |
|                1 | HLS          | 1             |          10 |         768 |       12 |              8 |        0.00231 |      0.9556 |         0.07 |
|                3 | YUV          | 2             |          10 |         729 |        9 |             12 |        0.00235 |      0.9556 |         0.25 |
|                1 | YCrCb        | 0             |          10 |         576 |        9 |              8 |        0.00239 |      0.9556 |         0.08 |
|                1 | LUV          | 0             |          10 |         144 |        9 |             16 |        0.00242 |      0.9556 |         0.13 |
|                1 | RGB          | 1             |          10 |         300 |       12 |             12 |        0.00243 |      0.9556 |         0.18 |
|                1 | YCrCb        | 1             |          10 |         150 |        6 |             12 |        0.00244 |      0.9556 |         0.05 |
|                2 | HLS          | 0             |          10 |         540 |       15 |             16 |        0.00245 |      0.9556 |         0.08 |
|                2 | RGB          | 2             |          10 |         216 |        6 |             16 |        0.00245 |      0.9556 |         0.17 |
|                4 | HLS          | ALL           |          10 |        1728 |        9 |             12 |        0.00246 |      0.9556 |         0.18 |
|                1 | HSV          | 0             |          10 |         192 |       12 |             16 |        0.00247 |      0.9556 |         0.02 |
|                1 | LUV          | 2             |          10 |         375 |       15 |             12 |        0.00247 |      0.9556 |         0.04 |
|                4 | YCrCb        | 2             |          10 |         240 |       15 |             16 |        0.00247 |      0.9556 |         0.07 |
|                3 | YUV          | 1             |          10 |         324 |        9 |             16 |        0.00247 |      0.9556 |         0.09 |
|                2 | HSV          | 2             |          10 |        2352 |       12 |              8 |        0.00247 |      0.9556 |         0.26 |
|                4 | YCrCb        | 1             |          10 |         144 |        9 |             16 |        0.00248 |      0.9556 |         0.06 |
|                3 | YCrCb        | 2             |          10 |         216 |        6 |             16 |        0.00248 |      0.9556 |         0.1  |
|                3 | LUV          | 2             |          10 |         486 |        6 |             12 |        0.00248 |      0.9556 |         0.21 |
|                3 | LUV          | 0             |          10 |         972 |       12 |             12 |        0.00248 |      0.9556 |         0.55 |
|                1 | LUV          | 1             |          10 |         375 |       15 |             12 |        0.00249 |      0.9556 |         0.04 |
|                3 | YUV          | 2             |          10 |         432 |       12 |             16 |        0.00249 |      0.9556 |         0.11 |
|                3 | HLS          | 0             |          10 |         432 |       12 |             16 |        0.00249 |      0.9556 |         0.14 |
|                1 | HLS          | 1             |          10 |         960 |       15 |              8 |        0.0025  |      0.9556 |         0.09 |
|                1 | RGB          | ALL           |          10 |        6912 |        9 |              4 |        0.0025  |      0.9556 |         0.82 |
|                3 | LUV          | 1             |          10 |         729 |        9 |             12 |        0.00251 |      0.9556 |         0.28 |
|                1 | RGB          | 0             |          10 |        3072 |       12 |              4 |        0.00251 |      0.9556 |         0.36 |
|                1 | YUV          | 2             |          10 |         300 |       12 |             12 |        0.00252 |      0.9556 |         0.02 |
|                2 | LUV          | 1             |          10 |         384 |        6 |             12 |        0.00252 |      0.9556 |         0.08 |
|                1 | HSV          | 2             |          10 |         144 |        9 |             16 |        0.00252 |      0.9556 |         0.17 |
|                3 | RGB          | ALL           |          10 |        8748 |        9 |              8 |        0.00255 |      0.9556 |         1.43 |
|                4 | HLS          | 1             |          10 |          96 |        6 |             16 |        0.0026  |      0.9556 |         0.14 |
|                3 | HLS          | 0             |          10 |         540 |       15 |             16 |        0.0026  |      0.9556 |         0.2  |
|                2 | LUV          | 0             |          10 |         768 |       12 |             12 |        0.00261 |      0.9556 |         0.23 |
|                1 | YCrCb        | 1             |          10 |         225 |        9 |             12 |        0.00265 |      0.9556 |         0.04 |
|                3 | HLS          | 1             |          10 |         540 |       15 |             16 |        0.00265 |      0.9556 |         0.3  |
|                4 | HLS          | 1             |          10 |       16224 |        6 |              4 |        0.00269 |      0.9556 |         2.28 |
|                3 | RGB          | 0             |          10 |        3888 |       12 |              8 |        0.00271 |      0.9556 |         0.51 |
|                2 | LUV          | 2             |          10 |        1764 |        9 |              8 |        0.00277 |      0.9556 |         0.16 |
|                4 | LUV          | 0             |          10 |        3600 |        9 |              8 |        0.00283 |      0.9556 |         0.51 |
|                3 | HSV          | 2             |          10 |         216 |        6 |             16 |        0.00421 |      0.9556 |         0.19 |
|                3 | YCrCb        | 2             |          10 |        2916 |        9 |              8 |        0.00476 |      0.9556 |         0.3  |
|                1 | YUV          | 0             |          10 |         144 |        9 |             16 |        0.00102 |      0.9533 |         0.13 |
|                3 | RGB          | 2             |          10 |        3888 |       12 |              8 |        0.00183 |      0.9533 |         0.43 |
|                1 | LUV          | 0             |          10 |         768 |       12 |              8 |        0.002   |      0.9533 |         0.08 |
|                4 | HSV          | 0             |          10 |         192 |       12 |             16 |        0.00219 |      0.9533 |         0.09 |
|                2 | HSV          | 2             |          10 |       10800 |       12 |              4 |        0.00225 |      0.9533 |         0.92 |
|                3 | RGB          | 2             |          10 |       10584 |        6 |              4 |        0.00229 |      0.9533 |         1.11 |
|                4 | LUV          | 2             |          10 |          96 |        6 |             16 |        0.0023  |      0.9533 |         0.08 |
|                2 | RGB          | 0             |          10 |       10800 |       12 |              4 |        0.00234 |      0.9533 |         0.95 |
|                2 | HLS          | 0             |          10 |         768 |       12 |             12 |        0.00238 |      0.9533 |         0.06 |
|                3 | RGB          | 0             |          10 |        4860 |       15 |              8 |        0.00238 |      0.9533 |         0.61 |
|                1 | HSV          | 0             |          10 |         144 |        9 |             16 |        0.00241 |      0.9533 |         0.1  |
|                1 | RGB          | 1             |          10 |         150 |        6 |             12 |        0.00241 |      0.9533 |         0.14 |
|                1 | HSV          | 0             |          10 |         576 |        9 |              8 |        0.00243 |      0.9533 |         0.05 |
|                1 | LUV          | 2             |          10 |        1536 |        6 |              4 |        0.00245 |      0.9533 |         0.11 |
|                2 | HSV          | 2             |          10 |        1176 |        6 |              8 |        0.00245 |      0.9533 |         0.25 |
|                1 | YUV          | 0             |          10 |         192 |       12 |             16 |        0.00247 |      0.9533 |         0.1  |
|                4 | RGB          | 0             |          10 |        3600 |        9 |              8 |        0.00247 |      0.9533 |         0.96 |
|                1 | YUV          | 1             |          10 |         192 |       12 |             16 |        0.00248 |      0.9533 |         0.04 |
|                2 | HLS          | 0             |          10 |         960 |       15 |             12 |        0.0025  |      0.9533 |         0.08 |
|                3 | YUV          | 0             |          10 |         216 |        6 |             16 |        0.0025  |      0.9533 |         0.18 |
|                2 | RGB          | ALL           |          10 |        1296 |       12 |             16 |        0.0025  |      0.9533 |         0.31 |
|                3 | HLS          | 1             |          10 |        4860 |       15 |              8 |        0.0025  |      0.9533 |         0.54 |
|                1 | YCrCb        | 0             |          10 |        1536 |        6 |              4 |        0.00251 |      0.9533 |         0.12 |
|                1 | RGB          | 2             |          10 |        3072 |       12 |              4 |        0.00251 |      0.9533 |         0.25 |
|                3 | LUV          | 0             |          10 |        1944 |        6 |              8 |        0.00251 |      0.9533 |         0.58 |
|                4 | LUV          | 2             |          10 |         240 |       15 |             16 |        0.00254 |      0.9533 |         0.1  |
|                4 | LUV          | 1             |          10 |         576 |        9 |             12 |        0.00254 |      0.9533 |         0.13 |
|                3 | HSV          | 0             |          10 |        1944 |        6 |              8 |        0.00254 |      0.9533 |         0.19 |
|                4 | YCrCb        | 2             |          10 |        2400 |        6 |              8 |        0.00257 |      0.9533 |         0.31 |
|                2 | RGB          | 1             |          10 |        2352 |       12 |              8 |        0.00262 |      0.9533 |         0.26 |
|                2 | LUV          | 0             |          10 |        1176 |        6 |              8 |        0.00265 |      0.9533 |         0.26 |
|                1 | RGB          | ALL           |          10 |        1125 |       15 |             12 |        0.00267 |      0.9533 |         0.23 |
|                3 | YCrCb        | 2             |          10 |        1944 |        6 |              8 |        0.00269 |      0.9533 |         0.2  |
|                2 | HLS          | 1             |          10 |       13500 |       15 |              4 |        0.00269 |      0.9533 |         1.19 |
|                4 | YUV          | 0             |          10 |        6000 |       15 |              8 |        0.0027  |      0.9533 |         0.83 |
|                3 | YUV          | 0             |          10 |        2916 |        9 |              8 |        0.00272 |      0.9533 |         0.36 |
|                4 | HLS          | 0             |          10 |          96 |        6 |             16 |        0.00279 |      0.9533 |         0.1  |
|                4 | RGB          | ALL           |          10 |       14400 |       12 |              8 |        0.00283 |      0.9533 |         1.94 |
|                4 | RGB          | 0             |          10 |       24336 |        9 |              4 |        0.00284 |      0.9533 |         3.21 |
|                4 | RGB          | ALL           |          10 |       18000 |       15 |              8 |        0.00293 |      0.9533 |         2.32 |
|                1 | HLS          | 1             |          10 |         576 |        9 |              8 |        0.00324 |      0.9533 |         0.08 |
|                3 | LUV          | 2             |          10 |         540 |       15 |             16 |        0.00377 |      0.9533 |         0.18 |
|                2 | LUV          | 0             |          10 |         960 |       15 |             12 |        0.00422 |      0.9533 |         0.24 |
|                4 | LUV          | 2             |          10 |        6000 |       15 |              8 |        0.00494 |      0.9533 |         5.58 |
|                4 | RGB          | ALL           |          10 |         576 |       12 |             16 |        0.00528 |      0.9533 |         0.17 |
|                3 | LUV          | 2             |          10 |        1215 |       15 |             12 |        0.00725 |      0.9533 |         0.2  |
|                1 | HSV          | 2             |          10 |         576 |        9 |              8 |        0.00141 |      0.9511 |         0.11 |
|                2 | YCrCb        | 0             |          10 |         540 |       15 |             16 |        0.00151 |      0.9511 |         0.28 |
|                2 | HSV          | 2             |          10 |         960 |       15 |             12 |        0.0018  |      0.9511 |         0.26 |
|                3 | LUV          | 0             |          10 |        2916 |        9 |              8 |        0.00184 |      0.9511 |         0.44 |
|                2 | YCrCb        | 2             |          10 |        2352 |       12 |              8 |        0.00197 |      0.9511 |         0.14 |
|                4 | YCrCb        | 0             |          10 |        6000 |       15 |              8 |        0.002   |      0.9511 |         0.76 |
|                2 | HSV          | 0             |          10 |         384 |        6 |             12 |        0.00201 |      0.9511 |         0.11 |
|                1 | RGB          | ALL           |          10 |         720 |       15 |             16 |        0.00207 |      0.9511 |         0.24 |
|                1 | YUV          | 0             |          10 |         768 |       12 |              8 |        0.00212 |      0.9511 |         0.07 |
|                1 | RGB          | 2             |          10 |         960 |       15 |              8 |        0.00212 |      0.9511 |         0.08 |
|                2 | HLS          | 1             |          10 |        1176 |        6 |              8 |        0.00227 |      0.9511 |         0.24 |
|                4 | YCrCb        | 2             |          10 |         144 |        9 |             16 |        0.00231 |      0.9511 |         0.08 |
|                3 | YCrCb        | 0             |          10 |        3888 |       12 |              8 |        0.00238 |      0.9511 |         0.4  |
|                4 | HSV          | 0             |          10 |        4800 |       12 |              8 |        0.0024  |      0.9511 |         0.35 |
|                4 | HSV          | 0             |          10 |         240 |       15 |             16 |        0.00241 |      0.9511 |         0.12 |
|                2 | HLS          | 1             |          10 |         768 |       12 |             12 |        0.00241 |      0.9511 |         0.25 |
|                4 | HSV          | ALL           |          10 |        1728 |        9 |             12 |        0.00244 |      0.9511 |         0.19 |
|                1 | LUV          | 1             |          10 |         300 |       12 |             12 |        0.00246 |      0.9511 |         0.03 |
|                3 | YUV          | 2             |          10 |         216 |        6 |             16 |        0.00246 |      0.9511 |         0.09 |
|                1 | RGB          | 2             |          10 |        2304 |        9 |              4 |        0.00246 |      0.9511 |         0.17 |
|                3 | HSV          | 2             |          10 |        2916 |        9 |              8 |        0.00246 |      0.9511 |         0.39 |
|                3 | HSV          | 1             |          10 |        4860 |       15 |              8 |        0.00247 |      0.9511 |         0.74 |
|                2 | YUV          | 2             |          10 |        1176 |        6 |              8 |        0.00248 |      0.9511 |         0.09 |
|                2 | RGB          | 2             |          10 |        1176 |        6 |              8 |        0.00248 |      0.9511 |         0.36 |
|                1 | YUV          | 1             |          10 |         384 |        6 |              8 |        0.00249 |      0.9511 |         0.04 |
|                3 | YUV          | 2             |          10 |         324 |        9 |             16 |        0.0025  |      0.9511 |         0.13 |
|                3 | RGB          | 0             |          10 |        2916 |        9 |              8 |        0.0025  |      0.9511 |         0.45 |
|                4 | RGB          | 2             |          10 |         192 |       12 |             16 |        0.00253 |      0.9511 |         0.13 |
|                3 | RGB          | 2             |          10 |        2916 |        9 |              8 |        0.00253 |      0.9511 |         0.42 |
|                4 | RGB          | 2             |          10 |        3600 |        9 |              8 |        0.00254 |      0.9511 |         0.94 |
|                1 | HLS          | 0             |          10 |          96 |        6 |             16 |        0.00258 |      0.9511 |         0.06 |
|                4 | RGB          | ALL           |          10 |       10800 |        9 |              8 |        0.00258 |      0.9511 |         1.52 |
|                2 | YCrCb        | 0             |          10 |        2352 |       12 |              8 |        0.00264 |      0.9511 |         0.21 |
|                2 | YUV          | 0             |          10 |        2940 |       15 |              8 |        0.00276 |      0.9511 |         0.32 |
|                4 | RGB          | 1             |          10 |       24336 |        9 |              4 |        0.00295 |      0.9511 |         2.63 |
|                1 | YCrCb        | 2             |          10 |         768 |       12 |              8 |        0.00428 |      0.9511 |         0.05 |
|                1 | YCrCb        | 2             |          10 |        2304 |        9 |              4 |        0.00436 |      0.9511 |         0.14 |
|                1 | RGB          | 0             |          10 |        3840 |       15 |              4 |        0.00567 |      0.9511 |         0.33 |
|                3 | HSV          | 1             |          10 |       26460 |       15 |              4 |        0.0058  |      0.9511 |         2.5  |
|                1 | YCrCb        | 1             |          10 |          96 |        6 |             16 |        0.0014  |      0.9489 |         0.05 |
|                2 | YUV          | 0             |          10 |         960 |       15 |             12 |        0.00151 |      0.9489 |         0.3  |
|                1 | RGB          | 2             |          10 |        1536 |        6 |              4 |        0.00157 |      0.9489 |         0.13 |
|                4 | YUV          | 2             |          10 |         240 |       15 |             16 |        0.00173 |      0.9489 |         0.09 |
|                1 | LUV          | 1             |          10 |         240 |       15 |             16 |        0.00176 |      0.9489 |         0.06 |
|                3 | HLS          | 0             |          10 |         972 |       12 |             12 |        0.00182 |      0.9489 |         0.1  |
|                2 | YCrCb        | 0             |          10 |         576 |        9 |             12 |        0.00187 |      0.9489 |         0.37 |
|                3 | YCrCb        | 2             |          10 |         486 |        6 |             12 |        0.00189 |      0.9489 |         0.21 |
|                2 | LUV          | 0             |          10 |         432 |       12 |             16 |        0.00192 |      0.9489 |         0.21 |
|                3 | LUV          | 1             |          10 |         324 |        9 |             16 |        0.00197 |      0.9489 |         0.12 |
|                3 | RGB          | 1             |          10 |         216 |        6 |             16 |        0.00198 |      0.9489 |         0.18 |
|                2 | RGB          | ALL           |          10 |        1152 |        6 |             12 |        0.00202 |      0.9489 |         0.57 |
|                2 | LUV          | 0             |          10 |         216 |        6 |             16 |        0.00233 |      0.9489 |         0.17 |
|                1 | YUV          | 2             |          10 |        1536 |        6 |              4 |        0.00234 |      0.9489 |         0.1  |
|                1 | RGB          | 0             |          10 |        2304 |        9 |              4 |        0.00234 |      0.9489 |         0.18 |
|                2 | HSV          | 0             |          10 |         216 |        6 |             16 |        0.00239 |      0.9489 |         0.1  |
|                3 | HLS          | 1             |          10 |         432 |       12 |             16 |        0.00242 |      0.9489 |         0.22 |
|                2 | RGB          | 2             |          10 |         540 |       15 |             16 |        0.00242 |      0.9489 |         0.3  |
|                3 | YUV          | 1             |          10 |         216 |        6 |             16 |        0.00243 |      0.9489 |         0.07 |
|                1 | LUV          | 1             |          10 |        1536 |        6 |              4 |        0.00243 |      0.9489 |         0.1  |
|                3 | HSV          | 0             |          10 |         216 |        6 |             16 |        0.00243 |      0.9489 |         0.11 |
|                1 | RGB          | ALL           |          10 |         900 |       12 |             12 |        0.00243 |      0.9489 |         0.13 |
|                3 | RGB          | 1             |          10 |         324 |        9 |             16 |        0.00243 |      0.9489 |         0.17 |
|                3 | RGB          | 1             |          10 |        4860 |       15 |              8 |        0.00243 |      0.9489 |         0.59 |
|                1 | YUV          | 2             |          10 |         384 |        6 |              8 |        0.00244 |      0.9489 |         0.05 |
|                2 | YUV          | 2             |          10 |         576 |        9 |             12 |        0.00244 |      0.9489 |         0.09 |
|                1 | LUV          | 1             |          10 |          96 |        6 |             16 |        0.00245 |      0.9489 |         0.1  |
|                1 | RGB          | 2             |          10 |         144 |        9 |             16 |        0.00245 |      0.9489 |         0.12 |
|                3 | RGB          | ALL           |          10 |         648 |        6 |             16 |        0.00248 |      0.9489 |         0.32 |
|                2 | RGB          | ALL           |          10 |        1728 |        9 |             12 |        0.00249 |      0.9489 |         0.39 |
|                4 | RGB          | 1             |          10 |         960 |       15 |             12 |        0.00249 |      0.9489 |         0.43 |
|                2 | LUV          | 2             |          10 |        1176 |        6 |              8 |        0.0025  |      0.9489 |         0.12 |
|                3 | HSV          | 0             |          10 |         432 |       12 |             16 |        0.00252 |      0.9489 |         0.14 |
|                2 | HSV          | 1             |          10 |        2940 |       15 |              8 |        0.00252 |      0.9489 |         0.32 |
|                3 | YUV          | 0             |          10 |        1944 |        6 |              8 |        0.00253 |      0.9489 |         0.47 |
|                4 | LUV          | 0             |          10 |        4800 |       12 |              8 |        0.00253 |      0.9489 |         0.6  |
|                2 | HLS          | 1             |          10 |       10800 |       12 |              4 |        0.00257 |      0.9489 |         0.91 |
|                2 | YCrCb        | 2             |          10 |        1176 |        6 |              8 |        0.00263 |      0.9489 |         0.11 |
|                3 | YUV          | 0             |          10 |       10584 |        6 |              4 |        0.00263 |      0.9489 |         1.31 |
|                1 | LUV          | 1             |          10 |         225 |        9 |             12 |        0.00264 |      0.9489 |         0.09 |
|                4 | RGB          | 1             |          10 |        6000 |       15 |              8 |        0.00272 |      0.9489 |         0.86 |
|                2 | HLS          | 1             |          10 |        1764 |        9 |              8 |        0.00275 |      0.9489 |         0.23 |
|                4 | RGB          | 0             |          10 |       32448 |       12 |              4 |        0.00312 |      0.9489 |         3.79 |
|                2 | RGB          | ALL           |          10 |        2304 |       12 |             12 |        0.00359 |      0.9489 |         0.45 |
|                1 | RGB          | 0             |          10 |        1536 |        6 |              4 |        0.00177 |      0.9467 |         0.14 |
|                1 | HLS          | 0             |          10 |         192 |       12 |             16 |        0.00208 |      0.9467 |         0.04 |
|                1 | YCrCb        | 0             |          10 |          96 |        6 |             16 |        0.00211 |      0.9467 |         0.15 |
|                4 | RGB          | ALL           |          10 |         288 |        6 |             16 |        0.00236 |      0.9467 |         0.18 |
|                3 | RGB          | 2             |          10 |        4860 |       15 |              8 |        0.00236 |      0.9467 |         0.48 |
|                4 | RGB          | 2             |          10 |         144 |        9 |             16 |        0.00238 |      0.9467 |         0.13 |
|                3 | YUV          | 0             |          10 |        4860 |       15 |              8 |        0.00238 |      0.9467 |         0.52 |
|                4 | RGB          | 0             |          10 |        4800 |       12 |              8 |        0.00238 |      0.9467 |         0.82 |
|                4 | HLS          | 0             |          10 |         768 |       12 |             12 |        0.00239 |      0.9467 |         0.14 |
|                4 | HLS          | 1             |          10 |         240 |       15 |             16 |        0.0024  |      0.9467 |         0.14 |
|                4 | HLS          | 0             |          10 |         240 |       15 |             16 |        0.00242 |      0.9467 |         0.13 |
|                2 | LUV          | 0             |          10 |         540 |       15 |             16 |        0.00242 |      0.9467 |         0.26 |
|                2 | RGB          | 1             |          10 |         540 |       15 |             16 |        0.00242 |      0.9467 |         0.31 |
|                1 | LUV          | 0             |          10 |         300 |       12 |             12 |        0.00244 |      0.9467 |         0.13 |
|                1 | YUV          | 0             |          10 |         384 |        6 |              8 |        0.00245 |      0.9467 |         0.13 |
|                1 | YCrCb        | 2             |          10 |         240 |       15 |             16 |        0.00247 |      0.9467 |         0.05 |
|                4 | LUV          | 2             |          10 |         192 |       12 |             16 |        0.00247 |      0.9467 |         0.06 |
|                1 | YUV          | 2             |          10 |         192 |       12 |             16 |        0.00247 |      0.9467 |         0.08 |
|                4 | YUV          | 2             |          10 |         144 |        9 |             16 |        0.00248 |      0.9467 |         0.07 |
|                4 | RGB          | 1             |          10 |        2400 |        6 |              8 |        0.00248 |      0.9467 |         0.84 |
|                3 | RGB          | ALL           |          10 |        2187 |        9 |             12 |        0.0025  |      0.9467 |         0.84 |
|                3 | HLS          | 1             |          10 |        3888 |       12 |              8 |        0.00251 |      0.9467 |         0.43 |
|                1 | HLS          | 1             |          10 |        3840 |       15 |              4 |        0.00252 |      0.9467 |         0.29 |
|                3 | YUV          | 0             |          10 |         729 |        9 |             12 |        0.00252 |      0.9467 |         0.39 |
|                3 | YCrCb        | 0             |          10 |         216 |        6 |             16 |        0.00253 |      0.9467 |         0.2  |
|                3 | RGB          | 2             |          10 |         216 |        6 |             16 |        0.00255 |      0.9467 |         0.2  |
|                3 | HSV          | 0             |          10 |        1215 |       15 |             12 |        0.00256 |      0.9467 |         0.19 |
|                2 | YCrCb        | 2             |          10 |         576 |        9 |             12 |        0.0026  |      0.9467 |         0.16 |
|                3 | RGB          | ALL           |          10 |        1296 |       12 |             16 |        0.0026  |      0.9467 |         0.56 |
|                3 | YUV          | 0             |          10 |         486 |        6 |             12 |        0.00265 |      0.9467 |         0.26 |
|                3 | YCrCb        | 0             |          10 |         540 |       15 |             16 |        0.00269 |      0.9467 |         0.26 |
|                1 | LUV          | 0             |          10 |         576 |        9 |              8 |        0.00271 |      0.9467 |         0.06 |
|                1 | HLS          | 2             |          10 |        3072 |       12 |              4 |        0.00271 |      0.9467 |         0.24 |
|                4 | RGB          | 1             |          10 |       16224 |        6 |              4 |        0.0029  |      0.9467 |         2.12 |
|                3 | HLS          | 1             |          10 |         324 |        9 |             16 |        0.00301 |      0.9467 |         0.18 |
|                2 | HLS          | 1             |          10 |        5400 |        6 |              4 |        0.00302 |      0.9467 |         0.54 |
|                2 | HSV          | 2             |          10 |         324 |        9 |             16 |        0.00398 |      0.9467 |         0.18 |
|                3 | YCrCb        | 0             |          10 |        2916 |        9 |              8 |        0.00178 |      0.9444 |         0.57 |
|                3 | LUV          | 0             |          10 |         729 |        9 |             12 |        0.00185 |      0.9444 |         0.4  |
|                4 | LUV          | 0             |          10 |         144 |        9 |             16 |        0.00189 |      0.9444 |         0.12 |
|                1 | RGB          | 1             |          10 |         960 |       15 |              8 |        0.00192 |      0.9444 |         0.07 |
|                3 | HSV          | 2             |          10 |         972 |       12 |             12 |        0.00199 |      0.9444 |         0.57 |
|                3 | HSV          | 2             |          10 |        4860 |       15 |              8 |        0.00202 |      0.9444 |         0.52 |
|                4 | RGB          | 2             |          10 |        6000 |       15 |              8 |        0.00204 |      0.9444 |         0.84 |
|                2 | HLS          | 1             |          10 |         216 |        6 |             16 |        0.0021  |      0.9444 |         0.15 |
|                1 | YUV          | 0             |          10 |         576 |        9 |              8 |        0.00213 |      0.9444 |         0.06 |
|                2 | HLS          | 2             |          10 |       10800 |       12 |              4 |        0.00214 |      0.9444 |         0.95 |
|                3 | RGB          | 0             |          10 |       10584 |        6 |              4 |        0.00231 |      0.9444 |         1.29 |
|                3 | HLS          | 1             |          10 |       10584 |        6 |              4 |        0.00232 |      0.9444 |         1.28 |
|                4 | YCrCb        | 0             |          10 |        4800 |       12 |              8 |        0.00236 |      0.9444 |         0.61 |
|                4 | HSV          | 0             |          10 |          96 |        6 |             16 |        0.0024  |      0.9444 |         0.09 |
|                2 | YCrCb        | 2             |          10 |        8100 |        9 |              4 |        0.00241 |      0.9444 |         0.63 |
|                2 | LUV          | 0             |          10 |         576 |        9 |             12 |        0.00243 |      0.9444 |         0.26 |
|                2 | RGB          | 0             |          10 |         540 |       15 |             16 |        0.00247 |      0.9444 |         0.31 |
|                3 | HSV          | 2             |          10 |        1944 |        6 |              8 |        0.00247 |      0.9444 |         0.35 |
|                1 | LUV          | 1             |          10 |         576 |        9 |              8 |        0.00249 |      0.9444 |         0.06 |
|                2 | YUV          | 2             |          10 |         384 |        6 |             12 |        0.00249 |      0.9444 |         0.13 |
|                1 | LUV          | 0             |          10 |          96 |        6 |             16 |        0.00249 |      0.9444 |         0.14 |
|                4 | YUV          | 2             |          10 |        2400 |        6 |              8 |        0.0025  |      0.9444 |         0.28 |
|                4 | HSV          | 2             |          10 |        6000 |       15 |              8 |        0.0025  |      0.9444 |         0.83 |
|                2 | RGB          | 1             |          10 |         384 |        6 |             12 |        0.00251 |      0.9444 |         0.2  |
|                1 | RGB          | ALL           |          10 |         675 |        9 |             12 |        0.00251 |      0.9444 |         0.33 |
|                2 | HLS          | 1             |          10 |         960 |       15 |             12 |        0.00252 |      0.9444 |         0.23 |
|                4 | RGB          | 1             |          10 |        3600 |        9 |              8 |        0.00253 |      0.9444 |         0.68 |
|                3 | HSV          | 0             |          10 |        2916 |        9 |              8 |        0.00258 |      0.9444 |         0.29 |
|                1 | YUV          | 0             |          10 |          96 |        6 |             16 |        0.00259 |      0.9444 |         0.13 |
|                4 | LUV          | 0             |          10 |        2400 |        6 |              8 |        0.0026  |      0.9444 |         0.85 |
|                2 | LUV          | 1             |          10 |         216 |        6 |             16 |        0.00262 |      0.9444 |         0.08 |
|                3 | YCrCb        | 0             |          10 |         432 |       12 |             16 |        0.00263 |      0.9444 |         0.21 |
|                2 | HSV          | 1             |          10 |        5400 |        6 |              4 |        0.00265 |      0.9444 |         0.56 |
|                2 | RGB          | 0             |          10 |        2352 |       12 |              8 |        0.00269 |      0.9444 |         0.24 |
|                4 | RGB          | 0             |          10 |       16224 |        6 |              4 |        0.00279 |      0.9444 |         2.09 |
|                3 | HSV          | 0             |          10 |         540 |       15 |             16 |        0.00284 |      0.9444 |         0.19 |
|                4 | LUV          | 0             |          10 |        6000 |       15 |              8 |        0.00286 |      0.9444 |         0.93 |
|                2 | YCrCb        | 0             |          10 |        1176 |        6 |              8 |        0.00302 |      0.9444 |         0.23 |
|                2 | RGB          | 1             |          10 |         432 |       12 |             16 |        0.00599 |      0.9444 |         0.21 |
|                2 | YCrCb        | 2             |          10 |         216 |        6 |             16 |        0.00158 |      0.9422 |         0.11 |
|                1 | LUV          | 2             |          10 |         240 |       15 |             16 |        0.00169 |      0.9422 |         0.12 |
|                3 | HLS          | 1             |          10 |         972 |       12 |             12 |        0.00179 |      0.9422 |         0.51 |
|                1 | YUV          | 0             |          10 |        1536 |        6 |              4 |        0.00183 |      0.9422 |         0.11 |
|                1 | LUV          | 2             |          10 |         192 |       12 |             16 |        0.00187 |      0.9422 |         0.08 |
|                1 | RGB          | 2             |          10 |         576 |        9 |              8 |        0.00188 |      0.9422 |         0.08 |
|                2 | HSV          | 0             |          10 |         576 |        9 |             12 |        0.00189 |      0.9422 |         0.11 |
|                3 | LUV          | 2             |          10 |         324 |        9 |             16 |        0.0019  |      0.9422 |         0.14 |
|                4 | RGB          | 0             |          10 |        6000 |       15 |              8 |        0.0019  |      0.9422 |         1.08 |
|                3 | RGB          | 1             |          10 |        3888 |       12 |              8 |        0.00204 |      0.9422 |         0.56 |
|                2 | HLS          | 2             |          10 |       13500 |       15 |              4 |        0.00227 |      0.9422 |         1.29 |
|                1 | HLS          | 0             |          10 |         144 |        9 |             16 |        0.00239 |      0.9422 |         0.09 |
|                2 | YUV          | 2             |          10 |         324 |        9 |             16 |        0.00239 |      0.9422 |         0.13 |
|                3 | RGB          | 0             |          10 |         324 |        9 |             16 |        0.0024  |      0.9422 |         0.21 |
|                1 | YUV          | 0             |          10 |         240 |       15 |             16 |        0.00241 |      0.9422 |         0.13 |
|                3 | HLS          | 0             |          10 |         486 |        6 |             12 |        0.00242 |      0.9422 |         0.23 |
|                1 | YCrCb        | 0             |          10 |         300 |       12 |             12 |        0.00243 |      0.9422 |         0.1  |
|                1 | YUV          | 2             |          10 |         240 |       15 |             16 |        0.00244 |      0.9422 |         0.05 |
|                1 | LUV          | 1             |          10 |         144 |        9 |             16 |        0.00244 |      0.9422 |         0.07 |
|                4 | HSV          | 2             |          10 |         144 |        9 |             16 |        0.00244 |      0.9422 |         0.13 |
|                4 | YCrCb        | 2             |          10 |         384 |        6 |             12 |        0.00244 |      0.9422 |         0.16 |
|                2 | LUV          | 2             |          10 |         960 |       15 |             12 |        0.00245 |      0.9422 |         0.11 |
|                3 | RGB          | 0             |          10 |         216 |        6 |             16 |        0.00245 |      0.9422 |         0.2  |
|                2 | HLS          | 2             |          10 |        2352 |       12 |              8 |        0.00247 |      0.9422 |         0.37 |
|                3 | YUV          | 0             |          10 |        1215 |       15 |             12 |        0.00247 |      0.9422 |         0.42 |
|                2 | YCrCb        | 2             |          10 |         324 |        9 |             16 |        0.00248 |      0.9422 |         0.15 |
|                4 | RGB          | 0             |          10 |        2400 |        6 |              8 |        0.00249 |      0.9422 |         0.71 |
|                2 | RGB          | 2             |          10 |         432 |       12 |             16 |        0.0025  |      0.9422 |         0.22 |
|                2 | HSV          | 1             |          10 |        2352 |       12 |              8 |        0.00251 |      0.9422 |         0.28 |
|                4 | HLS          | 0             |          10 |        3600 |        9 |              8 |        0.00251 |      0.9422 |         0.46 |
|                4 | YUV          | 0             |          10 |         192 |       12 |             16 |        0.00252 |      0.9422 |         0.11 |
|                2 | RGB          | 2             |          10 |        1764 |        9 |              8 |        0.00252 |      0.9422 |         0.23 |
|                2 | RGB          | 1             |          10 |         768 |       12 |             12 |        0.00253 |      0.9422 |         0.15 |
|                2 | YCrCb        | 0             |          10 |         960 |       15 |             12 |        0.00253 |      0.9422 |         0.17 |
|                3 | RGB          | ALL           |          10 |        1458 |        6 |             12 |        0.00254 |      0.9422 |         0.77 |
|                2 | HLS          | 0             |          10 |         216 |        6 |             16 |        0.00256 |      0.9422 |         0.09 |
|                2 | HSV          | 1             |          10 |       10800 |       12 |              4 |        0.00266 |      0.9422 |         0.96 |
|                4 | LUV          | 0             |          10 |         960 |       15 |             12 |        0.00272 |      0.9422 |         0.6  |
|                3 | HLS          | 2             |          10 |       26460 |       15 |              4 |        0.00301 |      0.9422 |         2.56 |
|                1 | RGB          | 1             |          10 |         375 |       15 |             12 |        0.00171 |      0.94   |         0.14 |
|                2 | YUV          | 2             |          10 |         960 |       15 |             12 |        0.0018  |      0.94   |         0.08 |
|                4 | RGB          | ALL           |          10 |        2880 |       15 |             12 |        0.00191 |      0.94   |         0.49 |
|                2 | HSV          | 2             |          10 |        1764 |        9 |              8 |        0.00197 |      0.94   |         0.2  |
|                3 | YUV          | 0             |          10 |         540 |       15 |             16 |        0.00201 |      0.94   |         0.32 |
|                1 | LUV          | 2             |          10 |         144 |        9 |             16 |        0.00235 |      0.94   |         0.11 |
|                2 | YUV          | 0             |          10 |        2352 |       12 |              8 |        0.00237 |      0.94   |         0.23 |
|                1 | HLS          | 0             |          10 |         225 |        9 |             12 |        0.00241 |      0.94   |         0.1  |
|                4 | HSV          | 2             |          10 |          96 |        6 |             16 |        0.00242 |      0.94   |         0.14 |
|                1 | RGB          | ALL           |          10 |         450 |        6 |             12 |        0.00243 |      0.94   |         0.27 |
|                2 | RGB          | 1             |          10 |        1176 |        6 |              8 |        0.00244 |      0.94   |         0.19 |
|                4 | YCrCb        | 2             |          10 |         960 |       15 |             12 |        0.00245 |      0.94   |         0.12 |
|                1 | LUV          | 2             |          10 |         576 |        9 |              8 |        0.00246 |      0.94   |         0.05 |
|                2 | RGB          | 1             |          10 |         576 |        9 |             12 |        0.00246 |      0.94   |         0.32 |
|                1 | HSV          | 2             |          10 |         960 |       15 |              8 |        0.00248 |      0.94   |         0.09 |
|                4 | YUV          | 0             |          10 |         144 |        9 |             16 |        0.00248 |      0.94   |         0.13 |
|                3 | HSV          | 1             |          10 |         486 |        6 |             12 |        0.00248 |      0.94   |         0.39 |
|                1 | YUV          | 2             |          10 |         144 |        9 |             16 |        0.00249 |      0.94   |         0.07 |
|                4 | LUV          | 2             |          10 |         576 |        9 |             12 |        0.00249 |      0.94   |         0.31 |
|                4 | HLS          | 1             |          10 |         768 |       12 |             12 |        0.00249 |      0.94   |         0.5  |
|                4 | RGB          | 0             |          10 |          96 |        6 |             16 |        0.0025  |      0.94   |         0.18 |
|                2 | HLS          | 0             |          10 |         576 |        9 |             12 |        0.00252 |      0.94   |         0.15 |
|                4 | HLS          | 1             |          10 |        2400 |        6 |              8 |        0.00257 |      0.94   |         1.04 |
|                3 | HLS          | 1             |          10 |         486 |        6 |             12 |        0.00258 |      0.94   |         0.32 |
|                1 | YUV          | 1             |          10 |          96 |        6 |             16 |        0.00261 |      0.94   |         0.05 |
|                3 | YUV          | 2             |          10 |        1944 |        6 |              8 |        0.00272 |      0.94   |         0.23 |
|                2 | YUV          | 0             |          10 |         768 |       12 |             12 |        0.00274 |      0.94   |         0.28 |
|                2 | RGB          | 1             |          10 |         960 |       15 |             12 |        0.00278 |      0.94   |         0.17 |
|                4 | HLS          | 2             |          10 |       40560 |       15 |              4 |        0.00308 |      0.94   |         4.27 |
|                2 | HSV          | 2             |          10 |         768 |       12 |             12 |        0.00532 |      0.94   |         0.19 |
|                1 | YCrCb        | 2             |          10 |          96 |        6 |             16 |        0.00147 |      0.9378 |         0.11 |
|                4 | HSV          | 0             |          10 |         960 |       15 |             12 |        0.00153 |      0.9378 |         0.22 |
|                1 | YCrCb        | 0             |          10 |         240 |       15 |             16 |        0.00175 |      0.9378 |         0.13 |
|                3 | LUV          | 0             |          10 |         324 |        9 |             16 |        0.00183 |      0.9378 |         0.17 |
|                3 | HLS          | 1             |          10 |        2916 |        9 |              8 |        0.00184 |      0.9378 |         0.38 |
|                1 | RGB          | 0             |          10 |         768 |       12 |              8 |        0.00186 |      0.9378 |         0.09 |
|                2 | RGB          | 1             |          10 |         216 |        6 |             16 |        0.00192 |      0.9378 |         0.16 |
|                2 | RGB          | 1             |          10 |        2940 |       15 |              8 |        0.00192 |      0.9378 |         0.29 |
|                3 | RGB          | ALL           |          10 |        2916 |       12 |             12 |        0.00193 |      0.9378 |         0.55 |
|                4 | RGB          | 1             |          10 |          96 |        6 |             16 |        0.00233 |      0.9378 |         0.13 |
|                1 | HLS          | 1             |          10 |         375 |       15 |             12 |        0.00241 |      0.9378 |         0.1  |
|                3 | YUV          | 0             |          10 |         324 |        9 |             16 |        0.00241 |      0.9378 |         0.17 |
|                2 | HLS          | 1             |          10 |         540 |       15 |             16 |        0.00242 |      0.9378 |         0.29 |
|                4 | RGB          | 2             |          10 |        4800 |       12 |              8 |        0.00242 |      0.9378 |         0.72 |
|                4 | RGB          | 1             |          10 |         768 |       12 |             12 |        0.00243 |      0.9378 |         0.4  |
|                1 | HSV          | 2             |          10 |         768 |       12 |              8 |        0.00244 |      0.9378 |         0.07 |
|                2 | HSV          | 2             |          10 |         540 |       15 |             16 |        0.00244 |      0.9378 |         0.3  |
|                2 | LUV          | 0             |          10 |         324 |        9 |             16 |        0.00245 |      0.9378 |         0.16 |
|                3 | RGB          | 1             |          10 |         540 |       15 |             16 |        0.00246 |      0.9378 |         0.28 |
|                3 | RGB          | 1             |          10 |        1215 |       15 |             12 |        0.00246 |      0.9378 |         0.37 |
|                3 | RGB          | 1             |          10 |         729 |        9 |             12 |        0.00246 |      0.9378 |         0.39 |
|                1 | YCrCb        | 2             |          10 |         144 |        9 |             16 |        0.00247 |      0.9378 |         0.1  |
|                1 | YCrCb        | 2             |          10 |         384 |        6 |              8 |        0.00248 |      0.9378 |         0.1  |
|                2 | YUV          | 0             |          10 |         576 |        9 |             12 |        0.00248 |      0.9378 |         0.31 |
|                2 | HSV          | 2             |          10 |         432 |       12 |             16 |        0.00249 |      0.9378 |         0.21 |
|                2 | RGB          | 0             |          10 |        1176 |        6 |              8 |        0.00249 |      0.9378 |         0.36 |
|                4 | HSV          | 0             |          10 |        3600 |        9 |              8 |        0.00249 |      0.9378 |         0.51 |
|                2 | RGB          | ALL           |          10 |         648 |        6 |             16 |        0.0025  |      0.9378 |         0.35 |
|                4 | RGB          | ALL           |          10 |        1728 |        9 |             12 |        0.0025  |      0.9378 |         1    |
|                1 | HSV          | 2             |          10 |         240 |       15 |             16 |        0.00251 |      0.9378 |         0.14 |
|                3 | RGB          | 2             |          10 |         432 |       12 |             16 |        0.00251 |      0.9378 |         0.22 |
|                3 | YCrCb        | 0             |          10 |         729 |        9 |             12 |        0.00251 |      0.9378 |         0.44 |
|                2 | RGB          | 0             |          10 |         324 |        9 |             16 |        0.00253 |      0.9378 |         0.21 |
|                3 | YUV          | 0             |          10 |         432 |       12 |             16 |        0.00261 |      0.9378 |         0.21 |
|                4 | HLS          | 0             |          10 |        2400 |        6 |              8 |        0.00263 |      0.9378 |         0.31 |
|                2 | RGB          | 0             |          10 |         960 |       15 |             12 |        0.00271 |      0.9378 |         0.34 |
|                4 | YUV          | 0             |          10 |         384 |        6 |             12 |        0.00274 |      0.9378 |         0.23 |
|                2 | RGB          | 0             |          10 |        5400 |        6 |              4 |        0.0039  |      0.9378 |         0.54 |
|                3 | YUV          | 0             |          10 |         972 |       12 |             12 |        0.0042  |      0.9378 |         0.38 |
|                2 | HLS          | 0             |          10 |         384 |        6 |             12 |        0.00441 |      0.9378 |         0.13 |
|                3 | LUV          | 0             |          10 |         432 |       12 |             16 |        0.00504 |      0.9378 |         0.21 |
|                3 | HSV          | 2             |          10 |         432 |       12 |             16 |        0.00508 |      0.9378 |         0.21 |
|                2 | RGB          | 0             |          10 |         216 |        6 |             16 |        0.00155 |      0.9356 |         0.19 |
|                1 | RGB          | 0             |          10 |          96 |        6 |             16 |        0.00177 |      0.9356 |         0.19 |
|                2 | RGB          | 0             |          10 |        1764 |        9 |              8 |        0.0018  |      0.9356 |         0.24 |
|                1 | YUV          | 0             |          10 |         375 |       15 |             12 |        0.00195 |      0.9356 |         0.19 |
|                1 | YCrCb        | 2             |          10 |         225 |        9 |             12 |        0.00199 |      0.9356 |         0.11 |
|                2 | HLS          | 2             |          10 |        5400 |        6 |              4 |        0.002   |      0.9356 |         0.63 |
|                4 | HSV          | 2             |          10 |         192 |       12 |             16 |        0.00204 |      0.9356 |         0.11 |
|                4 | YUV          | 2             |          10 |         384 |        6 |             12 |        0.00206 |      0.9356 |         0.15 |
|                4 | YUV          | 0             |          10 |         240 |       15 |             16 |        0.00232 |      0.9356 |         0.14 |
|                4 | YUV          | 0             |          10 |         576 |        9 |             12 |        0.00235 |      0.9356 |         0.31 |
|                4 | RGB          | 0             |          10 |         144 |        9 |             16 |        0.00237 |      0.9356 |         0.18 |
|                4 | RGB          | 1             |          10 |         240 |       15 |             16 |        0.00239 |      0.9356 |         0.14 |
|                1 | YUV          | 0             |          10 |         300 |       12 |             12 |        0.00243 |      0.9356 |         0.12 |
|                1 | RGB          | 1             |          10 |         240 |       15 |             16 |        0.00243 |      0.9356 |         0.13 |
|                1 | LUV          | 0             |          10 |         384 |        6 |              8 |        0.00245 |      0.9356 |         0.07 |
|                3 | RGB          | 2             |          10 |         324 |        9 |             16 |        0.00245 |      0.9356 |         0.18 |
|                1 | RGB          | 1             |          10 |        1536 |        6 |              4 |        0.00246 |      0.9356 |         0.11 |
|                1 | RGB          | 1             |          10 |         384 |        6 |              8 |        0.00246 |      0.9356 |         0.15 |
|                1 | HLS          | 1             |          10 |         144 |        9 |             16 |        0.00247 |      0.9356 |         0.13 |
|                3 | RGB          | 0             |          10 |         432 |       12 |             16 |        0.00247 |      0.9356 |         0.26 |
|                2 | HSV          | 1             |          10 |         540 |       15 |             16 |        0.00247 |      0.9356 |         0.34 |
|                3 | HLS          | 2             |          10 |        3888 |       12 |              8 |        0.00247 |      0.9356 |         0.61 |
|                2 | YUV          | 0             |          10 |         432 |       12 |             16 |        0.00248 |      0.9356 |         0.2  |
|                1 | RGB          | 1             |          10 |          96 |        6 |             16 |        0.00249 |      0.9356 |         0.12 |
|                1 | RGB          | 1             |          10 |         192 |       12 |             16 |        0.0025  |      0.9356 |         0.09 |
|                2 | YCrCb        | 2             |          10 |        1764 |        9 |              8 |        0.00251 |      0.9356 |         0.14 |
|                3 | RGB          | ALL           |          10 |        5832 |        6 |              8 |        0.00251 |      0.9356 |         0.83 |
|                4 | YCrCb        | 0             |          10 |        3600 |        9 |              8 |        0.00252 |      0.9356 |         0.68 |
|                4 | HSV          | 1             |          10 |        6000 |       15 |              8 |        0.00255 |      0.9356 |         0.99 |
|                4 | HLS          | 1             |          10 |         192 |       12 |             16 |        0.00259 |      0.9356 |         0.11 |
|                3 | HSV          | 2             |          10 |         324 |        9 |             16 |        0.00263 |      0.9356 |         0.21 |
|                1 | LUV          | 0             |          10 |         240 |       15 |             16 |        0.00273 |      0.9356 |         0.13 |
|                3 | HSV          | 1             |          10 |       15876 |        9 |              4 |        0.00273 |      0.9356 |         1.6  |
|                4 | HSV          | 1             |          10 |       32448 |       12 |              4 |        0.00306 |      0.9356 |         3.17 |
|                4 | HSV          | 1             |          10 |       40560 |       15 |              4 |        0.00423 |      0.9356 |         4.19 |
|                1 | HSV          | 2             |          10 |         375 |       15 |             12 |        0.00427 |      0.9356 |         0.18 |
|                3 | RGB          | 0             |          10 |        1215 |       15 |             12 |        0.00104 |      0.9333 |         0.9  |
|                2 | RGB          | 2             |          10 |         768 |       12 |             12 |        0.00168 |      0.9333 |         0.32 |
|                3 | LUV          | 2             |          10 |         216 |        6 |             16 |        0.0018  |      0.9333 |         0.1  |
|                1 | YCrCb        | 0             |          10 |         144 |        9 |             16 |        0.00193 |      0.9333 |         0.14 |
|                2 | YUV          | 2             |          10 |         216 |        6 |             16 |        0.00196 |      0.9333 |         0.09 |
|                1 | RGB          | 2             |          10 |         384 |        6 |              8 |        0.002   |      0.9333 |         0.13 |
|                1 | RGB          | 0             |          10 |         192 |       12 |             16 |        0.00221 |      0.9333 |         0.17 |
|                3 | HSV          | 1             |          10 |       21168 |       12 |              4 |        0.00231 |      0.9333 |         2.06 |
|                3 | HSV          | 2             |          10 |         486 |        6 |             12 |        0.00242 |      0.9333 |         0.29 |
|                2 | HSV          | 2             |          10 |         216 |        6 |             16 |        0.00244 |      0.9333 |         0.16 |
|                4 | RGB          | 2             |          10 |        2400 |        6 |              8 |        0.00245 |      0.9333 |         0.89 |
|                2 | LUV          | 0             |          10 |         384 |        6 |             12 |        0.00247 |      0.9333 |         0.2  |
|                1 | RGB          | 1             |          10 |         144 |        9 |             16 |        0.00248 |      0.9333 |         0.13 |
|                3 | RGB          | 2             |          10 |         540 |       15 |             16 |        0.00248 |      0.9333 |         0.28 |
|                2 | RGB          | 0             |          10 |         768 |       12 |             12 |        0.00249 |      0.9333 |         0.34 |
|                1 | YCrCb        | 0             |          10 |         150 |        6 |             12 |        0.0025  |      0.9333 |         0.13 |
|                2 | HSV          | 2             |          10 |         576 |        9 |             12 |        0.00251 |      0.9333 |         0.23 |
|                4 | HSV          | 0             |          10 |         384 |        6 |             12 |        0.00252 |      0.9333 |         0.19 |
|                2 | YCrCb        | 0             |          10 |         432 |       12 |             16 |        0.00253 |      0.9333 |         0.22 |
|                4 | RGB          | ALL           |          10 |        1152 |        6 |             12 |        0.00253 |      0.9333 |         0.7  |
|                4 | LUV          | 0             |          10 |         240 |       15 |             16 |        0.00254 |      0.9333 |         0.13 |
|                4 | HSV          | 1             |          10 |         192 |       12 |             16 |        0.00257 |      0.9333 |         0.15 |
|                4 | HSV          | 2             |          10 |        4800 |       12 |              8 |        0.00264 |      0.9333 |         0.89 |
|                1 | LUV          | 0             |          10 |         150 |        6 |             12 |        0.00265 |      0.9333 |         0.13 |
|                2 | YUV          | 0             |          10 |         540 |       15 |             16 |        0.00265 |      0.9333 |         0.25 |
|                1 | YUV          | 0             |          10 |         960 |       15 |              8 |        0.00269 |      0.9333 |         0.08 |
|                1 | HSV          | 1             |          10 |        3072 |       12 |              4 |        0.00269 |      0.9333 |         0.23 |
|                4 | YCrCb        | 0             |          10 |         576 |        9 |             12 |        0.00269 |      0.9333 |         0.32 |
|                3 | HSV          | 1             |          10 |       10584 |        6 |              4 |        0.00269 |      0.9333 |         1.17 |
|                3 | HLS          | 2             |          10 |       15876 |        9 |              4 |        0.00277 |      0.9333 |         1.84 |
|                1 | HLS          | 1             |          10 |         150 |        6 |             12 |        0.00282 |      0.9333 |         0.14 |
|                1 | HLS          | 2             |          10 |        3840 |       15 |              4 |        0.00604 |      0.9333 |         0.28 |
|                1 | HSV          | 2             |          10 |         300 |       12 |             12 |        0.00177 |      0.9311 |         0.18 |
|                4 | YUV          | 0             |          10 |         768 |       12 |             12 |        0.0019  |      0.9311 |         0.52 |
|                2 | HLS          | 1             |          10 |         324 |        9 |             16 |        0.00193 |      0.9311 |         0.18 |
|                1 | HSV          | 2             |          10 |         150 |        6 |             12 |        0.00194 |      0.9311 |         0.17 |
|                2 | HLS          | 1             |          10 |         432 |       12 |             16 |        0.00194 |      0.9311 |         0.21 |
|                2 | HSV          | 1             |          10 |        1764 |        9 |              8 |        0.00195 |      0.9311 |         0.31 |
|                1 | HSV          | 1             |          10 |        3840 |       15 |              4 |        0.00197 |      0.9311 |         0.31 |
|                4 | YCrCb        | 0             |          10 |         768 |       12 |             12 |        0.00205 |      0.9311 |         0.45 |
|                3 | YCrCb        | 0             |          10 |         486 |        6 |             12 |        0.00211 |      0.9311 |         0.27 |
|                4 | YUV          | 0             |          10 |         960 |       15 |             12 |        0.00215 |      0.9311 |         0.48 |
|                4 | HSV          | 1             |          10 |       24336 |        9 |              4 |        0.00238 |      0.9311 |         2.9  |
|                1 | YUV          | 2             |          10 |         150 |        6 |             12 |        0.0024  |      0.9311 |         0.06 |
|                3 | HSV          | 2             |          10 |        1215 |       15 |             12 |        0.00241 |      0.9311 |         0.47 |
|                4 | LUV          | 0             |          10 |         192 |       12 |             16 |        0.00243 |      0.9311 |         0.1  |
|                4 | RGB          | 2             |          10 |         576 |        9 |             12 |        0.00243 |      0.9311 |         0.36 |
|                4 | YUV          | 0             |          10 |          96 |        6 |             16 |        0.00244 |      0.9311 |         0.13 |
|                4 | HSV          | 2             |          10 |         240 |       15 |             16 |        0.00244 |      0.9311 |         0.15 |
|                1 | HSV          | 2             |          10 |         384 |        6 |              8 |        0.00245 |      0.9311 |         0.17 |
|                1 | YCrCb        | 0             |          10 |         384 |        6 |              8 |        0.00246 |      0.9311 |         0.13 |
|                4 | YCrCb        | 0             |          10 |        2400 |        6 |              8 |        0.00246 |      0.9311 |         0.59 |
|                1 | HLS          | 1             |          10 |         240 |       15 |             16 |        0.00247 |      0.9311 |         0.13 |
|                1 | HSV          | 2             |          10 |          96 |        6 |             16 |        0.00247 |      0.9311 |         0.15 |
|                3 | HSV          | 0             |          10 |         486 |        6 |             12 |        0.00247 |      0.9311 |         0.21 |
|                2 | HSV          | 1             |          10 |        8100 |        9 |              4 |        0.00247 |      0.9311 |         0.76 |
|                4 | RGB          | 1             |          10 |         576 |        9 |             12 |        0.00248 |      0.9311 |         0.32 |
|                1 | HSV          | 0             |          10 |         225 |        9 |             12 |        0.00249 |      0.9311 |         0.11 |
|                3 | RGB          | 1             |          10 |         432 |       12 |             16 |        0.00249 |      0.9311 |         0.2  |
|                2 | RGB          | 2             |          10 |         576 |        9 |             12 |        0.00249 |      0.9311 |         0.31 |
|                1 | LUV          | 1             |          10 |         384 |        6 |              8 |        0.0025  |      0.9311 |         0.07 |
|                2 | HSV          | 1             |          10 |         768 |       12 |             12 |        0.0025  |      0.9311 |         0.43 |
|                4 | RGB          | 2             |          10 |         384 |        6 |             12 |        0.00251 |      0.9311 |         0.29 |
|                1 | LUV          | 2             |          10 |         225 |        9 |             12 |        0.00257 |      0.9311 |         0.11 |
|                2 | RGB          | 1             |          10 |         324 |        9 |             16 |        0.00268 |      0.9311 |         0.16 |
|                2 | YCrCb        | 0             |          10 |         324 |        9 |             16 |        0.00315 |      0.9311 |         0.18 |
|                4 | RGB          | 2             |          10 |         768 |       12 |             12 |        0.00429 |      0.9311 |         0.6  |
|                3 | HSV          | 2             |          10 |         540 |       15 |             16 |        0.00439 |      0.9311 |         0.33 |
|                1 | HLS          | 1             |          10 |         300 |       12 |             12 |        0.00177 |      0.9289 |         0.17 |
|                2 | LUV          | 2             |          10 |         384 |        6 |             12 |        0.00194 |      0.9289 |         0.11 |
|                4 | HLS          | 0             |          10 |         192 |       12 |             16 |        0.00205 |      0.9289 |         0.09 |
|                1 | HLS          | 1             |          10 |         225 |        9 |             12 |        0.00208 |      0.9289 |         0.13 |
|                3 | HLS          | 2             |          10 |       10584 |        6 |              4 |        0.00214 |      0.9289 |         1.36 |
|                4 | HLS          | 1             |          10 |         144 |        9 |             16 |        0.0023  |      0.9289 |         0.12 |
|                2 | RGB          | 2             |          10 |         324 |        9 |             16 |        0.00241 |      0.9289 |         0.18 |
|                4 | RGB          | 2             |          10 |          96 |        6 |             16 |        0.00242 |      0.9289 |         0.13 |
|                4 | HLS          | 1             |          10 |         576 |        9 |             12 |        0.00244 |      0.9289 |         0.3  |
|                1 | LUV          | 0             |          10 |         192 |       12 |             16 |        0.00247 |      0.9289 |         0.1  |
|                2 | HSV          | 1             |          10 |         216 |        6 |             16 |        0.00247 |      0.9289 |         0.23 |
|                2 | HLS          | 1             |          10 |         576 |        9 |             12 |        0.00247 |      0.9289 |         0.32 |
|                3 | HSV          | 1             |          10 |        3888 |       12 |              8 |        0.00248 |      0.9289 |         0.58 |
|                4 | HSV          | 2             |          10 |         576 |        9 |             12 |        0.00249 |      0.9289 |         0.33 |
|                3 | YCrCb        | 0             |          10 |         324 |        9 |             16 |        0.0025  |      0.9289 |         0.18 |
|                1 | RGB          | 0             |          10 |         960 |       15 |              8 |        0.00257 |      0.9289 |         0.1  |
|                2 | RGB          | 2             |          10 |         960 |       15 |             12 |        0.00267 |      0.9289 |         0.21 |
|                2 | HSV          | 1             |          10 |       13500 |       15 |              4 |        0.0028  |      0.9289 |         1.32 |
|                3 | YCrCb        | 2             |          10 |         324 |        9 |             16 |        0.00416 |      0.9289 |         0.14 |
|                4 | YCrCb        | 0             |          10 |          96 |        6 |             16 |        0.0043  |      0.9289 |         0.13 |
|                3 | RGB          | 0             |          10 |         972 |       12 |             12 |        0.00156 |      0.9267 |         0.63 |
|                4 | HLS          | 1             |          10 |         960 |       15 |             12 |        0.00205 |      0.9267 |         0.49 |
|                1 | YCrCb        | 0             |          10 |         192 |       12 |             16 |        0.00237 |      0.9267 |         0.11 |
|                1 | LUV          | 2             |          10 |         384 |        6 |              8 |        0.00242 |      0.9267 |         0.12 |
|                1 | YUV          | 0             |          10 |         225 |        9 |             12 |        0.00243 |      0.9267 |         0.12 |
|                3 | RGB          | 2             |          10 |        1215 |       15 |             12 |        0.00243 |      0.9267 |         0.61 |
|                4 | HSV          | 2             |          10 |         384 |        6 |             12 |        0.00244 |      0.9267 |         0.27 |
|                1 | HSV          | 2             |          10 |        1536 |        6 |              4 |        0.00245 |      0.9267 |         0.12 |
|                4 | RGB          | 1             |          10 |         144 |        9 |             16 |        0.00246 |      0.9267 |         0.09 |
|                3 | HSV          | 1             |          10 |         729 |        9 |             12 |        0.00246 |      0.9267 |         0.47 |
|                3 | YCrCb        | 0             |          10 |         972 |       12 |             12 |        0.00248 |      0.9267 |         0.48 |
|                2 | LUV          | 2             |          10 |         216 |        6 |             16 |        0.00249 |      0.9267 |         0.1  |
|                2 | HLS          | 2             |          10 |        8100 |        9 |              4 |        0.00249 |      0.9267 |         0.87 |
|                3 | RGB          | 1             |          10 |         486 |        6 |             12 |        0.00252 |      0.9267 |         0.26 |
|                4 | YCrCb        | 0             |          10 |         144 |        9 |             16 |        0.00258 |      0.9267 |         0.12 |
|                2 | YCrCb        | 0             |          10 |         384 |        6 |             12 |        0.00259 |      0.9267 |         0.22 |
|                4 | YCrCb        | 0             |          10 |         192 |       12 |             16 |        0.00267 |      0.9267 |         0.11 |
|                4 | YCrCb        | 2             |          10 |        3600 |        9 |              8 |        0.00271 |      0.9267 |         0.45 |
|                3 | YCrCb        | 0             |          10 |        1215 |       15 |             12 |        0.00272 |      0.9267 |         0.26 |
|                4 | HSV          | 1             |          10 |        4800 |       12 |              8 |        0.00275 |      0.9267 |         0.8  |
|                4 | HLS          | 0             |          10 |         384 |        6 |             12 |        0.00401 |      0.9267 |         0.18 |
|                1 | LUV          | 0             |          10 |         375 |       15 |             12 |        0.00505 |      0.9267 |         0.16 |
|                3 | HLS          | 2             |          10 |         216 |        6 |             16 |        0.00184 |      0.9244 |         0.34 |
|                1 | RGB          | 2             |          10 |         192 |       12 |             16 |        0.00196 |      0.9244 |         0.1  |
|                1 | HLS          | 2             |          10 |        1536 |        6 |              4 |        0.00196 |      0.9244 |         0.14 |
|                1 | HLS          | 2             |          10 |         768 |       12 |              8 |        0.00201 |      0.9244 |         0.1  |
|                3 | HLS          | 1             |          10 |        1215 |       15 |             12 |        0.00223 |      0.9244 |         0.38 |
|                4 | LUV          | 0             |          10 |          96 |        6 |             16 |        0.00237 |      0.9244 |         0.12 |
|                1 | RGB          | ALL           |          10 |         288 |        6 |             16 |        0.00246 |      0.9244 |         0.18 |
|                1 | HSV          | 1             |          10 |         144 |        9 |             16 |        0.00246 |      0.9244 |         0.21 |
|                1 | HSV          | 1             |          10 |         768 |       12 |              8 |        0.00247 |      0.9244 |         0.09 |
|                3 | HSV          | 1             |          10 |         324 |        9 |             16 |        0.00247 |      0.9244 |         0.29 |
|                2 | HLS          | 2             |          10 |         216 |        6 |             16 |        0.00247 |      0.9244 |         0.3  |
|                4 | HSV          | 2             |          10 |        3600 |        9 |              8 |        0.0025  |      0.9244 |         0.63 |
|                4 | RGB          | 1             |          10 |         192 |       12 |             16 |        0.00251 |      0.9244 |         0.11 |
|                3 | RGB          | 2             |          10 |        1944 |        6 |              8 |        0.00252 |      0.9244 |         0.34 |
|                4 | HSV          | 2             |          10 |        2400 |        6 |              8 |        0.00255 |      0.9244 |         0.8  |
|                4 | HLS          | 2             |          10 |       32448 |       12 |              4 |        0.00291 |      0.9244 |         3.24 |
|                4 | HLS          | 0             |          10 |         960 |       15 |             12 |        0.00384 |      0.9244 |         0.16 |
|                4 | HLS          | 2             |          10 |       24336 |        9 |              4 |        0.0043  |      0.9244 |         3.17 |
|                3 | RGB          | 2             |          10 |         972 |       12 |             12 |        0.00198 |      0.9222 |         0.48 |
|                3 | HLS          | 2             |          10 |       21168 |       12 |              4 |        0.00229 |      0.9222 |         1.89 |
|                1 | RGB          | 2             |          10 |          96 |        6 |             16 |        0.00239 |      0.9222 |         0.14 |
|                4 | LUV          | 0             |          10 |         576 |        9 |             12 |        0.00239 |      0.9222 |         0.32 |
|                4 | RGB          | 1             |          10 |         384 |        6 |             12 |        0.00249 |      0.9222 |         0.27 |
|                4 | RGB          | 0             |          10 |         960 |       15 |             12 |        0.00249 |      0.9222 |         0.61 |
|                1 | RGB          | 2             |          10 |         225 |        9 |             12 |        0.0025  |      0.9222 |         0.12 |
|                2 | RGB          | 0             |          10 |         576 |        9 |             12 |        0.0025  |      0.9222 |         0.35 |
|                1 | HLS          | 2             |          10 |         960 |       15 |              8 |        0.00252 |      0.9222 |         0.1  |
|                3 | HLS          | 0             |          10 |         216 |        6 |             16 |        0.00264 |      0.9222 |         0.12 |
|                2 | HLS          | 1             |          10 |         384 |        6 |             12 |        0.00264 |      0.9222 |         0.21 |
|                1 | RGB          | 0             |          10 |         144 |        9 |             16 |        0.00415 |      0.9222 |         0.19 |
|                1 | RGB          | 2             |          10 |         150 |        6 |             12 |        0.00196 |      0.92   |         0.16 |
|                3 | HLS          | 0             |          10 |         729 |        9 |             12 |        0.00238 |      0.92   |         0.28 |
|                2 | LUV          | 2             |          10 |         324 |        9 |             16 |        0.00239 |      0.92   |         0.15 |
|                2 | YUV          | 0             |          10 |         324 |        9 |             16 |        0.00243 |      0.92   |         0.17 |
|                1 | LUV          | 2             |          10 |          96 |        6 |             16 |        0.00245 |      0.92   |         0.1  |
|                2 | RGB          | 0             |          10 |         432 |       12 |             16 |        0.00245 |      0.92   |         0.26 |
|                1 | HSV          | 1             |          10 |        2304 |        9 |              4 |        0.00246 |      0.92   |         0.17 |
|                3 | HSV          | 2             |          10 |         729 |        9 |             12 |        0.00248 |      0.92   |         0.42 |
|                1 | YCrCb        | 0             |          10 |         375 |       15 |             12 |        0.0025  |      0.92   |         0.14 |
|                2 | HLS          | 2             |          10 |        2940 |       15 |              8 |        0.0025  |      0.92   |         0.38 |
|                4 | RGB          | 0             |          10 |         192 |       12 |             16 |        0.00251 |      0.92   |         0.17 |
|                4 | LUV          | 0             |          10 |         384 |        6 |             12 |        0.00251 |      0.92   |         0.24 |
|                3 | HLS          | 1             |          10 |         729 |        9 |             12 |        0.00254 |      0.92   |         0.39 |
|                3 | HSV          | 1             |          10 |         432 |       12 |             16 |        0.00255 |      0.92   |         0.24 |
|                2 | HSV          | 1             |          10 |         432 |       12 |             16 |        0.00262 |      0.92   |         0.24 |
|                1 | HSV          | 2             |          10 |         192 |       12 |             16 |        0.00263 |      0.92   |         0.11 |
|                3 | RGB          | 2             |          10 |         486 |        6 |             12 |        0.00264 |      0.92   |         0.28 |
|                1 | RGB          | 2             |          10 |         375 |       15 |             12 |        0.00267 |      0.92   |         0.16 |
|                3 | HLS          | 0             |          10 |         324 |        9 |             16 |        0.0027  |      0.92   |         0.25 |
|                4 | HSV          | 1             |          10 |          96 |        6 |             16 |        0.00286 |      0.92   |         0.2  |
|                1 | RGB          | 1             |          10 |         225 |        9 |             12 |        0.00182 |      0.9178 |         0.12 |
|                3 | HLS          | 1             |          10 |         216 |        6 |             16 |        0.00234 |      0.9178 |         0.23 |
|                4 | RGB          | 0             |          10 |         768 |       12 |             12 |        0.00237 |      0.9178 |         0.48 |
|                1 | YUV          | 0             |          10 |         150 |        6 |             12 |        0.00241 |      0.9178 |         0.09 |
|                4 | RGB          | 0             |          10 |         576 |        9 |             12 |        0.00242 |      0.9178 |         0.36 |
|                1 | LUV          | 1             |          10 |         150 |        6 |             12 |        0.00245 |      0.9178 |         0.07 |
|                3 | HSV          | 0             |          10 |         729 |        9 |             12 |        0.00254 |      0.9178 |         0.38 |
|                1 | HSV          | 1             |          10 |        1536 |        6 |              4 |        0.00257 |      0.9178 |         0.14 |
|                1 | HLS          | 2             |          10 |         384 |        6 |              8 |        0.00402 |      0.9178 |         0.27 |
|                4 | YCrCb        | 0             |          10 |         240 |       15 |             16 |        0.0018  |      0.9156 |         0.13 |
|                3 | LUV          | 0             |          10 |         216 |        6 |             16 |        0.0018  |      0.9156 |         0.18 |
|                4 | HSV          | 0             |          10 |         144 |        9 |             16 |        0.00184 |      0.9156 |         0.18 |
|                1 | LUV          | 0             |          10 |         225 |        9 |             12 |        0.00189 |      0.9156 |         0.11 |
|                1 | HLS          | 1             |          10 |         192 |       12 |             16 |        0.00193 |      0.9156 |         0.11 |
|                2 | RGB          | 2             |          10 |         384 |        6 |             12 |        0.00194 |      0.9156 |         0.21 |
|                4 | YCrCb        | 0             |          10 |         960 |       15 |             12 |        0.00199 |      0.9156 |         0.36 |
|                1 | YCrCb        | 0             |          10 |         225 |        9 |             12 |        0.00231 |      0.9156 |         0.12 |
|                2 | HLS          | 2             |          10 |        1764 |        9 |              8 |        0.0024  |      0.9156 |         0.48 |
|                1 | YCrCb        | 2             |          10 |         150 |        6 |             12 |        0.00241 |      0.9156 |         0.08 |
|                4 | HSV          | 1             |          10 |         240 |       15 |             16 |        0.00244 |      0.9156 |         0.17 |
|                1 | HSV          | 1             |          10 |         192 |       12 |             16 |        0.00248 |      0.9156 |         0.13 |
|                3 | HSV          | 1             |          10 |        1944 |        6 |              8 |        0.00248 |      0.9156 |         0.58 |
|                1 | HSV          | 1             |          10 |         300 |       12 |             12 |        0.00249 |      0.9156 |         0.19 |
|                3 | RGB          | 0             |          10 |         540 |       15 |             16 |        0.00249 |      0.9156 |         0.31 |
|                4 | HSV          | 2             |          10 |         960 |       15 |             12 |        0.00249 |      0.9156 |         0.51 |
|                1 | HLS          | 2             |          10 |        2304 |        9 |              4 |        0.0028  |      0.9156 |         0.2  |
|                3 | LUV          | 0             |          10 |         486 |        6 |             12 |        0.00315 |      0.9156 |         0.26 |
|                2 | HLS          | 0             |          10 |         324 |        9 |             16 |        0.0043  |      0.9156 |         0.18 |
|                3 | HSV          | 0             |          10 |         324 |        9 |             16 |        0.01446 |      0.9156 |         0.24 |
|                2 | HSV          | 0             |          10 |         324 |        9 |             16 |        0.00201 |      0.9133 |         0.19 |
|                1 | RGB          | 2             |          10 |         240 |       15 |             16 |        0.00233 |      0.9133 |         0.12 |
|                1 | RGB          | 2             |          10 |         300 |       12 |             12 |        0.00243 |      0.9133 |         0.18 |
|                3 | HSV          | 1             |          10 |         216 |        6 |             16 |        0.00244 |      0.9133 |         0.25 |
|                4 | HSV          | 0             |          10 |         576 |        9 |             12 |        0.00248 |      0.9133 |         0.35 |
|                4 | HSV          | 2             |          10 |         768 |       12 |             12 |        0.00252 |      0.9133 |         0.49 |
|                3 | HSV          | 1             |          10 |        2916 |        9 |              8 |        0.00253 |      0.9133 |         0.69 |
|                4 | HLS          | 2             |          10 |          96 |        6 |             16 |        0.00261 |      0.9133 |         0.24 |
|                4 | HLS          | 2             |          10 |       16224 |        6 |              4 |        0.00288 |      0.9133 |         2.37 |
|                3 | RGB          | 2             |          10 |         729 |        9 |             12 |        0.00229 |      0.9111 |         0.41 |
|                2 | HSV          | 1             |          10 |         576 |        9 |             12 |        0.00246 |      0.9111 |         0.37 |
|                4 | HSV          | 1             |          10 |        3600 |        9 |              8 |        0.00251 |      0.9111 |         0.89 |
|                2 | HSV          | 1             |          10 |         324 |        9 |             16 |        0.00252 |      0.9111 |         0.21 |
|                1 | HLS          | 1             |          10 |         384 |        6 |              8 |        0.00266 |      0.9111 |         0.1  |
|                1 | RGB          | 0             |          10 |         576 |        9 |              8 |        0.00268 |      0.9111 |         0.14 |
|                3 | RGB          | 1             |          10 |         972 |       12 |             12 |        0.00432 |      0.9111 |         0.41 |
|                4 | HSV          | 1             |          10 |       16224 |        6 |              4 |        0.00229 |      0.9089 |         2.03 |
|                3 | HLS          | 2             |          10 |        4860 |       15 |              8 |        0.00242 |      0.9089 |         0.88 |
|                4 | LUV          | 0             |          10 |         768 |       12 |             12 |        0.00243 |      0.9089 |         0.42 |
|                3 | RGB          | 0             |          10 |         486 |        6 |             12 |        0.00244 |      0.9089 |         0.36 |
|                3 | HLS          | 2             |          10 |         324 |        9 |             16 |        0.00249 |      0.9089 |         0.42 |
|                4 | HLS          | 0             |          10 |         144 |        9 |             16 |        0.00251 |      0.9089 |         0.16 |
|                1 | HSV          | 1             |          10 |         960 |       15 |              8 |        0.00261 |      0.9089 |         0.13 |
|                4 | RGB          | ALL           |          10 |         432 |        9 |             16 |        0.00272 |      0.9089 |         0.25 |
|                4 | HLS          | 2             |          10 |         192 |       12 |             16 |        0.00198 |      0.9067 |         0.22 |
|                1 | RGB          | 0             |          10 |         150 |        6 |             12 |        0.00202 |      0.9067 |         0.23 |
|                3 | HSV          | 1             |          10 |         972 |       12 |             12 |        0.00209 |      0.9067 |         0.6  |
|                2 | HLS          | 2             |          10 |         768 |       12 |             12 |        0.00249 |      0.9067 |         0.51 |
|                4 | HLS          | 1             |          10 |         384 |        6 |             12 |        0.00259 |      0.9067 |         0.27 |
|                3 | RGB          | 0             |          10 |         729 |        9 |             12 |        0.00191 |      0.9044 |         0.44 |
|                4 | HLS          | 2             |          10 |         384 |        6 |             12 |        0.00194 |      0.9044 |         0.48 |
|                2 | HSV          | 1             |          10 |         384 |        6 |             12 |        0.00198 |      0.9044 |         0.28 |
|                4 | YCrCb        | 0             |          10 |         384 |        6 |             12 |        0.00209 |      0.9044 |         0.27 |
|                2 | HLS          | 2             |          10 |         432 |       12 |             16 |        0.0023  |      0.9044 |         0.3  |
|                2 | HLS          | 2             |          10 |         324 |        9 |             16 |        0.0024  |      0.9044 |         0.36 |
|                4 | RGB          | 2             |          10 |         960 |       15 |             12 |        0.00244 |      0.9044 |         0.57 |
|                2 | HSV          | 1             |          10 |        1176 |        6 |              8 |        0.00246 |      0.9044 |         0.38 |
|                4 | HSV          | 1             |          10 |         576 |        9 |             12 |        0.0025  |      0.9044 |         0.39 |
|                1 | HSV          | 1             |          10 |         150 |        6 |             12 |        0.00269 |      0.9044 |         0.24 |
|                4 | HSV          | 1             |          10 |         384 |        6 |             12 |        0.00251 |      0.9022 |         0.34 |
|                4 | HLS          | 0             |          10 |         576 |        9 |             12 |        0.00253 |      0.9022 |         0.32 |
|                1 | HLS          | 2             |          10 |          96 |        6 |             16 |        0.00281 |      0.9022 |         0.27 |
|                1 | RGB          | 0             |          10 |         375 |       15 |             12 |        0.00299 |      0.9022 |         0.2  |
|                1 | HLS          | 2             |          10 |         144 |        9 |             16 |        0.00195 |      0.9    |         0.24 |
|                4 | RGB          | 2             |          10 |         240 |       15 |             16 |        0.00211 |      0.9    |         0.15 |
|                1 | HLS          | 2             |          10 |         150 |        6 |             12 |        0.00222 |      0.9    |         0.25 |
|                4 | RGB          | 0             |          10 |         384 |        6 |             12 |        0.00244 |      0.9    |         0.39 |
|                2 | YUV          | 0             |          10 |         384 |        6 |             12 |        0.00251 |      0.9    |         0.19 |
|                2 | HLS          | 2             |          10 |        1176 |        6 |              8 |        0.00251 |      0.9    |         0.51 |
|                3 | HLS          | 2             |          10 |        1944 |        6 |              8 |        0.0028  |      0.9    |         0.81 |
|                3 | HSV          | 1             |          10 |        1215 |       15 |             12 |        0.00182 |      0.8978 |         0.75 |
|                4 | HSV          | 1             |          10 |         960 |       15 |             12 |        0.00198 |      0.8978 |         0.7  |
|                3 | HSV          | 1             |          10 |         540 |       15 |             16 |        0.00207 |      0.8978 |         0.34 |
|                2 | HLS          | 2             |          10 |         384 |        6 |             12 |        0.00212 |      0.8978 |         0.37 |
|                1 | HSV          | 1             |          10 |          96 |        6 |             16 |        0.00243 |      0.8978 |         0.18 |
|                4 | HSV          | 1             |          10 |        2400 |        6 |              8 |        0.00246 |      0.8978 |         1.24 |
|                1 | HLS          | 2             |          10 |         192 |       12 |             16 |        0.00264 |      0.8978 |         0.15 |
|                4 | HSV          | 1             |          10 |         768 |       12 |             12 |        0.00267 |      0.8978 |         0.53 |
|                4 | HLS          | 2             |          10 |        2400 |        6 |              8 |        0.00199 |      0.8956 |         1.63 |
|                2 | HSV          | 1             |          10 |         960 |       15 |             12 |        0.00243 |      0.8956 |         0.64 |
|                4 | HLS          | 2             |          10 |         144 |        9 |             16 |        0.00251 |      0.8956 |         0.29 |
|                4 | HLS          | 2             |          10 |         576 |        9 |             12 |        0.00252 |      0.8956 |         0.6  |
|                3 | HLS          | 2             |          10 |         972 |       12 |             12 |        0.00187 |      0.8933 |         0.68 |
|                3 | HLS          | 2             |          10 |        2916 |        9 |              8 |        0.00196 |      0.8933 |         0.76 |
|                3 | HLS          | 2             |          10 |         432 |       12 |             16 |        0.00243 |      0.8933 |         0.3  |
|                4 | HLS          | 2             |          10 |         768 |       12 |             12 |        0.00247 |      0.8933 |         0.54 |
|                2 | RGB          | 0             |          10 |         384 |        6 |             12 |        0.00249 |      0.8933 |         0.25 |
|                4 | HLS          | 2             |          10 |        3600 |        9 |              8 |        0.00249 |      0.8933 |         1.25 |
|                4 | HSV          | 1             |          10 |         144 |        9 |             16 |        0.00254 |      0.8933 |         0.17 |
|                4 | HLS          | 2             |          10 |         240 |       15 |             16 |        0.00266 |      0.8933 |         0.29 |
|                1 | HSV          | 1             |          10 |         240 |       15 |             16 |        0.00212 |      0.8911 |         0.16 |
|                4 | HLS          | 2             |          10 |        4800 |       12 |              8 |        0.00237 |      0.8911 |         0.85 |
|                1 | HSV          | 1             |          10 |         384 |        6 |              8 |        0.00247 |      0.8911 |         0.25 |
|                4 | HLS          | 2             |          10 |        6000 |       15 |              8 |        0.00247 |      0.8911 |         1.12 |
|                1 | HSV          | 1             |          10 |         225 |        9 |             12 |        0.00242 |      0.8889 |         0.15 |
|                1 | HLS          | 2             |          10 |         240 |       15 |             16 |        0.00242 |      0.8889 |         0.21 |
|                4 | RGB          | 0             |          10 |         240 |       15 |             16 |        0.00246 |      0.8889 |         0.15 |
|                3 | HLS          | 2             |          10 |         540 |       15 |             16 |        0.00249 |      0.8889 |         0.44 |
|                1 | HLS          | 2             |          10 |         300 |       12 |             12 |        0.00312 |      0.8889 |         0.18 |
|                2 | HLS          | 2             |          10 |         540 |       15 |             16 |        0.00245 |      0.8844 |         0.38 |
|                1 | HLS          | 2             |          10 |         225 |        9 |             12 |        0.00263 |      0.8822 |         0.29 |
|                3 | HLS          | 2             |          10 |         486 |        6 |             12 |        0.00727 |      0.8822 |         0.49 |
|                1 | RGB          | 0             |          10 |         300 |       12 |             12 |        0.00223 |      0.88   |         0.19 |
|                1 | HSV          | 2             |          10 |         225 |        9 |             12 |        0.0025  |      0.88   |         0.13 |
|                1 | HLS          | 2             |          10 |         375 |       15 |             12 |        0.00251 |      0.88   |         0.28 |
|                3 | HLS          | 2             |          10 |        1215 |       15 |             12 |        0.00225 |      0.8778 |         0.98 |
|                2 | HLS          | 2             |          10 |         576 |        9 |             12 |        0.00251 |      0.8778 |         0.47 |
|                1 | RGB          | 0             |          10 |         384 |        6 |              8 |        0.00243 |      0.8733 |         0.21 |
|                1 | HSV          | 1             |          10 |         375 |       15 |             12 |        0.00268 |      0.8733 |         0.26 |
|                1 | HSV          | 1             |          10 |         576 |        9 |              8 |        0.00258 |      0.8711 |         0.12 |
|                3 | HLS          | 2             |          10 |         729 |        9 |             12 |        0.00247 |      0.8689 |         0.62 |
|                2 | HLS          | 2             |          10 |         960 |       15 |             12 |        0.00251 |      0.8689 |         0.7  |
|                4 | HLS          | 2             |          10 |         960 |       15 |             12 |        0.00264 |      0.8689 |         0.76 |
|                1 | RGB          | 0             |          10 |         240 |       15 |             16 |        0.00227 |      0.8667 |         0.17 |
|                1 | HLS          | 2             |          10 |         576 |        9 |              8 |        0.00237 |      0.8622 |         0.2  |
|                1 | RGB          | 0             |          10 |         225 |        9 |             12 |        0.00244 |      0.8622 |         0.14 |

